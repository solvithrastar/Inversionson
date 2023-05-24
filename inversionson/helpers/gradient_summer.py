from __future__ import annotations
import os
import toml
from typing import Optional, Union, List
from pathlib import Path

from inversionson.project import Project
from inversionson.utils import sum_two_parameters_h5, write_xdmf


SUM_GRADIENTS_SCRIPT_PATH = (
    Path(__file__).parent.parent / "remote_scripts" / "gradient_summing.py"
)


class GradientSummer(object):
    """
    This class helps with summing gradients.
    Currently only implemented for remote summing.
    """

    def __init__(self, project: Project):
        """
        :param comm: Inversionson communicator
        """
        self.project = project

    def print(
        self,
        message: str,
        color: str = "green",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = ":nerd_face:",
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def sum_gradients(
        self,
        events: List[str],
        output_location: Union[str, Path],
        batch_average: bool = True,
        sum_vpv_vph: bool = True,
        store_norms: bool = True,
        iteration: Optional[str] = None,
    ):
        """
        Sum gradients on the HPC.

        :param events: List of events to be summed.
        :type events: list
        :param output_location: local file path for the end result
        :type: output_location: bool
        :param batch_average: Average the summed gradients
        :type batch_average: bool
        :param sum_vpv_vph: sum vpv and vph
        :type: sum_vpv_vph: bool
        :param store_norms: Store the gradient norms that are computed while
        summing.
        :type: store_norms: bool
        """
        gradient_paths = []

        iteration = iteration or self.project.current_iteration

        for event in events:
            if self.project.config.meshing.multi_mesh:
                job = self.project.flow.get_job(event, "gradient_interp", iteration)
                gradient_path = os.path.join(
                    str(job.stderr_path.parent), "output/mesh.h5"
                )

            else:
                job = self.project.flow.get_job(event, "adjoint", iteration)

                output_files = job.get_output_files()
                gradient_path = output_files[0][
                    ("adjoint", "gradient", "output_filename")
                ]
            gradient_paths.append(str(gradient_path))
        # Connect to daint
        hpc_cluster = self.project.flow.hpc_cluster

        remote_inversionson_dir = self.project.remote_paths.sum_dir
        remote_output_path = remote_inversionson_dir / "summed_gradient.h5"
        remote_norms_path = remote_inversionson_dir / f"{iteration}_gradient_norms.toml"

        # copy summing script to hpc
        remote_script = remote_inversionson_dir / "gradient_summing.py"
        if not hpc_cluster.remote_exists(remote_script):
            self.project.flow.safe_put(SUM_GRADIENTS_SCRIPT_PATH, remote_script)

        info = dict(
            filenames=gradient_paths,
            parameters=self.project.config.inversion.inversion_parameters,
            output_gradient=str(remote_output_path),
            events_list=events,
            gradient_norms_path=str(remote_norms_path),
            batch_average=batch_average,
        )

        toml_filename = "gradient_sum.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # Copy toml to HPC and remove locally
        remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
        self.project.flow.safe_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Call script
        self.print("Remote summing of gradients started...")
        hpc_cluster.run_ssh_command(f"python {remote_script} {remote_toml}")
        self.print("Remote summing completed...")

        if store_norms:
            self._store_norms(remote_norms_path)
        self.project.flow.safe_get(remote_output_path, output_location)

        # Only sum the raw gradient in AdamOpt, not the update
        if sum_vpv_vph:
            sum_two_parameters_h5(output_location, ["VPV", "VPH"])
        write_xdmf(output_location)

    def _store_norms(self, remote_norms_path: str):
        norm_dict_toml = self.project.paths.gradient_norms_path()

        self.project.flow.safe_get(remote_norms_path, norm_dict_toml)
        all_norms_path = self.project.paths.all_gradient_norms_toml

        norm_dict = toml.load(all_norms_path) if os.path.exists(all_norms_path) else {}
        norm_iter_dict = toml.load(norm_dict_toml)
        for event, norm in norm_iter_dict.items():
            norm_dict[event] = float(norm)

        with open(all_norms_path, "w") as fh:
            toml.dump(norm_dict, fh)
