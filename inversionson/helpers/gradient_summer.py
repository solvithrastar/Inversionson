import inspect
import os
import toml
from typing import Union, List

from salvus.flow.api import get_site
from inversionson.utils import sum_two_parameters_h5

SUM_GRADIENTS_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "remote_scripts",
    "gradient_summing.py",
)


class GradientSummer(object):
    """
    This class helps with summing gradients.
    Currently only implemented for remote summing.

    #TODO: Add option for local summing (not needed at the moment)
    """

    def __init__(self, comm):
        """
        :param comm: Inversionson communicator
        """
        self.comm = comm

    def print(
        self,
        message: str,
        color: str = "green",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = ":nerd_face:",
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def sum_gradients(
        self,
        events,
        output_location,
        batch_average=True,
        sum_vpv_vph=True,
        store_norms=True,
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
        iteration = self.comm.project.current_iteration

        for event in self.comm.project.events_in_iteration:
            if self.comm.project.meshes == "multi-mesh":
                job = self.comm.salvus_flow.get_job(event, "gradient_interp")
                gradient_path = os.path.join(
                    str(job.stderr_path.parent), "output/mesh.h5"
                )

            else:
                job = self.comm.salvus_flow.get_job(event, "adjoint")

                output_files = job.get_output_files()
                gradient_path = output_files[0][
                    ("adjoint", "gradient", "output_filename")
                ]
            gradient_paths.append(str(gradient_path))
        # Connect to daint
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_inversionson_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "SUMMING"
        )
        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        remote_output_path = os.path.join(remote_inversionson_dir, "summed_gradient.h5")
        remote_norms_path = os.path.join(
            remote_inversionson_dir, f"{iteration}_gradient_norms.toml"
        )

        # copy summing script to hpc
        remote_script = os.path.join(remote_inversionson_dir, "gradient_summing.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(SUM_GRADIENTS_SCRIPT_PATH, remote_script)

        info = dict(
            filenames=gradient_paths,
            parameters=self.comm.project.inversion_params,
            output_gradient=remote_output_path,
            events_list=events,
            gradient_norms_path=remote_norms_path,
            batch_average=batch_average,
        )

        toml_filename = f"gradient_sum.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # Copy toml to HPC and remove locally
        remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
        hpc_cluster.remote_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Call script
        self.print("Remote summing of gradients started...")
        hpc_cluster.run_ssh_command(f"python {remote_script} {remote_toml}")
        self.print("Remote summing completed...")

        if store_norms:
            doc_path = os.path.join(
                self.comm.project.paths["inversion_root"], "DOCUMENTATION"
            )
            norm_dict_toml = os.path.join(doc_path, f"{iteration}_gradient_norms.toml")

            hpc_cluster.remote_get(remote_norms_path, norm_dict_toml)
            all_norms_path = os.path.join(doc_path, "all_norms.toml")
            if not os.path.exists(all_norms_path):
                norm_dict = {}
                with open(all_norms_path, "w") as fh:
                    toml.dump(norm_dict, fh)
            else:
                norm_dict = toml.load(all_norms_path)

            norm_iter_dict = toml.load(norm_dict_toml)
            for event, norm in norm_iter_dict.items():
                norm_dict[event] = float(norm)

            with open(all_norms_path, "w") as fh:
                toml.dump(norm_dict, fh)

        hpc_cluster.remote_get(remote_output_path, output_location)

        # Only sum the raw gradient in AdamOpt, not the update
        if sum_vpv_vph:
            sum_two_parameters_h5(output_location, ["VPV", "VPH"])

    # def sum_local_gradients(self):
    #     from inversionson.utils import sum_gradients
    #     events = self.comm.project.events_in_iteration
    #     grad_mesh = self.comm.lasif.find_gradient(
    #         iteration=self.comm.project.current_iteration,
    #         event=None,
    #         summed=True,
    #         smooth=False,
    #         just_give_path=True,
    #     )
    #     if os.path.exists(grad_mesh):
    #         print("Gradient has already been summed. Moving on")
    #         return
    #
    #     gradients = []
    #     for event in events:
    #         gradients.append(
    #             self.comm.lasif.find_gradient(
    #                 iteration=self.comm.project.current_iteration,
    #                 event=event,
    #                 summed=False,
    #                 smooth=False,
    #                 just_give_path=False,
    #             )
    #         )
    #     shutil.copy(gradients[0], grad_mesh)
    #     sum_gradients(mesh=grad_mesh, gradients=gradients)
