from __future__ import annotations
import os
from typing import List, TYPE_CHECKING
from .component import Component

if TYPE_CHECKING:
    from inversionson.project import Project

import salvus.flow.simple_config as sc  # type: ignore
from salvus.opt import smoothing  # type: ignore


class Smoother(Component):
    """
    A class which handles all dealings with the salvus smoother.
    """

    def __init__(self, project: Project):
        super().__init__(project)

    def get_sims_for_smoothing_task(
        self,
        reference_model: str,
        model_to_smooth: str,
        smoothing_lengths: List[float],
        smoothing_parameters: List[str],
    ) -> List[sc.simulation.Diffusion]:
        """
        Writes diffusion models based on a reference model and smoothing
        lengths. Then ploads them to the remote cluster if they don't exist there
        yet.
        and returns a list of simulations that can then be submitted
        as usual.

        The model_to_smooth [a
        Returns a list of simulation objects

        :param reference_model: Mesh file with the velocities on which smoothing lengths are based.
        This file should be locally present.
        :type reference_model: str
        :param model_to_smooth: Mesh file with the fields that require smoothing
        This may either be a file that is currently located on the HPC already
        or a file that stills needs to be uploaded. If it is located
        on the remote already, please pass a path starts with: "Remote:"
        :type model_to_smooth: str
        :param smoothing_lengths: List of floats that specify the smoothing lengths
        :type smoothing_lengths: list
        :param smoothing_parameters: List of strings that specify which parameters need smoothing
        :type smoothing_parameters: list
        """
        ref_model_name = ".".join(reference_model.split("/")[-1].split(".")[:-1])
        freq = 1.0 / self.project.lasif_settings.min_period
        hpc_cluster = self.project.flow.hpc_cluster

        if "REMOTE:" not in model_to_smooth:
            print(
                f"Uploading initial values from: {model_to_smooth} " f"for smoothing."
            )
            file_name = model_to_smooth.split("/")[-1]
            remote_file_path = self.project.remote_paths.diff_dir / file_name
            self.project.flow.safe_put(model_to_smooth, remote_file_path)
            model_to_smooth = f"REMOTE:{remote_file_path}"

        sims = []
        for param in smoothing_parameters:
            if param.startswith("V"):
                reference_velocity = param
            # If it is not some velocity, use P velocities
            elif not param.startswith("V"):
                if "VPV" in self.project.config.inversion.inversion_parameters:
                    reference_velocity = "VPV"
                elif "VP" in self.project.config.inversion.inversion_parameters:
                    reference_velocity = "VP"
                else:
                    raise NotImplementedError(
                        "Inversionson always expects" "to get models with at least VP"
                    )

            unique_id = (
                "_".join([str(i).replace(".", "") for i in smoothing_lengths])
                + "_"
                + str(self.project.lasif_settings.min_period)
            )

            fname = f"{unique_id}_diff_model_{ref_model_name}_{param}.h5"
            remote_diff_model = self.project.remote_paths.diff_dir / fname
            diff_model_path = self.project.paths.diff_model_dir / fname

            if not os.path.exists(diff_model_path):
                smooth = smoothing.AnisotropicModelDependent(
                    reference_frequency_in_hertz=freq,
                    smoothing_lengths_in_wavelengths=smoothing_lengths,
                    reference_model=reference_model,
                    reference_velocity=reference_velocity,
                )
                diff_model = smooth.get_diffusion_model(reference_model)
                diff_model.write_h5(diff_model_path)

            if not hpc_cluster.remote_exists(remote_diff_model):
                self.project.flow.safe_put(diff_model_path, remote_diff_model)

            sim = sc.simulation.Diffusion(mesh=diff_model_path)
            sim.domain.polynomial_order = self.project.tensor_order
            sim.physics.diffusion_equation.courant_number = 0.06

            sim.physics.diffusion_equation.initial_values.filename = model_to_smooth
            sim.physics.diffusion_equation.initial_values.format = "hdf5"
            sim.physics.diffusion_equation.initial_values.field = f"{param}"
            sim.physics.diffusion_equation.final_values.filename = f"{param}.h5"

            sim.domain.mesh.filename = f"REMOTE:{remote_diff_model}"
            sim.domain.model.filename = f"REMOTE:{remote_diff_model}"
            sim.domain.geometry.filename = f"REMOTE:{remote_diff_model}"
            sim.validate()

            # append sim to array
            sims.append(sim)

        return sims
