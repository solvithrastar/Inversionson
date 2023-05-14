from __future__ import annotations
import pathlib
from inversionson import InversionsonError
from salvus.flow.sites import job, remote_io_site  # type: ignore
import salvus.flow.api as sapi  # type: ignore
from .component import Component
import os
import inspect
import shutil
import multi_mesh.api as mapi  # type: ignore
from pathlib import Path

import toml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inversionson.project import Project
from typing import Optional, Union, List

REMOTE_SCRIPT_PATHS = Path(__file__).parent.parent / "remote_scripts"


class MultiMesh(Component):
    """
    Class to deal with tasks related to MultiMesh
    """

    def __init__(self, project: Project):
        super().__init__(project)

    def print(
        self,
        message: str,
        color: Optional[str] = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = None,
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def find_model_file(self, iteration: str):
        """
        Find the mesh which contains the model for this iteration

        :param iteration: Name of iteration
        :type iteration: str
        """
        optimizer = self.project.get_optimizer()
        model = optimizer.model_path
        val_it_itv = self.project.config.monitoring.iterations_between_validation_checks

        if "validation_" in iteration:
            iteration = iteration.replace("validation_", "")
            if (
                val_it_itv > 1
                and iteration != "it0000_model"
                and iteration != "model_00000"
            ):
                it_number = optimizer.iteration_number
                old_it = it_number - val_it_itv + 1
                model = (
                    self.project.salvus_mesher.average_meshes
                    / f"it_{old_it}_to_{it_number}"
                    / "mesh.h5"
                )
        return model

    def prepare_forward(
        self,
        event: str,
    ):
        """
        Interpolate current master model to a simulation mesh.

        :param event: Name of event
        :type event: str
        """
        job = self.construct_remote_interpolation_job(
            event=event,
            gradient=False,
        )
        if job is not None:
            self.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["name"]',
                new_value=job.job_name,
            )
            job.launch()
            self.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["submitted"]',
                new_value=True,
            )
            self.print(
                f"Prepare forward job for event {event} submitted",
                emoji_alias=":white_check_mark:",
            )

    def interpolate_gradient_to_model(self, event: str):
        """
        Interpolate gradient parameters from simulation mesh to master
        dicretisation. In minibatch approach gradients are not summed,
        they are all interpolated to the same discretisation and salvus opt
        deals with them individually.

        :param event: Name of event
        :type event: str
        :param smooth: Whether the smoothed gradient should be used
        :type smooth: bool, optional
        :param interp_folder: Pass a path if you want the matrix of the
        interpolation to be saved and then it can be used later on. Also
        pass this if the directory exists and you want to use the matrices
        """

        self._create_and_launch_remote_interp_job(event)

    def _create_and_launch_remote_interp_job(self, event: str):
        job = self.construct_remote_interpolation_job(
            event=event,
            gradient=True,
        )
        self.project.change_attribute(
            attribute=f'gradient_interp_job["{event}"]["name"]',
            new_value=job.job_name,
        )
        job.launch()
        self.project.change_attribute(
            attribute=f'gradient_interp_job["{event}"]["submitted"]',
            new_value=True,
        )
        self.print(
            f"Interpolation job for event {event} submitted",
            emoji_alias=":white_check_mark:",
        )
        self.project.update_iteration_toml()

    def construct_remote_interpolation_job(self, event: str, gradient: bool = False):
        """
        Construct a custom Salvus job which can be submitted to an HPC cluster
        The job can either do an interpolation of model or gradient

        :param event: Name of event
        :type event: str
        :param gradient: Are we interpolating the gradient?, defaults to False
        :type gradient: bool, optional
        """

        description = "Interpolation of " + ("gradient " if gradient else "model ")
        description += f"for event {event}"

        wall_time = 0.0
        if self.project.config.meshing.multi_mesh:
            wall_time += self.project.config.hpc.model_interp_wall_time

        if not gradient:
            hpc_cluster = self.project.flow.hpc_cluster
            remote_processed_dir = self.project.remote_paths.proc_data_dir
            proc_filename = (
                f"preprocessed_{int(self.project.lasif_settings.min_period)}s"
                f"_to_{int(self.project.lasif_settings.max_period)}s.h5"
            )
            remote_proc_file_name = f"{event}_{proc_filename}"
            remote_proc_path = remote_processed_dir / remote_proc_file_name

            # ALso add a check if the forward_dict exists here
            forward_simulation_dict = Path(
                self.project.lasif.lasif_comm.project.paths["salvus_files"]
                / "SIMULATION_DICTS"
                / event
                / "simulation_dict.toml"
            )
            # Submit a job either if the local dict is missing or
            # if the processed data is missing on the remote
            if (
                not hpc_cluster.remote_exists(remote_proc_path)
                or not forward_simulation_dict.exists()
            ):
                wall_time += self.project.config.hpc.data_proc_wall_time
            elif not self.project.config.meshing.multi_mesh:
                self.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["submitted"]',
                    new_value=True,
                )
                self.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                return None

        if gradient:
            wall_time = self.project.config.hpc.grad_interp_wall_time

        return job.Job(
            site=self.project.flow.hpc_cluster,
            commands=self.get_interp_commands(event=event, gradient=gradient),
            job_type="interpolation",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )

    def prepare_interpolation_toml(self, gradient, event):
        hpc_cluster = self.project.flow.hpc_cluster
        toml_name = "gradient_interp.toml" if gradient else "prepare_forward.toml"
        toml_filename = (
            self.project.inversion_root / "INTERPOLATION" / event / toml_name
        )
        if not os.path.exists(toml_filename.parent):
            os.makedirs(toml_filename.parent)
        tag = "GRADIENTS" if gradient else "MODELS"

        remote_weights_path = os.path.join(
            self.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            tag,
            event,
        )

        information = toml.load(toml_filename) if os.path.exists(toml_filename) else {}
        information["gradient"] = gradient
        information["mesh_info"] = {
            "event_name": event,
            "mesh_folder": str(self.project.fast_mesh_dir),
            "long_term_mesh_folder": str(self.project.remote_mesh_dir),
            "min_period": self.project.lasif_settings.min_period,
            "elems_per_quarter": self.project.elem_per_quarter,
            "interpolation_weights": remote_weights_path,
            "elems_per_wavelength": self.project.elem_per_wavelength,
        }
        information["data_processing"] = bool(
            not gradient and self.project.remote_data_processing
        )
        information["multi-mesh"] = self.project.meshes == "multi-mesh"
        # Provide information for cut and clipping
        if gradient:
            information["cutout_radius_in_km"] = self.project.cut_source_radius
            information["source_location"] = self.project.lasif.get_source(
                event_name=event
            )
            information["clipping_percentile"] = self.project.clip_gradient
            information["parameters"] = self.project.inversion_params
        else:
            proc_filename = f"preprocessed_{int(self.project.lasif_settings.min_period)}s_to_{int(self.project.lasif_settings.max_period)}s.h5"
            remote_proc_path = f"{event}_{proc_filename}"
            remote_processed_dir = self.project.remote_paths.proc_data_dir
            remote_proc_path = remote_processed_dir / remote_proc_path

            processing_info = {
                "minimum_period": self.project.lasif_settings.min_period,
                "maximum_period": self.project.lasif_settings.max_period,
                "npts": self.project.lasif_settings["number_of_time_steps"],
                "dt": self.project.time_step,
                "start_time_in_s": self.project.start_time,
                "asdf_input_filename": "raw_event_data.h5",
                "asdf_output_filename": remote_proc_path,
                "preprocessing_tag": self.project.lasif.lasif_comm.waveforms.preprocessing_tag,
            }
            information["processing_info"] = processing_info

            remote_receiver_dir = self.project.remote_paths.receiver_dir
            information["receiver_json_path"] = os.path.join(
                remote_receiver_dir, f"{event}_receivers.json"
            )

            # If we have a dict already, we can just update it with the proper
            # remote mesh files and also we don't need to create the simulation
            # dict again in the interpolation job.
            local_simulation_dict = (
                (
                    self.project.lasif.lasif_comm.project.paths["salvus_files"]
                    / "SIMULATION_DICTS"
                )
                / event
            ) / "simulation_dict.toml"
            # Only create simulation dict when we don't have it yet.
            information["create_simulation_dict"] = not os.path.exists(
                local_simulation_dict
            )

            if not gradient:
                if self.project.ellipticity:
                    information["ellipticity"] = 0.0033528106647474805
                if self.project.topography["use"]:
                    information["mesh_info"]["topography"] = self.project.topography
                if self.project.ocean_loading["use"]:
                    information["mesh_info"][
                        "ocean_loading"
                    ] = self.project.ocean_loading
                source_info = self.project.lasif.get_source(event_name=event)
                if isinstance(source_info, list):
                    source_info = source_info[0]
                source_info["side_set"] = (
                    "r1_ol"
                    if self.project.ocean_loading["use"]
                    and self.project.meshes != "multi-mesh"
                    else "r1"
                )
                source_info["stf"] = str(
                    self.project.remote_inversionson_dir
                    / "SOURCE_TIME_FUNCTIONS"
                    / self.project.current_iteration
                    / "stf.h5"
                )
                information["source_info"] = source_info

                if (
                    not os.path.exists(toml_filename)
                    and not self.project.remote_data_processing
                ):  # this is a slow step, so let's skip it if we can
                    receivers = self.project.lasif.get_receivers(event_name=event)
                    information["receiver_info"] = receivers
                if self.project.absorbing_boundaries:
                    if (
                        "inner_boundary"
                        in self.project.lasif.lasif_comm.project.domain.get_side_set_names()
                    ):
                        side_sets = ["inner_boundary"]
                    else:
                        side_sets = [
                            "r0",
                            "t0",
                            "t1",
                            "p0",
                            "p1",
                        ]
                else:
                    side_sets = []

                information["simulation_info"] = {
                    "end_time": self.project.end_time,
                    "time_step": self.project.time_step,
                    "start_time": self.project.start_time,
                    "minimum_period": self.project.lasif.lasif_comm.project.simulation_settings[
                        "minimum_period_in_s"
                    ],
                    "simulation_time_step": self.project.simulation_time_step,
                    "attenuation": self.project.attenuation,
                    "absorbing_boundaries": self.project.absorbing_boundaries,
                    "side_sets": side_sets,
                    "absorbing_boundary_length": self.project.abs_bound_length * 1000.0,
                }

        with open(toml_filename, "w") as fh:
            toml.dump(information, fh)
        return toml_filename

    def move_toml_to_hpc(self, toml_filename: pathlib.Path, event: str) -> pathlib.Path:
        """
        Move information file to HPC so that it can perform mesh generation
        and interpolation

        :param toml_filename: path to local toml
        :type toml_filename: pathlib.Path
        :param event: name of event
        :type event: str
        """
        hpc_cluster = self.project.flow.hpc_cluster
        remote_path = self.project.remote_paths.mesh_dir / event / toml_filename.name
        if not hpc_cluster.remote_exists(remote_path.parent):
            hpc_cluster.remote_mkdir(remote_path.parent)
        self.project.flow.safe_put(toml_filename, remote_path)
        return remote_path

    def get_interp_commands(
        self,
        event: str,
        gradient: bool,
    ) -> list:
        """
        Get the interpolation commands needed to do remote interpolations.
        If not gradient, we will look for a smoothie mesh and create it if needed.
        """
        average_model = bool(
            self.project.is_validation_event(event)
            and self.project.config.monitoring.use_model_averaging
            and "00000" not in self.project.current_iteration
        )
        optimizer = self.project.get_optimizer()
        mesh_to_interpolate_from = (
            self.project.lasif.find_remote_mesh(
                event=event,
                gradient=True,
                interpolate_to=False,
                validation=False,
            )
            if gradient
            else optimizer.get_remote_model_path(model_average=average_model)
        )
        hpc_cluster = self.project.flow.hpc_cluster
        interpolation_toml = self.prepare_interpolation_toml(
            gradient=gradient, event=event
        )
        remote_toml = self.move_toml_to_hpc(
            toml_filename=interpolation_toml,
            event=event,
        )

        commands = [
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {remote_toml} ./interp_info.toml",
                execute_with_mpi=False,
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {mesh_to_interpolate_from} ./from_mesh.h5",
                execute_with_mpi=False,
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {self.project.remote_paths.interp_script} ./interpolate.py",
                execute_with_mpi=False,
            ),
            remote_io_site.site_utils.RemoteCommand(
                command="mkdir output", execute_with_mpi=False
            ),
            remote_io_site.site_utils.RemoteCommand(
                command="python interpolate.py ./interp_info.toml",
                execute_with_mpi=False,
            ),
        ]

        if not gradient:
            hpc_cluster = self.project.flow.hpc_cluster
            remote_processed_dir = self.project.remote_paths.proc_data_dir
            proc_filename = f"preprocessed_{int(self.project.lasif_settings.min_period)}s_to_{int(self.project.lasif_settings.max_period)}s.h5"
            remote_proc_file_name = f"{event}_{proc_filename}"
            remote_proc_path = remote_processed_dir / remote_proc_file_name

            if not hpc_cluster.remote_exists(remote_proc_path):
                raw_file = self.project.config.hpc.remote_data_dir / f"{event}.h5"

                copy_data_command = [
                    remote_io_site.site_utils.RemoteCommand(
                        command=f"cp {raw_file} raw_event_data.h5",
                        execute_with_mpi=False,
                    )
                ]
                commands = copy_data_command + commands

        if self.project.config.hpc.conda_env_name:
            conda_command = [
                remote_io_site.site_utils.RemoteCommand(
                    command=f"conda activate {self.project.config.hpc.conda_env_name}",
                    execute_with_mpi=False,
                )
            ]
            commands = conda_command + commands

            if self.project.config.hpc.conda_location:
                source_command = [
                    remote_io_site.site_utils.RemoteCommand(
                        command=f"source {self.project.config.hpc.conda_location}",
                        execute_with_mpi=False,
                    )
                ]
                commands = source_command + commands

        return commands
