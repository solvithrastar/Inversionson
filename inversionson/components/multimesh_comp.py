from inversionson import InversionsonError
from salvus.flow.sites import job, remote_io_site
import salvus.flow.api as sapi
from lasif.components.component import Component
import os
import inspect
import shutil
import multi_mesh.api as mapi
import lasif.api as lapi
from salvus.flow.api import get_site
import pathlib
import toml
from typing import Union, List

REMOTE_SCRIPT_PATHS = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "remote_scripts",
)


class MultiMeshComponent(Component):
    """
    Communication with Lasif
    """

    def __init__(self, communicator, component_name):
        super(MultiMeshComponent, self).__init__(communicator, component_name)

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
    ):
        self.comm.storyteller.printer.print(
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
        optimizer = self.comm.project.get_optimizer()
        model = optimizer.model_path

        if "validation_" in iteration:
            iteration = iteration.replace("validation_", "")
            if (
                self.comm.project.when_to_validate > 1
                and iteration != "it0000_model"
                and iteration != "model_00000"
            ):
                it_number = optimizer.iteration_number
                old_it = it_number - self.comm.project.when_to_validate + 1
                model = (
                    self.comm.salvus_mesher.average_meshes
                    / f"it_{old_it}_to_{it_number}"
                    / "mesh.h5"
                )
        return model

    def add_fields_for_interpolation_to_mesh(self, gradient=False):
        """
        In order to do a layered interpolation, we need some fields to be
        present in the model.

        :param gradient: We preparing for gradient interpolation?
            defaults to False
        :type gradient: bool, optional
        """
        iteration = self.comm.project.current_iteration
        if gradient:
            raise InversionsonError("Not yet implemented")
        else:
            model = self.find_model_file(iteration)
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="layer",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="fluid",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="moho_idx",
                global_string=True,
                overwrite=False,
            )

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
            self.comm.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["name"]',
                new_value=job.job_name,
            )
            job.launch()
            self.comm.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["submitted"]',
                new_value=True,
            )
            self.print(
                f"Prepare forward job for event {event} submitted",
                emoji_alias=":white_check_mark:",
            )


    def interpolate_gradient_to_model(
        self, event: str, smooth=True, interp_folder=None
    ):
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
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.interpolation_mode
        if mode == "remote":
            job = self.construct_remote_interpolation_job(
                event=event,
                gradient=True,
            )
            self.comm.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["name"]',
                new_value=job.job_name,
            )
            job.launch()
            self.comm.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["submitted"]',
                new_value=True,
            )
            self.print(
                f"Interpolation job for event {event} submitted",
                emoji_alias=":white_check_mark:",
            )
            self.comm.project.update_iteration_toml()
        else:
            gradient = self.comm.lasif.find_gradient(iteration, event, smooth=smooth)
            simulation_mesh = self.comm.lasif.get_simulation_mesh(event_name=event)

            master_model = self.comm.lasif.get_master_model()

            master_disc_gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=True,
                just_give_path=True,
            )
            shutil.copy(master_model, master_disc_gradient)
            self.comm.salvus_mesher.fill_inversion_params_with_zeroes(
                mesh=master_disc_gradient
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=simulation_mesh,
                to_mesh=gradient,
                field_name="layer",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=simulation_mesh,
                to_mesh=gradient,
                field_name="fluid",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=master_model,
                to_mesh=gradient,
                field_name="moho_idx",
                global_string=True,
                overwrite=False,
            )
            # Dangerous here when we copy something and it maintains the values from before.
            # Make sure that the core values are not fixed there
            mapi.gll_2_gll_layered(
                from_gll=gradient,
                to_gll=master_disc_gradient,
                nelem_to_search=20,
                layers="nocore",
                parameters=self.comm.project.inversion_params,
                stored_array=interp_folder,
            )
            self.comm.salvus_mesher.write_xdmf(master_disc_gradient)

    def construct_remote_interpolation_job(self, event: str, gradient=False):
        """
        Construct a custom Salvus job which can be submitted to an HPC cluster
        The job can either do an interpolation of model or gradient

        :param event: Name of event
        :type event: str
        :param gradient: Are we interpolating the gradient?, defaults to False
        :type gradient: bool, optional
        """

        description = "Interpolation of "
        description += "gradient " if gradient else "model "
        description += f"for event {event}"

        wall_time = 0.0
        if self.comm.project.meshes == "multi-mesh":
            wall_time += self.comm.project.model_interp_wall_time

        # Add wall time when the data needs to be processed, this way
        # we can get through the queue faster for jobs that were finished already.
        if self.comm.project.remote_data_processing:
            hpc_cluster = get_site(self.comm.project.site_name)
            remote_processed_dir = os.path.join(
                self.comm.project.remote_inversionson_dir, "PROCESSED_DATA"
            )
            proc_filename = (
                f"preprocessed_{int(self.comm.project.min_period)}s"
                f"_to_{int(self.comm.project.max_period)}s.h5"
            )
            remote_proc_file_name = f"{event}_{proc_filename}"
            remote_proc_path = os.path.join(remote_processed_dir, remote_proc_file_name)

            # ALso add a check if the forward_dict exists here
            forward_simulation_dict = (
                    self.comm.lasif.lasif_comm.project.paths["salvus_files"]
                    / f"SIMULATION_DICTS"
                    / event
                    / "simulation_dict.toml"
            )
            # Submit a job either if the local dict is missing or
            # if the processed data is missing on the remote
            if not hpc_cluster.remote_exists(remote_proc_path) \
                    or not os.path.exists(forward_simulation_dict):
                wall_time += self.comm.project.remote_data_proc_wall_time
            else:
                self.comm.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["submitted"]',
                    new_value=True,
                )
                self.comm.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                return None

        if gradient:
            wall_time = self.comm.project.grad_interp_wall_time

        int_job = job.Job(
            site=sapi.get_site(self.comm.project.interpolation_site),
            commands=self.get_interp_commands(event=event, gradient=gradient),
            job_type="interpolation",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )
        return int_job

    def prepare_interpolation_toml(self, gradient, event, hpc_cluster=None):
        toml_name = "gradient_interp.toml" if gradient else "prepare_forward.toml"
        toml_filename = (
            self.comm.project.inversion_root / "INTERPOLATION" / event / toml_name
        )
        if not os.path.exists(toml_filename.parent):
            os.makedirs(toml_filename.parent)
        tag = "GRADIENTS" if gradient else "MODELS"

        remote_weights_path = os.path.join(
            self.comm.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            tag,
            event,
        )

        if os.path.exists(
            toml_filename
        ):  # if exists, we can update the important parameters. and skip the rest.
            information = toml.load(toml_filename)
        else:
            information = {}
        information["gradient"] = gradient
        information["mesh_info"] = {
            "event_name": event,
            "mesh_folder": str(self.comm.project.fast_mesh_dir),
            "long_term_mesh_folder": str(self.comm.project.remote_mesh_dir),
            "min_period": self.comm.project.min_period,
            "elems_per_quarter": self.comm.project.elem_per_quarter,
            "interpolation_weights": remote_weights_path,
            "elems_per_wavelength": self.comm.project.elem_per_wavelength,
        }
        if not gradient and self.comm.project.remote_data_processing:
            information["data_processing"] = True
        else:
            information["data_processing"] = False

        if self.comm.project.meshes == "multi-mesh":
            information["multi-mesh"] = True
        else:
            information["multi-mesh"] = False

        # Provide information for cut and clipping
        if gradient:
            information["cutout_radius_in_km"] = self.comm.project.cut_source_radius
            information["source_location"] = self.comm.lasif.get_source(
                event_name=event
            )
            information["clipping_percentile"] = self.comm.project.clip_gradient
            information["parameters"] = self.comm.project.inversion_params
        else:
            proc_filename = f"preprocessed_{int(self.comm.project.min_period)}s_to_{int(self.comm.project.max_period)}s.h5"
            remote_proc_path = f"{event}_{proc_filename}"
            if hpc_cluster is None:
                hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
            remote_processed_dir = os.path.join(
                self.comm.project.remote_inversionson_dir, "PROCESSED_DATA"
            )
            remote_proc_path = os.path.join(remote_processed_dir, remote_proc_path)

            if not hpc_cluster.remote_exists(remote_processed_dir):
                hpc_cluster.remote_mkdir(remote_processed_dir)

            processing_info = {
                "minimum_period": self.comm.project.min_period,
                "maximum_period": self.comm.project.max_period,
                "npts": self.comm.project.simulation_dict["number_of_time_steps"],
                "dt": self.comm.project.time_step,
                "start_time_in_s": self.comm.project.start_time,
                "asdf_input_filename": "raw_event_data.h5",
                "asdf_output_filename": remote_proc_path,
                "preprocessing_tag": self.comm.lasif.lasif_comm.waveforms.preprocessing_tag,
            }
            information["processing_info"] = processing_info

            remote_receiver_dir = os.path.join(
                self.comm.project.remote_inversionson_dir, "RECEIVERS"
            )
            if not hpc_cluster.remote_exists(remote_receiver_dir):
                hpc_cluster.remote_mkdir(remote_receiver_dir)
            information["receiver_json_path"] = os.path.join(
                remote_receiver_dir, f"{event}_receivers.json"
            )

            # If we have a dict already, we can just update it with the proper
            # remote mesh files and also we don't need to create the simulation
            # dict again in the interpolation job.
            local_simulation_dict = (
                self.comm.lasif.lasif_comm.project.paths["salvus_files"]
                / f"SIMULATION_DICTS"
                / event
                / "simulation_dict.toml"
            )
            # Only create simulation dict when we don't have it yet.
            information["create_simulation_dict"] = (
                False if os.path.exists(local_simulation_dict) else True
            )

            if not gradient:
                if self.comm.project.ellipticity:
                    information["ellipticity"] = 0.0033528106647474805
                if self.comm.project.topography["use"]:
                    information["mesh_info"][
                        "topography"
                    ] = self.comm.project.topography
                if self.comm.project.ocean_loading["use"]:
                    information["mesh_info"][
                        "ocean_loading"
                    ] = self.comm.project.ocean_loading
                source_info = self.comm.lasif.get_source(event_name=event)
                if isinstance(source_info, list):
                    source_info = source_info[0]
                source_info["side_set"] = (
                    "r1_ol"
                    if self.comm.project.ocean_loading["use"]
                    and not self.comm.project.meshes == "multi-mesh"
                    else "r1"
                )
                source_info["stf"] = str(
                    self.comm.project.remote_inversionson_dir
                    / "SOURCE_TIME_FUNCTIONS"
                    / self.comm.project.current_iteration
                    / "stf.h5"
                )
                information["source_info"] = source_info

                if (
                    not os.path.exists(toml_filename)
                    and not self.comm.project.remote_data_processing
                ):  # this is a slow step, so let's skip it if we can
                    receivers = self.comm.lasif.get_receivers(event_name=event)
                    information["receiver_info"] = receivers
                if self.comm.project.absorbing_boundaries:
                    if (
                        "inner_boundary"
                        in self.comm.lasif.lasif_comm.project.domain.get_side_set_names()
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
                    "end_time": self.comm.project.end_time,
                    "time_step": self.comm.project.time_step,
                    "start_time": self.comm.project.start_time,
                    "minimum_period": self.comm.lasif.lasif_comm.project.simulation_settings[
                        "minimum_period_in_s"
                    ],
                    "attenuation": self.comm.project.attenuation,
                    "absorbing_boundaries": self.comm.project.absorbing_boundaries,
                    "side_sets": side_sets,
                    "absorbing_boundary_length": self.comm.project.abs_bound_length
                    * 1000.0,
                }

        with open(toml_filename, "w") as fh:
            toml.dump(information, fh)
        return toml_filename

    def move_toml_to_hpc(
        self, toml_filename: pathlib.Path, event: str, hpc_cluster=None
    ):
        """
        Move information file to HPC so that it can perform mesh generation
        and interpolation

        :param toml_filename: path to local toml
        :type toml_filename: pathlib.Path
        :param event: name of event
        :type event: str
        :param hpc_cluster: the cluster site object, defaults to None
        :type hpc_cluster: Salvus.site, optional
        """
        if hpc_cluster is None:
            hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
        remote_path = (
            pathlib.Path(self.comm.project.remote_mesh_dir) / event / toml_filename.name
        )
        if not hpc_cluster.remote_exists(remote_path.parent):
            hpc_cluster.remote_mkdir(remote_path.parent)
        hpc_cluster.remote_put(toml_filename, remote_path)
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

        # TODO Add average model option here
        mesh_to_interpolate_from = self.comm.lasif.get_remote_model_path()
        interpolation_script = self.find_interpolation_script()
        hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
        interpolation_toml = self.prepare_interpolation_toml(
            gradient=gradient, event=event, hpc_cluster=hpc_cluster
        )
        remote_toml = self.move_toml_to_hpc(
            toml_filename=interpolation_toml,
            event=event,
            hpc_cluster=hpc_cluster,
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
                command=f"cp {interpolation_script} ./interpolate.py",
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

        if self.comm.project.remote_data_processing and not gradient:
            hpc_cluster = get_site(self.comm.project.site_name)
            remote_processed_dir = os.path.join(
                self.comm.project.remote_inversionson_dir, "PROCESSED_DATA"
            )
            proc_filename = f"preprocessed_{int(self.comm.project.min_period)}s_to_{int(self.comm.project.max_period)}s.h5"
            remote_proc_file_name = f"{event}_{proc_filename}"
            remote_proc_path = os.path.join(remote_processed_dir, remote_proc_file_name)

            if not hpc_cluster.remote_exists(remote_proc_path):
                raw_file = os.path.join(
                    self.comm.project.remote_raw_data_dir, f"{event}.h5"
                )
                copy_data_command = [
                    remote_io_site.site_utils.RemoteCommand(
                        command=f"cp {raw_file} raw_event_data.h5",
                        execute_with_mpi=False,
                    )
                ]
                commands = copy_data_command + commands

        if self.comm.project.remote_conda_env:
            conda_command = [
                remote_io_site.site_utils.RemoteCommand(
                    command=f"conda activate {self.comm.project.remote_conda_env}",
                    execute_with_mpi=False,
                )
            ]
            commands = conda_command + commands

        return commands

    def find_interpolation_script(self) -> str:
        """
        Check to see if remote interpolation script is available.
        If not, create one and put it there
        """
        remote_script_path = os.path.join(
            self.comm.project.remote_inversionson_dir,
            "SCRIPTS",
            "interpolation.py",
        )

        return remote_script_path

    def get_remote_field_moving_script_path(self):
        site = get_site(self.comm.project.interpolation_site)
        username = site.config["ssh_settings"]["username"]

        remote_inversionson_scripts = os.path.join("/users", username, "scripts")

        if not site.remote_exists(remote_inversionson_scripts):
            site.remote_mkdir(remote_inversionson_scripts)

        # copy processing script to daint
        remote_script = os.path.join(remote_inversionson_scripts, "move_fields.py")
        if not site.remote_exists(remote_script):
            site.remote_put(
                os.path.join(REMOTE_SCRIPT_PATHS, "cut_and_clip.py"), remote_script
            )
        return remote_script

    def _make_remote_interpolation_script(self, hpc_cluster):
        """
        Executed if remote interpolation script can not be found
        We see if it exists locally.
        If not, we create it locally and copy to cluster.
        """
        if hpc_cluster.config["site_type"] == "local":
            remote_script_dir = os.path.join(
                self.comm.project.remote_diff_model_dir, "..", "scripts"
            )
        else:
            username = hpc_cluster.config["ssh_settings"]["username"]
            remote_script_dir = os.path.join("/users", username, "scripts")
        local_script = os.path.join(
            self.comm.project.paths["inversion_root"], "interpolation.py"
        )

        if not hpc_cluster.remote_exists(remote_script_dir):
            hpc_cluster.remote_mkdir(remote_script_dir)

        self.print("New interpolation script will be generated")
        if not os.path.exists(local_script):
            interp_script = f"""import multi_mesh.api
fm = "from_mesh.h5"
tm = "to_mesh.h5"
multi_mesh.api.gll_2_gll_layered_multi(
    fm,
    tm,
    nelem_to_search=20,
    layers="nocore",
    parameters={self.comm.project.inversion_params},
    stored_array=".",
)
            """
            with open(local_script, "w+") as fh:
                fh.write(interp_script)

        remote_interp_script = os.path.join(remote_script_dir, "interpolation.py")
        if not hpc_cluster.remote_exists(remote_interp_script):
            hpc_cluster.remote_put(local_script, remote_interp_script)
