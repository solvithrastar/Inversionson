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
from typing import Union
from inversionson.optimizers.adam_optimizer import AdamOptimizer

CUT_SOURCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
    ),
    "remote_scripts",
    "move_fields.py",
)


class MultiMeshComponent(Component):
    """
    Communication with Lasif
    """

    def __init__(self, communicator, component_name):
        super(MultiMeshComponent, self).__init__(communicator, component_name)
        self.physical_models = self.comm.salvus_opt.models

    def find_model_file(self, iteration: str):
        """
        Find the mesh which contains the model for this iteration

        :param iteration: Name of iteration
        :type iteration: str
        """
        model = os.path.join(self.physical_models, iteration + ".h5")
        if "validation_" in iteration:
            iteration = iteration.replace("validation_", "")

            if (
                self.comm.project.when_to_validate > 1
                and iteration != "it0000_model"
            ):
                it_number = (
                    self.comm.salvus_opt.get_number_of_newest_iteration()
                )
                old_it = it_number - self.comm.project.when_to_validate + 1
                model = (
                    self.comm.salvus_mesher.average_meshes
                    / f"it_{old_it}_to_{it_number}"
                    / "mesh.h5"
                )
            else:
                if self.AdamOpt:
                    adam_opt = AdamOptimizer(inversion_root=
                                             self.comm.project.
                                             paths["inversion_root"])
                    model = adam_opt.get_model_path()
                else:
                    model = os.path.join(self.physical_models, iteration + ".h5")
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

    def interpolate_to_simulation_mesh(
        self, event: str, interp_folder=None,
    ):
        """
        Interpolate current master model to a simulation mesh.

        :param event: Name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.interpolation_mode
        if mode == "remote":
            job = self.construct_remote_interpolation_job(
                event=event, gradient=False, interp_folder=interp_folder,
            )
            self.comm.project.change_attribute(
                attribute=f'model_interp_job["{event}"]["name"]',
                new_value=job.job_name,
            )
            job.launch()
            self.comm.project.change_attribute(
                attribute=f'model_interp_job["{event}"]["submitted"]',
                new_value=True,
            )
            print(f"Interpolation job for event {event} submitted")
        else:
            simulation_mesh = lapi.get_simulation_mesh(
                self.comm.lasif.lasif_comm, event, iteration
            )
            model = self.find_model_file(iteration)

            # There are many more knobs to tune but for now lets stick to
            # defaults.
            mapi.gll_2_gll_layered(
                from_gll=model,
                to_gll=simulation_mesh,
                layers="nocore",
                nelem_to_search=20,
                parameters=self.comm.project.modelling_params,
                stored_array=interp_folder,
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
                interp_folder=interp_folder,
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
            print(f"Interpolation job for event {event} submitted")
            self.comm.project.update_iteration_toml()
        else:
            gradient = self.comm.lasif.find_gradient(
                iteration, event, smooth=smooth
            )
            simulation_mesh = self.comm.lasif.get_simulation_mesh(
                event_name=event
            )

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

    def construct_remote_interpolation_job(
        self, event: str, gradient=False, interp_folder=None
    ):
        """
        Construct a custom Salvus job which can be submitted to an HPC cluster
        The job can either do an interpolation of model or gradient

        :param event: Name of event
        :type event: str
        :param gradient: Are we interpolating the gradient?, defaults to False
        :type gradient: bool, optional
        :param interp_folder: A folder to save interpolation weights,
            if interpolation has been done before, these weights can be stored,
            defaults to None
        :type interp_folder: str, optional
        """

        description = "Interpolation of "
        description += "gradient " if gradient else "model "
        description += f"for event {event}"

        wall_time = self.comm.project.model_interp_wall_time
        if gradient:
            wall_time = self.comm.project.grad_interp_wall_time

        int_job = job.Job(
            site=sapi.get_site(self.comm.project.interpolation_site),
            commands=self.get_interp_commands(
                event=event, gradient=gradient, interp_folder=interp_folder
            ),
            job_type="interpolation",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )
        return int_job

    def get_interp_commands(
        self,
        event: str,
        gradient: bool,
        interp_folder: Union[str, pathlib.Path],
    ) -> list:
        """
        Get the interpolation commands needed to do remote interpolations
        """
        iteration = self.comm.project.current_iteration
        if "validation_" in iteration:
            validation = True
        else:
            validation = False
        if iteration == "validation_it0000_model":
            validation = False  # Here there can't be any mesh averaging

        hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
        interp_info_file = pathlib.Path(interp_folder) / "interp_info.h5"
        weights_exists = hpc_cluster.remote_exists(interp_info_file)

        mesh_to_interpolate_to = self.comm.lasif.find_remote_mesh(
            event=event, gradient=gradient, interpolate_to=True,
        )
        mesh_to_interpolate_from = self.comm.lasif.find_remote_mesh(
            event=event,
            gradient=gradient,
            interpolate_to=False,
            validation=validation,
        )
        interpolation_script = self.find_interpolation_script()
        # if gradient:
        #     mesh_to_get_fields_from = str(
        #         self.comm.lasif.find_remote_mesh(
        #             event=event,
        #             iteration=iteration,
        #             already_interpolated=True,
        #         )
        #     )
        #     move_fields_script = self.get_remote_field_moving_script_path()
        commands = [
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {mesh_to_interpolate_from} ./from_mesh.h5",
                execute_with_mpi=False,
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {mesh_to_interpolate_to} ./to_mesh.h5",
                execute_with_mpi=False,
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"cp {interpolation_script} ./interpolate.py",
                execute_with_mpi=False,
            ),
        ]
        # if weights_exists:
        #     commands.append(
        #         remote_io_site.site_utils.RemoteCommand(
        #             command=f"cp {interp_info_file} ./interp_info.h5",
        #             execute_with_mpi=False,
        #         )
        #     )
        # if gradient:
        #     commands.append(
        #         remote_io_site.site_utils.RemoteCommand(
        #             command=f"python {move_fields_script} {mesh_to_get_fields_from} ./from_mesh.h5 layer elemental",
        #             execute_with_mpi=False,
        #         )
        #     )
        #     commands.append(
        #         remote_io_site.site_utils.RemoteCommand(
        #             command=f"python {move_fields_script} {mesh_to_get_fields_from} ./from_mesh.h5 fluid elemental",
        #             execute_with_mpi=False,
        #         )
        #     )
        #     commands.append(
        #         remote_io_site.site_utils.RemoteCommand(
        #             command=f"python {move_fields_script} {mesh_to_get_fields_from} ./from_mesh.h5 z_node_1D nodal",
        #             execute_with_mpi=False,
        #         )
        #     )
        commands.append(
            remote_io_site.site_utils.RemoteCommand(
                command="mkdir output", execute_with_mpi=False
            )
        )
        commands.append(
            remote_io_site.site_utils.RemoteCommand(
                command="which python", execute_with_mpi=False
            )
        )
        commands.append(
            remote_io_site.site_utils.RemoteCommand(
                command="python interpolate.py", execute_with_mpi=False
            )
        )
        commands.append(
            remote_io_site.site_utils.RemoteCommand(
                command="mv ./to_mesh.h5 ./output/mesh.h5",
                execute_with_mpi=False,
            ),
        )
        # if not weights_exists:
        #     commands.append(
        #         remote_io_site.site_utils.RemoteCommand(
        #             command=f"mv ./interp_info.h5 {interp_info_file}",
        #             execute_with_mpi=False,
        #         )
        #     )
        return commands

    def find_interpolation_script(self) -> str:
        """
        Check to see if remote interpolation script is available.
        If not, create one and put it there
        """
        # get_remote
        hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
        username = hpc_cluster.config["ssh_settings"]["username"]

        remote_script_path = os.path.join(
            "/users", username, "scripts", "interpolation.py"
        )
        if not hpc_cluster.remote_exists(remote_script_path):
            self._make_remote_interpolation_script(hpc_cluster)
        return remote_script_path

    def get_remote_field_moving_script_path(self):
        site = get_site(self.comm.project.interpolation_site)
        username = site.config["ssh_settings"]["username"]

        remote_inversionson_scripts = os.path.join(
            "/users", username, "scripts"
        )

        if not site.remote_exists(remote_inversionson_scripts):
            site.remote_mkdir(remote_inversionson_scripts)

        # copy processing script to daint
        remote_script = os.path.join(
            remote_inversionson_scripts, "move_fields.py"
        )
        if not site.remote_exists(remote_script):
            site.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)
        return remote_script

    def _make_remote_interpolation_script(self, hpc_cluster):
        """
        Executed if remote interpolation script can not be found
        We see if it exists locally.
        If not, we create it locally and copy to cluster.
        """

        # get_remote
        username = hpc_cluster.config["ssh_settings"]["username"]

        remote_script_dir = os.path.join("/users", username, "scripts")
        local_script = os.path.join(
            self.comm.project.paths["inversion_root"], "interpolation.py"
        )

        if not hpc_cluster.remote_exists(remote_script_dir):
            hpc_cluster.remote_mkdir(remote_script_dir)

        print("New interpolation script will be generated")
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

        remote_interp_script = os.path.join(
            remote_script_dir, "interpolation.py"
        )
        if not hpc_cluster.remote_exists(remote_interp_script):
            hpc_cluster.remote_put(local_script, remote_interp_script)
