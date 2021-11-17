from .component import Component
from inversionson import InversionsonError

import toml
import os
from salvus.opt import smoothing


class SalvusSmoothComponent(Component):
    """
    A class which handles all dealings with the salvus smoother.
    I will start with fixed parameters
    TODO: Add a regularization toml file which includes dropout rate and smoothing lengths
    """

    def __init__(self, communicator, component_name):
        super(SalvusSmoothComponent, self).__init__(
            communicator, component_name
        )
        # self.smoother_path = self.comm.project.paths["salvus_smoother"]

    def generate_smoothing_config(self, event: str = None) -> dict:
        """
        Generate a dictionary which contains smoothing objects for each
        parameter to be smoothed.

        :param event: Name of event
        :type event: str
        :return: Dictonary which points each parameter to a smoothing object
        :rtype: dict
        """

        # The mesh used as diffusion model is the event_mesh with the 1D model
        diff_model = self.comm.lasif.find_event_mesh(event=event)

        smoothing_config = {}
        freq = 1.0 / self.comm.project.min_period
        smoothing_lengths = self.comm.project.smoothing_lengths

        # Loop through parameters to assign smoothing objects to parameters.
        for param in self.comm.project.inversion_params:
            if param.startswith("V"):
                reference_velocity = param
            elif param == "RHO":
                if "VP" in self.comm.project.inversion_params:
                    reference_velocity = "VP"
                elif "VPV" in self.comm.project.inversion_params:
                    reference_velocity = "VPV"
                else:
                    raise InversionsonError(
                        f"Unexpected case while smoothing {param}. "
                        f"Take a closer look"
                    )
            if self.comm.project.smoothing_mode == "anisotropic":
                smooth = smoothing.AnisotropicModelDependent(
                    reference_frequency_in_hertz=freq,
                    smoothing_lengths_in_wavelengths=smoothing_lengths,
                    reference_model=diff_model,
                    reference_velocity=reference_velocity,
                )
            elif self.comm.project.smoothing_mode == "isotropic":
                smooth = smoothing.IsotropicModelDependent(
                    reference_frequency_in_hertz=freq,
                    smoothing_length_in_wavelengths=smoothing_lengths,
                    reference_model=diff_model,
                    reference_velocity=reference_velocity,
                )
            smoothing_config[param] = smooth

        return smoothing_config

    def retrieve_smooth_gradient(self, event_name: str, iteration=None):
        """
        Retrieve the smoothed gradient from a specific event and iteration.

        :param event_name: Name of event, can be None if mono-batch
        :type event_name: str
        :param iteration: Name of iteration, defaults to None (current)
        :type iteration: str, optional
        """
        from salvus.opt.smoothing import get_smooth_model
        import salvus.flow.api

        if iteration is None:
            iteration = self.comm.project.current_iteration
        job_name = self.comm.salvus_flow.get_job_name(
            event=event_name,
            sim_type="smoothing",
            iteration=iteration,
        )
        salvus_job = salvus.flow.api.get_job_array(
            site_name=self.comm.project.site_name,
            job_array_name=job_name,
        )

        if self.comm.project.inversion_mode == "mono-batch":
            smooth_grad = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=None,
                smooth=True,
                summed=True,
                just_give_path=True,
            )
        elif self.comm.project.meshes == "multi-mesh":
            smooth_grad = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event_name,
                smooth=True,
                inversion_grid=True,
                just_give_path=True,
            )
        else:
            smooth_grad = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event_name,
                smooth=True,
                inversion_grid=False,
                just_give_path=True,
            )

        if not os.path.exists(os.path.dirname(smooth_grad)):
            os.mkdir(os.path.dirname(smooth_grad))

        smooth_gradient = get_smooth_model(
            job=salvus_job,
            model=self.comm.lasif.get_master_model(),
        )
        smooth_gradient.write_h5(smooth_grad)
        if "VPV" in list(smooth_gradient.element_nodal_fields.keys()):
            self.comm.salvus_mesher.sum_two_fields_on_a_mesh(
                mesh=smooth_grad,
                fieldname_1="VPV",
                fieldname_2="VPH",
            )

    def run_smoother(
        self, smoothing_config: dict, event: str, iteration: str = None
    ):
        """
        Run the Smoother, the settings are specified in inversion toml. Make
        sure that the smoothing config has already been generated

        :param smoothing_config: Dictionary with objects for each parameter
        :type smoothing_config: dict
        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration, if None it will give the
            current iteration, defaults to None
        :type iteration: str, optional
        """
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        if iteration is None:
            iteration = self.comm.project.current_iteration

        if self.comm.project.remote_gradient_processing and event is not None:
            job = self.comm.salvus_flow.get_job(event, "adjoint")
            output_files = job.get_output_files()
            grad = output_files[0][("adjoint", "gradient", "output_filename")]
            mesh = UnstructuredMesh.from_h5(str(grad))
        else:
            if self.comm.project.inversion_mode == "mini-batch":
                mesh = UnstructuredMesh.from_h5(
                    self.comm.lasif.find_gradient(
                        iteration=iteration, event=event
                    )
                )
            else:
                mesh = UnstructuredMesh.from_h5(
                    self.comm.lasif.find_gradient(
                        iteration=iteration,
                        summed=True,
                        smooth=False,
                        event=None,
                    )
                )
            mesh.attach_global_variable(
                name="reference_frame", data="spherical"
            )

        job = smoothing.run_async(
            model=mesh,
            smoothing_config=smoothing_config,
            time_step_in_seconds=self.comm.project.smoothing_timestep,
            site_name=self.comm.project.smoothing_site_name,
            ranks_per_job=self.comm.project.smoothing_ranks,
            wall_time_in_seconds_per_job=self.comm.project.smoothing_wall_time,
        )
        if self.comm.project.inversion_mode == "mini-batch":
            self.comm.project.change_attribute(
                f'smoothing_job["{event}"]["name"]', job.job_array_name
            )
            self.comm.project.change_attribute(
                f'smoothing_job["{event}"]["submitted"]', True
            )
        else:
            self.comm.project.change_attribute(
                'smoothing_job["name"]', job.job_array_name
            )
            self.comm.project.change_attribute(
                'smoothing_job["submitted"]', True
            )
        self.comm.project.update_iteration_toml()

    def run_remote_smoother(
        self,
        event: str,
    ):
        """
        Run the Smoother, the settings are specified in inversion toml. Make
        sure that the smoothing config has already been generated

        :param event: Name of event
        :type event: str
        """
        from salvus.opt import smoothing
        import salvus.flow.simple_config as sc
        from salvus.flow.api import get_site
        from salvus.flow import api as sapi

        int_mode = self.comm.project.interpolation_mode
        if self.comm.project.meshes == "multi-mesh":
            mesh = self.comm.lasif.get_master_model()
        else:
            mesh = self.comm.lasif.get_simulation_mesh(event)
        freq = 1.0 / self.comm.project.min_period
        smoothing_lengths = self.comm.project.smoothing_lengths

        # get remote gradient filename
        if int_mode == "remote" and self.comm.project.meshes == "multi-mesh":
            # The gradient we want has been interpolated
            job = self.comm.salvus_flow.get_job(event, "gradient_interp")
            remote_grad = str(job.stdout_path.parent / "output" / "mesh.h5")
        else:
            job = self.comm.salvus_flow.get_job(event, "adjoint")
            output_files = job.get_output_files()
            remote_grad = str(
                output_files[0][("adjoint", "gradient", "output_filename")]
            )

        # make site stuff (hardcoded for now)
        # This needs to be modified by anyone not using daint
        hpc_cluster = get_site(self.comm.project.site_name)
        remote_diff_dir = self.comm.project.remote_diff_model_dir
        # username = daint.config["ssh_settings"]["username"]
        # remote_diff_dir = os.path.join(
        #     "/scratch/snx3000", username, "diff_models"
        # )
        local_diff_model_dir = "DIFF_MODELS"

        if not os.path.exists(local_diff_model_dir):
            os.mkdir(local_diff_model_dir)

        if not hpc_cluster.remote_exists(remote_diff_dir):
            hpc_cluster.remote_mkdir(remote_diff_dir)

        sims = []
        for param in self.comm.project.inversion_params:
            if param.startswith("V"):
                reference_velocity = param
            elif param == "RHO":
                if "VPV" in self.comm.project.inversion_params:
                    reference_velocity = "VPV"
                elif "VP" in self.comm.project.inversion_params:
                    reference_velocity = "VP"

            unique_id = (
                "_".join([str(i).replace(".", "") for i in smoothing_lengths])
                + "_"
                + str(self.comm.project.min_period)
            )

            diff_model_file = unique_id + f"diff_model_{param}.h5"
            # if self.comm.project.meshes == "multi-mesh":
            #     diff_model_file = event + "_" + diff_model_file

            remote_diff_model = os.path.join(remote_diff_dir, diff_model_file)

            diff_model_file = os.path.join(
                local_diff_model_dir, diff_model_file
            )

            if not os.path.exists(diff_model_file):
                smooth = smoothing.AnisotropicModelDependent(
                    reference_frequency_in_hertz=freq,
                    smoothing_lengths_in_wavelengths=smoothing_lengths,
                    reference_model=mesh,
                    reference_velocity=reference_velocity,
                )
                diff_model = smooth.get_diffusion_model(mesh)
                diff_model.write_h5(diff_model_file)

            if not hpc_cluster.remote_exists(remote_diff_model):
                hpc_cluster.remote_put(diff_model_file, remote_diff_model)

            sim = sc.simulation.Diffusion(mesh=diff_model_file)

            tensor_order = 2  # Hard coded now but should be fixed

            sim.domain.polynomial_order = tensor_order
            sim.physics.diffusion_equation.time_step_in_seconds = (
                self.comm.project.smoothing_timestep
            )
            sim.physics.diffusion_equation.courant_number = 0.06

            sim.physics.diffusion_equation.initial_values.filename = (
                "REMOTE:" + remote_grad
            )
            sim.physics.diffusion_equation.initial_values.format = "hdf5"
            sim.physics.diffusion_equation.initial_values.field = f"{param}"
            sim.physics.diffusion_equation.final_values.filename = (
                f"{param}.h5"
            )

            sim.domain.mesh.filename = "REMOTE:" + remote_diff_model
            sim.domain.model.filename = "REMOTE:" + remote_diff_model
            sim.domain.geometry.filename = "REMOTE:" + remote_diff_model

            sim.validate()

            # append sim to array
            sims.append(sim)

        job = sapi.run_many_async(
            input_files=sims,
            site_name=self.comm.project.smoothing_site_name,
            ranks_per_job=self.comm.project.smoothing_ranks,
            wall_time_in_seconds_per_job=self.comm.project.smoothing_wall_time,
        )
        if self.comm.project.inversion_mode == "mini-batch":
            print(f"Submitted smoothing for event {event}")
            self.comm.project.change_attribute(
                f'smoothing_job["{event}"]["name"]', job.job_array_name
            )
            self.comm.project.change_attribute(
                f'smoothing_job["{event}"]["submitted"]', True
            )
        else:
            self.comm.project.change_attribute(
                'smoothing_job["name"]', job.job_array_name
            )
            self.comm.project.change_attribute(
                'smoothing_job["submitted"]', True
            )
        self.comm.project.update_iteration_toml()
