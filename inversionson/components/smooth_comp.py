from .component import Component
from inversionson import InversionsonError

import h5py  # Might be needed when it comes to space dependent smoothing
import toml
import os
import subprocess
import sys
import pathlib
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

    def generate_diffusion_object(
        self, gradient: str, mesh: object, par: str, movie=False
    ) -> object:
        """
        Generate the input object that the smoother requires
        
        :param gradient: Path to the gradient file to be smoothed
        :type gradient: str
        :param mesh: Mesh object with diffusion parameters
        :type mesh: UnstructuredMesh object
        :param movie: If a movie should be saved, defaults to False
        :type movie: bool
        """
        import salvus.flow.simple_config as sc

        seperator = "/"
        # grad_folder, _ = os.path.split(gradient)
        # smoothing_fields_mesh = os.path.join(grad_folder, "smoothing_fields.h5")
        sim = sc.simulation.Diffusion(mesh=mesh)
        output_file = seperator.join(gradient.split(seperator)[:-1])
        movie_file = output_file + "/smoothing_movie.h5"
        output_file += "/smooth_gradient.h5"

        sim.physics.diffusion_equation.time_step_in_seconds = 1e-5
        sim.physics.diffusion_equation.initial_values.filename = gradient
        sim.physics.diffusion_equation.initial_values.format = "hdf5"
        sim.physics.diffusion_equation.initial_values.field = par  # Temporary

        sim.physics.diffusion_equation.final_values.filename = output_file
        if movie:
            sim.output.volume_data.filename = movie_file
            sim.output.volume_data.format = "hdf5"
            sim.output.volume_data.fields = ["VS"]
            sim.output.volume_data.sampling_interval_in_time_steps = 10

        sim.validate()

        return sim

    # Now I need to make some stuff based on the new salvus.opt and it's smoothing configurations

    def generate_smoothing_config(self, event: str) -> dict:
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
        import toml

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
        with open("./smoothing_config.toml", "w") as fh:
            toml.dump(smoothing_config, fh)
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
        from salvus.mesh.unstructured_mesh import UnstructuredMesh
        import salvus.flow.api

        if iteration is None:
            iteration = self.comm.project.current_iteration
        job_name = self.comm.salvus_flow.get_job_name(
            event=event_name, sim_type="smoothing", iteration=iteration,
        )
        salvus_job = salvus.flow.api.get_job_array(
            site_name=self.comm.project.site_name, job_array_name=job_name,
        )

        if self.comm.project.inversion_mode == "mono-batch":
            smooth_grad = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=None,
                smooth=True,
                summed=True,
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
        smooth_gradient = get_smooth_model(
            job=salvus_job,
            model=self.comm.lasif.find_event_mesh(event=event_name),
        )
        smooth_gradient.write_h5(smooth_grad)

    def generate_input_toml(self, gradient: str, movie=False):
        """
        Generate the input file that the smoother requires
        
        :param gradient: Path to the gradient file to be smoothed
        :type gradient: str
        :param movie: If a movie should be saved, defaults to False
        :type movie: bool
        """
        # Define a few paths
        seperator = "/"
        if movie:
            movie_file = seperator.join(gradient.split(seperator)[:-1])
            movie_file += "/smooth_movie.h5"
        output_file = seperator.join(gradient.split(seperator)[:-1])
        output_file += "/smooth_gradient.h5"

        grad_folder, _ = os.path.split(gradient)
        smoothing_fields_mesh = os.path.join(
            grad_folder, "smoothing_fields.h5"
        )
        # Domain dictionary
        mesh = {"filename": gradient, "format": "hdf5"}
        domain = {
            "dimension": 3,
            "polynomial-order": 4,
            "mesh": mesh,
            "model": mesh,
            "geometry": mesh,
        }

        # Physics dictionary
        diffusion_equation = {
            "start-time-in-seconds": 0.0,
            "end-time-in-seconds": 1.0,
            "time-step-in-seconds": 0.001,
            "time-stepping-scheme": "euler",
            "initial-values": {
                "filename": gradient,
                "format": "hdf5",
                "field": self.comm.project.inversion_params,
            },
            "final-values": {"filename": output_file},
        }

        physics = {"diffusion-equation": diffusion_equation}

        # Output dict
        if movie:
            volume_data = {
                "fields": ["VS"],
                "sampling-interval-in-time-steps": 10,
                "filename": movie_file,
                "format": "hdf5",
            }
            output = {"volume-data": volume_data}

        input_dict = {"domain": domain, "physics": physics}
        if movie:
            input_dict["output"] = output

        # Write toml file
        toml_dir = seperator.join(gradient.split(seperator)[:-1])
        toml_filename = "input.toml"
        toml_path = os.path.join(toml_dir, toml_filename)
        with open(toml_path, "w+") as fh:
            toml.dump(input_dict, fh)

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
        from salvus.flow.sites import SiteConfig
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        if iteration is None:
            iteration = self.comm.project.current_iteration
        mesh = UnstructuredMesh.from_h5(
            self.comm.lasif.find_gradient(iteration=iteration, event=event)
        )
        mesh.attach_global_variable(name="reference_frame", data="spherical")
        # site_config = SiteConfig(
        #     site_name=self.comm.project.smoothing_site_name,
        #     ranks=self.comm.project.smoothing_ranks,
        #     wall_time_in_seconds=self.comm.project.smoothing_wall_time,
        # )
        job = smoothing.run_async(
            model=mesh,
            smoothing_config=smoothing_config,
            # site_config=site_config,
            time_step_in_seconds=1.0e-5,
            site_name=self.comm.project.smoothing_site_name,
            ranks_per_job=self.comm.project.smoothing_ranks,
            wall_time_in_seconds_per_job=self.comm.project.smoothing_wall_time,
        )

        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["name"]', job.job_array_name
        )
        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["submitted"]', True
        )
