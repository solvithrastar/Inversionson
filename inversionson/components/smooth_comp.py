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

    def generate_smoothing_config(
        self, event: str, gradient: bool = True, iteration: str = None,
    ) -> dict:
        """
        Generate a dictionary which contains smoothing objects for each 
        parameter to be smoothed.

        :param event: Name of event
        :type event: str
        :param gradient: We are smoothing the gradient right?, defaults to True
        :type gradient: bool
        :param iteration: Name of iteration, if none, current is used,
            defaults to None
        :type iteration: str, optional
        :return: Dictonary which points each parameter to a smoothing object
        :rtype: dict
        """

        if not iteration:
            iteration_name = self.comm.project.current_iteration
        if gradient:
            gradient_file = self.comm.lasif.find_gradient(
                iteration=iteration_name, event=event,
            )
        # The mesh used as diffusion model is the event_mesh with the 1D model
        diff_model = self.comm.lasif.find_event_mesh(event=event)
        smoothing_config = {}
        freq = 1.0 / self.comm.project.period_low
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
            smooth = smoothing.AnisotropicModelDependent(
                reference_frequency_in_hertz=freq,
                smoothing_lengths_in_wavelengths=smoothing_lengths,
                reference_model=diff_model,
                reference_velocity=reference_velocity,
            )
            smoothing_config[param] = smooth
        return smooth

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
        if not iteration:
            iteration = self.comm.project.current_iteration
        mesh = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event
        )
        job = smoothing.run_async(
            model=mesh,
            smoothing_config=smoothing_config,
            ranks=self.comm.project.smoothing_ranks,
            wall_time_in_seconds=self.comm.project.smoothing_wall_time
        )
        
        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["name"]', job.job_array_name
        )
        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["submitted"]', True
        )