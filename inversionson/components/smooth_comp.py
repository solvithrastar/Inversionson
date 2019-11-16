from .component import Component
from inversionson import InversionsonError

import h5py  # Might be needed when it comes to space dependent smoothing
import toml
import os
import subprocess
import sys


class SalvusSmoothComponent(Component):
    """
    A class which handles all dealings with the salvus smoother.
    I will start with fixed parameters
    TODO: Add a regularization toml file which includes dropout rate and smoothing lengths
    """

    def __init__(self, communicator, component_name):
        super(SalvusSmoothComponent, self).__init__(communicator, component_name)
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
        import salvus_flow.simple_config as sc

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
        smoothing_fields_mesh = os.path.join(grad_folder, "smoothing_fields.h5")
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

    def run_smoother(self, gradient: str):
        """
        A method which runs the salvus smoother and waits until it is
        done. Currently this is always run on a local computer. If things
        get heavy, this can be modified to run on a supercomputer.

        :param gradient: Path to gradient file
        :type gradient: str
        """
        seperator = "/"
        input_toml = seperator.join(gradient.split(seperator)[:-1])
        input_toml = os.path.join(input_toml, "input.toml")
        if not os.path.exists(input_toml):
            self.generate_input_toml(gradient)

        # Add something to make it run on more cores.
        command = f"{self.smoother_path} --input {input_toml}"

        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, bufsize=1
        )
        print("Smoother is running")
        # This is to print out to command line, what salvus smooth prints
        for line in process.stdout:
            print(line, end="\n", flush=True)
            # sys.stdout.write(line)

        process.wait()
        print("Smoother is done")
        print(process.returncode)
