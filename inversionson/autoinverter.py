"""
A class which takes care of a Full-Waveform Inversion using multiple meshes.
It uses Salvus, Lasif and Multimesh to perform most of its actions.
This is a class which wraps the three packages together to perform an
automatic Full-Waveform Inversion
"""
import numpy as np
import pyasdf
import multi_mesh.api as mapi
from storyteller import Storyteller


class autoinverter(object):
    """
    Ok lets do this.
    We need something that reads Salvus opt
    Something that talks to salvus flow
    Something that creates meshes
    Something that talks to lasif
    Something that talks to multimesh
    Something interacting with dp-s random selection (No need now)
    Something talking to the smoother.
    So let's create a few files:
    salvus_opt communicator
    salvus_flow communicator
    salvus_mesher (not necessary now)
    lasif communicator
    multimesh communicator
    I can regularly save the inversion_dict as a toml file and reload it
    """

    def __init__(self, info_dict: dict, simulation_dict: dict,
                 inversion_dict: dict):
        self.info = info_dict
        self.comm = self._find_project_comm()
        self.storyteller = Storyteller()
        self.iteration_dict = {}

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        from inversionson.components.project import ProjectComponent

        return ProjectComponent(self.info).get_communicator()

    def _validate_inversion_project(self):
        """
        Make sure everything is correctly set up in order to perform inversion.

        :param info_dict: Information needed
        :type info_dict: dict
        :param simulation_dict: Information regarding simulations
        :type simulation_dict: dict
        """
        import pathlib

        if "inversion_name" not in self.info.keys():
            raise ValueError(
                "The inversion needs a name")

        # Salvus Opt
        if "salvus_opt_dir" not in self.info.keys():
            raise ValueError(
                "Information on salvus_opt_dir is missing from information")
        else:
            folder = pathlib.Path(self.info["salvus_opt_dir"])
            if not (folder / "inversion.toml").exists():
                raise ValueError("Salvus opt inversion not initiated")

        # Lasif
        if "lasif_project" not in self.info.keys():
            raise ValueError(
                "Information on lasif_project is missing from information")
        else:
            folder = pathlib.Path(self.info["lasif_project"])
            if not (folder / "lasif_config.toml").exists():
                raise ValueError("Lasif project not initialized")

        # Simulation parameters:
        if "end_time_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding end time of simulation missing")

        if "time_step_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding time step of simulation missing")

        if "start_time_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding start time of simulation missing")

    def initialize_inversion(self):
        """
        Set up everything regarding the inversion. Make sure everything
        is correct and that information is there.
        Make this check status of salvus opt, the inversion does not have
        to be new to call this method.
        """
        # Will do later.

    def prepare_iteration(self):
        """
        Prepare iteration.
        Get iteration name from salvus opt
        Modify name in inversion status
        Create iteration
        Pick events
        Make meshes if needed
        Update information in iteration dictionary.
        """
        it_name = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.current_iteration = it_name

        events = self.comm.lasif.get_minibatch(it_name)  # Sort this out.
        self.comm.lasif.set_up_iteration(it_name, events)

        for event in events:
            if not self.comm.lasif.has_mesh(event):
                self.comm.salvus_mesher.create_mesh(event)
                self.comm.lasif.move_mesh(event, it_name)
            else:
                self.comm.lasif.move_mesh(event, it_name)

        self.comm.project.create_iteration_toml(it_name, events)

        # Storyteller

    def interpolate_model(self, event: str):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        """
        self.comm.multi_mesh.interpolate_to_simulation_mesh(event)

    def interpolate_gradient(self, event: str, first: bool):
        """
        Interpolate gradient to master mesh

        :param event: Name of event
        :type event: str
        :param first: First iteration gradient to be interolated?
        :type first: bool
        """
        self.comm.multi_mesh.interpolate_gradient_to_model(event, first=first)

    def run_forward_simulation(self, event: str, iteration: str) -> object:
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :return: receiver object
        :return type: object
        """
        receivers = self.salvus_flow.get_receivers(event)
        source = self.salvus_flow.get_source_object(iteration, event)
        w = self.salvus_flow.construct_simulation(iteration, event, source, receivers)
        self.salvus_flow.submit_job()


    def run_adjoint_simulation(self):
        """
        Submit adjoint simulation to daint and possibly monitor
        """

    def misfit_quantification(self, adjoint: bool):
        """
        Compute misfit (and adjoint source) for iteration

        :param adjoint: Compute adjoint source?
        :type adjoint: bool
        """

    def perform_task(self, task: str):
        """
        Input a task and send to correct function

        :param task: task issued by salvus opt
        :type task: str
        """
        if task == "compute_misfit_and_gradient":
            self.prepare_iteration()
            # Figure out a way to do this on a per event basis.
            self.interpolate_model(event)
            self.run_forward_simulation()
            self.misfit_quantification(adjoint=True)
            self.run_adjoint_simulation()
            self.storyteller.document_task(task)
            self.salvus_opt.close_salvus_opt_task()
            task = self.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "compute_misfit":
            self.prepare_iteration()
            self.run_forward_simulation()
            self.misfit_quantification(adjoint=True)
            self.salvus_opt.close_salvus_opt_task()
            task = self.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "compute_gradient":
            self.run_adjoint_simulation()
            # Cut sources and receivers?
            self.interpolate_gradient()
            # Smooth gradients
            self.salvus_opt.move_gradient_to_salvus_opt_folder(event)
            self.salvus_opt.close_salvus_opt_task()
            task = self.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "finalize_iteration":
            self.salvus_opt.close_salvus_opt_task()
            task = self.salvus_opt.read_salvus_opt()
            self.perform_task(task)
            # Possibly delete wavefields

        else:
            raise ValueError(f"Salvus Opt task {task} not known")

    def run_inversion(self):
        """
        This is where the inversion runs.
        1. Check status of inversion
        2. Continue inversion

        Make iteration, select events for it.
        See whether events have meshes
        If not, make meshes

        Interpolate model to meshes
        as soon as an individual interpolation is done, submit job.

        If event not in control group, select windows.
        Retrieve results, calculate adjoint sources, submit adjoint jobs.

        Retrieve gradients, interpolate back, smooth.

        Update Model.

        Workflow:
                Read Salvus opt,
                Perform task,
                Document it
                Close task, repeat.
        """
        # Always do this as a first thing, Might write a different function for checking status
        self.initialize_inversion()

        task = self.salvus_opt.read_salvus_opt()

        self.perform_task(task)
