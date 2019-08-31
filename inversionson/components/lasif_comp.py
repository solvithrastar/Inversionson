from __future__ import absolute_import

from .component import Component
import lasif.api as lapi
import os
from inversionson import InversionsonError, InversionsonWarning
import warnings
import subprocess
import sys

class LasifComponent(Component):
    """
    Communication with Lasif
    """

    def __init__(self, communicator, component_name):
        super(LasifComponent, self).__init__(communicator, component_name)
        self.lasif_root = self.comm.project.lasif_root
        self.lasif_comm = self._find_project_comm()

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        import pathlib
        from lasif.components.project import Project

        folder = pathlib.Path(self.lasif_root).absolute()
        max_folder_depth = 4

        for _ in range(max_folder_depth):
            if (folder / "lasif_config.toml").exists():
                return Project(folder).get_communicator()
            folder = folder.parent
        raise ValueError(f"Path {self.lasif_root} is not a LASIF project")

    def has_iteration(self, it_name: str) -> bool:
        """
        See if lasif project has the iteration already

        :param it_name: name of iteration
        :type name: str
        :return: True if lasif has the iteration
        """
        iterations = lapi.list_iterations(self.lasif_comm, output=True)
        if isinstance(iterations, list):
            if it_name in iterations:
                return True
        else:
            return False

    def set_up_iteration(self, name: str, events=[]):
        """
        Create a new iteration in the lasif project

        :param name: Name of iteration
        :type name: str
        :param events: list of events used in iteration, defaults to []
        :type events: list, optional
        """
        iterations = lapi.list_iterations(self.lasif_comm, output=True)
        if isinstance(iterations, list):
            if name in iterations:
                warnings.warn(
                    f"Iteration {name} already exists", InversionsonWarning)

        lapi.set_up_iteration(self.lasif_comm, iteration=name, events=events)

    def get_minibatch(self, first=False) -> list:
        """
        Get a batch of events to use in the coming iteration.
        This is still under development

        :param first: First batch of inversion?
        :type first: bool
        :return: A fresh batch of earthquakes
        :rtype: list
        """
        if first:
            blocked_events = []
            use_these = None
            # TODO: There must be a better way of defining number of events.
            count = self.comm.salvus_opt.read_salvus_opt()
            count = count["task"][0]["input"]["num_events"]
            events = self.list_events()
            batch = lapi.get_subset(
                self.lasif_comm,
                count=count,
                events=events,
                existing_events=None
            )
            return batch
        else:
            blocked_events, use_these = self.comm.salvus_opt.find_blocked_events()
            count = self.comm.salvus_opt.get_batch_size()
        events = self.list_events()
        prev_iter = self.comm.salvus_opt.get_previous_iteration_name()
        prev_iter_info = self.comm.project.get_old_iteration_info(prev_iter)
        existing = prev_iter_info["new_control_group"]
        count -= existing
        if use_these:
            count -= len(use_these)
            batch = use_these + existing
            avail_events = list(
                set(events) - set(blocked_events) - set(use_these))
            existing = list(set(existing + use_these))
        else:
            batch = existing
            if len(blocked_events) == 0:
                rand_batch = self.comm.salvus_opt.get_random_event(
                    n=self.comm.project.n_random_events_picked,
                    existing=existing
                )
                batch = list(set(batch) + set(rand_batch))
                existing = batch
            avail_events = list(set(events) - set(blocked_events))
        # TODO: existing events should only be the control group.
        # events should exclude the blocked events because that's what
        # are the options to choose from. The existing go into the poisson disc
        add_batch = lapi.get_subset(
            self.lasif_comm,
            count=count,
            events=avail_events,
            existing_events=existing
        )
        batch = list(set(batch) + set(add_batch))
        return batch

    def list_events(self, iteration=None):
        """
        Make lasif list events, supposed to be used when all events
        are used per iteration. IF only for an iteration, pass 
        an iteration value.
        """
        return lapi.list_events(
            self.lasif_comm,
            just_list=True,
            iteration=iteration,
            output=True)

    def has_mesh(self, event: str) -> bool:
        """
        Check whether mesh has been constructed for respective event

        :param event: Name of event
        :type event: str
        :return: Answer whether mesh exists
        :rtype: bool
        """
        has, _ = lapi.find_event_mesh(self.lasif_comm, event)

        return has

    def move_mesh(self, event: str, iteration: str):
        """
        Move mesh to simulation mesh path, where model will be added to it

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        """
        import shutil
        has, event_mesh = lapi.find_event_mesh(self.lasif_comm, event)

        if not has:
            raise ValueError(f"{event_mesh} does not exist")
        else:
            event_iteration_mesh = lapi.get_simulation_mesh(
                self.lasif_comm, event, iteration)
            shutil.copy(event_mesh, event_iteration_mesh)
            print(
                f"Mesh for event: {event} has been moved to correct path for "
                f"iteration: {iteration} and is ready for interpolation.")

    def find_gradient(self, iteration: str, event: str, smooth=False) -> str:
        """
        Find the path to a gradient produced by an adjoint simulation.

        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event
        :type event: str
        :param smooth: Do you want the smoothed gradient, defaults to False
        :type smooth: bool
        :return: Path to a gradient
        :rtype: str
        """
        gradients = self.lasif_comm.project.paths["gradients"]
        if smooth:
            gradient = os.path.join(gradients, f"ITERATION_{iteration}",
                                    event, "smooth_gradient.h5")
        else:
            gradient = os.path.join(gradients, f"ITERATION_{iteration}",
                                    event, "gradient.h5")
        if os.path.exists(gradient):
            return gradient
        else:
            raise ValueError(f"File: {gradient} does not exist.")

    def plot_iteration_events(self) -> str:
        """
        Return the path to a file containing an illustration of
        event distribution for the current iteration

        :return: Path to figure
        :rtype: str
        """
        lapi.plot_events(
            self.lasif_comm,
            type="map",
            iteration=self.comm.project.current_iteration,
            save=True
        )
        filename = os.path.join(
            self.lasif_root,
            "OUTPUT",
            "event_plots",
            "events",
            f"events_{self.comm.project.current_iteration}.png")
        return filename

    def plot_iteration_raydensity(self) -> str:
        """
        Return the path to a file containing an illustration of
        event distribution for the current iteration

        :return: Path to figure
        :rtype: str
        """
        lapi.plot_raydensity(
                self.lasif_comm,
                iteration=self.comm.project.current_iteration)
        filename = os.path.join(
                self.lasif_root,
                "OUTPUT",
                "raydensity_plots",
                f"ITERATION_{iteration}",
                "raydensity.png")
        return filename

    def get_source(self, event_name: str) -> dict:
        """
        Get information regarding source used in simulation

        :param event_name: Name of source
        :type event_name: str
        :return: Dictionary with source information
        :rtype: dict
        """
        return lapi.get_source(
            self.lasif_comm,
            event_name,
            self.comm.project.current_iteration)

    def get_receivers(self, event_name: str) -> dict:
        """
        Locate receivers and get them in a format that salvus flow
        can use

        :param event_name: Name of event
        :type event_name: str
        :return: A list of receiver dictionaries
        :rtype: dict
        """
        return lapi.get_receivers(self.lasif_comm, event_name)

    def get_simulation_mesh(self, event_name: str) -> str:
        """
        Get path to correct simulation mesh for a simulation

        :param event_name: Name of event
        :type event_name: str
        :return: Path to a mesh
        :rtype: str
        """
        return lapi.get_simulation_mesh(
            self.lasif_comm,
            event_name,
            self.comm.project.current_iteration)

    def calculate_station_weights(self, event: str):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        # Name weight set after event to know it
        weight_set_name = event
        # If set exists, we don't recalculate it
        if self.lasif_comm.weights.has_weight_set(weight_set_name):
            print(f"Weight set already exists for event {event}")
            return

        lapi.compute_station_weights(
            self.lasif_comm,
            weight_set=event,
            events=[event]
        )

    def misfit_quantification(self, event: str, mpi=True, n=6):
        """
        Quantify misfit and calculate adjoint sources.

        :param event: Name of event
        :type event: str
        :param mpi: If you want to run with MPI, default True
        :type mpi: bool
        :param n: How many ranks to run on
        :type n: int
        """
        iteration = self.comm.project.current_iteration
        window_set = iteration + "_" + event
        # Check if adjoint sources exist:
        adjoint_path = os.path.join(
                self.lasif_root,
                "ADJOINT_SOURCES",
                f"ITERATION_{iteration}",
                event,
                "stf.h5")
        if os.path.exists(adjoint_path):   
            print(f"Adjoint source exists for event: {event} ")
            print("Will not be recalculated. If you want them "
                    f"calculated, delete file: {adjoint_path}")
        elif mpi:
            os.chdir(self.comm.project.lasif_root)
            command = f"mpirun -n {n} lasif calculate_adjoint_sources "
            command += f"{iteration} "
            command += f"{window_set} {event}"
            process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, bufsize=1)
            for line in process.stdout:
                print(line, end="\n", flush=True)
            process.wait()
            print(process.returncode)
            os.chdir(self.comm.project.inversion_root)

        else:
            lapi.calculate_adjoint_sources(
                self.lasif_comm,
                iteration=iteration,
                window_set=window_set,
                weight_set=event,
                events=[event])
        # See if misfit has already been written into iteration toml
        if self.comm.project.misfits[event] == 0.0:
            misfit = self.lasif_comm.adj_sources.get_misfit_for_event(
                event=event,
                iteration=iteration,
                weight_set_name=event
            )
        else:
            misfit = self.comm.project.misfits[event]
            print(f"Misfit for {event} has already been computed. ")
            print("If you want it recomputed, change it to 0.0 in iteration "
                  "toml file")
        return misfit

    def get_adjoint_source_file(self, event: str, iteration: str) -> str:
        """
        Find the path to the correct asdf file containing the adjoint sources

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :return: Path to adjoint source file
        :rtype: str
        """
        adjoint_filename = "stf.h5"
        adj_sources = self.lasif_comm.project.paths["adjoint_sources"]
        it_name = self.lasif_comm.iterations.get_long_iteration_name(iteration)
        return os.path.join(adj_sources, it_name, event, adjoint_filename)

    def _already_processed(self, event: str) -> bool:
        """
        Looks for processed data for a certain event

        :param event: Name of event
        :type event: str
        :return: True/False regarding the alreadyness of the processed data.
        :rtype: bool
        """
        low_period = self.comm.project.period_low
        high_period = self.comm.project.period_high
        processed_filename = "preprocessed_" + \
            str(int(low_period)) + "s_to_" + str(int(high_period)) + "s.h5"
        processed_data_folder = self.lasif_comm.project.paths["preproc_eq_data"]

        return os.path.exists(os.path.join(
            processed_data_folder, event, processed_filename
        ))

    def process_data(self, event: str):
        """
        Process the data for the periods specified in Lasif project.

        :param event: Name of event to be processed
        :type event: str
        """
        if self._already_processed(event):
            return

        lapi.process_data(
            self.lasif_comm,
            events=[event])

    def select_windows(self, window_set_name: str, event: str, mpi=True, n=6):
        """
        Select window for a certain event in an iteration.

        :param window_set_name: Name of window set
        :type window_set_name: str
        :param event: Name of event to pick windows on
        :type event: str
        :param mpi: Switch on/off for running with MPI
        :type mpi: bool
        :param n: How many ranks to use
        :type n: int
        """
        # Check if window set exists:
        path = os.path.join(
                self.lasif_root,
                "SETS",
                "WINDOWS",
                f"{window_set_name}.sqlite")
        if os.path.exists(path):
            print(f"Window set for event {event} exists.")
            return
        
        if mpi:
            os.chdir(self.comm.project.lasif_root)
            command = f"mpirun -n 6 lasif select_windows "
            command += f"{self.comm.project.current_iteration} "
            command += f"{window_set_name} {event}"
            process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE,
                    bufsize=1)
            for line in process.stdout:
                print(line, end=' \n', flush=True)
            process.wait()
            print(process.returncode)
            os.chdir(self.comm.project.inversion_root)
        else:
            lapi.select_windows(
                self.lasif_comm,
                iteration=self.comm.project.current_iteration,
                window_set=window_set_name,
                events=[event]
            )

    def find_seismograms(self, event: str, iteration: str) -> str:
        """
        Find path to seismograms

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :return: str
        """
        if not iteration.startswith("ITERATION_"):
            iteration = f"ITERATION_{iteration}"

        event_folder = os.path.join(
                self.lasif_root,
                "SYNTHETICS",
                "EARTHQUAKES",
                iteration,
                event)
        if not os.path.exists(event_folder):
            os.mkdir(event_folder)

        return  os.path.join(event_folder, "receivers.h5")
