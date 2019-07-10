from __future__ import absolute_import

from .component import Component
import lasif.api as lapi
import os


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

    def set_up_iteration(self, name: str, events=[]):
        """
        Create a new iteration in the lasif project

        :param name: Name of iteration
        :type name: str
        :param events: list of events used in iteration, defaults to []
        :type events: list, optional
        """
        lapi.set_up_iteration(self.lasif_comm, name=name, events=events)

    def get_minibatch(self, it_name):
        """
        Will do later
        """
        return []

    def list_events(self):
        """
        Make lasif list events, supposed to be used when all events
        are used per iteration.
        """
        return lapi.list_events(self.lasif_comm, list=True, iteration=None)

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

    def find_gradient(self, iteration: str, event: str) -> str:
        """
        Find the path to a gradient produced by an adjoint simulation.

        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event
        :type event: str
        :return: Path to a gradient
        :rtype: str
        """
        gradients = self.lasif_comm.project.Project["gradients"]
        gradient = os.path.join(gradients, "ITERATION_" +
                                iteration, event, "gradient.h5")
        if os.path.exist(gradient):
            return gradient
        else:
            raise ValueError(f"File: {gradient} does not exist.")

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
        if self.lasif_comm.has_weight_set(weight_set_name):
            return

        lapi.compute_station_weights(
            self.lasif_comm,
            weight_set=event,
            events=[event]
        )

    def misfit_quantification(self, event: str):
        """
        Quantify misfit and calculate adjoint sources.

        :param event: Name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        window_set = iteration + "_" + event
        # Not sure how to keep track of the window set yet.
        lapi.calculate_adjoint_sources(
            self.lasif_comm,
            iteration=iteration,
            window_set=window_set,
            weight_set=event,
            events=[event])

        misfit = self.lasif_comm.adjoint_sources.get_misfit_for_event(
            event=event,
            iteration=iteration,
            weight_set_name=event
        )
        return misfit
        # Somehow write misfit into a file and such.

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
        adjoint_filename = "ADJ_SRC_" + event + ".h5"
        adj_sources = self.lasif_comm.project.paths["adjoint_sources"]
        it_name = self.lasif_comm.iterations.get_long_iteration_name(iteration)
        return os.path.join(adj_sources, it_name, adjoint_filename)

    def _already_processed(self, event: str) -> bool:
        """
        Looks for processed data for a certain event

        :param event: Name of event
        :type event: str
        :return: True/False regarding the alreadyness of the processed data.
        :rtype: bool
        """
        low_period = self.comm.project.low_period
        high_period = self.comm.project.high_period
        processed_filename = "preprocessed_" + \
            str(int(low_period)) + "s_to_" + str(int(high_period)) + "s.h5"
        processed_data_folder = self.lasif_comm.project.paths["processed_data"]
        
        return os.path.exists(os.path.join(
            processed_data_folder, "EARTHQUAKES", event, processed_filename
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

    def select_windows(self, window_set_name: str, event: str):
        """
        Select window for a certain event in an iteration.
        
        :param window_set_name: Name of window set
        :type window_set_name: str
        :param event: Name of event to pick windows on
        :type event: str
        """
        lapi.select_windows(
            self.lasif_comm,
            iteration=self.comm.project.current_iteration,
            window_set=window_set_name,
            events=[event]
        )
