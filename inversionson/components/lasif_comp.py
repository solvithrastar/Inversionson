from __future__ import absolute_import

from .component import Component
import lasif.api as lapi


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
