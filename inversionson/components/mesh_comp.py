from __future__ import absolute_import
from .component import Component


class SalvusMeshComponent(Component):
    """
    Communications with Salvus Mesh.
    This will have to be done in a temporary way to begin with
    as it is not possible to make smoothiesem meshes through
    a nice config as things stand.

    :param infodict: Information related to inversion project
    :type infodict: Dictionary
    """
    def __init__(self, communicator, component_name):
        super(SalvusMeshComponent, self).__init__(communicator, component_name)

    def create_mesh(self, lasif: object, event: str):
        """
        Create a smoothiesem mesh for an event
        
        :param lasif: lasif communicator
        :type lasif: object
        :param event: Name of event
        :type event: str
        """
        # Get relavant information from lasif. Use mesher to make a mesh
        # Put it into the correct directory.
        # I need to import something from a private code.
        # How do I do that?
