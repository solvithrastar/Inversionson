from __future__ import absolute_import
from .component import Component
import numpy as np


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

    def add_smoothing_fields(self, event: str):
        """
        The diffusion equation smoothing needs certain parameters for 
        smoothing. These parameters need to be appended to the mesh as fields.
        Currently we only use constant smoothing lengths but that will change.
        
        :param event: name of event
        :type event: str
        """
        from salvus_mesh.unstructured_mesh import UnstructuredMesh
        iteration = self.comm.project.current_iteration
        gradient = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event,
            smooth=False)
        
        smoothing_length = 5000.0 * 1000.0  # Hardcoded for now
        mesh = UnstructuredMesh.from_h5(gradient)

        M0 = np.ones(mesh.npoint) * 2 * np.sqrt(smoothing_length)
        M1 = np.copy(M0)

        mesh.attach_field(name="M0", M0)
        mesh.attach_field(name="M1", M1)
        mesh.map_nodal_fields_to_element_nodal()
        mesh.write_h5_tensorized_model(gradient)
        print(f"Smoothing fields M0 and M1 added to gradient for "
              f"event {event}")
