from __future__ import absolute_import
from .component import Component
import numpy as np
# import os


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

    def add_smoothing_fields(self, event: str) -> object:
        """
        The diffusion equation smoothing needs certain parameters for
        smoothing. These parameters need to be appended to the mesh as fields.
        Currently we only use constant smoothing lengths but that will change.

        :param event: name of event
        :type event: str
        """
        import shutil
        import os
        import h5py
        # TODO: Make the smoothing fields be a different mesh.
        from salvus_mesh.unstructured_mesh import UnstructuredMesh
        iteration = self.comm.project.current_iteration
        gradient = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event,
            smooth=False)

        # grad_folder, _ = os.path.split(gradient)
        # smoothing_fields_mesh = os.path.join(grad_folder, "smoothing_fields.h5")
        
        # shutil.copyfile(gradient, smoothing_fields_mesh)
        smoothing_length = 500.0 * 1000.0  # Hardcoded for now
        # dimstr = '[ ' + ' | '.join(["M0", "M1"]) + ' ]'
        
        # with h5py.File(smoothing_fields_mesh, "r+") as fh:
        #     if "MODEL/data" in fh:
        #         M0 = np.ones(shape=(fh["MODEL/data"].shape[0], 1, fh["MODEL/data"].shape[2]))
        #         # M0 = fh["MODEL/data"][:, 0, :]
        #         # M0 = np.ones(M0.shape)
        #         M1 = np.copy(M0) * 2 * np.sqrt(smoothing_length)
        #         smoothing = np.concatenate((M0, M1), axis=1)
        #         print(f"Smoothing shape: {smoothing.shape}")
        #         del fh["MODEL/data"]
        #         fh.create_dataset("MODEL/data", data=smoothing)
        #         fh['MODEL/data'].dims[0].label = 'element'
        #         fh['MODEL/data'].dims[1].label = dimstr
        #         fh['MODEL/data'].dims[2].label = 'point'
                

        mesh = UnstructuredMesh.from_h5(gradient)
        mesh.elemental_fields = {}
        smooth_gradient = UnstructuredMesh.from_h5(gradient)
        smooth_gradient.elemental_fields = {}

        mesh.attach_field('M0', np.ones_like(mesh.get_element_nodes()[:, :, 0]))
        mesh.attach_field('M1', 0.5 * smoothing_length ** 2 * np.ones_like(mesh.get_element_nodes()[:, :, 0]))
        mesh.attach_field('fluid', np.ones(mesh.nelem))

        print(f"Smoothing fields M0 and M1 added to gradient for "
              f"event {event}")
        return mesh, smooth_gradient

    def write_xdmf(self, filename: str):
        """
        A hacky way to write an xdmf file for the hdf5 file
        :param filename: path to hdf5 file
        :return:
        """
        from salvus_mesh.unstructured_mesh import UnstructuredMesh

        mesh = UnstructuredMesh.from_h5(filename)
        mesh.write_h5(filename)
