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

    def add_field_from_one_mesh_to_another(self, from_mesh: str, to_mesh: str,
                                           field_name: str):
        """
        Add one field from a specific mesh to another mesh. The two meshes
        need to have identical discretisations
        
        :param from_mesh: Path of mesh to copy field from
        :type from_mesh: str
        :param to_mesh: Path of mesh to copy field to
        :type to_mesh: str
        :param field_name: Name of the field to copy between them.
        :type field_name: str
        """
        from salvus_mesh.unstructred_mesh import UnstructuredMesh
        import os
        import shutil

        if not os.path.exists(to_mesh):
            print(f"Mesh {to_mesh} does not exist. Will create new one.")
            shutil.copy(from_mesh, to_mesh)
            tm = UnstructuredMesh.from_h5(to_mesh)
            tm.nodal_fields = {}
        else:
            tm = UnstructuredMesh.from_h5(to_mesh)
        fm = UnstructuredMesh.from_h5(from_mesh)

        field = fm.nodal_fields[field_name]
        tm.attach_field(field_name, field)
        print(f"Attached field {field_name} to mesh {to_mesh}")

    def write_xdmf(self, filename: str):
        """
        A hacky way to write an xdmf file for the hdf5 file
        :param filename: path to hdf5 file
        :return:
        """
        from salvus_mesh.unstructured_mesh import UnstructuredMesh

        mesh = UnstructuredMesh.from_h5(filename)
        mesh.write_h5(filename)

    def add_fluid_and_roi_from_lasif_mesh(self):
        """
        For some reason the salvus opt meshes don't have all the necessary info.
        I need this to get them simulation ready. I will write them into the
        lasif folder afterwards.
        As this is a quickfix, I will make it for my specific case.
        """
        from salvus_mesh.unstructured_mesh import UnstructuredMesh
        import os
        import numpy as np

        initial_model = os.path.join(self.comm.project.lasif_root,
                                     "MODELS",
                                     "Globe3D_csem_100.h5")
        iteration = self.comm.project.current_iteration
        opt_mesh = os.path.join(
                self.comm.project.paths["salvus_opt"],
                "PHYSICAL_MODELS",
                f"{iteration}.h5")
        m_opt = UnstructuredMesh.from_h5(opt_mesh)
        m_init = UnstructuredMesh.from_h5(initial_model)
        
        fluid = m_init.elemental_fields['fluid']
        roi = np.abs(1.0 - fluid)

        m_opt.attach_field(name="fluid", data=fluid)
        m_opt.attach_field(name="ROI", data=roi)

        iteration_mesh = os.path.join(
                self.comm.project.lasif_root,
                "MODELS",
                f"ITERATION_{iteration}",
                "mesh.h5")
        if not os.path.exists(os.path.dirname(iteration_mesh)):
            os.makedirs(os.path.dirname(iteration_mesh))
        m_opt.write_h5(iteration_mesh)
