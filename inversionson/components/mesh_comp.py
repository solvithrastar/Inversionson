from __future__ import absolute_import
from .component import Component
import numpy as np
import sys
import shutil
from pathlib import Path
import os
from inversionson import InversionsonError


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
        self.meshes = Path(self.comm.project.lasif_root) / "MODELS"
        self.event_meshes = self.meshes / "EVENT_MESHES"
        self.average_meshes = self.meshes / "AVERAGE_MESHES"

    def create_mesh(self, event: str, n_lat: int):
        """
        Create a smoothiesem mesh for an event. I'll keep refinements fixed
        for now.
        
        :param event: Name of event
        :type event: str
        :param n_lat: Elements per quarter in azimuthal dimension
        :type n_lat: int
        """
        sys.path.append("/home/solvi/workspace/Meshing/inversionson_smoothie/")
        from global_mesh_smoothiesem import mesh as tmp_mesher
        from salvus.mesh import simple_mesh

        info = {}
        info["period"] = self.comm.project.min_period
        source_info = self.comm.lasif.get_source(event_name=event)
        if isinstance(source_info, list):
            source_info = source_info[0]
        info["latitude"] = source_info["latitude"]
        info["longitude"] = source_info["longitude"]
        info["n_lat"] = n_lat
        info["event_name"] = event
        sm = simple_mesh.AxiSEM()
        sm.basic.model = "prem_ani_one_crust"  # Maybe set as an import param
        sm.basic.min_period_in_seconds = self.comm.project.min_period
        sm.basic.elements_per_wavelength = 1.5
        sm.validate()
        print(sm)

        m = tmp_mesher(sm.get_dictionary(), tensor_order=1)

        theta_min_lat_refine = [40.0]
        theta_max_lat_refine = [140.0]
        r_min_lat_refine = [6250.0 / 6371.0]  # Should adapt this to 1D model

        m = tmp_mesher(
            sm.get_dictionary(),
            tensor_order=4,
            theta_max_lat_refine=theta_max_lat_refine,
            theta_min_lat_refine=theta_min_lat_refine,
            r_min_lat_refine=r_min_lat_refine,
            n_lat=n_lat,
            src_lat=source_info["latitude"],
            src_lon=source_info["longitude"],
        )
        mesh_file = self.event_meshes / event / "mesh.h5"
        if not os.path.exists(os.path.dirname(mesh_file)):
            os.makedirs(os.path.dirname(mesh_file))
        m.write_h5(mesh_file)

    def add_smoothing_fields(
        self, event: str
    ) -> object:  # @TODO: Need to rewrite this whole thing for new smoothing interface
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
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        iteration = self.comm.project.current_iteration
        gradient = self.comm.lasif.find_gradient(
            iteration=iteration, event=event, smooth=False
        )

        # grad_folder, _ = os.path.split(gradient)
        # smoothing_fields_mesh = os.path.join(grad_folder, "smoothing_fields.h5")

        # shutil.copyfile(gradient, smoothing_fields_mesh)
        smoothing_length = 350.0 * 1000.0  # Hardcoded for now
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

        mesh.attach_field(
            "M0", np.ones_like(mesh.get_element_nodes()[:, :, 0])
        )
        mesh.attach_field(
            "M1",
            0.5
            * smoothing_length ** 2
            * np.ones_like(mesh.get_element_nodes()[:, :, 0]),
        )
        mesh.attach_field("fluid", np.ones(mesh.nelem))

        print(
            f"Smoothing fields M0 and M1 added to gradient for "
            f"event {event}"
        )
        return mesh, smooth_gradient

    def add_field_from_one_mesh_to_another(
        self, from_mesh: str, to_mesh: str, field_name: str
    ):
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
        from salvus.mesh.unstructured_mesh import UnstructuredMesh
        import os
        import shutil

        if not os.path.exists(to_mesh):
            print(f"Mesh {to_mesh} does not exist. Will create new one.")
            shutil.copy(from_mesh, to_mesh)
            tm = UnstructuredMesh.from_h5(to_mesh)
            tm.elemental_nodal_fields = {}
        else:
            tm = UnstructuredMesh.from_h5(to_mesh)
        fm = UnstructuredMesh.from_h5(from_mesh)

        field = fm.element_nodal_fields[field_name]
        tm.attach_field(field_name, field)
        tm.write_h5(to_mesh)
        print(f"Attached field {field_name} to mesh {to_mesh}")

    def write_xdmf(self, filename: str):
        """
        A hacky way to write an xdmf file for the hdf5 file
        :param filename: path to hdf5 file
        :return:
        """
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        mesh = UnstructuredMesh.from_h5(filename)
        mesh.write_h5(filename)

    def add_fluid_and_roi_from_lasif_mesh(self):
        """
        For some reason the salvus opt meshes don't have all the necessary info.
        I need this to get them simulation ready. I will write them into the
        lasif folder afterwards.
        As this is a quickfix, I will make it for my specific case.
        """
        from salvus.mesh.unstructured_mesh import UnstructuredMesh
        import os
        import numpy as np

        initial_model = self.comm.lasif.lasif_comm.project.lasif_config[
            "domain_settings"
        ]["domain_file"]
        iteration = self.comm.project.current_iteration
        opt_mesh = os.path.join(
            self.comm.project.paths["salvus_opt"],
            "PHYSICAL_MODELS",
            f"{iteration}.h5",
        )
        m_opt = UnstructuredMesh.from_h5(opt_mesh)
        m_init = UnstructuredMesh.from_h5(initial_model)

        fluid = m_init.elemental_fields["fluid"]
        roi = np.abs(1.0 - fluid)

        m_opt.attach_field(name="fluid", data=fluid)
        m_opt.attach_field(name="ROI", data=roi)

        iteration_mesh = os.path.join(
            self.comm.project.lasif_root,
            "MODELS",
            f"ITERATION_{iteration}",
            "mesh.h5",
        )
        if not os.path.exists(os.path.dirname(iteration_mesh)):
            os.makedirs(os.path.dirname(iteration_mesh))
        m_opt.write_h5(iteration_mesh)

    def get_average_model(self, iteration_range: tuple) -> Path:
        """
        Get an average model between a list of iteration numbers.
        Can be used to get a smoother misfit curve for validation
        data set.
        
        :param iteration_range: From iteration to iteration tuple
        :type iterations: tuple
        """
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        folder_name = f"it_{iteration_range[0]}_to_{iteration_range[1]}"
        full_path = self.average_meshes / folder_name / "mesh.h5"
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        # We copy the newest mesh from SALVUS_OPT to LASIF and write the
        # average fields onto those.
        model = self.comm.salvus_opt.get_model_path()
        shutil.copy(model, full_path)

        m = UnstructuredMesh.from_h5(full_path)
        fields = m.element_nodal_fields
        new_fields = {}
        for field in fields.keys():
            new_fields[field] = np.zeros_like(fields[field])
        m.element_nodal_fields = {}
        for iteration in range(iteration_range[0], iteration_range[1]):
            it = self.comm.salvus_opt.get_name_for_accepted_iteration_number(
                number=iteration
            )
            model_path = self.comm.salvus_opt.get_model_path(iteration=it)
            m_tmp = UnstructuredMesh.from_h5(model_path)
            for field_name, field in new_fields.items():
                field += m_tmp.element_nodal_fields[field_name]

        for field_name, field in new_fields.items():
            field /= len(range(iteration_range[0], iteration_range[1]))
            m.attach_field(field_name, field)
        m.write_h5(full_path)
        print(
            f"Wrote and average model of iteration {iteration_range[0]} to"
            f" iteration {iteration_range[1]} onto mesh: {full_path}"
        )

        return full_path

        # Save similar looking fields as zeros
        # Delete fields
        # Sequentially go through old models and get fields
        # Add them to the fields of zeros
        # Divide all the new fields by len(range(iteration_range))
        # Add fields to mesh object
        # Write mesh out as hdf5.

    def add_region_of_interest(self, event: str):
        """
        Region of interest is the region where the gradient is computed
        and outside the region, it is not computed.
        Currently we add the region of interest as an elemental field
        which is the oposite of the fluid field.
        
        :param event: Name of event
        :type event: str
        """
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        mesh = self.comm.lasif.find_event_mesh(event)
        m = UnstructuredMesh.from_h5(mesh)
        fluid = m.elemental_fields["fluid"]
        roi = 1.0 - fluid
        m.attach_field("ROI", roi)
        m.write_h5(mesh)

    def write_new_opt_fields_to_simulation_mesh(self):
        """
        Salvus opt makes a mesh which has the correct velocities but
        it does not have everything which is needed to run a simulation.
        We will thus write it's fields on to our simulation mesh.
        """
        if self.comm.project.meshes == "multi-mesh":
            raise InversionsonError(
                "Multi-mesh inversion should not use this function. Only "
                "Mono-mesh."
            )
        print("Writing new fields to simulation mesh")
        iteration = self.comm.project.current_iteration
        if "validation" in iteration:
            iteration = iteration[11:]  # We don't need a special mesh
        opt_model = os.path.join(
            self.comm.salvus_opt.models, f"{iteration}.h5"
        )
        simulation_mesh = self.comm.lasif.get_simulation_mesh(
            event_name=None, iteration="current"
        )
        if os.path.exists(simulation_mesh):
            print("Mesh already exists, will not add fields")
            return
        else:
            shutil.copy(
                self.comm.lasif.lasif_comm.project.lasif_config[
                    "domain_settings"
                ]["domain_file"],
                simulation_mesh,
            )
        fields = self.comm.project.inversion_params
        for field in fields:
            print(f"Writing field: {field}")
            self.add_field_from_one_mesh_to_another(
                from_mesh=opt_model, to_mesh=simulation_mesh, field_name=field,
            )
