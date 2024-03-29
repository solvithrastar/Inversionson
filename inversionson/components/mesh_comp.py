from __future__ import absolute_import
import h5py
import numpy as np
import shutil
import os
from pathlib import Path
from inversionson import InversionsonError
from salvus.mesh.unstructured_mesh import UnstructuredMesh
from typing import Union, List

from lasif.components.component import Component


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

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = ":globe_with_meridians:",
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def create_mesh(self, event: str):
        """
        Create a smoothiesem mesh for an event. I'll keep refinements fixed
        for now.

        :param event: Name of event
        :type event: str
        """

        from salvus.mesh.simple_mesh import SmoothieSEM

        source_info = self.comm.lasif.get_source(event_name=event)
        if isinstance(source_info, list):
            source_info = source_info[0]
        sm = SmoothieSEM()
        sm.basic.model = "prem_ani_one_crust"
        sm.basic.min_period_in_seconds = self.comm.project.min_period
        sm.basic.elements_per_wavelength = 1.7
        sm.basic.number_of_lateral_elements = self.comm.project.elem_per_quarter
        sm.advanced.tensor_order = 4
        if self.comm.project.ellipticity:
            sm.spherical.ellipticity = 0.0033528106647474805
        if self.comm.project.ocean_loading["use"]:
            sm.ocean.bathymetry_file = self.comm.project.ocean_loading["file"]
            sm.ocean.bathymetry_varname = self.comm.project.ocean_loading["variable"]
            sm.ocean.ocean_layer_style = "loading"
            sm.ocean.ocean_layer_density = 1025.0
        if self.comm.project.topography["use"]:
            sm.topography.topography_file = self.comm.project.topography["file"]
            sm.topography.topography_varname = self.comm.project.topography["variable"]
        sm.source.latitude = source_info["latitude"]
        sm.source.longitude = source_info["longitude"]
        sm.refinement.lateral_refinements.append(
            {"theta_min": 40.0, "theta_max": 140.0, "r_min": 6250.0}
        )
        m = sm.create_mesh()
        mesh_file = self.event_meshes / event / "mesh.h5"
        if not os.path.exists(os.path.dirname(mesh_file)):
            os.makedirs(os.path.dirname(mesh_file))
        m.write_h5(mesh_file)

    def _check_if_mesh_has_field(
        self,
        check_mesh: str,
        field_name: str,
        elemental: bool,
        global_string: bool,
        side_sets: bool,
    ) -> bool:
        """
        Use h5py to quickly check whether field exists on mesh

        :param check_mesh: path to mesh to check
        :type check_mesh: str
        :param field_name: Name of field
        :type field_name: str
        :param elemental: Is field an elemental field
        :type elemental: bool
        :param global_string: Is it a global string
        :type global_string: bool
        :param side_sets: Are we checking for side sets? Provide the name,
        :type side_sets: str
        """
        with h5py.File(check_mesh, mode="r") as mesh:
            if global_string:
                global_strings = list(mesh["MODEL"].attrs.keys())
                if field_name in global_strings:
                    return True
                else:
                    return False
            if elemental:
                if "element_data" in mesh["MODEL"].keys():
                    elemental_fields = mesh["MODEL/element_data"].attrs.get(
                        "DIMENSION_LABELS"
                    )[1]
                    elemental_fields = elemental_fields[2:-2]
                    if not type(elemental_fields) == str:
                        elemental_fields = elemental_fields.decode()
                    elemental_fields = elemental_fields.replace(" ", "").split("|")
                    if field_name in elemental_fields:
                        return True
                    else:
                        return False
                else:
                    return False
            if side_sets:
                if "SIDE_SETS" in mesh.keys():
                    return True
                else:
                    return False
            else:
                # Here we assume it's an element_nodal_field
                nodal_fields = mesh["MODEL/data"].attrs.get("DIMENSION_LABELS")[1]
                nodal_fields = nodal_fields[2:-2].replace(" ", "").split("|")
                if field_name in nodal_fields:
                    return True
                else:
                    return False

    def add_field_from_one_mesh_to_another(
        self,
        from_mesh: str,
        to_mesh: str,
        field_name: str,
        elemental: bool = False,
        global_string: bool = False,
        side_sets: bool = False,
        overwrite: bool = True,
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
        :param elemental: If the field is elemental make true, defaults to
            False
        :type elemental: bool, optional
        :param global_string: If the field is a global variable, defaults
            to False
        :type global_string: bool, optional
        :param overwrite: We check whether field is existing, if overwrite
            is True, we write it anyway, defaults to True
        :type bool, optional
        """
        import os
        import shutil

        # has_field = self._check_if_mesh_has_field(
        #     check_mesh=from_mesh,
        #     field_name=field_name,
        #     elemental=elemental,
        #     global_string=global_string,
        # )
        has_field = self._check_if_mesh_has_field(
            check_mesh=to_mesh,
            field_name=field_name,
            elemental=elemental,
            global_string=global_string,
            side_sets=side_sets,
        )
        if has_field and not overwrite:
            self.print(f"Field: {field_name} already exists on mesh")
            return
        attach_field = True
        if not os.path.exists(to_mesh):
            self.print(f"Mesh {to_mesh} does not exist. Will create new one.")
            shutil.copy(from_mesh, to_mesh)
            tm = UnstructuredMesh.from_h5(to_mesh)
            # tm.element_nodal_fields = {}
        else:
            tm = UnstructuredMesh.from_h5(to_mesh)
        fm = UnstructuredMesh.from_h5(from_mesh)
        if global_string:
            # if field_name in tm.global_strings.keys():
            #     if not overwrite:
            #         print(f"Field {field_name} already exists on mesh")
            #         return
            field = fm.global_strings[field_name]
            tm.attach_global_variable(name=field_name, data=field)
            tm.write_h5(to_mesh)
            self.print(f"Attached field {field_name} to mesh {to_mesh}")
            return
        elif elemental:
            # if field_name in tm.elemental_fields.keys():
            #     if not overwrite:
            #         print(f"Field {field_name} already exists on mesh")
            #         return
            field = fm.elemental_fields[field_name]
        elif side_sets:
            for side_set in fm.side_sets.keys():
                tm.define_side_set(
                    name=side_set,
                    element_ids=fm.side_sets[side_set][0],
                    side_ids=fm.side_sets[side_set][1],
                )
                self.print(f"Attached side set {side_set} to mesh {to_mesh}")
            attach_field = False

        else:
            # if field_name in tm.element_nodal_fields.keys():
            #     if not overwrite:
            #         print(f"Field {field_name} already exists on mesh")
            #         return
            field = fm.element_nodal_fields[field_name]
        if attach_field:
            tm.attach_field(field_name, field)
            self.print(f"Attached field {field_name} to mesh {to_mesh}")
        tm.write_h5(to_mesh)

    def write_xdmf(self, filename: str):
        """
        A hacky way to write an xdmf file for the hdf5 file
        :param filename: path to hdf5 file
        :return:
        """

        mesh = UnstructuredMesh.from_h5(filename)
        mesh.write_h5(filename)

    def add_fluid_and_roi_from_lasif_mesh(self):
        """
        For some reason the salvus opt meshes don't have all the necessary info.
        I need this to get them simulation ready. I will write them into the
        lasif folder afterwards.
        As this is a quickfix, I will make it for my specific case.
        """
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
        # I have to make sure the I am consistent with naming of things, might be a bit off there

        folder_name = f"it_{iteration_range[0]}_to_{iteration_range[1]}"
        full_path = self.average_meshes / folder_name / "mesh.h5"
        if not os.path.exists(full_path.parent):
            os.makedirs(full_path.parent)
        if os.path.exists(full_path):
            return full_path

        # We copy the newest mesh from SALVUS_OPT to LASIF and write the
        # average fields onto those.
        # No, that may be an error, we copy the master model from LASIF
        # and put average field onto that one.

        # model = self.comm.salvus_opt.get_model_path()
        model = self.comm.lasif.get_master_model()
        shutil.copy(model, full_path)

        m = UnstructuredMesh.from_h5(full_path)
        fields = m.element_nodal_fields
        new_fields = {}
        for field in fields.keys():
            new_fields[field] = np.zeros_like(fields[field])
        # m.element_nodal_fields = {}
        optimizer = self.comm.project.get_optimizer()
        for iteration in range(iteration_range[0], iteration_range[1] + 1):
            model_path = optimizer.get_path_for_iteration(
                iteration, optimizer.model_path
            )
            m_tmp = UnstructuredMesh.from_h5(model_path)
            for field_name, field in new_fields.items():
                field += m_tmp.element_nodal_fields[field_name]

        for field_name, field in new_fields.items():
            field /= len(range(iteration_range[0], iteration_range[1] + 1))
            m.attach_field(field_name, field)
        m.write_h5(full_path)
        self.print(
            f"Wrote and average model of iteration {iteration_range[0]} to"
            f" iteration {iteration_range[1]} onto mesh: {full_path}"
        )

        return full_path

    def add_region_of_interest(self, event: str):
        """
        Region of interest is the region where the gradient is computed
        and outside the region, it is not computed.
        Currently we add the region of interest as an elemental field
        which is the oposite of the fluid field.

        :param event: Name of event
        :type event: str
        """

        mesh = self.comm.lasif.find_event_mesh(event)
        m = UnstructuredMesh.from_h5(mesh)
        mesh_layers = np.sort(np.unique(m.elemental_fields["layer"]))[::-1].astype(int)
        layers = m.elemental_fields["layer"]
        o_core_idx = layers[np.where(m.elemental_fields["fluid"] == 1)[0][0]]
        o_core_idx = np.where(mesh_layers == o_core_idx)[0][0]
        correct_layers = mesh_layers[o_core_idx:]
        roi = np.zeros_like(layers)
        for layer in correct_layers:
            roi = np.logical_or(roi, layers == layer)

        m.attach_field("ROI", roi)
        m.write_h5(mesh)

    def sum_two_fields_on_a_mesh(
        self,
        mesh: str,
        fieldname_1: str,
        fieldname_2: str,
        newname: str = None,
        delete_old_fields: bool = False,
    ):
        """
        Take two fields on a mesh and sum them together. If no newname is
        given the summed field will be written into both the old fields.
        If newname is given the summed field will be written in there
        and if delete_old_fields is true they will be deleted of course.

        :param mesh: Path to mesh to be used
        :type mesh: str or Path
        :param fieldname_1: Name of field to be summed
        :type fieldname_1: str
        :param fieldname_2: Name of other field to be summed
        :type fieldname_2: str
        :param newname: Name of field to store summed field, defaults to None
        :type newname: str, optional
        :param delete_old_fields: Whether old fields should be deleted,
            defaults to False. Currently not implemented
        :type delete_old_fields: bool, optional
        """

        m = UnstructuredMesh.from_h5(mesh)

        available_fields = list(m.element_nodal_fields.keys())
        if fieldname_1 not in available_fields:
            raise InversionsonError(
                f"Field {fieldname_1} not available on mesh {mesh}. "
                f"Only available fields are: {available_fields}"
            )
        if fieldname_2 not in available_fields:
            raise InversionsonError(
                f"Field {fieldname_2} not available on mesh {mesh}. "
                f"Only available fields are: {available_fields}"
            )

        if delete_old_fields:
            if newname is not None:
                raise InversionsonError(
                    "If you want to delete old fields you need to write the "
                    "summed one into a new field"
                )

        summed_field = np.copy(m.element_nodal_fields[fieldname_1])
        summed_field += m.element_nodal_fields[fieldname_2]

        if newname is None:
            m.attach_field(fieldname_1, summed_field)
            m.attach_field(fieldname_2, summed_field)
            m.write_h5(mesh)

        else:
            m.attach_field(newname, summed_field)

    def fill_inversion_params_with_zeroes(self, mesh: str):
        """
        This is done because we don't interpolate every layer and then
        we want to make sure there is nothing sneaking into the gradients

        :param mesh: Path to mesh
        :type mesh: str
        """
        self.print("Filling inversion parameters with zeros before interpolation")
        m = UnstructuredMesh.from_h5(mesh)
        parameters = self.comm.project.inversion_params
        zero_nodal = np.zeros_like(m.element_nodal_fields[parameters[0]])

        for param in parameters:
            m.attach_field(param, zero_nodal)
        m.write_h5(mesh)
