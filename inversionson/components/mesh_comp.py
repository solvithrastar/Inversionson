from __future__ import annotations
import h5py  # type: ignore
import numpy as np
import shutil
import os
from pathlib import Path
from inversionson import InversionsonError
from salvus.mesh.unstructured_mesh import UnstructuredMesh  # type: ignore
from typing import Optional, Union, List, TYPE_CHECKING

from .component import Component

if TYPE_CHECKING:
    from inversionson.project import Project


class Mesh(Component):
    """
    Communications with Salvus Mesh.
    This will have to be done in a temporary way to begin with
    as it is not possible to make smoothiesem meshes through
    a nice config as things stand.

    :param infodict: Information related to inversion project
    :type infodict: Dictionary
    """

    def __init__(self, project: Project):
        super().__init__(project)
        self.meshes = self.project.config.lasif_root / "MODELS"
        self.event_meshes = self.meshes / "EVENT_MESHES"
        self.average_meshes = self.meshes / "AVERAGE_MESHES"

    def print(
        self,
        message: str,
        color: Optional[str] = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = ":globe_with_meridians:",
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

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
                return field_name in global_strings
            if elemental:
                if "element_data" not in mesh["MODEL"].keys():
                    return False
                elemental_fields = mesh["MODEL/element_data"].attrs.get(
                    "DIMENSION_LABELS"
                )[1]
                elemental_fields = elemental_fields[2:-2]
                if type(elemental_fields) != str:
                    elemental_fields = elemental_fields.decode()
                elemental_fields = elemental_fields.replace(" ", "").split("|")
                return field_name in elemental_fields
            if side_sets:
                return "SIDE_SETS" in mesh.keys()
            # Here we assume it's an element_nodal_field
            nodal_fields = mesh["MODEL/data"].attrs.get("DIMENSION_LABELS")[1]
            nodal_fields = nodal_fields[2:-2].replace(" ", "").split("|")
            return field_name in nodal_fields

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
            # tm.element_nodal_fields = {}
        tm = UnstructuredMesh.from_h5(to_mesh)
        fm = UnstructuredMesh.from_h5(from_mesh)
        if global_string:
            field = fm.global_strings[field_name]
            tm.attach_global_variable(name=field_name, data=field)
            tm.write_h5(to_mesh)
            self.print(f"Attached field {field_name} to mesh {to_mesh}")
            return
        elif elemental:
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

    def sum_two_fields_on_a_mesh(
        self,
        mesh: str,
        fieldname_1: str,
        fieldname_2: str,
        newname: Optional[str] = None,
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

        if delete_old_fields and newname is not None:
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

    def fill_inversion_params_with_zeroes(self, mesh: Union[Path, str]):
        """
        This is done because we don't interpolate every layer and then
        we want to make sure there is nothing sneaking into the gradients

        :param mesh: Path to mesh
        :type mesh: str
        """
        self.print("Filling inversion parameters with zeros before interpolation")
        m = UnstructuredMesh.from_h5(mesh)
        parameters = self.project.config.inversion.inversion_parameters
        zero_nodal = np.zeros_like(m.element_nodal_fields[parameters[0]])

        for param in parameters:
            m.attach_field(param, zero_nodal)
        m.write_h5(mesh)
