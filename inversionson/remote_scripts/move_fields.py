import h5py
import sys
import numpy as np


def get_elemental_parameter_indices(mesh):
    if len(mesh["MODEL/element_data"].attrs.get("DIMENSION_LABELS")) == 1:
        return []
    return (
        mesh["MODEL/element_data"]
        .attrs.get("DIMENSION_LABELS")[1]
        .replace(" ", "")[1:-1]
        .split("|")
    )


def get_nodal_parameter_indices(mesh):
    return (
        mesh["MODEL/data"]
        .attrs.get("DIMENSION_LABELS")[1]
        .replace(" ", "")[1:-1]
        .split("|")
    )


def create_dimension_labels(gll, parameters: list, nodal=False):
    """
    Create the dimstring which is needed in the h5 meshes.
    :param gll_model: The gll mesh which needs the new dimstring
    :param parameters: The parameters which should be in the dimstring
    """
    dimstr = "[ " + " | ".join(parameters) + " ]"
    if nodal:
        gll["MODEL/data"].dims[0].label = "element"
        gll["MODEL/data"].dims[1].label = dimstr
        gll["MODEL/data"].dims[2].label = "point"
    else:
        gll["MODEL/element_data"].dims[0].label = "element"
        gll["MODEL/element_data"].dims[1].label = dimstr


def move_elemental_field_from_mesh_to_another(from_mesh, to_mesh, field):
    print(f"Moving field {field}")
    with h5py.File(from_mesh, "r") as fm:
        fm_indices = get_elemental_parameter_indices(fm)
        field_index = fm_indices.index(field)
        fm_field = fm["MODEL/element_data"][:, field_index]
    with h5py.File(to_mesh, "r+") as tm:
        model = tm["MODEL"]
        if "element_data" not in model.keys():
            tm.create_dataset(
                "MODEL/element_data", data=fm_field.reshape(len(fm_field), 1)
            )
            parameters = [field]
            create_dimension_labels(tm, parameters)
        else:
            tm_indices = get_elemental_parameter_indices(tm)
            if field not in tm_indices:
                shape = tm["MODEL/element_data"].shape
                if len(shape) == 1:
                    old_data = tm["MODEL/element_data"][()]
                    old_data = old_data.reshape(len(old_data), 1)
                    fm_field = fm_field.reshape(len(fm_field), 1)
                    data = np.concatenate((old_data, fm_field), axis=1)
                    parameters = ["something", field]
                    del tm["MODEL/element_data"]
                    tm.create_dataset("MODEL/element_data", data=data)
                else:
                    parameters = tm_indices
                    old_data = tm["MODEL/element_data"][()]
                    fm_field = fm_field.reshape(len(fm_field), 1)
                    data = np.concatenate((old_data, fm_field), axis=1)
                    del tm["MODEL/element_data"]
                    tm.create_dataset("MODEL/element_data", data=data)
                    parameters.append(field)

                create_dimension_labels(tm, parameters)


def move_nodal_field_from_mesh_to_another(from_mesh, to_mesh, field):
    print(f"Moving field {field}")
    with h5py.File(from_mesh, "r") as fm:
        fm_indices = get_nodal_parameter_indices(fm)
        field_index = fm_indices.index(field)
        fm_field = fm["MODEL/data"][:, field_index, :]
        fm_field = fm_field.reshape(fm_field.shape[0], 1, fm_field.shape[1])
    with h5py.File(to_mesh, "r+") as tm:
        tm_indices = get_nodal_parameter_indices(tm)
        if field not in tm_indices:
            parameters = tm_indices
            old_data = tm["MODEL/data"][()]
            data = np.concatenate((old_data, fm_field), axis=1)
            del tm["MODEL/data"]
            tm.create_dataset("MODEL/data", data=data)
            parameters.append(field)
            create_dimension_labels(tm, parameters, nodal=True)
            # else:
            #     field_index = tm_indices.index(field)
            #     tm["MODEL/element_data"][:, field_index] = fm_field


if __name__ == "__main__":
    """
    Call with python name_of_script from_mesh to_mesh field field_type

    field_type can be either "elemental" or "nodal"
    """
    from_mesh = sys.argv[1]
    to_mesh = sys.argv[2]
    field = sys.argv[3]
    field_type = sys.argv[4]
    if field_type == "elemental":
        move_elemental_field_from_mesh_to_another(from_mesh, to_mesh, field)
    else:
        move_nodal_field_from_mesh_to_another(from_mesh, to_mesh, field)
