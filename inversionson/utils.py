"""
A collection of useful scripts which don't really fall into one of
the components.
Stuff like cutting away sources and receivers and such.
Maybe one day some of these will be moved to a handyman component
or something like that.
"""

from typing import Union, Tuple
import numpy as np
import pathlib
import os
import h5py

__FILE_TEMPLATES_DIR = pathlib.Path(__file__).parent / "file_templates"


def write_xdmf(filename: Union[pathlib.Path, str]):
    """
    Takes a path to an h5 file and writes the accompanying xdmf file.

    :param filename: Filename of the h5 file that needs an xdmf file.
    Should be a pathlib.Path or str.
    """
    filename = pathlib.Path(filename)

    xdmf_attribute_path = __FILE_TEMPLATES_DIR / "attribute.xdmf"
    base_xdmf_path = __FILE_TEMPLATES_DIR / "base_xdmf.xdmf"

    # Get relevant info from h5 file
    with h5py.File(filename, "r") as h5:
        num_big_elements = h5["MODEL"]["coordinates"].shape[0]
        nodes_per_element = h5["MODEL"]["coordinates"].shape[1]
        dimension = h5["MODEL"]["coordinates"].shape[2]
        total_points = num_big_elements * nodes_per_element
        tensor_order = round(nodes_per_element ** (1 / dimension) - 1)
        num_sub_elements = h5["TOPOLOGY"]["cells"].shape[0] * tensor_order**3

        dim_labels = h5["MODEL/data"].attrs.get("DIMENSION_LABELS")[1][1:-1]
        if type(dim_labels) != str:
            dim_labels = dim_labels.decode()
        dim_labels = dim_labels.replace(" ", "").split("|")

    # Write all atrributes
    with open(xdmf_attribute_path, "r") as fh:
        attribute_string = fh.read()
    all_attributes = ""

    final_filename = filename.name
    for i, attribute in enumerate(dim_labels):
        if i != 0:
            all_attributes += "\n"
        all_attributes += attribute_string.format(
            num_points=total_points,
            parameter=attribute,
            parameter_idx=i,
            num_elements=num_big_elements,
            nodes_per_element=nodes_per_element,
            num_parameters=len(dim_labels),
            filename=final_filename,
        )
    with open(base_xdmf_path, "r") as fh:
        base_string = fh.read().format(
            number_of_sub_elements=num_sub_elements,
            filename=final_filename,
            num_points=total_points,
            attributes=all_attributes,
        )

    xdmf_filename = ".".join(final_filename.split(".")[:-1]) + ".xdmf"
    complete_xdmf_filename = filename.parent / xdmf_filename

    with open(complete_xdmf_filename, "w") as fh:
        fh.write(base_string)


def get_window_filename(event: str, iteration: str) -> str:
    return f"{event}_{iteration}_windows.json"


def get_misfits_filename(event: str, iteration: str) -> str:
    return f"{event}_{iteration}_misfits.json"


def latlondepth_to_cartesian(
    lat: float, lon: float, depth_in_km: float = 0.0
) -> Tuple[float, float, float]:
    """
    Go from lat, lon, depth to cartesian coordinates

    :param lat: Latitude
    :type lat: float
    :param lon: Longitude
    :type lon: float
    :param depth_in_km: Depth in kilometers, Optional
    :type depth_in_km: float, defaults to 0.0
    :return: x,y,z coordinates
    :rtype: np.ndarray
    """
    R = (6371.0 - depth_in_km) * 1000.0
    lat *= np.pi / 180.0
    lon *= np.pi / 180.0
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return x, y, z


def cut_source_region_from_gradient(
    mesh: str, source_location: dict, radius_to_cut: float
):
    """
    Sources often show unreasonable sensitivities. This function
    brings the value of the gradient down to zero for that region.
    I recommend doing this before smoothing.

    :param mesh: Path to the mesh
    :type mesh: str
    :param source_location: Source latitude, longitude and depth
    :type source_location: dict
    :param radius_to_cut: Radius to cut in km
    :type radius_to_cut: float
    """
    gradient = h5py.File(mesh, "r+")
    coordinates = gradient["MODEL/coordinates"]
    data = gradient["MODEL/data"]
    # TODO: Maybe I should implement this in a way that it uses predefined
    # params. Then I only need to find out where they are
    if isinstance(source_location, list):
        source_location = source_location[0]
    s_x, s_y, s_z = latlondepth_to_cartesian(
        lat=source_location["latitude"],
        lon=source_location["longitude"],
        depth_in_km=source_location["depth_in_m"] / 1000.0,
    )

    dist = np.sqrt(
        (coordinates[:, :, 0] - s_x) ** 2
        + (coordinates[:, :, 1] - s_y) ** 2
        + (coordinates[:, :, 2] - s_z) ** 2
    ).ravel()

    cut_indices = np.where(dist < radius_to_cut * 1000.0)

    for i in range(data.shape[1]):
        tmp_dat = data[:, i, :].ravel()
        tmp_dat[cut_indices] = 0.0
        tmp_dat = np.reshape(tmp_dat, (data.shape[0], 1, data.shape[2]))
        if i == 0:
            cut_data = tmp_dat.copy()
        else:
            cut_data = np.concatenate((cut_data, tmp_dat), axis=1)
    data[:, :, :] = cut_data

    gradient.close()


def get_h5_parameter_indices(filename, parameters):
    """Get indices in h5 file for parameters in filename"""
    with h5py.File(filename, "r") as h5:
        h5_data = h5["MODEL/data"]
        # Get dimension indices of relevant parameters
        # These should be constant for all gradients, so this is only done
        # once.
        dim_labels = h5_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
        if type(dim_labels) != str:
            dim_labels = dim_labels.decode()
        dim_labels = dim_labels.replace(" ", "").split("|")
        indices = [dim_labels.index(param) for param in parameters]
    return indices


def sum_two_parameters_h5(filename: Union[str, pathlib.Path], parameters):
    """sum two parameters in h5 file. Mostly used for summing VPV and VPH"""
    if not os.path.exists(filename):
        raise FileNotFoundError("only works on existing files.")

    indices = get_h5_parameter_indices(filename, parameters)
    indices.sort()

    if len(indices) != 2:
        raise ValueError("Only implemented for 2 fields.")

    with h5py.File(filename, "r+") as h5:
        dat = h5["MODEL/data"]
        # TODO: DP it seems the below can be simplified. Look into this at some point.
        data_copy = dat[:, :, :].copy()
        par_sum = data_copy[:, indices[0], :] + data_copy[:, indices[1], :]
        data_copy[:, indices[0], :] = par_sum
        data_copy[:, indices[1], :] = par_sum

        dat[:, indices, :] = data_copy[:, indices, :]
