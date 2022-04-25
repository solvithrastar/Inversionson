"""
A collection of useful scripts which don't really fall into one of
the components.
Stuff like cutting away sources and receivers and such.
Maybe one day some of these will be moved to a handyman component
or something like that.
"""

import numpy as np
import os, sys
import h5py
import time


def _print(
    comm,
    message,
    color="white",
    line_above=False,
    line_below=False,
    emoji_alias=":ear:",
):
    comm.storyteller.printer.print(
        message=message,
        color=color,
        line_above=line_above,
        line_below=line_below,
        emoji_alias=emoji_alias,
    )


def sleep_or_process(comm):
    """
    This functions tries to process a random unprocessed event
    or sleeps if all are processed.
    """
    if (
        comm.project.random_event_processing
        and not comm.lasif.process_random_unprocessed_event()
    ):
        _print(
            comm,
            f"Seems like there is nothing to do now "
            f"I might as well process some random event.",
            emoji_alias=None,
        )
    else:
        _print(
            comm,
            f"Waiting {comm.project.sleep_time_in_s} seconds before trying again",
            emoji_alias=":clock:",
        )
        time.sleep(comm.project.sleep_time_in_s)


def latlondepth_to_cartesian(lat: float, lon: float, depth_in_km=0.0) -> np.ndarray:
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


def find_parameters_in_dataset(dataset) -> list:
    """
    Figure out which parameter is in dataset and where

    :param dataset: hdf5 dataset with a dimension labels
    :type dataset: hdf5 dataset
    :return: parameters
    :rtype: list
    """
    dims = dataset.attrs.get("DIMENSION_LABELS")[1].decode()
    params = dims[2:-2].split("|").replace(" ", "")
    return params


def add_dimension_labels(mesh, parameters: list):
    """
    Label the dimensions in a newly created dataset

    :param dataset: Loaded mesh as an hdf5 file
    :type dataset: hdf5 file
    :param parameters: list of parameters
    :type parameters: list
    """
    dimstr = "[ " + " | ".join(parameters) + " ]"
    mesh["MODEL/data"].dims[0].label = "element"
    mesh["MODEL/data"].dims[1].label = dimstr
    mesh["MODEL/data"].dims[2].label = "point"


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


def cut_receiver_regions_from_gradient(
    mesh: str, receivers: dict, radius_to_cut: float
):
    """
    Remove regions around receivers from gradients. Receivers often have an
    imprint on a model and this aims to fight that effect.

    :param mesh: Path to a mesh with a gradient
    :type mesh: str
    :param receivers: key: receivers{'lat': , 'lon':}
    :type receivers: dict
    :param radius_to_cut: Radius to cut gradient in km
    :type radius_to_cut: float
    """

    gradient = h5py.File(mesh, "r+")
    coordinates = gradient["MODEL/coordinates"]
    data = gradient["MODEL/data"]
    # TODO: Maybe I should implement this in a way that it uses predefined
    # params. Then I only need to find out where they are

    for _i, rec in enumerate(receivers):
        x_r, y_r, z_r = latlondepth_to_cartesian(
            lat=rec["latitude"], lon=rec["longitude"]
        )
        dist = np.sqrt(
            (coordinates[:, :, 0] - x_r) ** 2
            + (coordinates[:, :, 1] - y_r) ** 2
            + (coordinates[:, :, 2] - z_r) ** 2
        ).ravel()
        if _i == 0:
            close_by = np.where(dist < radius_to_cut * 1000.0)[0]
        else:
            tmp_close = np.where(dist < radius_to_cut * 1000.0)
            if tmp_close[0].shape[0] == 0:
                continue
            if close_by.shape[0] == 0:
                close_by = tmp_close[0]
                continue
            close_by = np.concatenate((close_by, tmp_close[0]))

    close_by = np.unique(close_by)

    for i in range(data.shape[1]):
        parameter = data[:, i, :].ravel()
        parameter[close_by] = 0.0
        parameter = np.reshape(parameter, (data.shape[0], 1, data.shape[2]))
        if i == 0:
            cut_data = parameter.copy()
        else:
            cut_data = np.concatenate((cut_data, parameter), axis=1)
    data[:, :, :] = cut_data

    gradient.close()


def clip_gradient(mesh: str, percentile: float, parameters: list):
    """
    Clip the gradient to remove abnormally high/low values from it.
    Discrete gradients sometimes have the problem of unphysically high
    values, especially at source/receiver locations so this should be
    taken care of by cutting out a region around these.

    :param mesh: Path to mesh containing gradient
    :type mesh: str
    :param percentile: The percentile at which you want to clip the gradient
    :type percentile: float
    :param parameters: Parameters to clip
    :type parameters: list
    """
    gradient = h5py.File(mesh, "r+")
    data = gradient["MODEL/data"]
    dim_labels = (
        data.attrs.get("DIMENSION_LABELS")[1].decode()[1:-1].replace(" ", "").split("|")
    )
    indices = []
    for param in parameters:
        indices.append(dim_labels.index(param))
    clipped_data = data[:, :, :].copy()

    for i in indices:
        clipped_data[:, i, :] = np.clip(
            data[:, i, :],
            a_min=np.quantile(data[:, i, :], 1.0 - percentile),
            a_max=np.quantile(data[:, i, :], percentile),
        )
    data[:, :, :] = clipped_data
    gradient.close()


def get_h5_parameter_indices(filename, parameters):
    """Get indices in h5 file for parameters in filename"""
    with h5py.File(filename, "r") as h5:
        h5_data = h5["MODEL/data"]
        # Get dimension indices of relevant parameters
        # These should be constant for all gradients, so this is only done
        # once.
        dim_labels = h5_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
        if not type(dim_labels) == str:
            dim_labels = dim_labels.decode()
        dim_labels = dim_labels.replace(" ", "").split("|")
        indices = []
        for param in parameters:
            indices.append(dim_labels.index(param))
    return indices


def sum_two_parameters_h5(filename, parameters):
    """sum two parameters in h5 file. Mostly used for summing VPV and VPH"""
    if not os.path.exists(filename):
        raise Exception("only works on existing files.")

    indices = get_h5_parameter_indices(filename, parameters)
    indices.sort()

    if len(indices) != 2:
        raise Exception("Only implemented for 2 fields.")

    with h5py.File(filename, "r+") as h5:
        dat = h5["MODEL/data"]
        data_copy = dat[:, :, :].copy()
        par_sum = data_copy[:, indices[0], :] + data_copy[:, indices[1], :]
        data_copy[:, indices[0], :] = par_sum
        data_copy[:, indices[1], :] = par_sum

        dat[:, indices, :] = data_copy[:, indices, :]


def sum_gradients(mesh: str, gradients: list):
    """
    Sum the parameters on gradients for a list of events in an iteration

    :param mesh: Path to a mesh to be used to store the summed gradients on
    make sure it exists and is of the same dimensions as the others.
    :type mesh: str
    :param gradients: List of paths to gradients to be summed
    :type gradients: list
    """
    # Read in the fields for these gradients and sum them accordingly
    # store on a single mesh.
    from salvus.mesh.unstructured_mesh import UnstructuredMesh

    m = UnstructuredMesh.from_h5(mesh)
    fields = UnstructuredMesh.from_h5(gradients[0]).element_nodal_fields.keys()

    for _i, gradient in enumerate(gradients):
        print(f"Adding gradient {_i+1} of {len(gradients)}")
        grad = UnstructuredMesh.from_h5(gradient)
        for field in fields:
            if _i == 0:
                m.attach_field(field, np.zeros_like(grad.element_nodal_fields[field]))
            m.element_nodal_fields[field] += grad.element_nodal_fields[field]

    m.write_h5(mesh)


def double_fork():
    print("\n \n Attempting to DoubleFork \n \n")
    try:
        pid = os.fork()
        if pid > 0:
            print("I am in parent process and I will exit")
            sys.exit(0)
    except OSError:
        print("Fork failed")
        sys.exit(1)

    os.chdir("/")
    os.setsid()
    os.umask(0)

    # second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError:
        print("Fork 2 failed")
