"""
A collection of useful scripts which don't really fall into one of
the components.
Stuff like cutting away sources and receivers and such.
Maybe one day some of these will be moved to a handyman component
or something like that.
"""

import numpy as np
import os
import h5py


def latlondepth_to_cartesian(lat: float, lon: float,
                             depth_in_km=0.0) -> np.ndarray:
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
    lat *= (np.pi / 180.0)
    lon *= (np.pi / 180.0)
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
    dimstr = '[ ' + ' | '.join(parameters) + ' ]'
    mesh['MODEL/data'].dims[0].label = 'element'
    mesh['MODEL/data'].dims[1].label = dimstr
    mesh['MODEL/data'].dims[2].label = 'point'


def cut_source_region_from_gradient(mesh: str, source_location: dict,
                                    radius_to_cut: float):
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
        depth_in_km=source_location["depth_in_m"] * 1000.0
    )

    dist = np.sqrt((coordinates[:, :, 0] - s_x) ** 2 +
                   (coordinates[:, :, 1] - s_y) ** 2 +
                   (coordinates[:, :, 2] - s_z) ** 2).ravel()

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


def cut_receiver_regions_from_gradient(mesh: str, receivers: dict,
                                       radius_to_cut: float):
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
            lat=rec["latitude"],
            lon=rec["longitude"]
        )
        dist = np.sqrt((coordinates[:, :, 0] - x_r) ** 2 +
                       (coordinates[:, :, 1] - y_r) ** 2 +
                       (coordinates[:, :, 2] - z_r) ** 2).ravel()
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


def clip_gradient(mesh: str, percentile: float):
    """
    Clip the gradient to remove abnormally high/low values from it.
    Discrete gradients sometimes have the problem of unphysically high
    values, especially at source/receiver locations so this should be
    taken care of by cutting out a region around these.

    :param mesh: Path to mesh containing gradient
    :type mesh: str
    :param percentile: The percentile at which you want to clip the gradient
    :type percentile: float
    """
    gradient = h5py.File(mesh, "r+")
    data = gradient["MODEL/data"]

    clipped_data = data[:, :, :].copy()

    for i in range(data.shape[1]):
        clipped_data[:, i, :] = np.clip(data[:, i, :],
                                        a_min=np.quantile(
                                            data[:, i, :],
                                            1.0 - percentile),
                                        a_max=np.quantile(
                                            data[:, i, :],
                                            percentile))
    data[:, :, :] = clipped_data
    gradient.close()

