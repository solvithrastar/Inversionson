import h5py
import numpy as np


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
        data.attrs.get("DIMENSION_LABELS")[1][1:-1]
        .replace(" ", "")
        .split("|")
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


def latlondepth_to_cartesian(
    lat: float, lon: float, depth_in_km=0.0
) -> np.ndarray:
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

