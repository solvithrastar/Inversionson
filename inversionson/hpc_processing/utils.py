import obspy
import pyasdf
import os
import json
import math


def elliptic_to_geocentric_latitude(
    lat: float, axis_a: float = 6378137.0, axis_b: float = 6356752.314245
) -> float:
    """
    Convert latitudes defined on an ellipsoid to a geocentric one.
    Based on Salvus Seismo

    :param lat: Latitude to convert
    :type lat: float
    :param axis_a: Major axis of planet in m, defaults to 6378137.0
    :type axis_a: float, optional
    :param axis_b: Minor axis of planet in m, defaults to 6356752.314245
    :type axis_b: float, optional
    :return: Converted latitude
    :rtype: float

    >>> elliptic_to_geocentric_latitude(0.0)
    0.0
    >>> elliptic_to_geocentric_latitude(90.0)
    90.0
    >>> elliptic_to_geocentric_latitude(-90.0)
    -90.0
    """
    _f = (axis_a - axis_b) / axis_a
    if abs(lat) < 1e-6 or abs(lat - 90) < 1e-6 or abs(lat + 90) < 1e-6:
        return lat

    E_2 = 2 * _f - _f**2
    return math.degrees(math.atan((1 - E_2) * math.tan(math.radians(lat))))


def build_or_get_receiver_info(receiver_json_path, asdf_file_path):
    """
    Returns a list of dict with receiver information that
    is compiled from informatio in an ASDF file. If the file exists already,
    it simply returns the list of dicts without compiling it first.

    :param receiver_json_path: path where the receiver file should be found
    or written to
    :type receiver_json_path: str
    :param asdf_file_path: Path to the asdf file from which the receiver info
    is extracted.
    :type asdf_file_path; str
    :tyoe
    """

    if not os.path.exists(receiver_json_path):
        with pyasdf.ASDFDataSet(asdf_file_path, mode="r") as ds:
            all_coords = ds.get_all_coordinates()

            # build list of dicts
            all_recs = []
            for station in all_coords.keys():
                net, sta = station.split(".")
                lat = all_coords[station]["latitude"]
                lon = all_coords[station]["longitude"]

                rec = {
                    "latitude": elliptic_to_geocentric_latitude(lat),
                    "longitude": lon,
                }
                rec["network-code"] = net
                rec["station-code"] = sta
                all_recs.append(rec)

        with open(receiver_json_path, "w") as outfile:
            json.dump(all_recs, outfile)
    else:
        # Opening JSON file
        with open(receiver_json_path, "r") as openfile:
            # Reading from json file
            all_recs = json.load(openfile)

    return all_recs


def select_component_from_stream(st: obspy.core.Stream, component: str):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.

    :param st: Obspy stream
    :type st: obspy.core.Stream
    :param component: Name of component of stream
    :type component: str
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise Exception(f"Component {component} not found in Stream.")
    elif len(component) > 1:
        raise Exception(
            f"More than 1 Trace with component {component} found in Stream."
        )
    return component[0]
