import obspy
import pyasdf
import os
import json


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
                rec = {}
                net, sta = station.split(".")
                lat = all_coords[station]["latitude"]
                lon = all_coords[station]["longitude"]

                rec["latitude"] = lat
                rec["longitude"] = lon
                rec["network-code"] = net
                rec["station-code"] = sta
                all_recs.append(rec)

        with open(receiver_json_path, "w") as outfile:
            json.dump(all_recs, outfile)
        return all_recs
    else:
        # Opening JSON file
        with open(receiver_json_path, 'r') as openfile:
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
        raise Exception(
            "Component %s not found in Stream." % component
        )
    elif len(component) > 1:
        raise Exception(
            "More than 1 Trace with component %s found "
            "in Stream." % component
        )
    return component[0]