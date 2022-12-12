import copy
import json
import sys
import toml
import shutil
import numpy as np
import h5py
from obspy.geodetics import locations2degrees
from inversionson.hpc_processing.window_selection import select_windows
from inversionson.hpc_processing.source_time_function import \
    source_time_function
from inversionson.hpc_processing.utils import select_component_from_stream, build_or_get_receiver_info
from inversionson.hpc_processing.adjoint_source import calculate_adjoint_source
from tqdm import tqdm
from salvus.flow.simple_config import simulation, source, stf, receiver
import multiprocessing
import warnings
import pyasdf
import os


def calculate_station_weight(lat_1: float, lon_1: float, locations: np.ndarray):
    """
    Calculates the weight set for a set of stations for one event

    :param lat_1: latitude of station
    :type lat_1: float
    :param lon_1: longitude of station
    :type lon_1: float
    :param locations: array of latitudes and longitudes of other stations
    :type locations: numpy.ndarray
    :return: weight. weight for this specific station
    :rtype: float
    """

    distance = 1.0 / (
            1.0
            + locations2degrees(lat_1, lon_1, locations[0, :], locations[1, :])
    )
    factor = np.sum(distance) - 1.0
    weight = 1.0 / factor
    assert weight > 0.0
    return weight


def get_adjoint_source_object(event_name, adjoint_filename,
                              receiver_json_path, proc_filename,
                              misfits, forward_meta_json_filename) -> object:
    """
    Generate the adjoint source object for the respective event

    :param event_name: Name of event
    :type event_name: str
    :return: Adjoint source object for salvus
    :rtype: object
    """
    receivers = build_or_get_receiver_info(receiver_json_path, proc_filename)
    adjoint_recs = list(misfits[event_name].keys())

    # Need to make sure I only take receivers with an adjoint source
    adjoint_sources = []
    for rec in receivers:
        if (
                rec["network-code"] + "_" + rec["station-code"] in adjoint_recs
                or rec["network-code"] + "." + rec[
            "station-code"] in adjoint_recs
        ):
            adjoint_sources.append(rec)

    # Build meta_info_dict
    with open(forward_meta_json_filename) as json_file:
        data = json.load(json_file)

    meta_recs = data["forward_run_input"]["output"]["point_data"]["receiver"]
    meta_info_dict = {}
    for rec in meta_recs:
        if (
                rec["network_code"] + "_" + rec["station_code"] in adjoint_recs
                or rec["network_code"] + "." + rec[
            "station_code"] in adjoint_recs
        ):
            rec_name = rec["network_code"] + "_" + rec["station_code"]
            meta_info_dict[rec_name] = {}
            # this is the rotation from XYZ to ZNE,
            # we still need to transpose to get ZNE -> XYZ
            meta_info_dict[rec_name]["rotation_on_input"] = {
                "matrix": np.array(
                    rec["rotation_on_output"]["matrix"]).T.tolist()
            }
            meta_info_dict[rec_name]["location"] = rec["location"]

    adj_src = [
        source.cartesian.VectorPoint3D(
            x=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                "location"
            ][0],
            y=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                "location"
            ][1],
            z=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                "location"
            ][2],
            fx=1.0,
            fy=1.0,
            fz=1.0,
            source_time_function=stf.Custom(
                filename=adjoint_filename,
                dataset_name="/" + rec["network-code"] + "_" + rec[
                    "station-code"],
            ),
            rotation_on_input=meta_info_dict[
                rec["network-code"] + "_" + rec["station-code"]
                ]["rotation_on_input"],
        )
        for rec in adjoint_sources
    ]
    return adj_src


def construct_adjoint_simulation(parameterization,
                                 forward_meta_json_filename,
                                 adj_src: object):
    """
    Create the adjoint simulation object that salvus flow needs
    """
    print("Constructing Adjoint Simulation now")

    with open(forward_meta_json_filename, "r") as fh:
        meta_info = json.load(fh)
    forward_mesh = meta_info["forward_run_input"]["domain"]["mesh"]["filename"]
    gradient = "gradient.h5"
    w = simulation.Waveform(mesh=forward_mesh)
    w.adjoint.forward_meta_json_filename = f"REMOTE:{forward_meta_json_filename}"
    w.adjoint.gradient.parameterization = parameterization
    w.adjoint.gradient.output_filename = gradient
    w.adjoint.gradient.format = "hdf5-full"
    w.adjoint.point_source = adj_src
    with open("output/adjoint_simulation_dict.toml", "w") as fh:
        toml.dump(w.get_dictionary(), fh)


def get_station_weights(list_of_stations, processed_data,
                        receiver_json_path):
    """
    The plan here is to compute the station weights based
    on a list of stations, that are in turn based on the selected windwos.
    :param list_of_stations: List of windows
    :type list_of_stations: List
    """
    weight_set = {}
    if len(list_of_stations) == 1:
        weight_set[list_of_stations[0]]["station_weight"] = 1.0
        return weight_set
    print("Getting station weights...")
    list_of_recs = build_or_get_receiver_info(receiver_json_path, processed_data)
    coordinates = {}
    for rec in list_of_recs:
        station_name = rec["network-code"] + "." + rec["station-code"]
        coordinates[station_name] = {"latitude": rec["latitude"],
                                     "longitude": rec["longitude"]}

    # Make reduced list:
    stations = {}
    for station in list_of_stations:
        stations[station] = {}
        stations[station]["latitude"] = coordinates[station]["latitude"]
        stations[station]["longitude"] = coordinates[station]["longitude"]

    locations = np.zeros((2, len(stations)), dtype=np.float64)

    for _i, station in enumerate(stations):
        locations[0, _i] = stations[station]["latitude"]
        locations[1, _i] = stations[station]["longitude"]

    sum_value = 0.0
    weight_set = {}
    for station in tqdm(stations):
        weight_set[station] = {}
        weight = calculate_station_weight(
            lat_1=stations[station]["latitude"],
            lon_1=stations[station]["longitude"],
            locations=locations,
        )
        sum_value += weight
        weight_set[station]["station_weight"] = weight

    for station in stations:
        weight_set[station]["station_weight"] *= (
                len(stations) / sum_value
        )

    print("Station weights computed")
    return weight_set


def run(info):
    """
    This function takes a processed data file, a synthetic data file
    and an info toml and it writes the adjoint sources and a misfit toml.
    This avoids the need to download the synthetics and upload the adjoint
    sources, and offloads computations to the remote.
    """
    warnings.filterwarnings("ignore")

    num_processes = multiprocessing.cpu_count()

    global _window_select
    global _process

    # READ INPUT DICT
    processed_filename = info["processed_filename"]
    synthetic_filename = info["synthetic_filename"]
    event_name = info["event_name"]
    delta = info["delta"]
    npts = info["npts"]

    minimum_period = info["minimum_period"]
    maximum_period = info["maximum_period"]
    freqmin = 1.0 / minimum_period
    freqmax = 1.0 / maximum_period
    start_time_in_s = info["start_time_in_s"]

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if "windowing_needed" in info.keys():
        windowing_needed = info["windowing_needed"]
    else:
        windowing_needed = True

    if "window_path" in info.keys():
        window_path = os.path.join(output_folder, "windows.json")
    else:
        window_path = None

    if "misfit_json_filename" in info.keys():
        misfit_json_filename = os.path.join(output_folder, "misfits.json")
    else:
        misfit_json_filename = None

    if "scale_data_to_synthetics" in info.keys():
        scale_data_to_synthetics = info["scale_data_to_synthetics"]
    else:
        scale_data_to_synthetics = True

    if not os.path.exists(processed_filename):
        raise Exception(f"File {processed_filename} does not exists.")

    if not os.path.exists(synthetic_filename):
        raise Exception(f"File {synthetic_filename} does not exists.")

    # Get source time function (required for windowing algorithm)
    stf_trace = source_time_function(
        npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax
    )

    # Get source info.
    with pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False) as ds:
        event = ds.events[0]
        org = event.preferred_origin() or event.origins[0]
        event_latitude = org.latitude
        event_longitude = org.longitude

    def _window_select(station):
        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
        ds_synth = pyasdf.ASDFDataSet(
            synthetic_filename, mode="r", mpi=False
        )
        observed_station = ds.waveforms[station]
        synthetic_station = ds_synth.waveforms[station]

        obs_tag = observed_station.get_waveform_tags()
        syn_tag = synthetic_station.get_waveform_tags()

        try:
            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                    "Station: %s - Requires 1 observed waveform tag. Has %i."
                    % (observed_station._station_name, len(obs_tag))
            )
            assert len(syn_tag) == 1, (
                    "Station: %s - Requires 1 synthetic waveform tag. Has %i."
                    % (observed_station._station_name, len(syn_tag))
            )
        except AssertionError:
            return {station: None}

        obs_tag = obs_tag[0]
        syn_tag = syn_tag[0]

        # Finally get the data.
        st_obs = observed_station[obs_tag]
        st_syn = synthetic_station[syn_tag]

        # Extract coordinates once.
        try:
            coordinates = observed_station.coordinates
        except Exception as e:
            print(e)
            return {station: None}

        all_windows = {}
        for tr in st_syn:
            tr.stats.starttime = (
                    org.time.timestamp + start_time_in_s)
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(st_obs, component)
                synth_tr = select_component_from_stream(st_syn, component)

                # I THINK I SHOULD SAMPLE at the period of the data
                # make sure traces match in length and sampling rate.
                data_tr.interpolate(sampling_rate=synth_tr.stats.sampling_rate,
                                     method="linear")
                data_tr.trim(endtime=synth_tr.stats.endtime)
                synth_tr.trim(endtime=data_tr.stats.endtime)
                if scale_data_to_synthetics:
                    scaling_factor = (
                            synth_tr.data.ptp() / data_tr.data.ptp()
                    )
                    # Store and apply the scaling.
                    data_tr.stats.scaling_factor = scaling_factor
                    data_tr.data *= scaling_factor

            except Exception as e:
                continue

            windows = None
            try:
                windows = select_windows(
                    data_tr,
                    synth_tr,
                    stf_trace,
                    event_latitude,
                    event_longitude,
                    coordinates["latitude"],
                    coordinates["longitude"],
                    minimum_period=minimum_period,
                    maximum_period=maximum_period,
                )
            except Exception as e:
                print(e)

            if not windows:
                continue
            all_windows[data_tr.id] = windows

        if all_windows:
            return {station: all_windows}
        else:
            return {station: None}

    # Generate task list
    with pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False) as ds:
        task_list = ds.waveforms.list()

    if len(task_list) < 1:
        raise Exception("No processed data found.")

    # Use at most num_processes workers
    number_processes = min(num_processes, len(task_list))

    if windowing_needed:
        # Open Pool of workers
        print("Starting window selection", flush=True)
        with multiprocessing.Pool(number_processes) as pool:
            all_windows = {}
            with tqdm(total=len(task_list)) as pbar:
                for i, r in enumerate(
                        pool.imap_unordered(_window_select, task_list)
                ):
                    pbar.update()
                    k, v = r.popitem()
                    all_windows[k] = v

            pool.close()
            pool.join()
        if window_path:
            with open(window_path, "w") as outfile:
                json.dump(all_windows, outfile)
            shutil.copy(window_path, info["window_path"])
    else:
        if not info["window_path"]:
            raise Exception("I need at least a path to windows "
                            "if we don't select them.")
        with open(info["window_path"]) as json_file:
            all_windows = json.load(json_file)


    # Write files with a single worker
    print("Finished window selection", flush=True)
    sta_with_windows = [k for k, v in all_windows.items() if v is not None]
    num_sta_with_windows = len(sta_with_windows)
    print(
        f"Selected windows for {num_sta_with_windows} out of "
        f"{len(task_list)} stations."
    )

    station_weights = get_station_weights(sta_with_windows, processed_filename,
                                          info["receiver_json_path"])

    #TODO: windows are now in timestamp format. For use with obspy we need to convert to UTCDatetime

    ###########################################################################
    # ADJOINT SOURCE CALCULATIONS
    ###########################################################################

    if "ad_src_type" in info.keys():
        ad_src_type = info["ad_src_type"]
    else:
        ad_src_type = "tf_phase_misfit"

    env_scaling = False

    if len(all_windows.keys()) == 0:
        raise Exception(f"No windows were found")

    def _process(station):
        ds = pyasdf.ASDFDataSet(processed_filename, mode="r", mpi=False)
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r",
                                      mpi=False)
        observed_station = ds.waveforms[station]
        synthetic_station = ds_synth.waveforms[station]

        # print(observed_station, synthetic_station)
        obs_tag = observed_station.get_waveform_tags()
        syn_tag = synthetic_station.get_waveform_tags()
        adjoint_sources = {}
        try:
            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                    "Station: %s - Requires 1 observed waveform tag. Has %i."
                    % (observed_station._station_name, len(obs_tag))
            )
            assert len(syn_tag) == 1, (
                    "Station: %s - Requires 1 synthetic waveform tag. Has %i."
                    % (observed_station._station_name, len(syn_tag))
            )
        except AssertionError:
            return {station: adjoint_sources}

        obs_tag = obs_tag[0]
        syn_tag = syn_tag[0]

        # Finally get the data.
        st_obs = observed_station[obs_tag]
        st_syn = synthetic_station[syn_tag]

        # Set the same starttime, this is important for the window_trace function
        st_syn = copy.deepcopy(st_syn)
        for tr in st_syn:
            tr.stats.starttime = (
                    org.time.timestamp + start_time_in_s)
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(st_obs, component)
                synth_tr = select_component_from_stream(st_syn, component)
                # make sure sampled at the same rate
                # Make sure synthetics is sampled at the same
                # rate and data matches the synthetics in terms of endtime
                # start time should happen automatically.
                data_tr.interpolate(sampling_rate=synth_tr.stats.sampling_rate,
                                    method="linear")
                data_tr.trim(endtime=synth_tr.stats.endtime)
                synth_tr.trim(endtime=data_tr.stats.endtime)

            except Exception as e:
                continue

            if scale_data_to_synthetics:
                scaling_factor = (
                        synth_tr.data.ptp() / data_tr.data.ptp()
                )
                # Store and apply the scaling.
                data_tr.stats.scaling_factor = scaling_factor
                data_tr.data *= scaling_factor

            net, sta, cha = data_tr.id.split(".", 2)
            station = net + "." + sta

            if station not in all_windows:
                continue
            if all_windows[station] == None:
                continue
            if data_tr.id not in all_windows[station]:
                continue
            # Collect all.
            windows = all_windows[station][data_tr.id]
            try:
                # for window in windows:
                misfit, adj_source = calculate_adjoint_source(
                    observed=data_tr,
                    synthetic=synth_tr,
                    window=windows,
                    min_period=minimum_period,
                    max_period=maximum_period,
                    adj_src_type=ad_src_type,
                    taper_ratio=0.15,
                    taper_type="cosine",
                    envelope_scaling=env_scaling,
                )

            except Exception as e:
                print(e)
                continue

            adjoint_sources[data_tr.id] = {
                "misfit": misfit,
                "adj_source": adj_source,
            }
        # TODO figure out what happens when no adjoint source is calculated
        adj_dict = {station: adjoint_sources}
        return adj_dict

    # Only loop over stations with windows.
    task_list = sta_with_windows

    # Use at most num_processes
    number_processes = min(num_processes, len(task_list))

    if len(task_list) < 1:
        raise Exception("At least one window is needed to compute"
                        "an adjoint source.")

    print("Starting adjoint source calculation")
    with multiprocessing.Pool(number_processes) as pool:
        all_adj_srcs = {}
        with tqdm(total=len(task_list)) as pbar:
            for i, r in enumerate(pool.imap_unordered(_process, task_list)):
                pbar.update()
                k, v = r.popitem()
                all_adj_srcs[k] = v

        pool.close()
        pool.join()

    # Write adjoint sources # TODO add station weighting
    sta_with_sources = [k for k, v in all_adj_srcs.items() if v]
    num_sta_with_sources = len(sta_with_sources)

    if num_sta_with_sources < 1:
        raise Exception("No adjoint sources calculated, Please consider "
                        "what to do.")

    print(f"Calculated adjoint sources for {num_sta_with_sources} out of "
          f"{len(task_list)} stations with windows.")

    print("Writing adjoint sources")
    adjoint_source_file_name = os.path.join(output_folder, "stf.h5")
    f = h5py.File(adjoint_source_file_name, "w")

    for station in all_adj_srcs.keys():
        all_sta_channels = list(all_adj_srcs[station].keys())

        if len(all_sta_channels) > 0:
            e_comp = np.zeros_like(
                all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)
            n_comp = np.zeros_like(
                all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)
            z_comp = np.zeros_like(
                all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)

            for channel in all_sta_channels:
                # check channel and set component
                if channel[-1] == "E":
                    e_comp = all_adj_srcs[station][channel]["adj_source"].data
                elif channel[-1] == "N":
                    n_comp = all_adj_srcs[station][channel]["adj_source"].data
                elif channel[-1] == "Z":
                    z_comp = all_adj_srcs[station][channel]["adj_source"].data

            zne = np.array((z_comp, n_comp, e_comp)) * \
                  station_weights[station]["station_weight"]
            # replace name to match the forward output run from salvus
            new_station_name = station.replace(".", "_")
            source = f.create_dataset(new_station_name, data=zne.T)
            source.attrs["dt"] = delta
            source.attrs["sampling_rate_in_hertz"] = 1.0 / delta
            source.attrs["spatial-type"] = np.string_("vector")
            source.attrs["start_time_in_seconds"] = start_time_in_s

    # Write misfits (currently unused in the ADAM workflow other than for
    # figuring out which stations should have adjoint sources.
    misfit_dict = {}
    total_misfit = 0
    for station in all_adj_srcs.keys():
        all_sta_channels = list(all_adj_srcs[station].keys())
        if not len(all_sta_channels) > 0:
            continue
        misfit_dict[station] = {}
        for trace in all_adj_srcs[station].keys():
            station_tr_misfit = all_adj_srcs[station][trace]["misfit"] * \
                station_weights[station]["station_weight"]

            misfit_dict[station][trace] = station_tr_misfit
            total_misfit += station_tr_misfit

    misfit_dict["total_misfit"] = total_misfit
    event_misfit_dict = {event_name: misfit_dict}

    if misfit_json_filename:
        print("Writing misfit dict...")
        with open(misfit_json_filename, "w") as outfile:
            json.dump(event_misfit_dict, outfile)
        shutil.copy(misfit_json_filename, info["misfit_json_filename"])


    # now create adjoint source simulation object
    adjoint_filename = "REMOTE:" + os.path.abspath(adjoint_source_file_name)
    adj_src = get_adjoint_source_object(event_name,
                                        adjoint_filename,
                                        info["receiver_json_path"],
                                        processed_filename,
                                        event_misfit_dict,
                                        info["forward_meta_json_filename"])

    construct_adjoint_simulation(info["parameterization"],
                                 info["forward_meta_json_filename"], adj_src)


if __name__ == "__main__":
    toml_filename = sys.argv[1]
    # toml_filename = "/Users/dirkphilip/Software/Inversionson/test_remote_window/info_dict.toml"
    info = toml.load(toml_filename)
    run(info)
