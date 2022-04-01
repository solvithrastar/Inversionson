import copy
import sys
import toml
import numpy as np
import h5py


def run(info):
    """
    This function takes a processed data file, a synthetic data file
    and an info toml and it writes the adjoint sources and a misfit toml.
    This avoids the need to download the synthetics and upload the adjoint
    sources, and offloads computations to the remote.
    """
    from inversionson.hpc_processing.window_selection import select_windows
    from inversionson.hpc_processing.source_time_function import \
        source_time_function
    from inversionson.hpc_processing.utils import select_component_from_stream
    from inversionson.hpc_processing.adjoint_source import calculate_adjoint_source
    from tqdm import tqdm
    import multiprocessing
    import warnings
    import pyasdf
    import os

    warnings.filterwarnings("ignore")

    num_processes = multiprocessing.cpu_count()

    global _window_select
    global _process


    # READ INPUT DICT
    processed_filename = info["processed_filename"]
    synthetic_filename = info["synthetic_filename"]
    window_set_name = info["window_set_name"]
    event_name = info["event_name"]
    delta = info["delta"]
    npts = info["npts"]
    iteration_name = info["iteration_name"]
    minimum_period = info["minimum_period"]
    maximum_period = info["maximum_period"]
    freqmin = 1.0 / minimum_period
    freqmax = 1.0 / maximum_period
    start_time_in_s = info["start_time_in_s"]

    if "weight_set" in info.keys():
        weight_set = info["weight_set"]
    else:
        weight_set = False

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
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(st_obs, component)
                synth_tr = select_component_from_stream(st_syn, component)

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

    # Open Pool of workers
    print("Starting window selection", flush=True)
    with multiprocessing.Pool(number_processes) as pool:
        results = {}
        with tqdm(total=len(task_list)) as pbar:
            for i, r in enumerate(
                pool.imap_unordered(_window_select, task_list)
            ):
                pbar.update()
                k, v = r.popitem()
                results[k] = v

        pool.close()
        pool.join()

    # Write files with a single worker
    print("Finished window selection", flush=True)
    sta_with_windows = [k for k, v in results.items() if v is not None]
    num_sta_with_windows = len(sta_with_windows)
    print(
        f"Selected windows for {num_sta_with_windows} out of "
        f"{len(task_list)} stations."
    )

    all_windows = results
    # Toml dumping the windows doesn't quite work because they are objects.
    # I should probably dump the timestamps instead, but do we really need to
    # keep the windows?
    # with open(window_set_name, "w") as fh:
    #     toml.dump(results, fh)

    ###########################################################################
    # ADJOINT SOURCE CALCULATIONS
    ###########################################################################

    if "ad_src_type" in info.keys():
        ad_src_type = info["ad_src_type"]
    else:
        ad_src_type = "tf_phase_misfit"

    if ad_src_type != "tf_phase_misfit":
        raise NotImplemented()

    # TODO make it work for more adjoint sources later
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
                    window_set=window_set_name,
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

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print("Writing adjoint sources")
    adjoint_source_file_name = os.path.join(output_folder, "stf.h5")
    f = h5py.File(adjoint_source_file_name, "w")

    for station in all_adj_srcs.keys():
        all_sta_channels = list(all_adj_srcs[station].keys())

        if len(all_sta_channels) > 0:
            e_comp = np.zeros_like(all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)
            n_comp = np.zeros_like(all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)
            z_comp = np.zeros_like(all_adj_srcs[station][all_sta_channels[0]]["adj_source"].data)

            for channel in all_sta_channels:
                # check channel and set component
                if channel[-1] == "E":
                    e_comp = all_adj_srcs[station][channel]["adj_source"].data
                elif channel[-1] == "N":
                    n_comp = all_adj_srcs[station][channel]["adj_source"].data
                elif channel[-1] == "Z":
                    z_comp = all_adj_srcs[station][channel]["adj_source"].data

            zne = np.array((z_comp, n_comp, e_comp))
            # replace name to match the forward output run from salvus
            new_station_name = station.replace(".", "_")
            source = f.create_dataset(new_station_name, data=zne.T)
            source.attrs["dt"] = delta
            source.attrs["sampling_rate_in_hertz"] = 1.0 / delta
            source.attrs["spatial-type"] = np.string_("vector")
            source.attrs["start_time_in_seconds"] = start_time_in_s

    # Write misfits # TODO add station weighting
    misfit_dict = {}
    for station in all_adj_srcs.keys():
        misfit_dict[station] = {}
        for trace in all_adj_srcs[station].keys():
            misfit_dict[station][trace] = all_adj_srcs[station][trace]["misfit"]
    event_misfit_dict = {event_name: misfit_dict}
    print("Writing misfit dict")

    misfit_filename = os.path.join(output_folder, "misfit_dict.toml")
    with open(misfit_filename, "w") as fh:
        toml.dump(event_misfit_dict, fh)

if __name__ == "__main__":
    toml_filename = sys.argv[1]
    # toml_filename = "/Users/dirkphilip/Software/Inversionson/test_remote_window/info_dict.toml"
    info = toml.load(toml_filename)
    run(info)


