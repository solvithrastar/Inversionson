
import numpy as np
import obspy
import warnings

import obspy.signal.filter
from inversionson.hpc_processing.adjoint_utils import window_trace
from obspy.core.utcdatetime import UTCDateTime

def calculate_adjoint_source(
    adj_src_type,
    observed,
    synthetic,
    window,
    min_period=None,
    max_period=None,
    taper=True,
    taper_type="cosine",
    **kwargs,
):
    """
    Computes the adjoint source,

    :param adj_src_type: The type of adjoint source to calculate.
    :type adj_src_type: str
    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param min_period: The minimum period of the spectral content of the data.
    :type min_period: float
    :param window: starttime and endtime of window(s) potentially including
        weighting for each window.
    :type window: list of tuples
    """
    observed, synthetic = _sanity_checks(observed, synthetic)
    # Keep these as they will need to be imported later

    # window variable should be a list of windows, if it is not make it into
    # a list.
    if not isinstance(window, list):
        window = [window]

    if adj_src_type == "tf_phase_misfit":
        from inversionson.hpc_processing.tf_phase_misfit import calculate_adjoint_source as fct
    elif adj_src_type == "ccc":
        from inversionson.hpc_processing.ccc import calculate_adjoint_source as fct
    else:
        raise Exception("Not implemented error")

    full_ad_src = None
    trace_misfit = 0.0

    original_observed = observed.copy()
    original_synthetic = synthetic.copy()

    if adj_src_type == "envelope_misfit" or "envelope_scaling" in kwargs \
            and kwargs["envelope_scaling"]:
        # scale data to same amplitude range and to 1
        # such that weak earthquakes count equally much
        scaling_factor_syn = 1.0 / original_synthetic.data.ptp()
        scaling_factor_data = 1.0 / original_observed.data.ptp()
        original_synthetic.data *= scaling_factor_syn
        original_observed.data *= scaling_factor_data

        # At this point they have the same ptp amplitude range.
        # Now we want to downweight high amplitude surface waves
        # and upweight body waves by dividing by the envelope + a reg term.
        envelope = obspy.signal.filter.envelope(original_observed.data)
        env_weighting = 1.0 / (envelope + np.max(envelope) * 0.3)
        original_observed.data *= env_weighting
        original_synthetic.data *= env_weighting

    for win_tuple in window:
        # Convert to UTCDateTime
        win = [UTCDateTime(win_tuple[0]),
               UTCDateTime(win_tuple[1]), win_tuple[2]]
        taper_ratio = 0.5 * (min_period / (win[1] - win[0]))
        observed = original_observed.copy()
        synthetic = original_synthetic.copy()

        # The window trace function tapers the trace and this modifies it.
        dt = 1.0 / observed.stats.sampling_rate
        observed = window_trace(
            trace=observed,
            window=win,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
        )
        synthetic = window_trace(
            trace=synthetic,
            window=win,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
        )
        # s, e = observed.stats.starttime, observed.stats.endtime
        # observed.trim(win[0] - dt*1500, win[1] + dt * 1500)
        # synthetic.trim(win[0] - dt*1500, win[1] + dt * 1500) # with padding to the trim, this matches
        # window is set to false, because we already taper here.
        adjoint = fct(
            observed=observed,
            synthetic=synthetic,
            window=False,
            min_period=min_period,
            max_period=max_period,
            adjoint_src=True,
            plot=True,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
        )
        # repad it to full length
        # adjoint["adjoint_source"].trim(s, e, pad=True, fill_value=0.0)
        # print(max(adjoint["adjoint_source"].data))
        adjoint["adjoint_source"] = window_trace(
            trace=adjoint["adjoint_source"],
            window=win,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
        )
        if win_tuple == window[0]:
            full_ad_src = adjoint["adjoint_source"]
            # print(max(full_ad_src.data))
        else:
            full_ad_src.data = full_ad_src.data + adjoint["adjoint_source"].data

        trace_misfit += adjoint["misfit"]

    # adjoint source requires an additional factor due to chain rule
    if adj_src_type == "envelope_misfit" or "envelope_scaling" in kwargs \
            and kwargs["envelope_scaling"]:
        full_ad_src.data *= (scaling_factor_syn * env_weighting)

    return trace_misfit, full_ad_src

def _sanity_checks(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid
    in a certain sense.

    It checks the types of both, the start time, sampling rate, number of
    samples, ...

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`

    :raises: :class:`~lasif.LASIFError`
    """
    if not isinstance(observed, obspy.Trace):
        # Also accept Stream objects.
        if isinstance(observed, obspy.Stream) and len(observed) == 1:
            observed = observed[0]
        else:
            raise Exception(
                "Observed data must be an ObsPy Trace object., not {}"
                "".format(observed)
            )
    if not isinstance(synthetic, obspy.Trace):
        if isinstance(synthetic, obspy.Stream) and len(synthetic) == 1:
            synthetic = synthetic[0]
        else:
            raise Exception("Synthetic data must be an ObsPy Trace object.")

    if observed.stats.npts != synthetic.stats.npts:
        raise Exception(
            "Observed and synthetic data must have the "
            "same number of samples."
        )

    sr1 = observed.stats.sampling_rate
    sr2 = synthetic.stats.sampling_rate

    if abs(sr1 - sr2) / sr1 >= 1e-5:
        raise Exception(
            "Observed and synthetic data must have the " "same sampling rate."
        )

    # Make sure data and synthetics start within half a sample interval.
    if (
        abs(observed.stats.starttime - synthetic.stats.starttime)
        > observed.stats.delta * 0.5
    ):
        raise Exception(
            "Observed and synthetic data must have the " "same starttime."
        )

    ptp = sorted([observed.data.ptp(), synthetic.data.ptp()])
    if ptp[1] / ptp[0] >= 5:
        warnings.warn(
            "The amplitude difference between data and "
            "synthetic is fairly large.",
        )

    # Also check the components of the data to avoid silly mistakes of
    # users.
    if (
        len(
            set(
                [
                    observed.stats.channel[-1].upper(),
                    synthetic.stats.channel[-1].upper(),
                ]
            )
        )
        != 1
    ):
        warnings.warn(
            "The orientation code of synthetic and observed "
            "data is not equal."
        )

    observed = observed.copy()
    synthetic = synthetic.copy()
    observed.data = np.require(
        observed.data, dtype=np.float64, requirements=["C"]
    )
    synthetic.data = np.require(
        synthetic.data, dtype=np.float64, requirements=["C"]
    )

    return observed, synthetic
