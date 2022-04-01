"""
An implementation of the time frequency phase misfit and adjoint source after
Fichtner et al. (2008).

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import warnings

import numexpr as ne
import numpy as np
import obspy
from obspy.signal.interpolation import lanczos_interpolation
from inversionson.hpc_processing import adjoint_utils as utils
from inversionson.hpc_processing import time_frequency

eps = np.spacing(1)
VERBOSE_NAME = "Time Frequency Phase Misfit"

DESCRIPTION = r"""
Time Frequency Phase Misfit is modelled after Fichtner et al. 2008. It measures
a misfit in phase between synthetic seismograms (:math: `u_i`) and recorded 
seismograms (:math: `u^0_i`). It is based on the time-frequency transform of 
both data and synthetics. The phase misfit is defined by:

.. math::

    E^n_p(u^0_i, u_i) := \int_{\mathbb{R}^2}W^n_p(t,\omega)[\phi_i(t,\omega)
    -\phi^0_i(t,\omega)]^n dtd\omega

Which is then used to define the adjoint source for displacement seismograms:

.. math::

    f^{\dagger}(\tau) = E^{1-n}_{p}\Im \int_{\mathbb{R}^2} W^n_p(t,\omega)
    [\phi_i(t,\omega) - phi^0_i(t,\omega)]^(n-1)[\frac{\hat{u}_i(t,\omega)}
    {\left \hat{u}_i(t,\omega) \right ^2}h(\tau - t)eˆ{\mathbf{i}\omega\tau}]dt
    d\omega

Where :math: `h(\tau - t)` is a sliding window function (e.g. a gaussian).
An advantage of the time frequency phase misfit over the cross correlation time
shift (which also measures phase/time shift) is that this gives a measurement
of misfit as a function of time and frequency while the cross correlation only
gives a single valued estimate for each window.

For a more detailed description, we refer to Fichtner et al. 2008.

"""  # NOQA

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
#  rest of the architecture.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"cosine"``.
"""


def calculate_adjoint_source(
    observed,
    synthetic,
    window,
    min_period,
    max_period,
    adjoint_src,
    plot=False,
    max_criterion=7.0,
    taper=True,
    taper_ratio=0.15,
    taper_type="cosine",
    **kwargs
):
    """
    :rtype: dictionary
    :returns: Return a dictionary with three keys:
        * adjoint_source: The calculated adjoint source as a numpy array
        * misfit: The misfit value
        * messages: A list of strings giving additional hints to what happened
            in the calculation.
    """
    # Assumes that t starts at 0. Pad your data if that is not the case -
    # Parts with zeros are essentially skipped making it fairly efficient.
    t = observed.times(type="relative")
    assert t[0] == 0

    ret_dict = {}

    if window:
        if len(window) == 2:
            window_weight = 1.0
        else:
            window_weight = window[2]
    else:
        window_weight = 1.0

    # Work on copies of the original data
    observed = observed.copy()
    synthetic = synthetic.copy()

    if window:
        observed = utils.window_trace(
            trace=observed,
            window=window,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
            **kwargs
        )
        synthetic = utils.window_trace(
            trace=synthetic,
            window=window,
            taper=taper,
            taper_ratio=taper_ratio,
            taper_type=taper_type,
            **kwargs
        )

    messages = []

    # Internal sampling interval. Some explanations for this "magic" number.
    # LASIF's preprocessing allows no frequency content with smaller periods
    # than min_period / 2.2 (see function_templates/preprocesssing_function.py
    # for details). Assuming most users don't change this, this is equal to
    # the Nyquist frequency and the largest possible sampling interval to
    # catch everything is min_period / 4.4.
    #
    # The current choice is historic as changing does (very slightly) chance
    # the calculated misfit and we don't want to disturb inversions in
    # progress. The difference is likely minimal in any case. We might have
    # same aliasing into the lower frequencies but the filters coupled with
    # the TF-domain weighting will get rid of them in essentially all
    # realistically occurring cases.
    dt_new = max(float(int(min_period / 4.0)), t[1] - t[0])
    dt_old = t[1] - t[0]

    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Make sure its odd - that avoid having to deal with some issues
    # regarding frequency bin interpolation. Now positive and negative
    # frequencies will always be all symmetric. Data is assumed to be
    # tapered in any case so no problem are to be expected.
    if not len(ti) % 2:
        ti = ti[:-1]

    # Interpolate both signals to the new time axis - this massively speeds
    # up the whole procedure as most signals are highly oversampled. The
    # adjoint source at the end is re-interpolated to the original sampling
    # points.
    data = lanczos_interpolation(
        data=observed.data,
        old_start=t[0],
        old_dt=t[1] - t[0],
        new_start=t[0],
        new_dt=dt_new,
        new_npts=len(ti),
        a=8,
        window="blackmann",
    )
    synthetic = lanczos_interpolation(
        data=synthetic.data,
        old_start=t[0],
        old_dt=t[1] - t[0],
        new_start=t[0],
        new_dt=dt_new,
        new_npts=len(ti),
        a=8,
        window="blackmann",
    )
    original_time = t
    t = ti

    # -------------------------------------------------------------------------
    # Compute time-frequency representations

    # Window width is twice the minimal period.
    width = 2.0 * min_period

    # Compute time-frequency representation of the cross-correlation
    _, _, tf_cc = time_frequency.time_frequency_cc_difference(
        t, data, synthetic, width
    )
    # Compute the time-frequency representation of the synthetic
    tau, nu, tf_synth = time_frequency.time_frequency_transform(
        t, synthetic, width
    )

    # -------------------------------------------------------------------------
    # compute tf window and weighting function

    # noise taper: down-weight tf amplitudes that are very low
    tf_cc_abs = np.abs(tf_cc)
    m = tf_cc_abs.max() / 10.0  # NOQA
    weight = ne.evaluate("1.0 - exp(-(tf_cc_abs ** 2) / (m ** 2))")
    nu_t = nu.T

    # highpass filter (periods longer than max_period are suppressed
    # exponentially)
    weight *= 1.0 - np.exp(-((nu_t * max_period) ** 2))

    # lowpass filter (periods shorter than min_period are suppressed
    # exponentially)
    nu_t_large = np.zeros(nu_t.shape)
    nu_t_small = np.zeros(nu_t.shape)
    thres = nu_t <= 1.0 / min_period
    nu_t_large[np.invert(thres)] = 1.0
    nu_t_small[thres] = 1.0
    weight *= (
        np.exp(-10.0 * np.abs(nu_t * min_period - 1.0)) * nu_t_large
        + nu_t_small
    )

    # normalisation
    weight /= weight.max()

    # computation of phase difference, make quality checks and misfit ---------

    # Compute the phase difference.
    # DP = np.imag(np.log(m + tf_cc / (2 * m + np.abs(tf_cc))))
    DP = np.angle(tf_cc)

    # Attempt to detect phase jumps by taking the derivatives in time and
    # frequency direction. 0.7 is an emperical value.
    abs_weighted_DP = np.abs(weight * DP)
    _x = abs_weighted_DP.max()  # NOQA
    test_field = ne.evaluate("weight * DP / _x")

    criterion_1 = np.sum([np.abs(np.diff(test_field, axis=0)) > 0.7])
    criterion_2 = np.sum([np.abs(np.diff(test_field, axis=1)) > 0.7])
    criterion = np.sum([criterion_1, criterion_2])
    # Compute the phase misfit
    dnu = nu[1] - nu[0]

    i = ne.evaluate("sum(weight ** 2 * DP ** 2)")

    phase_misfit = np.sqrt(i * dt_new * dnu) * window_weight

    # Sanity check. Should not occur.
    if np.isnan(phase_misfit):
        msg = "The phase misfit is NaN."
        raise Exception(msg)

    # The misfit can still be computed, even if not adjoint source is
    # available.
    if criterion > max_criterion:
        warning = (
            "Possible phase jump detected. Misfit included. No "
            "adjoint source computed. Criterion: %.1f - Max allowed "
            "criterion: %.1f" % (criterion, max_criterion)
        )
        warnings.warn(warning)
        messages.append(warning)

        ret_dict = {
            "adjoint_source": obspy.Trace(
                data=np.zeros_like(observed.data), header=observed.stats
            ),
            "misfit": phase_misfit * 2.0,
            "details": {"messages": messages},
        }

        return ret_dict
    if adjoint_src:
        # Make kernel for the inverse tf transform
        idp = ne.evaluate(
            "weight ** 2 * DP * tf_synth / (m + abs(tf_synth) ** 2)"
        )

        # Invert tf transform and make adjoint source
        ad_src, it, I = time_frequency.itfa(tau, idp, width)

        # Interpolate both signals to the new time axis
        ad_src = lanczos_interpolation(
            # Pad with a couple of zeros in case some where lost in all
            # these resampling operations. The first sample should not
            # change the time.
            data=np.concatenate([ad_src.imag, np.zeros(100)]),
            old_start=tau[0],
            old_dt=tau[1] - tau[0],
            new_start=original_time[0],
            new_dt=original_time[1] - original_time[0],
            new_npts=len(original_time),
            a=8,
            window="blackmann",
        )

        # Divide by the misfit and change sign.
        ad_src /= phase_misfit + eps
        ad_src = ad_src / ((t[1] - t[0]) ** 2) * dt_old

        # Reverse time and add a leading zero so the adjoint source has the
        # same length as the input time series.
        # ad_src = ad_src[::-1]

        # Calculate actual adjoint source. Not time reversed
        adj_src = obspy.Trace(
            data=ad_src * window_weight, header=observed.stats
        )
        if window:
            adj_src = utils.window_trace(
                trace=adj_src,
                window=window,
                taper=taper,
                taper_ratio=taper_ratio,
                taper_type=taper_type,
                **kwargs
            )

    ret_dict = {
        "adjoint_source": adj_src,
        "misfit": phase_misfit,
        "details": {"messages": messages},
    }

    return ret_dict
