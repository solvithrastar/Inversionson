import inspect
import matplotlib.pyplot as plt
import os
import numpy as np

import obspy


EXAMPLE_DATA_PDIFF = (800, 900)
EXAMPLE_DATA_SDIFF = (1500, 1600)


def window_trace(trace, window, taper, taper_ratio, taper_type, **kwargs):
    """
    Helper function to taper a window within a data trace.

    This function modifies the passed trace object in-place.

    :param trace: The trace to be tapered.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window: Tuples with UCTDateTime objects for start and end time
        and potentially a weight as well
    :type window: Tuple with UCTDateTime objects
    :param taper: True if you want to apply tapering
    :type taper: binary
    :param taper_percentage: Decimal percentage of taper at one end (ranging
        from ``0.0`` (0%) to ``0.5`` (50%)).
    :type taper_percentage: float
    :param taper_type: The taper type, supports anything
        :meth:`obspy.core.trace.Trace.taper` can use.
    :type taper_type: str

    Any additional keyword arguments are passed to the
    :meth:`obspy.core.trace.Trace.taper` method.
    """
    s, e = trace.stats.starttime, trace.stats.endtime
    trace.trim(window[0], window[1])
    if taper:
        trace.taper(max_percentage=taper_ratio, type=taper_type, **kwargs)
    trace.trim(s, e, pad=True, fill_value=0.0)
    # Enable method chaining.
    return trace


def get_example_data():
    """
    Helper function returning example data for SalvusMisft.

    :returns: Tuple of observed and synthetic streams
    :rtype: tuple of :class:`obspy.core.stream.Stream` objects

    .. rubric:: Example

    >>> from lasif.tools.adjoint.utils import get_example_data
    >>> observed, synthetic = get_example_data()
    >>> print(observed)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO.S3.MXR | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXT | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXZ | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    >>> print(synthetic)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO..LXR   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXT   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXZ   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    """
    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "example_data",
    )
    observed = obspy.read(os.path.join(path, "observed_processed.mseed"))
    observed.sort()
    synthetic = obspy.read(os.path.join(path, "synthetic_processed.mseed"))
    synthetic.sort()

    return observed, synthetic


def generic_adjoint_source_plot(
    observed, synthetic, time, adjoint_source, misfit, adjoint_source_name
):
    """
    Generic plotting function for adjoint sources and data.

    Many types of adjoint sources can be represented in the same manner.
    This is a convenience function that can be called by different
    the implementations for different adjoint sources.

    :param observed: The observed data, windowed.
    :type observed: numpy array
    :param synthetic: The synthetic data, windowed.
    :type synthetic: numpy array
    :param stats: Stats from obspy trace
    :type stats: Dictionary
    :param adjoint_source: The adjoint source.
    :type adjoint_source: `numpy.ndarray`
    :param misfit: The associated misfit value.
    :float misfit: misfit value
    :param adjoint_source_name: The name of the adjoint source.
    :type adjoint_source_name: str
    """

    plt.subplot(211)
    plt.plot(time, observed, color="0.2", label="Observed", lw=2)
    plt.plot(time, synthetic, color="#bb474f", label="Synthetic", lw=2)
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.subplot(212)
    plt.plot(
        time, adjoint_source, color="#2f8d5b", lw=2, label="Adjoint Source"
    )
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.suptitle(
        "%s Adjoint Source with a Misfit of %.3g"
        % (adjoint_source_name, misfit)
    )


def matlab_range(start, stop, step):
    """
    Simple function emulating the behaviour of Matlab's colon notation.

    This is very similar to np.arange(), except that the endpoint is included
    if it would be the logical next sample. Useful for translating Matlab code
    to Python.
    """
    # Some tolerance
    if (abs(stop - start) / step) % 1 < 1e-7:
        return np.linspace(
            start, stop, int(round((stop - start) / step)) + 1, endpoint=True
        )
    return np.arange(start, stop, step)


def get_dispersed_wavetrain(
    dw=0.001,
    distance=1500.0,
    t_min=0,
    t_max=900,
    a=4,
    b=1,
    c=1,
    body_wave_factor=0.01,
    body_wave_freq_scale=0.5,
    dt=1.0,
):
    """
    :type dw: float, optional
    :param dw: Angular frequency spacing. Defaults to 1E-3.
    :type distance: float, optional
    :param distance: The event-receiver distance in kilometer. Defaults to
        1500.
    :type t_min: float, optional
    :param t_min: The start time of the returned trace relative to the event
        origin in seconds. Defaults to 0.
    :type t_max: float, optional
    :param t_max: The end time of the returned trace relative to the event
        origin in seconds. Defaults to 900.
    :type a: float, optional
    :param a: Offset of dispersion curve. Defaults to 4.
    :type b: float, optional
    :param b: Linear factor of the dispersion curve. Defaults to 1.
    :type c: float, optional
    :param c: Quadratic factor of the dispersion curve. Defaults to 1.
    :type body_wave_factor: float, optional
    :param body_wave_factor: The factor of the body waves. Defaults to 0.01.
    :type body_wave_freq_scale: float, optional
    :param body_wave_freq_scale:  Determines the frequency of the body waves.
        Defaults to 0.5
    :returns: The time array t and the displacement array u.
    :rtype: Tuple of two numpy arrays
    """
    # Time and frequency axes
    w_min = 2.0 * np.pi / 50.0
    w_max = 2.0 * np.pi / 10.0
    w = matlab_range(w_min, w_max, dw)
    t = matlab_range(t_min, t_max, dt)

    # Define the dispersion curves.
    c = a - b * w - c * w ** 2

    # Time integration
    u = np.zeros(len(t))

    for _i in range(len(t)):
        u[_i] = np.sum(w * np.cos(w * t[_i] - w * distance / c) * dw)

    # Add body waves
    u += (
        body_wave_factor
        * np.sin(body_wave_freq_scale * t)
        * np.exp(-((t - 250) ** 2) / 500.0)
    )

    return t, u


def cross_correlation(f, g):
    """
    Computes a cross correlation similar to numpy's "full" correlation, except
    shifted indices.

    :type f: numpy array
    :param f: function 1
    :type g: numpy array
    :param g: function 1
    """
    cc = np.correlate(f, g, mode="full")
    N = len(cc)
    cc_new = np.zeros(N)

    cc_new[0 : (N + 1) // 2] = cc[(N + 1) // 2 - 1 : N]
    cc_new[(N + 1) // 2 : N] = cc[0 : (N + 1) // 2 - 1]
    return cc_new


def gaussian_window(y, width):
    """
    Returns a simple gaussian window along a given axis.

    :type y: numpy array
    :param y: The values at which to compute the window.
    :param width: float
    :param width: variance = (width ^ 2) / 2
    """
    return (
        1.0
        / (np.pi * width ** 2) ** (0.25)
        * np.exp(-0.5 * y ** 2 / width ** 2)
    )
