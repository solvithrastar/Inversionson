#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Cross correlation traveltime misfit.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch), 2018
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from obspy import Trace
import obspy.signal.cross_correlation as crosscorr

VERBOSE_NAME = "Zero-lag Cross-Correlation-Coefficient"

DESCRIPTION = r"""
This is the CCC misfit as used by famous seismologist Ya-Jian Gao in 
several of his publications.
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


def xcorr_shift(s, d, min_period):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace with subsample accuracy.
    """
    # Estimate shift and use it as a guideline for the subsample accuracy
    # shift.
    # the dt works if these are obspy traces, currently not sure
    shift = int(np.ceil(min_period / s.stats.delta))
    cc = crosscorr.correlate(s, d, shift=shift)
    time_shift = (cc.argmax() - shift) * s.stats.delta
    return time_shift


def calculate_adjoint_source(observed, synthetic, window, min_period,
                             max_period,
                             adjoint_src, plot=False, taper=True,
                             taper_ratio=0.15, taper_type="cosine",
                             **kwargs):

    ret_val = {}

    if window:
        if len(window) == 2:
            weight = 1.0
        else:
            weight = window[2]
    else:
        weight = 1.0

    # Work on copies of the original data
    observed = observed.copy()
    synthetic = synthetic.copy()
    CC = np.sum(observed.data * synthetic.data)

    oo = np.sum(observed.data * observed.data)
    ss = np.sum(synthetic.data * synthetic.data)
    weight_2 = np.sqrt(oo * ss)

    misfit = 1 - CC / weight_2


    ret_val["misfit"] = misfit

    # # Subsample accuracy time shift
    # time_shift = xcorr_shift(synthetic, observed, min_period)
    # if time_shift >= min_period / 2.0:
    #     ret_val["adjoint_source"] = Trace(data=np.zeros_like(observed.data),
    #                                       header=observed.stats)
    #     return ret_val

    if adjoint_src:
        A = CC / ss
        adj = (observed.data - A * synthetic.data) / weight_2
        adj_src = Trace(data=weight * adj *
                        synthetic.stats.delta, header=observed.stats)
        ret_val["adjoint_source"] = adj_src

    return ret_val
