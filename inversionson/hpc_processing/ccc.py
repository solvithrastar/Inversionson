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
from scipy.integrate import simps
import obspy.signal.cross_correlation as crosscorr

VERBOSE_NAME = "Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
Traveltime misfits simply measure the squared traveltime difference. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \left[ T^{obs} - T(\mathbf{m}) \right] ^ 2

:math:`T^{obs}` is the observed traveltime, and :math:`T(\mathbf{m})` the
predicted traveltime in Earth model :math:`\mathbf{m}`.

In practice traveltime are measured by cross correlating observed and
predicted waveforms. 

The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m})

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.


:math:`N` is a normalization factor given by


.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
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
    if len(window) == 2:
        weight = 1.0
    else:
        weight = window[2]

    # Work on copies of the original data
    observed = observed.copy()
    synthetic = synthetic.copy()
    CC = np.dot(observed, synthetic)
    weight_2 = np.sqrt(np.dot(observed, observed) * np.dot(synthetic, synthetic))

    misfit = 1 - CC / weight_2

    # Subsample accuracy time shift
    time_shift = xcorr_shift(synthetic, observed, min_period)
    ret_val["misfit"] = misfit
    
    if time_shift >= min_period / 2.0:
        ret_val["adjoint_source"] = Trace(data=np.zeros_like(observed.data),
                                          header=observed.stats)
        return ret_val

    if adjoint_src:
        A = np.dot(observed, synthetic) / np.dot(synthetic, synthetic)
        adj = (observed - A * synthetic) / weight_2
        adj_src = Trace(data=weight * adj *
                        synthetic.stats.delta, header=observed.stats)
        ret_val["adjoint_source"] = adj_src

    return ret_val



