#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Simple waveform misfit and adjoint source.

This file will also serve as an explanation of how to add new adjoint
sources to lasif.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch)
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from obspy import Trace
from scipy.integrate import simps


# This is the verbose and pretty name of the adjoint source defined in this
# function.

VERBOSE_NAME = "L2 Waveform Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX formulas.
DESCRIPTION = r"""
This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data. The misfit :math:`\chi(\mathbf{m})` for a
given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""

# Optional: document any additional parameters this particular adjoint source
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
# rest of the architecture of adjoint part of lasif.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`str`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"cosine"``.
"""


# Each adjoint source file must contain a calculate_adjoint_source()
# function. It must take observed, synthetic, min_period, max_period,
# left_window_border, right_window_border, adjoint_src, and figure as
# parameters. Other optional keyword arguments are possible.


def calculate_adjoint_source(
    observed,
    synthetic,
    window,
    min_period,
    max_period,
    adjoint_src,
    plot=False,
    taper=False,
    taper_ratio=0.15,
    taper_type="cosine",
    **kwargs
):  # NOQA
    # There is no need to perform any sanity checks on the passed trace
    # object. At this point they will be guaranteed to have the same
    # sampling rate, be sampled at the same points in time and a couple
    # other things.

    ret_val = {}
    scaling = 1e5
    weight = scaling * 1.0

    if window:
        if len(window) == 2:
            weight = 1.0 * scaling
        else:
            weight = window[2] * scaling

    diff = (observed.data - synthetic.data) * weight
    # 0.5 * (s-o) ** 2
    # (s-
    # Integrate with the composite Simpson's rule.
    ret_val["misfit"] = 0.5 * simps(y=diff ** 2, dx=observed.stats.delta)

    if adjoint_src is True:
        adj_src = Trace(
            data=diff * weight * synthetic.stats.delta, header=observed.stats
        )

        ret_val["adjoint_source"] = adj_src

    return ret_val
