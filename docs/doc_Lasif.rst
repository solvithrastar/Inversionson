Lasif Component
===============

LASIF is a **LA**\ rge-scale **S**\ eismic **I**\ nversion **F**\ ramework.
It takes care of the organization and management of all the documents
involved in a Full-Waveform Inversion (FWI).

The manual book keeping involved in an FWI study can be immense and it is
remarkable easy to make mistakes. That is why using such a framework like
LASIF is essential as the scale of the inversion grows.

LASIF takes care of finding earthquakes, downloading relevant data,
processing it and quantifying misfits between the data and the computed
synthetics using the automatically picked windows it picked. It also
geographically weights stations before the adjoint calculations.

The API that LASIF provides makes it easy to use for scripting and it
is thus relatively easy to write a wrapper around it like the one
described in this section.

For more information regarding LASIF... visit
https://dirkphilip.github.io/LASIF_2.0/

A description of the LasifComponent class can be found here below:

.. autoclass:: inversionson.components.lasif_comp.LasifComponent
    :members:
