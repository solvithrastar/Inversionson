AutoInverter
============

The AutoInverter class is what controls the workflow and calls the methods
from the other classes. It reads task from the non-linear optimization
algorithm (SalvusInvert) and does what is required to compute the relevant
task.

Autoinverter takes care of waiting until a simulation of a source is done to
perform certain tasks for the respective source and then making sure all
required steps have been performed in the correct order, in an optimal
time, to be able to proceed to the next step of the inversion.

.. autoclass:: inversionson.autoinverter.AutoInverter
    :members:
