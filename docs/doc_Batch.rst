Dynamic Mini-Batch Component
============================

Currently the only way of using Inversionson is by using a dynamic mini-batch
approach. The plan is to make Inversionson independent of this approach but
that's the way it is.

Details regarding the dynamic mini-batch approach in FWI can be found in
a paper written by Dirk-Philip van Herwaarden. The paper is currently being
reviewed but I think a preprint should be available on EarthArxiv. Wait let
me check that actually... Ohh, no it's not available. Sorry about that.

I'll write a little bit about it then. So far, in FWI, people have usually
used their whole dataset in each iteration. This makes each iteration a
computational challenge and makes the cost of an inversion directly scale
with the used data in the inversion. That makes it hard to invert for large
datasets and the convergence of such a heavy machinery is quite slow as a
function of simulations. In van Herwaarden et al, 2019 a different approach
is presented where in each iteration, the gradient of the full dataset is
approximated using a subset of the data (mini-batch).

The selection of the mini-batch is varied between iterations, while the misfit
is monitored using a control-group which is constant from one iteration to
the next one. The size of the control-group and mini-batch is a function of
how well the subset approximates the gradient and that's where the dynamic
in the name comes from.

Using this approach, much fewer simulations are needed to converge towards
a model and the inversion becomes somewhat independent of the amount of data
used in the inversion.

The methods used in this component are documented here below:

.. autoclass:: inversionson.components.batch_comp.BatchComponent
    :members:
