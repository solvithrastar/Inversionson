Salvus Opt Component
====================


The Salvus Opt component is used to communicate with a specific branch of
the Salvus Software suite (www.mondaic.com) which takes care of the
optimization algorithm.

In the current usage of `Inversionson`, it uses the Limited Memory BFGS
optimization algorithm which uses an approximation of the inverse Hessian
to configure the steps to take in search for a minimum of the misfit function.
On top of the L-BFGS approach it employs a trust region approach which
imposes further constraints on the steplengths in the iteration process.

Further information on L-BFGS can be found in Liu, D. C. and
Nocedal, J. (1989).

Salvus Opt operates in a way that given some initial conditions it spits
out tasks to be performed in each step of the iteration procedure. The
Salvus Opt component is thus central to the workflow of `Inversionson` as it
is the component which tells `Inversionson` which task to perform next.

The methods related to the Salvus Opt Component can be found here below.


.. autoclass:: inversionson.components.opt_comp.SalvusOptComponent
    :members: