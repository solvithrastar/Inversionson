Salvus Flow Component
=====================

To make `Inversionson` scalable for different hardwares, that part has been
abstracted away from everything through the usage of `Salvus Flow`. The Salvus
Flow Component handles all communication with `Salvus Flow`.

The component sends computational jobs to remote (or local) machines and
monitors its status. As soon as the status is finished, the results are
downloaded and the status of the job is updated in the inversion monitoring
module.

The methods associated with the Salvus Flow Component are documented below.

.. autoclass:: inversionson.components.flow_comp.SalvusFlowComponent
    :members:
