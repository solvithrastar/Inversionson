MultiMesh Component
===================

An essential part of the workflow described in Thrastarson et al, 2019 
(preprint: https://eartharxiv.org/v58cm/) is that there is a separation
between the inversion discretization, and the simulation discretization.
While that is the case, one needs some means of moving between different
discretizations in a consistant manner.

This is where MultiMesh comes in. The repository of MultiMesh is
(https://github.com/solvithrastar/MultiMesh). MultiMesh has an API which
can be used to create wrappers around it and that is what this component is.

It makes sure that in each situation, the meshes are prepared in a consistant
manner before interpolation and that MultiMesh always receives the correct
input parameters.

The progress of MultiMesh is monitored and the status of the inversion is
updated after each interpolation.

The documentation of the methods of the MultiMesh component are here below:

.. autoclass:: inversionson.components.multimesh_comp.MultiMeshComponent
    :members:
