Introduction
============

Inversionson is a workflow manager which automatically performs a
Full-waveform inversion of seismic data based on the workflow described in
Thrastarson et al, 2019 (submitted) with the addition of the dynamic
mini-batch implementation of van Herwaarden er al, 2019 (submitted).

The workflow uses wavefield adapted meshes for each earthquake which make the
simulations much faster than when using a classical cubed sphere mesh.
The dynamic mini-batch approach only uses a fraction of the dataset in
each iteration which speeds up convergence and acts as additional
regularization in the inversion.

A future plan is to make Inversionson more flexible in terms of workflows
but currently it is being developed for a single problem.

.. warning::
    The code is still very much under development. It is not
    usable in its current state.

Subpackages
^^^^^^^^^^^

Inversionson is built to tie together multiple softwares and libraries
which have been developed recently, namely:

- LASIF (https://github.com/dirkphilip/LASIF_2.0)
    - Project organisation
    - Data management
    - Misfit quantifier
    - Graphical user interface (for looking at waveforms)
    - Data processing toolkit
- MultiMesh (https://github.com/solvithrastar/MultiMesh)
    - Interpolate model/gradient between different discretisations
    - Interpolate model/gradient between different data formats
- Salvus (https://www.mondaic.com)
    - SalvusCompute
        - Physical wavefield simulations
        - Adjoint wavefield simulations
        - Diffusion equation simulations
    - SalvusMesh
        - Mesh generation
    - SalvusInvert
        - Non-linear optimization
    - SalvusFlow
        - Simulation manager
        - Communication with HPC client

Additionally Inversionson implements the dynamic mini-batch inversion
algorithm which is not currently a standalone library.

An automatic documentation algorithm is used to keep track of the inversion
and ideally the inversion should be able to run without any human input.
