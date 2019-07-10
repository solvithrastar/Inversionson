# Inversionson

Inversionson is a workflow manager which automatically performs a Full-waveform inversion of seismic data based on the workflow described in Thrastarson et al, 2019 (submitted).

The workflow uses wavefield adapted meshes for each earthquake which make the simulations much faster than when using a classical cubed sphere mesh.

The code is still very much under development so I would not reccommend trying to use it as it is right now. 

## Central Libraries

Inversionson ties together many libraries and softwares which have been developed in the Seismology and wave physics group at ETH Zurich. The libraries are [Lasif](https://dirkphilip.github.io/LASIF_2.0/) and [MultiMesh](https://github.com/solvithrastar/MultiMesh) and central to the inversion procedure is the software suite [Salvus](https://salvus.io/).

More information regaring Inversionson will be added later.