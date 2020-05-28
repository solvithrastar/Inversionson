# Inversionson

Inversionson is a workflow manager which automatically performs a Full-waveform inversion of seismic data based on the workflow described in [Thrastarson et al, 2020](https://academic.oup.com/gji/article/221/3/1591/5721256) with the addition of the mini-batch implementation of [van Herwaarden et al, 2020](https://academic.oup.com/gji/article/221/2/1427/5743423). It can also work with or without any combinations of the approaches described in the papers.

The workflow uses wavefield adapted meshes for each earthquake which make the simulations much faster than when using a classical cubed sphere mesh and only a fraction of the dataset in each iteration.

Inversionson is under development for using only the wavefield adapted meshes, only mini-batches or neither of the two for FWI aswell. It should more or less work for that but there might be a few kinks here and there as this is not a method which we use much these days.

## Central Libraries

Inversionson ties together many libraries and softwares which have been developed in the Seismology and wave physics group at ETH Zurich. The libraries are [Lasif](https://dirkphilip.github.io/LASIF_2.0/) and [MultiMesh](https://github.com/solvithrastar/MultiMesh) and central to the inversion procedure is the software suite [Salvus](https://mondaic.com/). Currently a specific version of the Salvus Optimization packages is needed to use inversionson. This might be improved later.

More information regaring Inversionson will be added later.

## Usage

To use Inversionson the best way is to create an info file in the directory where you want to keep your project and that can be done by running the __create_dummy_info_file.py__ and modifying that file at will according to what is needed.

A Lasif project is also needed so one has to initate a Lasif project where the signal processing parameters are specified. Inversionson communicates with Lasif in order to monitor and execute the inversion.

For any questions feel free to contact soelvi.thrastarson@erdw.ethz.ch
