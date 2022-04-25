# Inversionson

**Inversionson** is a workflow manager which fully automates FWI workflows, optimizing for both computational- and human time.
In collaboration with [Salvus](https://mondaic.com), it makes working with a combination of local machines and HPC clusters easy.
Setting up a large-scale seismic inversion and running it has never been easier and more efficient.

There exists an [open-access paper about Inversionson](https://eartharxiv.org/repository/view/2132/). If you use Inversionson, please consider citing it:

```bibtex
@article{thrastarson2021inversionson,
  title={Inversionson: Fully Automated Seismic Waveform Inversions},
  author={Thrastarson, Solvi and van Herwaarden, Dirk-Philip and Fichtner, Andreas},
  year={2021},
  publisher={EarthArXiv},
  doi = {10.31223/X5F31V},
  url = {https://doi.org/10.31223/X5F31V}
}
```

The paper describes workflows which were supported in [v0.0.1](https://github.com/solvithrastar/Inversionson/releases/tag/v0.0.1-minibatch) of Inversionson.
In the latest version, we have made some changes.
In the previous version, we used optimization routines from Salvus, but we have now stopped supporting them and created a basis for implementing our own optimization routines within Inversionson.
This release includes two versions of the [Adam](https://arxiv.org/abs/1412.6980) optimization method. More details on that later.
We plan on adding more optimization routines in the future, as we have built a basis to be able to do so relatively easily.

Inversionson has built in support for using a validation dataset, which is a dataset that is not explicitly a part of the inversion but is reserved for monitoring the inversion procedure with an independent dataset.
The validation dataset can be used to tune regularization for example. There is also support for reserving a test dataset to compute misfit for at the end of the inversion process.

The design principle of Inversionson is that it runs on a machine that can in principle be a laptop, but ideally it's a desktop machine.
This machine only serves as a controller and does not do any heavy computations. It submits jobs to HPC clusters to do everything that would normally take time.
This is done in order to make the workflow as parallel as possible, and it saves a lot of time.

## Central Libraries

Inversionson is not a standalone package but rather a wrapper around a few key FWI software.
It is thus important to have these software installed and available if one is interested in using Inversionson.
The main, non-standard software packaged that Inversionson requires are
[LASIF](https://dirkphilip.github.io/LASIF_2.0/),
[MultiMesh](https://github.com/solvithrastar/MultiMesh) and
[Salvus](https://mondaic.com/).
The recommended way of installing Inversionson is to follow the installation instructions of LASIF,
cloning the MultiMesh repository and installing that one, do the same thing with Inversionson,
and finally install Salvus into the same environment.

## Usage

Inversionson is designed to work in a way that initializing the project is the only thing that a user needs to spend time on.

A process which should get your project going:

1. Create the directory where you want to host your project (we do not recommend having this the same directory as the Inversionson code base).
1. Use LASIF to initialize a LASIF project or copy an existing project in here  `lasif init_project LASIF_PROJECT`.
    * Finish setting up the LASIF project (define domain, download data)
    * Inside LASIF, you need to figure out simulation parameters, such as frequency range, time step and length of simulations

1. Now go back to the folder of the Inversionson project and run:
    ```bash
    python -m inversionson.autoinverter
    ```
    * This will just create a file named `inversion_info.toml` in the project root, and exit.
1. Fill in the relevant fields in the `inversion_info.toml` file properly. The file contains comments to explain the fields but some of them will be further explained here.
    * __inversion_path__: Absolute path to the root Inversion folder. This is set automatically
    * __lasif_root__: The path to the LASIF project
    * __meshes__: Can be either "multi-mesh" (wavefield adapted meshes) or "mono-mesh" (same mesh for every simulation, defined by LASIF domain file)
    * __optimizer__: The optimization method. Can either be Adam or SGDM for stochastic gradient descent with momentum.
    * __inversion_parameters__: Parameters to invert for. Make sure these are the same ones as in the `inversion.toml` file in the `SALVUS_OPT` directory.
    * __modelling_parameters__: The parameters on the meshes you use for simulations.
    * __batch_size__: The number of events to use per iteration. If you don't want this to be stochastic, just put the size of the dataset you want to use.
    * __cut_source_region_from_gradient_in_km__: Gradients become unphysical next to the source and it can be good to cut the region out.
    * __clip_gradient__: You can clip gradient at some percentile so that the highest/lowest values are removed. 1.0 doesn't clip at all.
    * __absorbing_boundaries__: A true/false flag whether the absorbing boundaries specified in LASIF should be used.
    * __Meshing.elements_per_azimuthal_quarter__: Only relevant for "multi-mesh". Decides how many elements are used to sample the azimuthal dimension. See paper.
    * __Meshing.elements_per_wavelength__: Only relevant for "multi-mesh". Decides how many elements are used per wavelength in the wavefield-adapted meshes
    * __Meshing.ellipticity__: Only relevant for "multi-mesh". Do you want ellipticity in your mesh.
    * __Meshing.ocean_loading__: Make `use` True if you have ocean loading on your mesh. If you are using multi-mesh, you also need to supply a file and a parameter name to use as well as where you want this to be stored on the HPC cluster.
    * __Meshing.topography__: Make `use` True if you have topography on your mesh. If you are using multi-mesh, you also need to supply a file and a parameter name to use as well as where you want this to be stored on the HPC cluster.
    * __inversion_monitoring__: We recommend using a validation dataset to monitor the state of the inversion. 
    * __iterations_between_validation_checks__: When using a validation dataset, this decides with how many iterations are between each validation check. The models between checks are averaged. 0 means no check.
    * __validation_dataset__: Just a list of events in your LASIF project that you want to reserve for validation checks and will not be used in the inversion. Input event names.
    * __test_dataset__: Same principle as with the validation_dataset except that it is never used in the inversion.
    * __HPC.wave_propagation__: Settings regarding wavefield simulations. Inversionson asks for double that walltime in adjoint runs as they are more expensive
    * __HPC.diffusion_equation__: Settings regarding the smoothing computations.
    * __HPC.interpolation__: Settings regarding remote interpolations
    * __HPC.processing__: Settings regarding the processing of the results from the forward jobs

1. Run Inversionson again using
    ```bash
    python -m inversionson.autoinverter
    ```
    * This time, the optimization configurations are created under `OPTIMIZATION/opt_config.toml`
    * These settings need to be filled out before the inversion. The details of the parameters depend on the selected optimizer.
    * Once these are filled in, you are ready to go.

1. Run Inversionson again using
    ```bash
    python -m inversionson.autoinverter
    ```
    * This time it should already start the actual inversion process.
    * I would recommend running Inversionson with [tmux](https://tmuxcheatsheet.com/) as it keeps your shell running although you loose a connection with your computer or accidentally close your terminal window.

## Optimizers

Currently Inversionson comes with two optimizers.
We recommend reading up on them before starting, but the default parameters should give pretty good results.

Here we provide good resources for reading up on the optimizers.

### Adam

- [Original Adam Publication](https://arxiv.org/abs/1412.6980)
- [Adam weight decay](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)

### Stochastic Gradient Descent with Momentum

- [Blog Post](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)
- [Qian 1999](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf)

For any questions feel free to contact soelvi.thrastarson@erdw.ethz.ch

