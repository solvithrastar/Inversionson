# Inversionson

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

Inversionson is a workflow manager which automatically performs a Full-waveform inversion(FWI) of seismic data. It has built in support for various types of FWI workflows:
* A standard workflow like for example in [Krischer et al, 2018](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2017JB015289)
* The dynamic mini-batch workflow described by [van Herwaarden et al, 2020](https://academic.oup.com/gji/article/221/2/1427/5743423)
* The wavefield-adapted mesh workflow described by [Thrastarson et al, 2020](https://academic.oup.com/gji/article/221/3/1591/5721256) 
* A combination of dynamic mini-batches and wavefield adapted meshes.

It has built in support for using a validation dataset, which is a dataset that is not explicitly a part of the inversion but is reserved for monitoring the inversion procedure with an independent dataset.
The validation dataset can be used to tune regularization for example. There is also support for reserving a test dataset to compute misfit for at the end of the inversion process.

The inversion will be automatically documented in an easily readable MarkDown file. By setting some environment variables regarding Twilio, Inversionson can send you a Whatsapp message after each iteration.

## Central Libraries

Inversionson is not a standalone package but rather a wrapper around a few key FWI software. It is thus important to have these software installed and available if one is interested in using Inversionson. The main, non-standard software packaged that Inversionson requires are [LASIF](https://dirkphilip.github.io/LASIF_2.0/), [MultiMesh](https://github.com/solvithrastar/MultiMesh) and [Salvus](https://mondaic.com/). The recommended way of installing Inversionson is to follow the installation instructions of LASIF, cloning the MultiMesh repository and installing that one, do the same thing with Inversionson, and finally install Salvus into the same environment.

## Usage

Using Inversionson may seem a bit complicated at first, but once you get it working, it tends to run pretty smoothly. There are plans to make initializing an Inversionson project a much smoother process but that has not been done yet. The following is a description of how one can start an automatic FWI, using Inversionson.

A process which should get your project going:

1. Create the directory where you want to host your project.
2. Use LASIF to initialize a lasif project or copy an existing project in here.
    * Finish seting up the LASIF project (define domain, download data, define frequency range of synthetics)
3. Create a directory called `SALVUS_OPT` inside the Inversionson project directory
    * This directory is where the L-BFGS optimization routine will be carried out.
    * Move into this directory
4. Inside the `SALVUS_OPT` directory you need to run this code:
```bash
<Path to your Salvus binary> invert -x <Path to this folder>
```
5. Now create a file called `run_salvus_opt.sh` which has only one line in it:
```bash
<Path to your Salvus binary> invert -i ./inversion.toml
```
6. Salvus opt should now have created some files.
    * One of them is `inversion.toml` and you need to fill in some fields there, like initial model, parameters to invert for and whether you want to use batches of data or full gradients.
    * It's hard to assist with this file as it really depends on what you want to do but feel free to contact me if you are having troubles

7. Once you have filled in the `inversion.toml` file you should run
```bash
sh run_salvus_opt.sh
```
8. Now go back to the folder of the Inversionson project and run:
```bash
python <Path to inversionson code>/inversionson/create_dummy_info_file.py
```
9. Fill in the relevant fields in the `inversion_info.toml` file properly. The file contains comments to explain the fields but some of them will be further explained here.
    * __inversion_mode__: Can be either "mini-batch" (dynamic mini-batches) or "mono-batch" (full gradients)
    * __meshes__: Can be either "multi-mesh" (wavefield adapted meshes) or "mono-mesh" (same mesh for every simulation, defined by lasif domain file)
    * __model_interpolation_mode__: Actually only supports "gll_2_gll" right now so don't worry about that as long as you use `hdf5` meshes.
    * __inversion_parameters__: Parameters to invert for. Make sure these are the same ones as in the `inversion.toml` file in the `SALVUS_OPT` directory.
    * __modelling_parameters__: The parameters on the meshes you use for simulations.
    * __event_random_fraction__: Only relevant for "mini-batch" mode. Describes how many of the events selected in each batch are random, vs how many are selected based on spatial coverage.
    * __min_ctrl_group_size__: The minimum number of events used in control group, again only relevant for "mini-batch" mode.
    * __max_angular_change__: Used to decide how many events make it to the control group for the coming iteration in "mini-batch" mode.
    * __dropout_probability__: A form of regularization. Events in control group can be randomly dropped out with this probability so they don't get stuck there.
    * __initial_batch_size__: Make sure it's the same as in `inversion.toml` in "mini-batch" mode.
    * __cut_source_region_from_gradient_in_km__: Gradients become unphysical next to the source and it can be good to cut the region out.
    * __cut_receiver_region_from_gradient_in_km__: The same except receivers, and not nearly as bad of an unphysicality effect. This is currently quite slow and I would recommend just putting 0.0 here.
    * __clip_gradient__: You can clip gradient at some percentile so that the highest/lowest values are removed. 1.0 doesn't clip at all.
    * __absorbing_boundaries__: This is only a True/False flag, the actual absorbing boundaries are configured in the `lasif_config.toml`
    * __elements_per_azimuthal_quarter__: Only relevant for "multi-mesh". Decides how many elements are used to sample the azimuthal dimension. See paper.
    * __smoothing_mode__: isotropic or anisotropic. It's always model dependent and can be either direction dependent or not.
    * __smoothing_lengths__: How many wavelengths to smooth. If anisotropic the three values are: radial, lat, lon. For isotropic, only input one value.
    * __iterations_between_validation_checks__: When using a validation dataset, this decides with how many iterations are between each validation check. The models between checks are averaged. 0 means no check.
    * __validation_dataset__: Just a list of events in your lasif project that you want to reserve for validation checks and will not be used in the inversion. Input event names.
    * __test_dataset__: Same principle as with the validation_dataset
    * __HPC.wave_propagation__: Settings regarding wavefield simulations. Inversionson asks for double that walltime in adjoint runs as they are more expensive
    * __HPC.diffusion_equation__: Settings regarding the smoothing computations.

10. As the file is configured you should be able to start running Inversionson.
    * I would recommend running Inversionson with [tmux](https://tmuxcheatsheet.com/) as it keeps your shell running although you loose a connection with your computer or accidentally close your terminal window.

11. Run inversionson with this command:
```bash
python -m inversionson.autoinverter
```

For any questions feel free to contact soelvi.thrastarson@erdw.ethz.ch
