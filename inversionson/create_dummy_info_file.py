"""
A script to create an information toml file in a format that
Inversionson can read.
The values of the toml file can then be changed and used as
an actual information toml.
"""

import toml
import os


def create_info(root=None):

    info = {}
    if not root:
        root = os.getcwd()
    info["inversion_path"] = root
    info["lasif_root"] = os.path.join(root, "LASIF_PROJECT")
    info["inversion_id"] = "MY_INVERSION"
    info["inversion_mode"] = "mini-batch"
    inversion_mode_comment = "Pick between mini-batch and mono-batch"
    info["meshes"] = "multi-mesh"
    info["Meshing"] = {}
    info["Meshing"]["elements_per_azimuthal_quarter"] = 4
    info["Meshing"]["ellipticity"] = True
    info["Meshing"]["ocean_loading"] = {}
    info["Meshing"]["ocean_loading"]["use"] = False
    info["Meshing"]["ocean_loading"]["file"] = ""
    info["Meshing"]["ocean_loading"]["variable"] = ""
    info["Meshing"]["topography"] = {}
    info["Meshing"]["topography"]["use"] = False
    info["Meshing"]["topography"]["file"] = ""
    info["Meshing"]["topography"]["variable"] = ""
    meshes_comment = "Pick between 'multi-mesh' or 'mono-mesh'"
    info["interpolation_mode"] = "remote"
    info["HPC"] = {}
    info["HPC"]["wave_propagation"] = {
        "site_name": "daint",
        "wall_time": 3600,
        "ranks": 48,
    }
    info["HPC"]["diffusion_equation"] = {
        "site_name": "daint",
        "wall_time": 1000,
        "ranks": 24,
    }
    info["HPC"]["interpolation"] = {
        "site_name": "daint",
        "model_wall_time": 60 * 5,
        "gradient_wall_time": 60 * 15,
        "remote_mesh_directory": "/path_to_directory_containing_meshes",
    }
    info["inversion_parameters"] = [
        "VPV",
        "VPH",
        "VSV",
        "VSH",
        "RHO",
    ]
    info["modelling_parameters"] = [
        "VPV",
        "VPH",
        "VSV",
        "VSH",
        "RHO",
        "QKAPPA",
        "QMU",
        "ETA",
    ]
    info["Smoothing"] = {}
    info["Smoothing"]["smoothing_mode"] = "anisotropic"
    info["Smoothing"]["smoothing_lengths"] = [0.5, 1.0, 1.0]
    info["Smoothing"]["timestep"] = 1.0e-5
    info["random_event_fraction"] = 0.5
    info["min_ctrl_group_size"] = 2
    info["max_angular_change"] = 30.0
    info["dropout_probability"] = 0.15
    info["initial_batch_size"] = 4
    info["cut_source_region_from_gradient_in_km"] = 100.0
    info["cut_receiver_region_from_gradient_in_km"] = 0.0
    cut_stuff_gradient = "Put 0.0 if you don't want to cut anything"
    info["clip_gradient"] = 1.0
    info["absorbing_boundaries"] = True
    absorbing_boundaries = (
        "You specify the length of the absorbing boundaries in the "
        "lasif config"
    )
    info["inversion_monitoring"] = {}
    info["inversion_monitoring"]["iterations_between_validation_checks"] = 0
    info["inversion_monitoring"]["validation_dataset"] = []
    info["inversion_monitoring"]["test_dataset"] = []

    clip_grad_comment = "Values between 0.55 - 1.0. The number represents the "
    clip_grad_comment += "quantile where the gradient will be clipped. "
    clip_grad_comment += "If 1.0 nothing will be cut"
    info["comments"] = {}
    info["comments"]["interpolation_mode"] = "Either local or remote"
    info["comments"]["clip_gradient"] = clip_grad_comment
    info["comments"]["cut_gradient"] = cut_stuff_gradient
    info["comments"]["absorbing_boundaries"] = absorbing_boundaries
    info["comments"]["meshes"] = meshes_comment
    info["comments"]["inversion_mode"] = inversion_mode_comment
    info["comments"]["Smoothing"] = {}
    smoothing_mode_comment = (
        "isotropic or anisotropic. Smoothing is always model dependent but "
        "can be either isotropic or anisotropic meaning that different "
        "dimensions get smoothed with different wavelengths. "
        "Density is always smoothed based on some VP model. "
        "If you don't want any smoothing, write 'none'."
    )
    smoothing_lengths_comment = (
        "If smoothing_mode is isotropic, only one value is required, "
        "if smoothing_mode is anisotropic, three values in a list are used."
    )
    info["comments"]["Smoothing"]["smoothing_mode"] = smoothing_mode_comment
    info["comments"]["Smoothing"][
        "smoothing_lengths"
    ] = smoothing_lengths_comment
    info["comments"]["Meshing"] = {}
    epaq_comment = (
        "Only used for multi-mesh. Needs to be higher for "
        "more complex models."
    )

    info["comments"]["Meshing"][
        "elements_per_azimuthal_quarter"
    ] = epaq_comment
    info["comments"]["inversion_monitoring"] = {}
    validation_checks_comment = (
        "If you want to check misfit of validation set every few iterations "
        "you give a number N here if you want to do a validation check every "
        "N iterations. Put 0 if you do not want to perform a validation check."
    )
    validation_dataset_comment = (
        "A validation dataset is used to monitor state of inversion, it is "
        "used to tune parameters such as smoothing lengths. It is not used "
        "in the inversion in any other way. This parameter is optional. "
        "If you keep the above parameter at zero, you can still reserve "
        "sources for validation without it happening automatically."
    )
    test_dataset_comment = (
        "A test dataset is not used at all in the inversion and can only be "
        "used at the end of an inversion to test how reliable the inversion "
        "actually was. The only thing Inveversionson does with these events "
        "is to make sure they are not used in inversion. "
        "This parameter is optional."
    )
    info["comments"]["inversion_monitoring"][
        "iterations_between_validation_checks"
    ] = validation_checks_comment
    info["comments"]["inversion_monitoring"][
        "validation_dataset"
    ] = validation_dataset_comment
    info["comments"]["inversion_monitoring"][
        "test_dataset"
    ] = test_dataset_comment

    return info


if __name__ == "__main__":
    cwd = os.getcwd()
    info = create_info()
    filename = os.path.join(cwd, "inversion_info.toml")
    with open(filename, "w+") as fh:
        toml.dump(info, fh)
