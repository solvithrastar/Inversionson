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
    meshes_comment = "Pick between 'multi-mesh' or 'mono-mesh'"
    info["model_interpolation_mode"] = "gll_2_gll"
    info["gradient_interpolation_mode"] = "gll_2_gll"
    info["HPC"] = {}
    info["HPC"]["wave_propagation"] = {
        "site_name": "daint",
        "wall_time": 3600,
        "ranks": 1024,
    }
    info["HPC"]["diffusion_equation"] = {
        "site_name": "daint",
        "wall_time": 1000,
        "ranks": 512,
    }
    info["inversion_parameters"] = ["VP", "VS", "RHO"]
    info["modelling_parameters"] = ["VP", "VS", "RHO"]
    info["Smoothing"] = {}
    info["Smoothing"]["smoothing_mode"] = "anisotropic"
    info["Smoothing"]["smoothing_lengths"] = [0.5, 1.0, 1.0]
    info["n_random_events"] = 2
    info["min_ctrl_group_size"] = 2
    info["max_angular_change"] = 30.0
    info["dropout_probability"] = 0.15
    info["initial_batch_size"] = 4
    info["cut_source_region_from_gradient_in_km"] = 100.0
    info["cut_receiver_region_from_gradient_in_km"] = 10.0
    cut_stuff_gradient = "Put 0.0 if you don't want to cut anything"
    info["clip_gradient"] = 1.0
    clip_grad_comment = "Values between 0.55 - 1.0. The number represents the "
    clip_grad_comment += "quantile where the gradient will be clipped. "
    clip_grad_comment += "If 1.0 nothing will be cut"
    info["comments"] = {}
    info["comments"]["clip_gradient"] = clip_grad_comment
    info["comments"]["cut_gradient"] = cut_stuff_gradient
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

    info["comments"]["Meshing"]["elements_per_azimuthal_quarter"] epaq_comment

    return info


if __name__ == "__main__":
    cwd = os.getcwd()
    info = create_info()
    filename = os.path.join(cwd, "inversion_info.toml")
    with open(filename, "w+") as fh:
        toml.dump(info, fh)
