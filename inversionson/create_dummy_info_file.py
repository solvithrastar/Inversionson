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
    info["model_interpolation_mode"] = "gll_2_gll"
    info["gradient_interpolation_mode"] = "gll_2_gll"
    info["site_name"] = "daint"
    info["wall_time"] = 3600
    info["ranks"] = 1024
    info["inversion_parameters"] = ["VP", "VS", "RHO"]
    info["modelling_parameters"] = ["VP", "VS", "RHO"]
    info["n_random_events"] = 2
    info["max_ctrl_group_size"] = 4
    info["min_ctrl_group_size"] = 2
    info["max_angular_change"] = 30.0
    info["dropout_probability"] = 0.15

    return info


if __name__ == "__main__":
    cwd = os.getcwd()
    info = create_info
    filename = os.path.join(cwd, "inversion_info.toml")
    with open(filename, "w+") as fh:
        toml.dump(info, fh)
