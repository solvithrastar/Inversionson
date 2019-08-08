"""
A script to create an information toml file in a format that
Inversionson can read.
The values of the toml file can then be changed and used as
an actual information toml.
"""

import toml
import os

cwd = os.getcwd()
info = {}

info["inversion_path"] = cwd
info["lasif_root"] = os.path.join(cwd, "LASIF_PROJECT")
info["inversion_id"] = "MY_INVERSION"
info["model_interpolation_mode"] = "gll_2_gll"
info["gradient_interpolation_mode"] = "gll_2_gll"
info["site_name"] = "daint"
info["wall_time"] = 3600
info["ranks"] = 1024
info["inversion_parameters"] = ["VP", "VS", "RHO"]
info["modelling_parameters"] = ["VP", "VS", "RHO"]
info["n_random_events"] = 2
info["salvus_smoother"] = os.path.join(cwd, "salvus_smoother")

filename = os.path.join(cwd, "inversion_info.toml")
with open(filename, "w+") as fh:
    toml.dump(info, fh)
