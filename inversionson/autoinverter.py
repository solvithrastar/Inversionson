import os
import emoji
import toml
import inspect
import sys
from typing import List, Optional, Union

from pathlib import Path
from colorama import init
from salvus.flow.api import get_site
from inversionson.project import Project
from inversionson.optimizers.optson import OptsonLink

init()
__INTERPOLATION_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "inversionson",
    "remote_scripts",
    "interpolation.py",
)

__INVERSION_INFO_template = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "inversionson",
    "file_templates",
    "inversion_info_template.toml",
)


def get_project(info):
    """
    Get Inversionson communicator.
    """
    return Project(info)


class AutoInverter(object):
    _REMOTE_DIRECTORIES = [
        "DIFFUSION_MODELS",
        "SOURCE_TIME_FUNCTIONS",
        "INTERPOLATION_WEIGHTS",
        "MESHES",
        "WINDOWS",
        "MISFITS",
        "ADJOINT_SOURCES",
        "PROCESSED_DATA",
        "SCRIPTS",
    ]

    def __init__(self, info_dict: dict, manual_mode=False, verbose=True):
        self.info = info_dict
        self.project = get_project(self.info)
        self.print(
            message="All Good, let's go!",
            line_above=True,
            line_below=True,
            emoji_alias=":gun:",
            color="cyan",
        )
        if not manual_mode:
            self.run_inversion(verbose=verbose)

    def print(
        self,
        message: str,
        color: str = "cyan",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def _initialize_remote_directories(self, hpc_cluster):
        self.print("Initializing remote directories.")
        if not hpc_cluster.remote_exists(self.project.remote_inversionson_dir):
            hpc_cluster.remote_mkdir(self.project.remote_inversionson_dir)

        for directory in self._REMOTE_DIRECTORIES:
            if not hpc_cluster.remote_exists(
                self.project.remote_inversionson_dir / directory
            ):
                hpc_cluster.remote_mkdir(
                    self.project.remote_inversionson_dir / directory
                )

    @staticmethod
    def safe_put(hpc_cluster, local_path: str, remote_path: str):
        remote_parent = Path(remote_path).parent
        if not hpc_cluster.remote_exists(remote_parent):
            hpc_cluster.remote_mkdir(remote_parent)

        tmp_remote_path = f"{remote_path}__tmp"
        hpc_cluster.remote_put(local_path, tmp_remote_path)
        hpc_cluster.run_ssh_command(f"mv {tmp_remote_path} {remote_path}")

    def move_files_to_cluster(self):
        """
        Move all the remote scripts to hpc.
        Move the bathymetry and topography files if it makes sense.
        """
        hpc_cluster = get_site(self.project.site_name)
        self._initialize_remote_directories(hpc_cluster)

        if self.project.ocean_loading["use"] and self.project.meshes == "multi-mesh":
            if not hpc_cluster.remote_exists(self.project.ocean_loading["remote_path"]):
                self.safe_put(
                    hpc_cluster,
                    self.project.ocean_loading["file"],
                    self.project.ocean_loading["remote_path"],
                )
            else:
                self.print(
                    "Remote Bathymetry file is already uploaded",
                    emoji_alias=":white_check_mark:",
                )

        if self.project.topography["use"]:
            if not hpc_cluster.remote_exists(self.project.topography["remote_path"]):
                self.safe_put(
                    hpc_cluster,
                    self.project.topography["file"],
                    self.project.topography["remote_path"],
                )
            else:
                self.print(
                    "Remote Topography file is already uploaded",
                    emoji_alias=":white_check_mark:",
                )

        remote_interp_path = self.project.multi_mesh.find_interpolation_script()
        self.safe_put(__INTERPOLATION_SCRIPT_PATH, remote_interp_path)

        if self.project.meshes == "multi-mesh":
            self.project.lasif.move_gradient_to_cluster(
                hpc_cluster=hpc_cluster, overwrite=False
            )

    def run_inversion(self, verbose=True):
        self.move_files_to_cluster
        opt_link = OptsonLink(self.comm)
        opt_link.perform_task(verbose=verbose)


def _initialize_inversionson(root, info_toml_path):
    with open(__INVERSION_INFO_template, "r") as fh:
        toml_string = fh.read()
    toml_string = toml_string.format(INVERSION_PATH=str(root))
    with open(info_toml_path, "w") as fh:
        fh.write(toml_string)
    print(f"I created a dummy configuration file {str(info_toml_path)}")
    print("Please edit this file as needed and run me again.")
    sys.exit()


def read_info_toml(root: str):
    """
    Read the inversion config file inversion_info.toml into a dictionary.

    If the file does not exist yet, create a new config file and exit.

    :param root: the project root
    """
    info_toml = "inversion_info.toml"
    root = Path(root).resolve() if root else Path.cwd()
    if not root.is_dir():
        raise NotADirectoryError(f"Specified project root {root} is not a directory")
    info_toml_path = root / info_toml
    if not info_toml_path.is_file():
        _initialize_inversionson(root, info_toml_path)
    else:
        print(f"Using configuration file {str(info_toml_path)}")
    return toml.load(info_toml_path)


def run(root: Optional[str] = None):
    print(
        emoji.emojize(
            "\n :flag_for_Iceland: | Welcome to Inversionson | :flag_for_Iceland: \n",
            language="alias",
        )
    )
    info = read_info_toml(root)
    AutoInverter(info)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    run(root)
