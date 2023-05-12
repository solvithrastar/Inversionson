import emoji
import sys
from typing import List, Optional, Union

from pathlib import Path
from colorama import init
from inversionson.file_templates.inversion_info_template import InversionsonConfig
from inversionson.project import Project

init()


class AutoInverter(object):
    def __init__(
        self,
        config: InversionsonConfig,
        manual_mode: bool = False,
    ):
        self.project = Project(config)
        self.print(
            message="All Good, let's go!",
            line_above=True,
            line_below=True,
            emoji_alias=":gun:",
            color="cyan",
        )
        if not manual_mode:
            self.run_inversion()

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
        hpc_cluster = self.project.flow.hpc_cluster
        self.project.remote_paths.create_remote_directories(hpc_cluster)

        if (
            self.project.config.meshing.ocean_loading
            and self.project.config.meshing.multi_mesh
        ):
            if hpc_cluster.remote_exists(self.project.remote_paths.ocean_loading_f):
                self.print(
                    "Remote Bathymetry file is already uploaded",
                    emoji_alias=":white_check_mark:",
                )

            else:
                self.safe_put(
                    hpc_cluster,
                    self.project.config.meshing.ocean_loading_file,
                    self.project.remote_paths.ocean_loading_f,
                )
        if self.project.config.meshing.topography:
            if hpc_cluster.remote_exists(self.project.remote_paths.topography_f):
                self.print(
                    "Remote Topography file is already uploaded",
                    emoji_alias=":white_check_mark:",
                )

            else:
                self.safe_put(
                    hpc_cluster,
                    self.project.config.meshing.topography_file,
                    self.project.remote_paths.topography_f,
                )
        if not hpc_cluster.remote_exists(self.project.remote_paths.interp_script):
            local_interpolation_script = (
                Path(__file__).parent / "remote_scripts" / "interpolation.py"
            )
            self.safe_put(
                hpc_cluster,
                local_interpolation_script,
                self.project.remote_paths.interp_script,
            )

        if self.project.config.meshing.multi_mesh:
            self.project.lasif.move_gradient_to_cluster(
                hpc_cluster=hpc_cluster, overwrite=False
            )

    def run_inversion(self):
        self.move_files_to_cluster()


def _initialize_inversionson(root, info_toml_path):
    info_template = (
        Path(__file__).parent / "file_templates" / "inversion_info_template.py"
    )
    with open(info_template, "r") as fh:
        toml_string = fh.read()
    toml_string = toml_string.format(INVERSION_PATH=str(root))
    with open(info_toml_path, "w") as fh:
        fh.write(toml_string)
    print(f"I created a dummy configuration file {str(info_toml_path)}")
    print("Please edit this file as needed and run me again.")
    sys.exit()


def get_config(root: str) -> InversionsonConfig:
    """
    Read the inversion config file inversion_info.toml into a dictionary.

    If the file does not exist yet, create a new config file and exit.

    :param root: the project root
    """
    config_path = "inversion_config.py"
    root = Path(root).resolve() if root else Path.cwd()
    if not root.is_dir():
        raise NotADirectoryError(f"Specified project root {root} is not a directory")
    info_toml_path = root / config_path
    if not info_toml_path.is_file():
        _initialize_inversionson(root, config_path)
    else:
        print(f"Using configuration file {config_path}")
        return _get_inversionson_config(info_toml_path)


def _get_inversionson_config(config_path: Path) -> InversionsonConfig:
    import importlib

    inversion_config = importlib.import_module("inversion_config", config_path)
    config: InversionsonConfig = inversion_config.InversionsonConfig()
    return config


def run(root: Optional[str] = None):
    print(
        emoji.emojize(
            "\n :flag_for_Iceland: | Welcome to Inversionson | :flag_for_Iceland: \n",
            language="alias",
        )
    )
    AutoInverter(config=get_config(root))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    run(root)
