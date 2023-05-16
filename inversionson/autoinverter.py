import emoji  # type: ignore
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
        emoji_alias: Optional[Union[str, List[str]]] = None,
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

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
        if self.project.config.meshing.topography and not hpc_cluster.remote_exists(
            self.project.remote_paths.topography_f
        ):
            self.project.flow.safe_put(
                self.project.config.meshing.topography_file,
                self.project.remote_paths.topography_f,
            )

        if not hpc_cluster.remote_exists(self.project.remote_paths.interp_script):
            local_interpolation_script = (
                Path(__file__).parent / "remote_scripts" / "interpolation.py"
            )
            self.project.flow.safe_put(
                local_interpolation_script, self.project.remote_paths.interp_script
            )

        if self.project.config.meshing.multi_mesh:
            self.project.lasif.move_gradient_to_cluster()

    def _run_optson(self):
        """
        This function will be extracted and become user configurable.
        """
        from optson.optimizer import Optimizer  # type: ignore
        from optson.methods import AdamUpdate  # type: ignore
        from optson.stopping_criterion import BasicStoppingCriterion  # type: ignore
        from optson.monitor import BasicMonitor  # type: ignore
        from inversionson.optson_link.problem import Problem
        from inversionson.optson_link.helpers import mesh_to_vector

        sc = BasicStoppingCriterion(tolerance=1e-5, max_iterations=1)
        monitor = BasicMonitor(step=1)
        problem = Problem(project=self.project)

        opt = Optimizer(
            problem=problem,
            update=AdamUpdate(alpha=1.5),
            stopping_criterion=sc,
            monitor=monitor,
        )
        opt.iterate(
            x0=mesh_to_vector(
                self.project.lasif.master_mesh,
                params_to_invert=self.project.config.inversion.inversion_parameters,
            )
        )

    def run_inversion(self):
        self.move_files_to_cluster()
        self._run_optson()

        # Here, we initialize Optson


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


def get_config(root: Optional[Union[str, Path]] = None) -> InversionsonConfig:
    """
    Read the inversion config file inversion_info.toml into a dictionary.

    If the file does not exist yet, create a new config file and exit.

    :param root: the project root
    """
    config_path = "inversion_config.py"
    root = root or ""
    inversion_root = Path(root)

    if not inversion_root.is_dir():
        raise NotADirectoryError(
            f"Specified project root {inversion_root} is not a directory"
        )
    info_toml_path = inversion_root / config_path
    if not info_toml_path.is_file():
        _initialize_inversionson(inversion_root, config_path)
    print(f"Using configuration file {config_path}")
    return _get_inversionson_config(info_toml_path)


def _get_inversionson_config(config_path: Path) -> InversionsonConfig:
    import importlib

    inversion_config = importlib.import_module("inversion_config", str(config_path))
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
    root = sys.argv[1] if len(sys.argv) > 1 else ""
    run(root)
