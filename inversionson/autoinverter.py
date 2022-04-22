import os
import shutil
import emoji
import toml
import inspect
import sys

from pathlib import Path
from colorama import init
from colorama import Fore, Style
from salvus.flow.api import get_site
from inversionson import autoinverter_helpers as helpers
from inversionson.tasks import TaskManager
from inversionson import InversionsonError

init()
INTERPOLATION_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "inversionson",
    "remote_scripts",
    "interpolation.py",
)


def _find_project_comm(info):
    """
    Get Inversionson communicator.
    """
    from inversionson.components.project import ProjectComponent

    return ProjectComponent(info).get_communicator()


class AutoInverter(object):
    def __init__(self, info_dict: dict, manual_mode=False, verbose=False):
        self.info = info_dict
        print(Fore.RED + "Will make communicator now")
        self.comm = _find_project_comm(self.info)
        print(Fore.GREEN + "All Good, let's go")
        if not manual_mode:
            self.run_inversion(verbose=verbose)

    def move_files_to_cluster(self):
        """
        Move all the remote scripts to hpc.
        Move the bathymetry and topography files if it makes sense.
        """
        hpc_cluster = get_site(self.comm.project.site_name)

        if not hpc_cluster.remote_exists(self.comm.project.remote_inversionson_dir):
            hpc_cluster.remote_mkdir(self.comm.project.remote_inversionson_dir)
        for directory in [
            "DIFF_MODELS",
            "SOURCE_TIME_FUNCTIONS",
            "INTERPOLATION_WEIGHTS",
            "MESHES",
            "ADJOINT_SOURCES",
            "PROCESSED_DATA",
            "SCRIPTS",
        ]:
            if not hpc_cluster.remote_exists(
                self.comm.project.remote_inversionson_dir / directory
            ):
                hpc_cluster.remote_mkdir(
                    self.comm.project.remote_inversionson_dir / directory
                )

        if self.comm.project.ocean_loading["use"]:
            if hpc_cluster.remote_exists(
                self.comm.project.ocean_loading["remote_path"]
            ):
                print("Remote Bathymetry file is already uploaded")
            else:
                if not hpc_cluster.remote_exists(
                    Path(self.comm.project.ocean_loading["remote_path"]).parent
                ):
                    hpc_cluster.remote_mkdir(
                        Path(self.comm.project.ocean_loading["remote_path"]).parent
                    )
                hpc_cluster.remote_put(
                    self.comm.project.ocean_loading["file"],
                    self.comm.project.ocean_loading["remote_path"],
                )
        if self.comm.project.topography["use"]:
            if hpc_cluster.remote_exists(self.comm.project.topography["remote_path"]):
                print("Remote Topography file is already uploaded")
            else:
                if not hpc_cluster.remote_exists(
                    Path(self.comm.project.topography["remote_path"]).parent
                ):
                    hpc_cluster.remote_mkdir(
                        Path(self.comm.project.topography["remote_path"]).parent
                    )
                hpc_cluster.remote_put(
                    self.comm.project.topography["file"],
                    self.comm.project.topography["remote_path"],
                )
        remote_interp_path = self.comm.multi_mesh.find_interpolation_script()
        hpc_cluster.remote_put(INTERPOLATION_SCRIPT_PATH, remote_interp_path)

        self.comm.lasif.move_gradient_to_cluster(
            hpc_cluster=hpc_cluster, overwrite=False
        )

    def run_inversion(self, n_iterations=1000, verbose=True):
        taskmanager = TaskManager(
            optimization_method=self.comm.project.optimizer, comm=self.comm
        )
        print("Moving important files to cluster")
        self.move_files_to_cluster()
        n_tasks = taskmanager.get_n_tasks()
        n_tasks *= n_iterations
        for i in range(n_tasks):
            taskmanager.perform_task(verbose=verbose)
            taskmanager.get_new_task()


def read_info_toml(root):
    """
    Read the inversion config file inversion_info.toml into a dictionary.

    If the file does not exist yet, create a new config file and exit.

    :param root: the project root
    """
    info_toml = "inversion_info.toml"
    root = Path(root).resolve() if root else Path.cwd()
    if not root.is_dir():
        raise NotADirectoryError(
            "Specified project root {} is not a directory".format(root)
        )
    info_toml_path = root / info_toml
    if not info_toml_path.is_file():
        script_dir = Path(__file__).parent
        template_toml = script_dir / "inversion_info_template.toml"
        with open(template_toml, "r") as fh:
            toml_string = fh.read()
        toml_string = toml_string.format(INVERSION_PATH=str(root))
        with open(info_toml_path, "w") as fh:
            fh.write(toml_string)
        print("I created a dummy configuration file " + str(info_toml_path))
        print("Please edit this file as needed and run me again.")
        exit()
    else:
        print("Using configuration file " + str(info_toml_path))
    return toml.load(info_toml_path)


def run(root=None):
    print(
        emoji.emojize(
            "\n :flag_for_Iceland: | Welcome to Inversionson | :flag_for_Iceland: \n",
            use_aliases=True,
        )
    )
    info = read_info_toml(root)
    invert = AutoInverter(info)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    run(root)
