import os
import shutil
import emoji
import toml

from pathlib import Path
from colorama import init
from colorama import Fore, Style
from salvus.flow.api import get_site
from inversionson import autoinverter_helpers as helpers
from inversionson.tasks import TaskManager
from inversionson import InversionsonError

init()


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

    def run_inversion(self, n_iterations=1000, verbose=False):
        taskmanager = TaskManager(
            optimization_method=self.comm.project.optimizer, comm=self.comm
        )
        for i in range(n_iterations):
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
    invert = AutoInverterNew(info)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    run(root)
