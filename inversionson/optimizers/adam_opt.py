"""
Base optimization class. It covers all the basic things that most optimizers
have in common. The class serves the purpose of making it easy to add
a custom optimizer to Inversionson. Whenever the custom optimizer has a 
task which works the same as in the base class. It should aim to use that one.
"""
from pathlib import Path
import os
import toml
import numpy as np
import glob
import shutil
import h5py
from inversionson import InversionsonError, autoinverter_helpers as helpers
from inversionson.optimizers.optimizer import Optimize


class AdamOpt(Optimize):
    """
    A class that performs Adam optimization, if weight_decay is
    set to a non-zero value, it performs AdamW optimization. This is
    essentially a type of L2 smoothing.

    Useful references:
    https://arxiv.org/abs/1412.6980
    https://towardsdatascience.com/why-adamw-matters-736223f31b5d

    And somewhat unrelated, but relevant on Importance Sampling:
    https://arxiv.org/pdf/1803.00942.pdf

    The class inherits from the base optimizer class and is designed
    to facilitate any differences between the two classes.
    """

    def __init__(self, comm):
        self.current_task = self.read_current_task()
        self.available_tasks = [
            "prepare_iteration",
            "compute_gradient",
            "update_model",
            "documentation",
        ]
        self.comm = comm
        self.opt_folder = (
            Path(self.comm.project.paths["inversion_root"]) / "OPTIMIZATION"
        )
        self.models = self.opt_folder / "MODELS"

        if not os.path.exists(self.opt_folder):
            os.mkdir(self.opt_folder)
        self.config_file = self.opt_folder / "opt_config.toml"

        self.model_dir = self.opt_folder / "MODELS"
        self.raw_gradient_dir = self.opt_folder / "RAW_GRADIENTS"
        self.raw_update_dir = self.opt_folder / "RAW_UPDATES"
        self.smooth_update_dir = self.opt_folder / "SMOOTHED_UPDATES"
        self.first_moment_dir = self.opt_folder / "FIRST_MOMENTS"
        self.second_moment_dir = self.opt_folder / "SECOND_MOMENTS"
        self.task_dir = self.opt_folder / "TASKS"

        if not os.path.exists(self.config_file):
            self._write_initial_config()
            print(
                f"Please set config and provide initial model to "
                f"Adam optimizer in {self.config_file} \n"
                f"Then reinitialize the Adam Optimizer."
            )
            return
        self._read_config()

        if self.initial_model == "":
            print(
                f"Please set config and provide initial model to "
                f"Adam optimizer in {self.config_file} \n"
                f"Then reinitialize the Adam Optimizer."
            )
            return

        # Initialize folders if needed
        if not os.path.exists(self.get_model_path(iteration_number=0)):
            if self.initial_model is None:
                raise Exception(
                    "AdamOptimizer needs to be initialized with a "
                    "path to an initial model."
                )
            print("Initializing Adam...")
            self._init_directories()
            self._issue_first_task()
            self._read_task()

    @property
    def task_path(self):
        task_nums = glob.glob(f"{self.task_dir}/task_{self.iteration_number:05d}_*")
        if len(task_nums) == 0:
            task_nums = [0]
        else:
            task_nums = task_nums.split("_")[-1].split(".")[0].sort()
        return (
            self.task_dir
            / f"task_{self.iteration_number:05d}_{int(task_nums[-1]):02d}.toml"
        )

    @property
    def model_path(self):
        return self.model_dir / f"model_{self.iteration_number:05d}.h5"

    def _model_for_iteration(self, iteration_number):
        return self.model_dir / f"model_{iteration_number:05d}.h5"

    def _write_initial_config(self):
        """
        Writes the initial config file.
        """
        config = {
            "alpha": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "weight_decay": 0.001,
            "gradient_scaling_factor": 1e17,
            "epsilon": 1e-1,
            "parameters": ["VSV", "VSH", "VPV", "VPH"],
            "initial_model": "",
            "max_iterations": 1000,
        }
        with open(self.config_file, "w") as fh:
            toml.dump(config, fh)

        print(
            "Wrote a config file for the Adam optimizer. Please provide "
            "an initial model."
        )

    def _read_config(self):
        """Reads the config file."""

        if not os.path.exists(self.config_file):
            raise Exception("Can't read the ADAM config file")
        config = toml.load(self.config_file)
        self.initial_model = config["initial_model"]
        self.alpha = config["alpha"]
        self.beta_1 = config["beta_1"]  # decay factor for first moments
        self.beta_2 = config["beta_2"]  # decay factor for second moments
        # weight decay as percentage of # deviation from initial
        self.weight_decay = config["weight_decay"]
        # gradient scaling factor avoid issues with floats
        self.grad_scaling_fac = config["gradient_scaling_factor"]
        # Regularization parameter to avoid dividing by zero
        self.e = config["epsilon"]  # this is automatically scaled
        if "max_iterations" in config.keys():
            self.max_iterations = config["max_iterations"]
        else:
            self.max_iterations = None
        self.parameters = config["parameters"]

    def _init_directories(self):
        """
        Build directory structure.
        """
        folders = [
            self.model_dir,
            self.raw_gradient_dir,
            self.first_moment_dir,
            self.second_moment_dir,
            self.raw_update_dir,
            self.smooth_update_dir,
            self.task_dir,
        ]

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

        shutil.copy(self.initial_model, self.model_path())

    def _issue_first_task(self):
        """
        Create the initial task of preparing iteration
        """

        task_dict = {
            "task": "prepare_iteration",
            "model": self.model_path,
            "iteration_number": self.iteration_number,
            "iteration_name": f"model_{self.iteration_number:05d}",
            "finished": False,
        }

        with open(self.task_path, "w+") as fh:
            toml.dump(task_dict, fh)

    def _read_task_file(self):
        self.task_dict = toml.load(self.task_path)

    def _increase_task_number(self):
        task_file = self.task_path.stem
        task_number = int(task_file.split("_")[-1]) + 1
        return (
            self.task_dir / f"task_{self.iteration_number:05d}_{task_number:02d}.toml"
        )

    def _increase_iteration_number(self):
        task_file = self.task_path.stem
        iteration_number = int(task_file.split("_")[-2]) + 1
        return self.task_dir / f"task_{iteration_number:05d}_{0:02d}.toml"

    def _write_new_task(self):
        current_task = self._read_task_file()
        if not current_task["finished"]:
            raise InversionsonError(
                f"Task {current_task['task']} does not appear to be finished"
            )
        if current_task["task"] == "prepare_iteration":
            task_dict = {
                "task": "compute_gradient",
                "model": self.model_path,
                "raw_gradient_path": self.raw_gradient_path,
                "gradient_completed": False,
                "iteration_number": self.iteration_number,
                "finished": False,
            }
            task_file_path = self._increase_task_number()
        elif current_task["task"] == "compute_gradient":
            task_dict = {
                "task": "update_model",
                "model": self.model_path,
                "raw_update_path": self.raw_update_path,
                "smooth_update_path": self.smooth_path,
                "smoothing_completed": False,
                "iteration_finalized": False,
                "iteration_number": self.iteration_number,
                "finished": False,
            }
            task_file_path = self._increase_task_number()
        elif (
            current_task["task"] == "update_model"
        ):  # Here I'm assuming that there will already be created a newer model
            task_dict = {
                "task": "prepare_iteration",
                "model": self._model_for_iteration(self.iteration_number + 1),
                "iteration_number": self.iteration_number + 1,
                "finished": False,
            }
            task_file_path = self._increase_iteration_number()

        with open(task_file_path, "w+") as fh:
            toml.dump(task_dict, fh)

    def _update_task_file(self):
        with open(self.task_path, "w+") as fh:
            toml.dump(self.task_dict, fh)

    def read_and_write_task(self):
        """
        Checks task status and writes new task if task is already completed.
        """
        if self.iteration_number is None:
            task_path = self.get_latest_task()
        else:
            task_path = self.get_task_path(self.iteration_number)

        # If task exists, read it and see if model needs updating
        if os.path.exists(task_path):
            task_info = toml.load(task_path)
            if not task_info["iteration_finalized"]:
                print("Please complete task first")
        else:  # write task
            task_dict = {
                "task": "compute_gradient_for_adam",
                "model": self.get_model_path(),
                "raw_gradient_path": self.get_gradient_path(),
                "raw_update_path": self.get_raw_update_path(),
                "smooth_update_path": self.get_smooth_path(),
                "gradient_completed": False,
                "smoothing_completed": False,
                "iteration_finalized": False,
                "time_step": self.iteration_number,
            }

            with open(task_path, "w+") as fh:
                toml.dump(task_dict, fh)
