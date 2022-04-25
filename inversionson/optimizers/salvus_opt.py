from pathlib import Path
import os
from inversionson import InversionsonError
from inversionson.optimizers.optimizer import Optimize


class SalvusOpt(Optimize):
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
        self.available_tasks = [
            "prepare_iteration",
            "compute_gradient",
            "update_model",
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
            raise InversionsonError(
                f"Please set config and provide initial model to "
                f"Adam optimizer in {self.config_file} \n"
                f"Then reinitialize the Adam Optimizer."
            )

        # Initialize folders if needed
        if not os.path.exists(self._get_path_for_iteration(0, self.model_path)):
            if self.initial_model is None:
                raise InversionsonError(
                    "AdamOptimizer needs to be initialized with a "
                    "path to an initial model."
                )
            print("Initializing Adam...")
            self._init_directories()
            self._issue_first_task()
        self.tmp_model_path = self.opt_folder / "tmp_model.h5"
        self._read_task_file()
