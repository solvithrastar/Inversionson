from pathlib import Path
import os
import toml
import numpy as np
import glob
import shutil
import h5py
from inversionson import InversionsonError
from inversionson.helpers.interpolation_helper import InterpolationListener
from inversionson.optimizers.optimizer import Optimize
from inversionson.helpers.regularization_helper import RegularizationHelper
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.helpers.autoinverter_helpers import AdjointHelper


class AdamOpt(Optimize):
    """
    A class that performs Adam optimization, if model_decay is
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

    optimizer_name = "Adam"

    def __init__(self, comm):

        # Call the super init with all the common stuff
        super().__init__(comm)

    def _initialize_derived_class_folders(self):
        """These folder are needed only for Adam."""
        self.smooth_update_dir = self.opt_folder / "SMOOTHED_UPDATES"
        self.first_moment_dir = self.opt_folder / "FIRST_MOMENTS"
        self.second_moment_dir = self.opt_folder / "SECOND_MOMENTS"
        self.smoothed_model_dir = self.opt_folder / "SMOOTHED_MODELS"

    @property
    def task_path(self):
        task_files = glob.glob(f"{self.task_dir}/task_*_*")
        if len(task_files) <= 1:
            return self.task_dir / f"task_00000_00.toml"
        iteration_numbers = [int(Path(x).stem.split("_")[-2]) for x in task_files]
        task_nums = glob.glob(f"{self.task_dir}/task_{max(iteration_numbers):05d}_*")
        if len(task_nums) <= 1:
            task_nums = [0]
        else:
            task_nums = [int(Path(x).stem[-2:]) for x in task_nums]
        if max(task_nums) >= len(self.available_tasks):
            raise InversionsonError(
                f"{task_nums}, but also... {max(iteration_numbers)}"
            )
        return (
            self.task_dir
            / f"task_{max(iteration_numbers):05d}_{max(task_nums):02d}.toml"
        )

    @property
    def raw_gradient_path(self):
        return self.raw_gradient_dir / f"gradient_{self.iteration_number:05d}.h5"

    @property
    def first_moment_path(self):
        return self.first_moment_dir / f"first_moment_{self.iteration_number:05d}.h5"

    @property
    def second_moment_path(self):
        return self.second_moment_dir / f"second_moment_{self.iteration_number:05d}.h5"

    @property
    def smoothed_model_path(self):
        return (
            self.smoothed_model_dir / f"smoothed_model_{self.iteration_number:05d}.h5"
        )

    @property
    def raw_update_path(self):
        return self.raw_update_dir / f"raw_update_{self.iteration_number:05d}.h5"

    @property
    def smooth_update_path(self):
        return self.smooth_update_dir / f"smooth_update_{self.iteration_number:05d}.h5"

    @property
    def relative_perturbation_path(self):
        return (
            self.regularization_dir
            / f"relative_perturbations_{self.iteration_number:05d}.h5"
        )

    @property
    def gradient_norm_path(self):
        return (
            self.gradient_norm_dir / f"gradient_norms_{self.iteration_number:05d}.toml"
        )

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
            "perturbation_decay": 0.001,
            "roughness_decay_type": "relative_perturbation",  # or absolute
            "update_smoothing_length": [0.5, 1.0, 1.0],
            "roughness_decay_smoothing_length": [0.0, 0.0, 0.0],
            "gradient_scaling_factor": 1e17,
            "epsilon": 1e-1,
            "initial_model": "",
            "max_iterations": 1000,
            "smoothing_timestep": 1.0e-5,
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
        self.smoothing_timestep = config["smoothing_timestep"]

        # Perturbation decay per iteration as a percentage of the relative
        # deviation to the initial model
        self.perturbation_decay = config["perturbation_decay"]
        self.roughness_decay_type = config["roughness_decay_type"]
        if self.roughness_decay_type not in ["relative_perturbation", "absolute"]:
            raise Exception(
                "Roughness decay type should be either "
                "'relative_perturbation' or 'absolute'"
            )
        self.update_smoothing_length = config["update_smoothing_length"]
        self.roughness_decay_smoothing_length = config[
            "roughness_decay_smoothing_length"
        ]

        # Gradient scaling factor to avoid issues with floats, this should be constant throughout the inversion
        self.grad_scaling_fac = config["gradient_scaling_factor"]
        # Regularization parameter to avoid dividing by zero
        self.epsilon = config["epsilon"]  # this is automatically scaled
        if "max_iterations" in config.keys():
            self.max_iterations = config["max_iterations"]
        else:
            self.max_iterations = None

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
            self.regularization_dir,
            self.smoothed_model_dir,
            self.gradient_norm_dir,
        ]

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

        shutil.copy(self.initial_model, self.model_path)

    def _issue_first_task(self):
        """
        Create the initial task of preparing iteration
        """

        task_dict = {
            "task": "prepare_iteration",
            "model": str(self.model_path),
            "iteration_number": self.iteration_number,
            "iteration_name": f"model_{self.iteration_number:05d}",
            "finished": False,
        }

        with open(self.task_path, "w+") as fh:
            toml.dump(task_dict, fh)

    @staticmethod
    def _get_path_for_iteration(iteration_number, path):
        filename = path.stem
        separator = "_"
        reconstructed_filename = (
            separator.join(filename.split("_")[:-1])
            + f"_{iteration_number:05d}"
            + path.suffix
        )
        return path.parent / reconstructed_filename

    def _read_task_file(self):
        if not os.path.exists(self.task_path):
            self._issue_first_task()
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
        self._read_task_file()
        if not self.task_dict["finished"]:
            raise InversionsonError(
                f"Task {self.task_dict['task']} does not appear to be finished"
            )
        if self.task_dict["task"] == "prepare_iteration":
            task_dict = {
                "task": "compute_gradient",
                "model": str(self.model_path),
                "forward_submitted": False,
                "misfit_completed": False,
                "gradient_completed": False,
                "validated": False,
                "iteration_number": self.iteration_number,
                "finished": False,
            }
            task_file_path = self._increase_task_number()
        elif self.task_dict["task"] == "compute_gradient":
            task_dict = {
                "task": "update_model",
                "model": str(self.model_path),
                "raw_update_path": str(self.raw_update_path),
                "raw_gradient_path": str(self.raw_gradient_path),
                "smooth_update_path": str(self.smooth_update_path),
                "summing_completed": False,
                "raw_update_completed": False,
                "smoothing_completed": False,
                "smooth_update_completed": False,
                "iteration_finalized": False,
                "iteration_number": self.iteration_number,
                "finished": False,
            }
            task_file_path = self._increase_task_number()
        elif self.task_dict["task"] == "update_model":
            task_dict = {
                "task": "prepare_iteration",
                "model": str(self._model_for_iteration(self.iteration_number)),
                "iteration_number": self.iteration_number,
                "finished": False,
            }
            task_file_path = self._increase_iteration_number()

        with open(task_file_path, "w+") as fh:
            toml.dump(task_dict, fh)
        self.task_dict = task_dict

    def _update_task_file(self):
        with open(self.task_path, "w+") as fh:
            toml.dump(self.task_dict, fh)

    def _compute_raw_update(self):
        """Computes the raw update"""

        self.print("Adam: Computing raw update...", line_above=True)
        # Read task toml

        iteration_number = self.task_dict["iteration_number"] + 1

        indices = self.get_parameter_indices(self.raw_gradient_path)
        # scale the gradients, because they can be tiny and this leads to issues
        g_t = self.get_h5_data(self.raw_gradient_path) * self.grad_scaling_fac

        if np.sum(np.isnan(g_t)) > 1:
            raise Exception(
                "NaNs were found in the raw gradient." "Something must be wrong."
            )

        if iteration_number == 1:  # Initialize moments if needed
            shutil.copy(self.raw_gradient_path, self.first_moment_path)

            with h5py.File(self.first_moment_path, "r+") as h5:
                data = h5["MODEL/data"]

                # initialize with zeros
                for i in indices:
                    data[:, i, :] = np.zeros_like(data[:, i, :])

            # Also initialize second moments with zeros
            shutil.copy(self.first_moment_path, self.second_moment_path)

        m_t = (
            self.beta_1 * self.get_h5_data(self.first_moment_path)
            + (1 - self.beta_1) * g_t
        )

        # Store first moment
        shutil.copy(
            self.first_moment_path,
            self._get_path_for_iteration(
                self.iteration_number + 1, self.first_moment_path
            ),
        )
        self.set_h5_data(
            self._get_path_for_iteration(
                self.iteration_number + 1, self.first_moment_path
            ),
            m_t,
        )

        # v_t was sometimes becoming too small, so enforce double precision
        v_t = self.beta_2 * self.get_h5_data(self.second_moment_path) + (
            1 - self.beta_2
        ) * (g_t**2)

        # Store second moment
        shutil.copy(
            self.second_moment_path,
            self._get_path_for_iteration(
                self.iteration_number + 1, self.second_moment_path
            ),
        )
        self.set_h5_data(
            self._get_path_for_iteration(
                self.iteration_number + 1, self.second_moment_path
            ),
            v_t,
        )

        # Correct bias
        m_t = m_t / (1 - self.beta_1 ** (self.iteration_number + 1))
        v_t = v_t / (1 - self.beta_2 ** (self.iteration_number + 1))

        # ensure e is sufficiently small, even for the small gradient values
        # that we typically have.
        e = self.epsilon * np.mean(np.sqrt(v_t))

        update = self.alpha * m_t / (np.sqrt(v_t) + e)

        max_upd = np.max(np.abs(update))
        self.print(f"Max raw model update: {max_upd}")
        if max_upd > 3.0 * self.alpha:
            raise Exception("Raw update seems a bit large")
        if np.sum(np.isnan(update)) > 1:
            raise Exception(
                "NaNs were found in the raw update."
                "Check if the gradient is not excessively small"
            )

        # Write raw update to file for smoothing
        shutil.copy(self.raw_gradient_path, self.raw_update_path)
        self.set_h5_data(self.raw_update_path, update)

    def get_path_for_iteration(self, iteration_number, path):
        return self._get_path_for_iteration(iteration_number, path)

    def _apply_smooth_update(self):
        """
        Apply the smoothed update.
        """
        self.print("Adam: Applying smooth update...", line_above=True)

        raw_update = self.get_h5_data(self.raw_update_path)
        max_raw_update = np.max(np.abs(raw_update))
        update = self.get_h5_data(self.smooth_update_path)

        raw_update_norm = np.sqrt(np.sum(raw_update**2))
        update_norm = np.sqrt(np.sum(update**2))

        if np.sum(np.isnan(update)) > 1:
            raise Exception(
                "NaNs were found in the smoothed update."
                "Check the raw update and smoothing process."
            )

        max_upd = np.max(np.abs(update))
        print(f"Max smooth model update: {max_upd}")
        if max_upd > 4.0 * self.alpha:
            raise Exception(
                "Check smooth gradient, the update is larger than expected."
            )

        update_scaling_fac_norm = raw_update_norm / update_norm
        update_scaling_fac_alpha = self.alpha / max_upd
        update_scaling_fac_peak = max_raw_update / max_upd

        update_scaling_fac = min(
            update_scaling_fac_norm, update_scaling_fac_alpha, update_scaling_fac_peak
        )
        print(
            "Update scaling factor norm:",
            update_scaling_fac_norm,
            "Update scaling factor alpha:",
            update_scaling_fac_alpha,
            "Update scaling factor peak:",
            update_scaling_fac_peak,
        )

        self.print(
            f"Recaling based on lowest rescaling factor: {update_scaling_fac},"
            f"New maximum update is: {max_upd * update_scaling_fac}"
        )

        update *= update_scaling_fac

        # normalise theta and apply update
        theta_0 = self.get_h5_data(self._get_path_for_iteration(0, self.model_path))

        # Update parameters
        if max(self.roughness_decay_smoothing_length) > 0.0:
            theta_prev = self.get_h5_data(self.smoothed_model_path)

            # If relative perturbations are smoothed, make model physical
            if self.roughness_decay_type == "relative_perturbation":
                theta_prev = (theta_prev + 1) * theta_0

        else:
            theta_prev = self.get_h5_data(self.model_path)

        # Normalize the model and prevent division by zero in the outer core.
        theta_prev[theta_0 != 0] = theta_prev[theta_0 != 0] / theta_0[theta_0 != 0] - 1

        # Make sure that the model is only updated where theta is non_zero
        theta_new = np.zeros_like(theta_0)
        theta_new[theta_0 != 0] = (
            theta_prev[theta_0 != 0]
            - update[theta_0 != 0]
            - self.perturbation_decay * theta_prev[theta_0 != 0]
        )

        # Remove normalization from updated model and write physical model
        theta_physical = (theta_new + 1) * theta_0
        shutil.copy(
            self.model_path,
            self.tmp_model_path,
        )
        self.set_h5_data(
            self.tmp_model_path,
            theta_physical,
        )

    def _finalize_iteration(self):
        """
        Here we can do some documentation. Maybe just call a base function for that
        """
        super().delete_remote_files()
        self.comm.storyteller.document_task(task="adam_documentation")

    def _update_model(self, raw=True, smooth=False):
        """
        Apply an Adam style update to the model
        """
        if (raw and smooth) or (not raw and not smooth):
            raise InversionsonError("Adam updates can be raw or smooth, not both")
        if raw:
            gradient = (
                self.comm.lasif.lasif_comm.project.paths["gradients"]
                / f"ITERATION_{self.iteration_name}"
                / "summed_gradient.h5"
            )
            if not os.path.exists(self.raw_gradient_path):
                shutil.copy(gradient, self.raw_gradient_path)
            if not os.path.exists(self.raw_update_path):
                self._compute_raw_update()
        if smooth:
            self._apply_smooth_update()

    def ready_for_validation(self) -> bool:
        return "validated" in self.task_dict.keys() and not self.task_dict["validated"]

    def prepare_iteration(self):
        """Prepare the iteration"""
        self.comm.project.change_attribute("current_iteration", self.iteration_name)
        self.print("Picking data for iteration")
        events = self._pick_data_for_iteration()

        super().prepare_iteration(it_name=self.iteration_name, events=events)
        self.finish_task()

    def perform_smoothing(self):
        tasks = {}
        if max(self.update_smoothing_length) > 0.0:
            tasks["smooth_raw_update"] = {
                "reference_model": str(self.comm.lasif.get_master_model()),
                "model_to_smooth": str(self.raw_update_path),
                "smoothing_lengths": self.update_smoothing_length,
                "smoothing_parameters": self.parameters,
                "output_location": str(self.smooth_update_path),
            }

        if max(self.roughness_decay_smoothing_length) > 0.0:
            # We either smooth the physical model and then map the results back
            # to the internal parameterization

            # Or we smooth the relative perturbations with respect to
            if self.roughness_decay_type == "absolute":
                model_to_smooth = self.model_path
            else:
                model_to_smooth = os.path.join(
                    self.regularization_dir,
                    f"relative_perturbation_{self.iteration_name}.h5",
                )
                shutil.copy(self.model_path, model_to_smooth)

                # relative perturbation = (latest - start) / start
                theta_prev = self.get_h5_data(self.model_path)
                theta_0 = self.get_h5_data(
                    self._get_path_for_iteration(0, self.model_path)
                )

                theta_prev[theta_0 != 0] = (
                    theta_prev[theta_0 != 0] / theta_0[theta_0 != 0] - 1
                )
                self.set_h5_data(model_to_smooth, theta_prev)

            tasks["roughness_decay"] = {
                "reference_model": str(self.comm.lasif.get_master_model()),
                "model_to_smooth": str(model_to_smooth),
                "smoothing_lengths": self.roughness_decay_smoothing_length,
                "smoothing_parameters": self.parameters,
                "output_location": str(self.smoothed_model_path),
            }

        if len(tasks.keys()) > 0:
            reg_helper = RegularizationHelper(
                comm=self.comm, iteration_name=self.iteration_name, tasks=tasks
            )
            reg_helper.monitor_tasks()
        else:
            raise InversionsonError(
                "We require some sort of smoothing in Adam Optimization"
            )

    def compute_gradient(self, verbose=False):
        """
        This task does forward simulations and then gradient computations straight
        afterwards
        """
        if not self.task_dict["forward_submitted"]:
            self.run_forward(verbose=verbose)
            self.task_dict["forward_submitted"] = True
            self._update_task_file()
        else:
            self.print("Forwards already submitted")

        if not self.task_dict["misfit_completed"]:
            self.compute_misfit(adjoint=True, window_selection=True, verbose=verbose)
            self.task_dict["misfit_completed"] = True
            self._update_task_file()
        else:
            self.print("Misfit already computed")

        if not self.task_dict["gradient_completed"]:
            super().compute_gradient(verbose=verbose)
            self.task_dict["gradient_completed"] = True
            self._update_task_file()
        else:
            self.print("Gradients already computed")
        self.finish_task()

    def update_model(self, verbose=False):
        """
        This task takes the raw gradient and does all the regularisation and everything
        to update the model.
        """
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
            self.comm.lasif.move_gradient_to_cluster()
        else:
            interpolate = False

        if not self.task_dict["summing_completed"]:
            adjoint_helper = AdjointHelper(
                comm=self.comm, events=self.comm.project.non_val_events_in_iteration
            )
            # Dispatch anything that may have failed
            adjoint_helper.dispatch_adjoint_simulations()
            adjoint_helper.process_gradients(
                interpolate=interpolate,
                verbose=verbose,
            )
            assert adjoint_helper.assert_all_simulations_retrieved()
            interp_listener = InterpolationListener(
                comm=self.comm, events=self.comm.project.non_val_events_in_iteration
            )
            interp_listener.monitor_interpolations()

            grad_summer = GradientSummer(comm=self.comm)
            grad_summer.sum_gradients(
                events=self.comm.project.non_val_events_in_iteration,
                output_location=self.raw_gradient_path,
                batch_average=True,
                sum_vpv_vph=True,
                store_norms=True,
            )
            self.task_dict["summing_completed"] = True
            self._update_task_file()
        else:
            self.print("Summing already done")

        if not self.task_dict["raw_update_completed"]:
            self._update_model(raw=True, smooth=False)
            self.task_dict["raw_update_completed"] = True
            self._update_task_file()
        else:
            self.print("Raw updating already completed")

        if not self.task_dict["smoothing_completed"]:
            self.perform_smoothing()
            self.task_dict["smoothing_completed"] = True
            self._update_task_file()
        else:
            self.print("Smoothing already done")

        if not self.task_dict["smooth_update_completed"]:
            self._update_model(raw=False, smooth=True)
            self.task_dict["smooth_update_completed"] = True
            self._update_task_file()
        else:
            self.print("Smooth updating already completed")

        if not self.task_dict["iteration_finalized"]:
            self._finalize_iteration()
            self.task_dict["iteration_finalized"] = True
            self._update_task_file()
        else:
            self.print("Iteration already finalized")

        self.finish_task()

    def perform_task(self, verbose=False):
        """
        Look at which task is the current one and call the function which does it.
        """
        task_name = self.task_dict["task"]
        self.print(f"Current task is: {task_name}", line_above=True)

        if task_name == "prepare_iteration":
            if not self.task_dict["finished"]:
                if self.comm.lasif.has_iteration(self.iteration_name):
                    self.print(
                        f"Iteration {self.iteration_name} exists. Will load its attributes"
                    )
                    self.comm.project.get_iteration_attributes()
                    self.finish_task()
                else:
                    self.prepare_iteration()
            else:
                self.print("Iteration already prepared")
        elif task_name == "compute_gradient":
            if not self.task_dict["finished"]:
                self.comm.project.get_iteration_attributes()
                self.compute_gradient(verbose=verbose)
            else:
                self.print("Gradient already computed")
        elif task_name == "update_model":
            if not self.task_dict["finished"]:
                self.comm.project.get_iteration_attributes()
                self.update_model(verbose=verbose)
            else:
                self.print("Model already updated")
        else:
            raise InversionsonError(f"Task {task_name} is not recognized by AdamOpt")

    def get_new_task(self):
        if self.task_dict["finished"]:
            self._write_new_task()
            self.print(f"New task is: {self.task_dict['task']}", line_above=True)
        else:
            raise InversionsonError(f"Task: {self.task_dict['task']} is not finished.")

    def finish_task(self):
        paths = ["raw_update_path", "model", "smooth_update_path", "raw_gradient_path"]
        complete_checks = [
            "smoothing_completed",
            "gradient_completed",
            "iteration_finalized",
            "forward_submitted",
            "raw_update_completed",
            "smooth_update_completed",
            "misfit_completed",
            "summing_completed",
            "validated:",
        ]
        for path in paths:
            if path in self.task_dict.keys():
                if not os.path.exists(self.task_dict[path]):
                    raise InversionsonError(
                        f"Trying to finish task but it can't find {self.task_dict[path]}"
                    )

        for complete_check in complete_checks:
            if complete_check in self.task_dict.keys():
                if not self.task_dict[complete_check]:
                    raise InversionsonError(
                        f"Trying to finish task but {complete_check} is not completed"
                    )
        self.task_dict["finished"] = True
        if self.task_dict["task"] == "update_model":
            self._update_task_file()
            # Moving the new model into its place, moves the iteration property to the next one.
            shutil.move(
                self.tmp_model_path,
                self._get_path_for_iteration(
                    self.iteration_number + 1, self.model_path
                ),
            )
        else:
            self._update_task_file()
