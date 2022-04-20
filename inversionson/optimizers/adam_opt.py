from pathlib import Path
import os
import toml
import numpy as np
import glob
import shutil
import h5py
from inversionson import InversionsonError
from inversionson.optimizers.optimizer import Optimize
from inversionson.optimizers.regularization_helper import RegularizationHelper
from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset


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
    def raw_update_path(self):
        return self.raw_update_dir / f"raw_update_{self.iteration_number:05d}.h5"

    @property
    def smooth_update_path(self):
        return self.smooth_update_dir / f"smooth_update_{self.iteration_number:05d}.h5"

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
            "model_decay": 0.001,
            "gradient_scaling_factor": 1e17,
            "epsilon": 1e-1,
            "parameters": ["VSV", "VSH", "VPV", "VPH"],
            "initial_model": "",
            "prior_model": "",
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
        self.model_decay = config["model_decay"]
        self.prior_model = config["prior_model"]
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

    def _get_path_for_iteration(self, iteration_number, path):
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

    def _pick_data_for_iteration(self, validation=False):
        if validation:
            events = self.comm.project.validation_dataset
        else:
            all_events = self.comm.lasif.list_events()
            blocked_data = list(
                set(
                    self.comm.project.validation_dataset
                    + self.comm.project.test_dataset
                )
            )
            all_events = list(set(all_events) - set(blocked_data))
            n_events = self.comm.project.initial_batch_size
            doc_path = self.comm.project.paths["inversion_root"] / "DOCUMENTATION"
            all_norms_path = doc_path / "all_norms.toml"
            if os.path.exists(all_norms_path):
                norm_dict = toml.load(all_norms_path)
                unused_events = list(set(all_events).difference(set(norm_dict.keys())))
                list_of_vals = np.array(list(norm_dict.values()))
                max_norm = np.max(list_of_vals)

                # Assign high norm values to unused events to make them
                # more likely to be chosen
                for event in unused_events:
                    norm_dict[event] = max_norm
                events = get_random_mitchell_subset(
                    self.comm.lasif.lasif_comm, n_events, all_events, norm_dict
                )
            else:
                events = get_random_mitchell_subset(
                    self.comm.lasif.lasif_comm, n_events, all_events
                )

        return events

    def _compute_raw_update(self):
        """Computes the raw update"""

        print("Adam: Computing raw update...")
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
        e = self.e * np.mean(np.sqrt(v_t))

        update = self.alpha * m_t / (np.sqrt(v_t) + e)

        max_upd = np.max(np.abs(update))
        print(f"Max raw model update: {max_upd}")
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
        print("Adam: Applying smooth update...")

        raw_update = self.get_h5_data(self.raw_update_path)
        max_raw_update = np.max(np.abs(raw_update))
        update = self.get_h5_data(self.smooth_update_path)

        raw_update_norm = np.sqrt(np.sum(raw_update ** 2))
        update_norm = np.sqrt(np.sum(update ** 2))

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

        update_scaling_fac = min(update_scaling_fac_norm,
                                 update_scaling_fac_alpha,
                                 update_scaling_fac_peak)
        print("update_scaling_fac_norm", update_scaling_fac_norm,
              "update_scaling_fac_alpha", update_scaling_fac_alpha,
              "update_scaling_fac_peak", update_scaling_fac_peak,
              )

        print(f"Recaling based on lowest rescaling fac: {update_scaling_fac},"
              f"New maximum update is: {max_upd * update_scaling_fac}")

        update *= update_scaling_fac

        # Update parameters
        theta_prev = self.get_h5_data(self.model_path)
        theta_delta_prior = np.copy(theta_prev)

        # normalise theta and apply update
        theta_0 = self.get_h5_data(self._get_path_for_iteration(0, self.model_path))
        theta_prior = self.get_h5_data(self.prior_model)

        # Normalize the model and prevent division by zero in the outer core.
        theta_prev[theta_0 != 0] = theta_prev[theta_0 != 0] / theta_0[theta_0 != 0] - 1
        theta_delta_prior[theta_prior != 0] = theta_delta_prior[theta_prior != 0] / theta_prior[theta_prior != 0] - 1

        # only add model_decay at this stage
        theta_new = theta_prev - update - self.model_decay * theta_delta_prior

        # remove normalization from updated model and write physical model
        theta_physical = (theta_new + 1) * theta_0
        shutil.copy(
            self.model_path,
            self.tmp_model_path,
        )
        self.set_h5_data(
            self.tmp_model_path,
            theta_physical,
        )

    def _finalize_iteration(self, verbose: bool):
        """
        Here we can do some documentation. Maybe just call a base function for that
        """
        super().delete_remote_files()
        self.comm.storyteller.document_task(task="adam_documentation")

    def _update_model(self, verbose: bool, raw=True, smooth=False):
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
            smooth_update = self.comm.lasif.find_gradient(
                iteration=self.iteration_name,
                event=None,
                smooth=True,
                summed=True,
            )
            shutil.copy(smooth_update, self.smooth_update_path)
            self._apply_smooth_update()

    def ready_for_validation(self) -> bool:
        return "validated" in self.task_dict.keys() and not self.task_dict["validated"]

    def prepare_iteration(self, validation=False):
        if validation:
            it_name = f"validation_{self.iteration_name}"
        else:
            it_name = self.iteration_name

        move_meshes = "00000" in it_name if validation else True
        self.comm.project.change_attribute("current_iteration", it_name)

        if self.comm.lasif.has_iteration(it_name):
            raise InversionsonError(f"Iteration {it_name} already exists")

        print("Picking data for iteration")
        events = self._pick_data_for_iteration(validation=validation)

        super().prepare_iteration(
            it_name=it_name, move_meshes=move_meshes, events=events
        )
        if not validation:
            self.finish_task()

    def compute_gradient(self, verbose):
        """
        This task does forward simulations and then gradient computations straight
        afterwards
        """
        if not self.task_dict["forward_submitted"]:
            self.run_forward(verbose=verbose)
            self.task_dict["forward_submitted"] = True
            self._update_task_file()
        else:
            print("Forwards already submitted")

        if not self.task_dict["misfit_completed"]:
            self.compute_misfit(adjoint=True, window_selection=True, verbose=verbose)
            self.task_dict["misfit_completed"] = True
            self._update_task_file()
        else:
            print("Misfit already computed")

        if not self.task_dict["gradient_completed"]:
            super().compute_gradient(verbose=verbose)
            self.task_dict["gradient_completed"] = True
            self._update_task_file()
        else:
            print("Gradients already computed")
        self.finish_task()

    def update_model(self, verbose):
        """
        This task takes the raw gradient and does all the regularisation and everything
        to update the model.
        """
        if not self.task_dict["summing_completed"]:
            # This only interpolates and sums gradients
            self.regularization(
                smooth_individual=False,
                sum_gradients=True,
                smooth_update=True,
                verbose=verbose,
            )
            self.task_dict["summing_completed"] = True
            self._update_task_file()
        else:
            print("Summing already done")

        if not self.task_dict["raw_update_completed"]:
            self._update_model(raw=True, smooth=False, verbose=verbose)
            self.task_dict["raw_update_completed"] = True
            self._update_task_file()
        else:
            print("Raw updating already completed")

        if not self.task_dict["smoothing_completed"]:
            tasks = {"smooth_raw_update":
                         {"reference_model": str(self.comm.lasif.get_master_model()),
                          "model_to_smooth": str(self.raw_update_path),
                          "smoothing_lengths": self.comm.project.smoothing_lengths,
                          "smoothing_parameters": self.parameters,
                          "output_location": str(self.smooth_update_path)}}
            # Run the remote smoother with the raw update
            reg_helper = RegularizationHelper(
                comm=self.comm, iteration_name=self.iteration_name,
                tasks=tasks)
            reg_helper.monitor_tasks()

            # # This only smooths the update
            # self.regularization(
            #     smooth_individual=False,
            #     sum_gradients=False,
            #     smooth_update=True,
            #     verbose=verbose,
            # )
            self.task_dict["smoothing_completed"] = True
            self._update_task_file()
        else:
            print("Smoothing already done")

        if not self.task_dict["smooth_update_completed"]:
            self._update_model(raw=False, smooth=True, verbose=verbose)
            self.task_dict["smooth_update_completed"] = True
            self._update_task_file()
        else:
            print("Smooth updating already completed")

        if not self.task_dict["iteration_finalized"]:
            self._finalize_iteration(verbose=verbose)
            self.task_dict["iteration_finalized"] = True
            self._update_task_file()
        else:
            print("Iteration already finalized")

        self.finish_task()

    def do_validation_iteration(self, verbose=False):
        """
        This function computes the validation misfits.
        """
        if self.task_dict["validated"]:
            print("Validation misfit already computed")
            return

        it_name = f"validation_{self.iteration_name}"
        if not self.comm.lasif.has_iteration(it_name):
            self.prepare_iteration(validation=True)
        else:
            self.comm.project.get_iteration_attributes(validation=True)
        super().compute_validation_misfit(verbose=verbose)
        iteration = self.comm.project.current_iteration
        iteration = iteration[11:]
        self.comm.project.change_attribute(
            attribute="current_iteration", new_value=iteration
        )
        self.comm.project.get_iteration_attributes()

        self.task_dict["validated"] = True
        self._update_task_file()

    def perform_task(self, verbose=False):
        """
        Look at which task is the current one and call the function which does it.
        """
        task_name = self.task_dict["task"]
        print(f"Current task is: {task_name}")

        if task_name == "prepare_iteration":
            if not self.task_dict["finished"]:
                if self.comm.lasif.has_iteration(self.iteration_name):
                    print(
                        f"Iteration {self.iteration_name} exists. Will load its attributes"
                    )
                    self.comm.project.get_iteration_attributes(validation=False)
                    self.finish_task()
                else:
                    self.prepare_iteration(validation=False)
            else:
                print("Iteration already prepared")
        elif task_name == "compute_gradient":
            if not self.task_dict["finished"]:
                self.comm.project.get_iteration_attributes(validation=False)
                self.compute_gradient(verbose=verbose)
            else:
                print("Gradient already computed")
        elif task_name == "update_model":
            if not self.task_dict["finished"]:
                self.comm.project.get_iteration_attributes(validation=False)
                self.update_model(verbose=verbose)
            else:
                print("Model already updated")
        else:
            raise InversionsonError(f"Task {task_name} is not recognized by AdamOpt")

    def get_new_task(self):
        if self.task_dict["finished"]:
            self._write_new_task()
            print(f"New task is: {self.task_dict['task']}")
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
            "validated:"
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
