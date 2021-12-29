

from lasif.components.component import Component

import os
import toml
import numpy as np
import glob
import shutil
import h5py  # Use h5py to avoid lots of dependencies and slow reading

BOOL_ADAM = True


# TODO: To ensure smoothness, we may also just smooth the total update
# TODO: Add smart sampling

class AdamOptimizer:
    """
    A class that performs Adam optimization, if weight_decay is
    set to a non-zero value, it performs AdamW optimization. This is
    essentially a type of L2 smoothing.

    Useful references:
    https://arxiv.org/abs/1412.6980
    https://towardsdatascience.com/why-adamw-matters-736223f31b5d

    And somewhat unrelated, but relevant on Importance Sampling:
    https://arxiv.org/pdf/1803.00942.pdf
    """
    def __init__(self, inversion_root):
        """
        :param inversion_root: path to the folder that ADAM will use.
        :type inversion_root: str:
        """
        self.optimization_folder = os.path.join(inversion_root, "AdamOpt")
        if not os.path.exists(self.optimization_folder):
            os.mkdir(self.optimization_folder)
        self.config_file = os.path.join(self.optimization_folder,
                                        "adam_config.toml")
        # gradient scaling factor avoid issues with floats
        self.grad_scaling_fac = 10e20

        self.model_dir = os.path.join(self.optimization_folder,
                                      "MODELS")
        self.gradient_dir = os.path.join(self.optimization_folder,
                                         "RAW_GRADIENTS")
        self.raw_update_dir = os.path.join(self.optimization_folder,
                                           "RAW_UPDATES")
        self.smooth_update_dir = os.path.join(self.optimization_folder,
                                              "SMOOTHED_UPDATES")
        self.first_moment_dir = os.path.join(self.optimization_folder,
                                             "FIRST_MOMENTS")
        self.second_moment_dir = os.path.join(self.optimization_folder,
                                              "SECOND_MOMENTS")
        self.task_dir = os.path.join(self.optimization_folder,
                                     "TASKS")

        if not os.path.exists(self.config_file):
            self._write_initial_config()
            print(f"Please set config and provide initial model to "
                  f"Adam optimizer in {self.config_file} \n"
                  f"Then reinitialize the Adam Optimizer.")
            return
        self._read_config()

        if self.initial_model == "":
            print(f"Please set config and provide initial model to "
                  f"Adam optimizer in {self.config_file} \n"
                  f"Then reinitialize the Adam Optimizer.")
            return

        # Initialize folders if needed
        if not os.path.exists(self.get_model_path(time_step=0)):
            if self.initial_model is None:
                raise Exception(
                    "AdamOptimizer needs to be initialized with a "
                    "path to an initial model.")
            self.time_step = 0
            print("Initializing Adam...")
            self._init_directories()
            self.read_and_write_task()
        else:  # Figure out time step and read task
            self.get_latest_time_step()

    def _write_initial_config(self):
        """Writes the initial config file."""
        config = {"alpha": 0.001, "beta_1": 0.9, "beta_2": 0.999,
                  "weight_decay": 0.001, "epsilon": 10e-8,
                  "parameters": ["VSV", "VSH", "VPV", "VPH"],
                  "initial_model": ""}
        with open(self.config_file, "w") as fh:
            toml.dump(config, fh)

        print("Wrote a config file for the Adam optimizer. Please provide "
              "an initial model.")

    def _read_config(self):
        """ Reads the config file."""
        if not os.path.exists(self.config_file):
            raise Exception("Can't read the ADAM config file")
        config = toml.load(self.config_file)
        self.initial_model = config["initial_model"]
        self.alpha = config["alpha"]
        self.beta_1 = config["beta_1"]  # decay factor for first moments
        self.beta_2 = config["beta_2"]  # decay factor for second moments
        # weight decay as percentage of # deviation from initial
        self.weight_decay = config["weight_decay"]
        # Regularization parameter to avoid dividing by zero
        self.e = config["epsilon"]  # this is automatically scaled
        self.parameters = config["parameters"]

    def _init_directories(self):
        """
        Build directory structure.
        """
        folders = [self.model_dir, self.gradient_dir, self.first_moment_dir,
                   self.second_moment_dir, self.raw_update_dir,
                   self.smooth_update_dir, self.task_dir]

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

        shutil.copy(self.initial_model, self.get_model_path())

    def _get_parameter_indices(self, filename):
        """ Get parameter indices in h5 file"""
        with h5py.File(filename, "r") as h5:
            h5_data = h5["MODEL/data"]
            # Get dimension indices of relevant parameters
            # These should be constant for all gradients, so this is only done
            # once.
            dim_labels = (
                h5_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
            )
            if not type(dim_labels) == str:
                dim_labels = dim_labels.decode()
            dim_labels = dim_labels.replace(" ", "").split("|")
            indices = []
            for param in self.parameters:
                indices.append(dim_labels.index(param))
        return indices

    def get_latest_time_step(self):
        """ Extracts the latest time step from the model dir."""
        tasks = glob.glob(os.path.join(self.task_dir, "task_*.toml"))
        # if no task exists, get the latest status from the model_dir
        if len(tasks) < 1:
            tasks = glob.glob(os.path.join(self.model_dir, "model_*.h5"))

        tasks.sort()
        if len(tasks) < 1:
            raise Exception("No models found,"
                            "please initialize first.")
        self.time_step = int(tasks[-1].split("/")
                            [-1].split("_")[-1].split(".")[0])
        return self.time_step

    def get_latest_task(self):
        self.get_latest_time_step()
        return self.get_task_path()

    def get_task_path(self, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.task_dir, f"task_{time_step:05d}.toml")

    def get_gradient_path(self, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.gradient_dir, f"gradient_{time_step:05d}.h5")

    def get_first_moment_path(self, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.first_moment_dir,
                            f"first_moment_{time_step:05d}.h5")

    def get_second_moment_path(self, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.second_moment_dir,
                            f"second_moment_{time_step:05d}.h5")

    def get_raw_update_path(self, time_step=None):
        """Get path to raw update"""
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.raw_update_dir,
                            f"raw_update_{time_step:05d}.h5")

    def get_smooth_path(self, time_step=None):
        """Get the path of the smoothed gradient"""
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.smooth_update_dir,
                            f"smooth_update_{time_step:05d}.h5")

    def get_model_path(self, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        return os.path.join(self.model_dir, f"model_{time_step:05d}.h5")

    def get_h5_data(self, filename):
        """
        Returns the relevant data in the form of ND_array with all the data.
        """
        indices = self._get_parameter_indices(filename)

        with h5py.File(filename, "r") as h5:
            data = h5["MODEL/data"][:, :, :].copy()
            return data[:, indices, :]

    def set_h5_data(self, filename, data):
        """Writes the data with shape [:, indices :]. Requires existing file."""
        if not os.path.exists(filename):
            raise Exception("only works on existing files.")

        indices = self._get_parameter_indices(filename)

        with h5py.File(filename, "r+") as h5:
            dat = h5["MODEL/data"]
            data_copy = dat.copy()
            # avoid writing the file many times. work on array in memory
            for i in range(len(indices)):
                data_copy[:, indices[i], :] = data[:, i, :]

            # writing only works in sorted order. This sort can only happen after
            # the above executed to preserve the ordering that data came in
            indices.sort()
            dat[:, indices, :] = data_copy[:, indices, :]

    def compute_raw_update(self):
        """Computes the raw update"""
        print("Adam: Computing raw update...")
        # Read task toml
        task_info = toml.load(self.get_latest_task())

        if not task_info["gradient_completed"]:
            raise Exception("Gradient must be computed first. Compute gradient"
                            "and set gradient_completed to True.")

        time_step = task_info["time_step"] + 1
        gradient_path = task_info["raw_gradient_path"]
        raw_update_path = task_info["raw_update_path"]

        indices = self._get_parameter_indices(gradient_path)
        # scale the gradients, because they can be tiny and this leads to issues
        g_t = self.get_h5_data(gradient_path) * self.grad_scaling_fac

        if time_step == 1:  # Initialize moments if needed
            first_moment_path = self.get_first_moment_path(time_step=0)
            second_moment_path = self.get_second_moment_path(time_step=0)
            shutil.copy(gradient_path, first_moment_path)

            with h5py.File(first_moment_path, "r+") as h5:
                data = h5["MODEL/data"]

                # initialize with zeros
                for i in indices:
                    data[:, i, :] = np.zeros_like(data[:, i, :])

            # Also initialize second moments with zeros
            shutil.copy(first_moment_path, second_moment_path)

        m_t = self.beta_1 * self.get_h5_data(
            self.get_first_moment_path(time_step=time_step - 1)) + \
              (1 - self.beta_1) * g_t

        # Store first moment
        shutil.copy(self.get_first_moment_path(time_step=time_step - 1),
                    self.get_first_moment_path(time_step=time_step))
        self.set_h5_data(self.get_first_moment_path(), m_t)

        # v_t was sometimes becoming too small, so enforce double precision
        v_t = self.beta_2 * self.get_h5_data(
            self.get_second_moment_path(time_step=time_step - 1)) + \
              (1 - self.beta_2) * (g_t ** 2)

        # Store second moment
        shutil.copy(self.get_second_moment_path(time_step=time_step - 1),
                    self.get_second_moment_path(time_step=time_step))
        self.set_h5_data(self.get_second_moment_path(time_step=time_step), v_t)

        # Correct bias
        m_t = m_t / (1 - self.beta_1 ** time_step)
        v_t = v_t / (1 - self.beta_2 ** time_step)

        # Update parameters
        theta_prev = self.get_h5_data(self.get_model_path(
            time_step=time_step - 1))

        # normalise theta_ with initial
        # theta_prev = theta_prev / theta_initial -1
        theta_0 = self.get_h5_data(self.get_model_path(
            time_step=0))
        theta_prev = theta_prev / theta_0 - 1

        # ensure e is sufficiently small, even for the small gradient values
        # that we typically have.
        e = self.e * np.mean(np.sqrt(v_t))

        update = self.alpha * m_t / (
                    np.sqrt(v_t) + e) - self.weight_decay * theta_prev

        max_upd = np.max(np.abs(update))
        if max_upd > self.alpha * 1.05:
            raise Exception("Raw update seems to large")
        if np.sum(np.isnan(update)) > 1:
            raise Exception("NaNs were found.")

        # Write raw update to file for smoothing
        shutil.copy(gradient_path, raw_update_path)
        self.set_h5_data(raw_update_path,
                         update)

    def apply_smooth_update(self):
        """Apply the smoothed update
        # TODO: smmoothing might significantly reduce the step size,
        # so check this and maybe rescale the smoothed update back to the step size
        # this could be done based on the peak amplitudes in the raw and smoothed update
        # such that we don't scale too much.

        """
        print("Adam: Applying smooth update...")
        task_info = toml.load(self.get_latest_task())

        if not task_info["smoothing_completed"]:
            raise Exception("Smooth update first and set smoothing_completed to"
                            "True.")

        time_step = task_info["time_step"] + 1
        smooth_path = task_info["smooth_update_path"]
        update = self.get_h5_data(smooth_path)

        print("Maximum update step:", np.max(update))
        if np.max(np.abs(update)) > 1.05 * self.alpha:
            raise Exception("check smooth gradient, something seems off")

        # Update parameters
        theta_prev = self.get_h5_data(self.get_model_path(
            time_step=time_step - 1))

        # normalise theta and apply update
        theta_0 = self.get_h5_data(self.get_model_path(
            time_step=0))
        theta_prev = theta_prev / theta_0 - 1
        theta_new = theta_prev - update

        # remove normalization from updated model and write physical model
        theta_physical = (theta_new + 1) * theta_0
        shutil.copy(self.get_model_path(time_step=time_step - 1),
                    self.get_model_path(time_step=time_step))
        self.set_h5_data(self.get_model_path(time_step=time_step),
                         theta_physical)

        # Iteration still has to be finalized. This is done separately
        # for inversionson purposes, such that files can be cleaned.

    def finalize_iteration(self):
        """
        Finalize iteration, and write new task
        """
        task_info = toml.load(self.get_latest_task())
        task_info["iteration_finalized"] = True

        with open(self.get_task_path(time_step=self.time_step), "w") as fh:
            toml.dump(task_info, fh)

        self.time_step = task_info["time_step"] + 1
        print("writing task with timestep", self.time_step)
        self.read_and_write_task(time_step=self.time_step)

    def get_iteration_name(self):
        """ Get iteration name"""
        task_info = toml.load(self.get_latest_task())
        return task_info["model"].split("/")[-1].split(".")[0]

    def get_previous_iteration(self):
        time_step = self.get_latest_time_step()
        task_info = toml.load(self.get_task_path(time_step=time_step - 1))
        return task_info["model"].split("/")[-1].split(".")[0]

    def get_inversionson_task(self):
        """Gets the task for inversionson"""
        if not os.path.exists(self.get_latest_task()):
            raise Exception("Please ensure that the config file is filled and"
                            "the optimizer is reinitialized.")
        task_info = toml.load(self.get_latest_task())
        return task_info["task"]

    def set_gradient_task_to_finished(self):
        """ Set the gradient task to completed in latest task toml."""
        task_info = toml.load(self.get_latest_task())
        task_info["gradient_completed"] = True

        with open(self.get_latest_task(), "w") as fh:
            toml.dump(task_info, fh)

    def set_smoothing_task_to_finished(self):
        """ Set the smoothing task to completed in latest task toml."""
        task_info = toml.load(self.get_latest_task())
        task_info["smoothing_completed"] = True

        with open(self.get_latest_task(), "w") as fh:
            toml.dump(task_info, fh)

    def set_misfit(self, misfit):
        """ Set the misfit in latest toml. This is purely optional,
        because it is not used by ADAM itself."""
        task_info = toml.load(self.get_latest_task())
        task_info["misfit"] = float(misfit)

        with open(self.get_latest_task(), "w") as fh:
            toml.dump(task_info, fh)

    def read_and_write_task(self, time_step=None):
        """
        Checks task status and writes new task if task is already completed.
        """
        if time_step is None:
            task_path = self.get_latest_task()
        else:
            task_path = self.get_task_path(time_step)

        # If task exists, read it and see if model needs updating
        if os.path.exists(task_path):
            task_info = toml.load(task_path)
            if not task_info["iteration_finalized"]:
                print("Please complete task first")
        else:  # write task
            task_dict = {"task": "compute_gradient_for_adam", "misfit": "",
                         "model": self.get_model_path(),
                         "raw_gradient_path": self.get_gradient_path(),
                         "gradient_completed": False,
                         "raw_update_path": self.get_raw_update_path(),
                         "smooth_update_path": self.get_smooth_path(),
                         "smoothing_completed": False,
                         "iteration_finalized": False,
                         "time_step": self.time_step}

            with open(task_path, "w+") as fh:
                toml.dump(task_dict, fh)
