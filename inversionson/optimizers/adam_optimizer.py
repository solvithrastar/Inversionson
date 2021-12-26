"""
A class that performs ADAM optimization, also ensures
that we will be able to transition
"""

from lasif.components.component import Component

import os
import toml
import numpy as np
import glob
import shutil
import h5py  # Use h5py to avoid lots of dependencies and slow reading

BOOL_ADAM = True

# TODO add AdamW to reglarize weights
# TODO: To ensure smoothness, we may also just smooth the total update
# Essentially this means that in order to ramp the solution to our prior (let's say Prem)
# we need to
class AdamOptimizer:
    def __init__(self, opt_folder, initial_model=None):
        """
        initial_model: path to initial model, only required upon init
        init, also set init to True in that case.
        """

        self.initial_model = initial_model
        self.alpha = 0.001  # appstep size when parameters are normalized
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        # Regularization parameter to avoid dividing by zero
        self.e = 10e-8
        self.parameters = ["VSV", "VSH", "VPV", "VPH"]

        self.optimization_folder = opt_folder

        self.model_dir = os.path.join(self.optimization_folder, "MODELS")
        self.gradient_dir = os.path.join(self.optimization_folder, "GRADIENTS")
        self.first_moment_dir = os.path.join(self.optimization_folder,
                                             "FIRST_MOMENTS")
        self.second_moment_dir = os.path.join(self.optimization_folder,
                                              "SECOND_MOMENTS")
        self.task_dir = os.path.join(self.optimization_folder, "TASKS")

        # Initialize folders if needed
        if not os.path.exists(self.get_model_path(timestep=0)):
            if self.initial_model is None:
                raise Exception(
                    "AdamOptimizer needs to be initialized with a "
                    "path to an initial model.")
            self.timestep = 0

            task_file = os.path.join(self.task_dir, self.get_task_path())
            if os.path.exists(task_file):
                raise Exception("Init requires an empty TASKS folder.")
            self._init_directories()
            self.read_and_write_task()
        else:  # Figure out timestep and read task
            self.get_latest_timestep()

    def get_latest_timestep(self):
        models = glob.glob(os.path.join(self.task_dir, "task_*.toml"))
        models.sort()
        if len(models) < 1:
            raise Exception("No models found,"
                            "please initialize first.")
        self.timestep = int(models[-1].split("/")
                            [-1].split("_")[-1].split(".")[0])
        return self.timestep

    def get_latest_task(self):
        self.get_latest_timestep()
        return self.get_task_path()

    def _init_directories(self):
        """
        Build directory structure.
        """
        folders = [self.model_dir, self.gradient_dir, self.first_moment_dir,
                   self.second_moment_dir,
                   self.task_dir]

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

        shutil.copy(self.initial_model, self.get_model_path())

    def get_task_path(self, timestep=None):
        if timestep is None:
            timestep = self.timestep
        return os.path.join(self.task_dir, f"task_{timestep:05d}.toml")

    def get_gradient_path(self, timestep=None):
        if timestep is None:
            timestep = self.timestep
        return os.path.join(self.gradient_dir, f"gradient_{timestep:05d}.h5")

    def get_first_moment_path(self, timestep=None):
        if timestep is None:
            timestep = self.timestep
        return os.path.join(self.first_moment_dir,
                            f"first_moment_{timestep:05d}.h5")

    def get_second_moment_path(self, timestep=None):
        if timestep is None:
            timestep = self.timestep
        return os.path.join(self.second_moment_dir,
                            f"second_moment_{timestep:05d}.h5")

    def get_model_path(self, timestep=None):
        if timestep is None:
            timestep = self.timestep
        return os.path.join(self.model_dir, f"model_{timestep:05d}.h5")

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
            for i in range(len(indices)):
                dat[:, indices[i], :] = data[:, i, :]

    def _check_task_completion(self):
        task_path = self.get_latest_task()
        if os.path.exists(task_path):
            current_task = toml.load(task_path)
            if current_task["completed"]:
                return True
            else:
                return False

    def compute_update(self):
        if not self._check_task_completion():
            raise Exception("Task must be completed first")
        self.timestep += 1
        gradient_path = self.get_gradient_path(timestep=self.timestep - 1)

        indices = self._get_parameter_indices(gradient_path)
        g_t = self.get_h5_data(gradient_path)

        if self.timestep == 1:  # Initialize moments if needed
            first_moment_path = self.get_first_moment_path(timestep=0)
            second_moment_path = self.get_second_moment_path(timestep=0)
            shutil.copy(self.get_gradient_path(timestep=0), first_moment_path)

            with h5py.File(first_moment_path, "r+") as h5:
                data = h5["MODEL/data"]

                # initialize with zeros
                for i in indices:
                    data[:, i, :] = np.zeros_like(data[:, i, :])

            # Also initialize second moments with zeros
            shutil.copy(first_moment_path, second_moment_path)

        m_t = self.beta_1 * self.get_h5_data(
            self.get_first_moment_path(timestep=self.timestep - 1)) + \
              (1 - self.beta_1) * g_t

        # Store first moment
        shutil.copy(self.get_first_moment_path(timestep=self.timestep - 1),
                    self.get_first_moment_path())
        self.set_h5_data(self.get_first_moment_path(), m_t)

        v_t = self.beta_2 * self.get_h5_data(
            self.get_second_moment_path(timestep=self.timestep - 1)) + \
              (1 - self.beta_2) * (g_t ** 2)


        # Store second moment
        shutil.copy(self.get_second_moment_path(timestep=self.timestep - 1),
                    self.get_second_moment_path())
        self.set_h5_data(self.get_second_moment_path(), v_t)

        # Correct bias
        m_t = m_t / (1 - self.beta_1 ** self.timestep)
        v_t = v_t / (1 - self.beta_2 ** self.timestep)

        # Update parameters
        # TODO provide option to smooth update with smoother
        theta_prev = self.get_h5_data(self.get_model_path(
            timestep=self.timestep-1))

        # normalise theta_ with initial
        # theta_prev = theta_prev / theta_initial -1
        theta_0 = self.get_h5_data(self.get_model_path(
            timestep=0))
        theta_prev = theta_prev / theta_0 - 1

        # ensure e is sufficiently small, even for tiny gradient values
        e = self.e * np.mean(v_t)
        weight_decay = 0.001 #AdamW, prefer small weights
        theta_new = theta_prev - self.alpha * m_t / (np.sqrt(v_t) + e) - weight_decay * theta_prev

        # remove normalization
        theta_physical = (theta_new + 1) * theta_0
        shutil.copy(self.get_model_path(timestep=self.timestep - 1),
                    self.get_model_path())
        self.set_h5_data(self.get_model_path(), theta_physical)

        # Write next task.
        self.read_and_write_task()

    def get_iteration_name(self):
        """ Get iteration name"""
        task_info = toml.load(self.get_latest_task())
        return task_info["model"].split("/")[-1].split(".")[0]

    def get_previous_iteration(self):
        timestep = self.get_latest_timestep()
        task_info = toml.load(self.get_task_path(timestep=timestep - 1))
        return task_info["model"].split("/")[-1].split(".")[0]

    def get_inversionson_task(self):
        """Gets the task for inversionson"""
        task_info = toml.load(self.get_latest_task())
        return task_info["task"]

    def set_task_to_finished(self):
        """ Set the misfit in latest toml"""
        task_info = toml.load(self.get_latest_task())
        task_info["completed"] = True

        with open(self.get_latest_task(), "w") as fh:
            toml.dump(task_info, fh)

    def set_misfit(self, misfit):
        """ Set the misfit in latest toml"""
        task_info = toml.load(self.get_latest_task())
        task_info["misfit"] = float(misfit)

        with open(self.get_latest_task(), "w") as fh:
            toml.dump(task_info, fh)

    def read_and_write_task(self):
        """
        Checks task status and writes new task if task is already completed.
        """

        # Don't do this if task is completed already
        # it needs to have the latest timestep here:

        task_path = self.get_task_path()

        # If task exists, read it and see if model needs updating
        if os.path.exists(task_path):
            if not self._check_task_completion():
                print("Please complete task first")
        else: # write task
            task_dict = {"task": "compute_misfit_and_gradient", "misfit": "",
                         "model": self.get_model_path(),
                         "gradient": self.get_gradient_path(), "completed": False,
                         "timestep": self.timestep}

            with open(task_path, "w+") as fh:
                toml.dump(task_dict, fh)

