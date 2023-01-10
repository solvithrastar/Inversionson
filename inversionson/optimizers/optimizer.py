"""
Base optimization class. It covers all the basic things that most optimizers
have in common. The class serves the purpose of making it easy to add
a custom optimizer to Inversionson. Whenever the custom optimizer has a 
task which works the same as in the base class. It should aim to use that one.
"""
import sys
from abc import abstractmethod as _abstractmethod
from pathlib import Path
import os
import glob
import h5py
import pathlib
import toml
import numpy as np
from typing import List, Union

from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset
from salvus.flow.api import get_site
from inversionson import InversionsonError
from inversionson.utils import write_xdmf
import shutil


class Optimize(object):

    # Derived classes should add to this
    available_tasks = [
        "prepare_iteration",
        "run_forward",
        "compute_misfit",
        "compute_validation_misfit",
        "compute_gradient",
        "regularization",
        "update_model",
        "documentation",
    ]

    # Derived classes should override this
    optimizer_name = "BaseClass for optimizers. Don't instantiate. If you see this..."
    config_template_path = None

    def __init__(self, comm):

        # This init is only called by derived classes

        self.current_task = self.read_current_task()

        self.comm = comm
        self.opt_folder = (
            Path(self.comm.project.paths["inversion_root"]) / "OPTIMIZATION"
        )

        self.parameters = self.comm.project.inversion_params
        if not os.path.exists(self.opt_folder):
            os.mkdir(self.opt_folder)

        # These folders are universally needed
        self.model_dir = self.opt_folder / "MODELS"
        self.task_dir = self.opt_folder / "TASKS"
        self.average_model_dir = self.opt_folder / "AVERAGE_MODELS"
        self.raw_gradient_dir = self.opt_folder / "RAW_GRADIENTS"
        self.raw_update_dir = self.opt_folder / "RAW_UPDATES"
        self.regularization_dir = self.opt_folder / "REGULARIZATION"
        self.gradient_norm_dir = self.opt_folder / "GRADIENT_NORMS"

        # Do any folder initilization for the derived classes
        self._initialize_derived_class_folders()
        self.layer_mask = None

        self.config_file = self.opt_folder / "opt_config.toml"

        if not os.path.exists(self.config_file):
            self._write_initial_config()
            print(
                f"Please set config and provide initial model to "
                f"{self.optimizer_name} optimizer in {self.config_file} \n"
                f"Then reinitialize the {self.optimizer_name} optimizer."
            )
            sys.exit()
        self._read_config()

        if self.initial_model == "":
            print(
                f"Please set config and provide initial model to "
                f"{self.optimizer_name} optimizer in {self.config_file} \n"
                f"Then reinitialize the {self.optimizer_name} optimizer."
            )
            sys.exit()

        # Initialize folders if needed
        if not os.path.exists(self._get_path_for_iteration(0, self.model_path)):
            if self.initial_model is None:
                raise InversionsonError(
                    f"{self.optimizer_name} needs to be initialized with a "
                    "path to an initial model."
                )
            print(f"Initializing {self.optimizer_name}...")
            self._init_directories()
            self._issue_first_task()
        self.tmp_model_path = self.opt_folder / "tmp_model.h5"
        self._read_task_file()

        # Once this exits, continue with the derived class __init__().

    def print(
        self,
        message: str,
        color: str = "magenta",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = ":chart_with_downwards_trend:",
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    @staticmethod
    def _get_path_for_iteration(iteration_number, path):
        pass

    def _model_for_iteration(self, iteration_number):
        return self.model_dir / f"model_{iteration_number:05d}.h5"

    @_abstractmethod
    def _initialize_derived_class_folders(self):
        """You need to make this yourself. Can do nothing, if no extra folders are
        required"""
        pass

    @_abstractmethod
    def _init_directories(self):
        pass

    @_abstractmethod
    def _issue_first_task(self):
        pass

    @_abstractmethod
    def _read_task_file(self):
        pass

    def _write_initial_config(self):
        """
        Writes the initial config file.
        """
        shutil.copy(self.config_template_path, self.config_file)

        print(
            f"Wrote a config file for the {self.optimizer_name} optimizer. "
            f"Please provide an initial model."
        )

    def _read_config(self):
        """Reads the config file."""

        if not os.path.exists(self.config_file):
            raise Exception("Can't read the ADAM config file")
        config = toml.load(self.config_file)
        self.initial_model = config["initial_model"]
        self.step_length = config["step_length"]
        if "max_iterations" in config.keys():
            self.max_iterations = config["max_iterations"]
        else:
            self.max_iterations = None
        self.parameters = config["parameters"]

    def read_current_task(self):
        """
        Read the current task from file
        """
        pass

    @property
    def iteration_number(self):
        "Returns the number of the newest iteration"
        return max(self.find_iteration_numbers())

    @property
    def iteration_name(self):
        return f"model_{self.iteration_number:05d}"

    @property
    def model_path(self):
        return self.model_dir / f"model_{self.iteration_number:05d}.h5"

    @property
    def regularization_job_toml(self):
        return (
            self.regularization_dir / f"regularization_{self.iteration_number:05}.toml"
        )

    def find_iteration_numbers(self):
        models = glob.glob(f"{self.model_dir}/*.h5")
        print(models)
        if len(models) == 0:
            return [0]
        iteration_numbers = []
        for model in models:
            print(model)
            iteration_numbers.append(int(model.split(".")[0].split("_")[-1]))
        return iteration_numbers

    def _pick_data_for_iteration(self):
        all_events = self.comm.lasif.list_events()
        blocked_data = list(
            set(self.comm.project.validation_dataset + self.comm.project.test_dataset)
        )
        all_events = list(set(all_events) - set(blocked_data))
        n_events = self.comm.project.batch_size
        all_norms_path = self.gradient_norm_dir / "all_norms.toml"
        if os.path.exists(all_norms_path):
            norm_dict = toml.load(all_norms_path)
            unused_events = list(set(all_events).difference(set(norm_dict.keys())))
            list_of_vals = np.array(list(norm_dict.values()))
            # Set unused to 65%, so slightly above average
            max_norm = np.percentile(list_of_vals, 65.0)

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

        if self.time_for_validation():
            print(
                "This is a validation iteration. Will add validation events to the iteration"
            )
            events += self.comm.project.validation_dataset

        return events

    def delete_remote_files(self, iteration=None):
        if not iteration:
            iteration = self.iteration_name
        print("Deleting old iteration files...")
        self.comm.salvus_flow.delete_stored_wavefields(iteration, "forward")
        self.comm.salvus_flow.delete_stored_wavefields(iteration, "adjoint")

        if self.comm.project.prepare_forward:
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration, "prepare_forward"
            )
        if self.comm.project.meshes == "multi-mesh":
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration, "gradient_interp"
            )
        if self.comm.project.hpc_processing:
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration, "hpc_processing"
            )

    def prepare_iteration(
        self,
        it_name: str,
        events: List[str] = None,
    ):
        """
        Prepare iteration.

        :param it_name: Name of iteration
        :type it_name: "str", optional
        :param events: Pass a list of events if you want them to be predefined, defaults to None
        :type events: List[str], optional
        """
        self.comm.project.change_attribute("current_iteration", it_name)
        print("Preparing iteration for", it_name)
        if self.comm.lasif.has_iteration(it_name):
            raise InversionsonError(f"Iteration {it_name} already exists")

        self.comm.lasif.set_up_iteration(it_name, events)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes(it_name)

        optimizer = self.comm.project.get_optimizer()
        model = optimizer.model_path

        # WIP no average models being uploaded yet.
        remote_mesh_file = (
            self.comm.project.remote_inversionson_dir / "MODELS" / it_name / "mesh.h5"
        )
        hpc_cluster = get_site(self.comm.project.site_name)
        if not hpc_cluster.remote_exists(remote_mesh_file.parent):
            if not hpc_cluster.remote_exists(self.comm.project.remote_mesh_dir):
                hpc_cluster.remote_mkdir(self.comm.project.remote_mesh_dir)
            if not hpc_cluster.remote_exists(self.comm.project.remote_mesh_dir / "MODELS"):
                hpc_cluster.remote_mkdir(self.comm.project.remote_mesh_dir / "MODELS")
            hpc_cluster.remote_mkdir(remote_mesh_file.parent)
        self.print(
            f"Moving mesh to {self.comm.project.interpolation_site}",
            emoji_alias=":package:",
        )
        hpc_cluster.remote_put(model, remote_mesh_file)

        if self.time_for_validation() and self.comm.project.use_model_averaging\
                and self.iteration_number > 0:
            remote_avg_mesh_file = (
                    self.comm.project.remote_mesh_dir / "AVERAGE_MODELS" / it_name / "mesh.h5"
            )
            # this enters when the iteration number is 4
            print("writing average validation model")
            # 4 - 5 + 1 = 0
            starting_it_number = self.iteration_number - self.comm.project.val_it_interval + 1
            self.write_average_model(starting_it_number,
                                     self.iteration_number)
            self.print(
                f"Moving average_model to {self.comm.project.interpolation_site}",
                emoji_alias=":package:",
            )
            if not hpc_cluster.remote_exists(remote_avg_mesh_file.parent):
                hpc_cluster.remote_mkdir(remote_avg_mesh_file.parent)
            hpc_cluster.remote_put(
                self.get_average_model_name(starting_it_number, self.iteration_number),
                remote_avg_mesh_file
            )

        self.comm.lasif.upload_stf(iteration=it_name)

    def run_forward(self, verbose: bool = False):
        """
        Dispatch the forward simulations for all events

        :param verbose: You want to know the details?, defaults to False
        :type verbose: bool, optional
        """
        pass

    def select_new_windows(self):
        """
        Some logic that decides if new windows need to be selected or not.

        NOT FINISHED (OBVIOUSLY)
        """
        return True

    def get_remote_model_path(self, iteration=None, model_average=False):
        """ Gets the path to storage location of the remote
        model path. """
        if iteration is None:
            iteration = self.comm.project.current_iteration

        if model_average:
            remote_mesh_dir = pathlib.Path(self.comm.project.remote_mesh_dir)
            return remote_mesh_dir / "AVERAGE_MODELS" / iteration / "mesh.h5"
        else:
            remote_mesh_dir = pathlib.Path(self.comm.project.remote_inversionson_dir)
            return remote_mesh_dir / "MODELS" / iteration / "mesh.h5"

    def time_for_validation(self) -> bool:
        validation = False
        if self.comm.project.val_it_interval == 0:
            return False
        if self.iteration_number == 0:
            validation = True
        if (self.iteration_number + 1) % self.comm.project.val_it_interval == 0:
            validation = True

        return validation

    def compute_misfit(
        self,
        adjoint: bool = True,
        window_selection: bool = None,
        verbose: bool = False,
    ):
        """
        Retrieve and process the results of the forward simulations. Compute the misfits
        between synthetics and data.

        :param adjoint: Directly submit adjoint simulation, defaults to True
        :type adjoint: bool, optional
        :param window_selection: If windows should definitely be selected, if
            this is not clear, leave it at None. defaults to None
        :type window_selection: bool, optional
        :param verbose: You want to know the details?, defaults to False
        :type verbose: bool, optional
        """
        pass

    def compute_gradient(self, verbose=False):
        """
        Submit adjoint simulations to compute gradients.

        :param verbose: Do we want the details?, defaults to False
        :type verbose: bool, optional
        """
        pass

    def regularization(self):
        """
        To be implemented
        """
        pass

    def update_model(self):
        """
        Not yet implemented for the standard optimization.
        """
        pass

    def get_average_model_name(self, first_iteration_number=None,
                               last_iteration_number=None):
        """
        Gets the filename of the average model.
        """
        if first_iteration_number is None:
            first_iteration_number = self.iteration_number - \
                                     self.comm.project.val_it_interval + 1
        if last_iteration_number is None:
            self.iteration_number
        filename = f"average_model_{first_iteration_number}_to_{last_iteration_number}.h5"
        return os.path.join(self.average_model_dir, filename)

    def write_average_model(self, first_iteration_number, last_iteration_number):
        """
        Writes an average over the models from the first to last iteration
        number. Needs a minimum of 2 iterations to work and make sense.
        """
        import shutil
        total_num_models = 1
        average_model = self.get_h5_data(self._model_for_iteration(first_iteration_number))

        for i in range(first_iteration_number+1, last_iteration_number+1):
            average_model += self.get_h5_data(self._model_for_iteration(i))
            total_num_models += 1
        average_model /= total_num_models
        avg_model_name = self.get_average_model_name(first_iteration_number,
                                                last_iteration_number)
        shutil.copy(self._model_for_iteration(first_iteration_number),
                    avg_model_name)
        self.set_h5_data(filename=avg_model_name, data=average_model)

    def get_parameter_indices(self, filename, parameters=None):
        if not parameters:
            parameters = self.parameters
        """Get parameter indices in h5 file"""
        with h5py.File(filename, "r") as h5:
            h5_data = h5["MODEL/data"]
            # Get dimension indices of relevant parameters
            # These should be constant for all gradients, so this is only done
            # once.
            dim_labels = h5_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
            if not type(dim_labels) == str:
                dim_labels = dim_labels.decode()
            dim_labels = dim_labels.replace(" ", "").split("|")
            indices = []
            for param in parameters:
                indices.append(dim_labels.index(param))
        return indices

    def get_h5_data(self, filename, parameters=None):
        """
        Returns the relevant data in the form of ND_array with all the data.
        """
        if not parameters:
            parameters = self.parameters
        indices = np.array(self.get_parameter_indices(filename, parameters))
        argsort_idcs = np.argsort(indices)
        sorted_indices = indices[argsort_idcs]
        idx_in_original = np.arange(len(indices))[argsort_idcs]

        # get_layer_mask
        layer_idx = self.get_elemental_parameter_indices(filename, ["layer"])

        with h5py.File(filename, "r") as h5:
            if self.layer_mask is None:
                layer = h5["MODEL/element_data"][:, layer_idx]
                self.layer_mask = np.where(layer < 1.1, False, True).squeeze()
            data = h5["MODEL/data"][:, sorted_indices, :][self.layer_mask]
            return data[:, idx_in_original, :]
            data = h5["MODEL/data"][:, :, :][self.layer_mask].copy()
            return data[:, indices, :]

    def get_points(self, filename):
        """
        Returns the relevant data in the form of ND_array with all the data.
        """
        layer_idx = self.get_elemental_parameter_indices(filename, ["layer"])

        with h5py.File(filename, "r") as h5:
            layer = h5["MODEL/element_data"][:, layer_idx]
            layer_mask = np.where(layer < 1.1, False, True).squeeze()
            points =  h5["MODEL/coordinates"][:,:,:][layer_mask]
            return points

    def get_flat_non_duplicated_data(self, parameters:list, filename:str,
                                     pt_idcs:np.array):
        flat_pars = []
        all_data = self.get_h5_data(filename, parameters=parameters)
        for idx, param in enumerate(parameters):
            # first flatten and then select unique values...
            flat_pars.append(all_data[:, idx, :].flatten()[pt_idcs])
        return flat_pars

    def set_h5_data(self, filename, data, create_xdmf=True, parameters=None):
        """Writes the data with shape [:, indices :]. Requires existing file."""
        if not os.path.exists(filename):
            raise Exception("only works on existing files.")

        if not parameters:
            parameters=self.parameters
        indices = self.get_parameter_indices(filename, parameters)

        # get_layer_mask
        layer_idx = self.get_elemental_parameter_indices(filename, ["layer"])

        with h5py.File(filename, "r+") as h5:
            layer = h5["MODEL/element_data"][:, layer_idx]
            layer_mask = np.where(layer < 1.1, False, True).squeeze()

            dat = h5["MODEL/data"]
            data_copy = dat[:, :, :].copy()

            # avoid writing the file many times. work on array in memory
            for i in range(len(indices)):
                data_copy[:, indices[i], :][layer_mask] = data[:, i, :]

            # writing only works in sorted order. This sort can only happen after
            # the above executed to preserve the ordering that data came in
            indices.sort()
            dat[:, indices, :] = data_copy[:, indices, :]

        if create_xdmf:
            print("writing XDMF")
            write_xdmf(filename)

    def get_elemental_parameter_indices(self, filename, parameters):
        """Get parameter indices in h5 file"""
        with h5py.File(filename, "r") as h5:
            h5_data = h5["MODEL/element_data"]
            dim_labels = h5_data.attrs.get("DIMENSION_LABELS")[1][1:-1]
            if not type(dim_labels) == str:
                dim_labels = dim_labels.decode()
            dim_labels = dim_labels.replace(" ", "").split("|")
            indices = []
            for param in parameters:
                indices.append(dim_labels.index(param))
        return indices

    def get_tensor_order(self, filename):
        """
        Get the tensor order from a Salvus file.
        :param filename: filename
        :type filename: str
        """
        with h5py.File(filename, "r") as h5:
            num_gll = h5["MODEL"]["coordinates"].shape[1]
            dimension = h5["MODEL"]["coordinates"].shape[2]
        return round(num_gll ** (1 / dimension) - 1)
