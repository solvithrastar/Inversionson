"""
Base optimization class. It covers all the basic things that most optimizers
have in common. The class serves the purpose of making it easy to add
a custom optimizer to Inversionson. Whenever the custom optimizer has a 
task which works the same as in the base class. It should aim to use that one.
"""
from pathlib import Path
import os
import glob
import h5py
import toml
from typing import List, Tuple
from salvus.flow.api import get_site
from inversionson import InversionsonError, autoinverter_helpers as helpers


class Optimize(object):
    def __init__(self, comm):
        self.current_task = self.read_current_task()
        self.available_tasks = [
            "prepare_iteration",
            "run_forward",
            "compute_misfit",
            "compute_validation_misfit",
            "compute_gradient",
            "regularization",
            "update_model",
            "documentation",
        ]
        self.comm = comm
        self.opt_folder = (
            Path(self.comm.project.paths["inversion_root"]) / "OPTIMIZATION"
        )
        if not os.path.exists(self.opt_folder):
            os.mkdir(self.opt_folder)
        self.model_dir = self.opt_folder / "MODELS"
        self.task_dir = self.opt_folder / "TASKS"
        self.config_file = self.opt_folder / "opt_config.toml"
        if not os.path.exists(self.config_file):
            self._write_initial_config()
            print(
                f"Please set config and provide initial model to "
                f"Base optimizer in {self.config_file} \n"
                f"Then reinitialize the inversion."
            )
            return
        self._read_config()
        if self.initial_model == "":
            print(
                f"Please set config and provide initial model to "
                f"Base optimizer in {self.config_file} \n"
                f"Then reinitialize the inversion."
            )
            return

    def _write_initial_config(self):
        """
        Writes the initial config file.
        """
        config = {
            "step_length": 0.001,
            "parameters": ["VSV", "VSH", "VPV", "VPH"],
            "initial_model": "",
            "max_iterations": 1000,
        }
        with open(self.config_file, "w") as fh:
            toml.dump(config, fh)

        print(
            "Wrote a config file for the Base optimizer. Please provide "
            "an initial model."
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

    def find_iteration_numbers(self):
        models = glob.glob(f"{self.model_dir}/*.h5")
        if len(models) == 0:
            return [0]
        iteration_numbers = []
        for model in models:
            iteration_numbers.append(int(model.split(".")[0].split("_")[-1]))
        return iteration_numbers

    def delete_remote_files(self):
        self.comm.salvus_flow.delete_stored_wavefields(self.iteration_name, "forward")
        self.comm.salvus_flow.delete_stored_wavefields(self.iteration_name, "adjoint")
        self.comm.salvus_flow.delete_stored_wavefields(
            self.iteration_name, "model_interp"
        )
        self.comm.salvus_flow.delete_stored_wavefields(
            self.iteration_name, "gradient_interp"
        )

    def prepare_iteration(
        self,
        it_name: str = None,
        move_meshes: bool = False,
        first_try: bool = True,
        events: List[str] = None,
    ):
        """
        A base function for preparing iterations

        :param it_name: Name of iteration, will use autoname if None is passed, defaults to None
        :type it_name: "str", optional
        :param move_meshes: Do meshes need to be moved to remote, defaults to False
        :type move_meshes: bool, optional
        :param first_try: Only change in trust region methods if region is being reduced, defaults to True
        :type first_try: bool, optional
        :param events: Pass a list of events if you want them to be predefined, defaults to None
        :type events: List[str], optional
        """
        it_name = self.iteration_name if it_name is None else it_name
        self.comm.project.change_attribute("current_iteration", it_name)
        it_toml = os.path.join(
            self.comm.project.paths["iteration_tomls"], it_name + ".toml"
        )
        validation = "validation" in it_name

        if self.comm.lasif.has_iteration(it_name):
            raise InversionsonError(f"Iteration {it_name} already exists")

        if events is None and not validation:
            if self.comm.project.inversion_mode == "mini-batch":
                print("Getting minibatch")
                if it_name == "model_00000":
                    first = True
                else:
                    first = False
                events = self.comm.lasif.get_minibatch(first=first)
            else:
                events = self.comm.lasif.list_events()
        elif events is None and validation:
            events = self.comm.project.validation_dataset
        self.comm.lasif.set_up_iteration(it_name, events)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes(validation)

        if self.comm.project.meshes == "multi-mesh" and move_meshes:
            if self.comm.project.interpolation_mode == "remote":
                interp_site = get_site(self.comm.project.interpolation_site)
            else:
                interp_site = None
            self.comm.multi_mesh.add_fields_for_interpolation_to_mesh()
            self.comm.lasif.move_mesh(
                event=None, iteration=it_name, hpc_cluster=interp_site
            )
            # for event in events:
            #     if not self.comm.lasif.has_mesh(event, hpc_cluster=interp_site):
            #         self.comm.salvus_mesher.create_mesh(event=event)
            #         self.comm.lasif.move_mesh(event, it_name, hpc_cluster=interp_site)
            #     else:
            #         self.comm.lasif.move_mesh(event, it_name, hpc_cluster=interp_site)
        elif self.comm.project.meshes == "mono-mesh" and move_meshes:
            self.comm.lasif.move_mesh(event=None, iteration=it_name)

        self.comm.lasif.upload_stf(iteration=it_name)
        # Control group complications (update_control_group) should be done inside specific optimizer.

    def run_forward(self, verbose: bool = False):
        """
        Dispatch the forward simulations for all events

        :param verbose: You want to know the details?, defaults to False
        :type verbose: bool, optional
        """
        self.forward_helper = helpers.ForwardHelper(
            comm=self.comm, events=self.comm.project.events_in_iteration
        )
        self.forward_helper.dispatch_forward_simulations(verbose=verbose)
        assert self.forward_helper.assert_all_simulations_dispatched()

    def select_new_windows(self):
        """
        Some logic that decides if new windows need to be selected or not.

        NOT FINISHED (OBVIOUSLY)
        """
        return True

    def compute_validation_misfit(self, verbose: bool = False):
        """
        Compute misfits for validation dataset
        """
        if verbose:
            print("Creating average mesh for validation")
        if self.iteration_number != 0:
            to_it = self.iteration_number
            from_it = self.iteration_number - self.comm.project.when_to_validate + 1
            self.comm.salvus_mesher.get_average_model(iteration_range=(from_it, to_it))
            self.comm.multi_mesh.add_fields_for_interpolation_to_mesh()
            if self.comm.project.interapolation_mode == "remote":
                self.comm.lasif.move_mesh(
                    event=None,
                    iteration=None,
                    validation=True,
                )
        val_forward_helper = helpers.ForwardHelper(
            self.comm, self.comm.project.validation_dataset
        )
        assert "validation_" in self.comm.project.current_iteration
        val_forward_helper.dispatch_forward_simulations(verbose=verbose)
        assert val_forward_helper.assert_all_simulations_dispatched()
        val_forward_helper.retrieve_forward_simulations(
            adjoint=False, verbose=verbose, validation=True
        )
        val_forward_helper.report_total_validation_misfit()

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
        if window_selection is None:
            window_selection = self.select_new_windows()
        self.forward_helper = helpers.ForwardHelper(
            comm=self.comm, events=self.comm.project.events_in_iteration
        )
        self.forward_helper.retrieve_forward_simulations(
            adjoint=adjoint, windows=window_selection, verbose=verbose
        )
        assert self.forward_helper.assert_all_simulations_retrieved()

    def compute_gradient(self, verbose=False):
        """
        Submit adjoint simulations to compute gradients.

        :param verbose: Do we want the details?, defaults to False
        :type verbose: bool, optional
        """

        self.adjoint_helper = helpers.AdjointHelper(
            comm=self.comm, events=self.comm.project.events_in_iteration
        )
        self.adjoint_helper.dispatch_adjoint_simulations(verbose=verbose)
        assert self.adjoint_helper.assert_all_simulations_dispatched()

    def regularization(
        self,
        smooth_individual: bool = False,
        sum_gradients: bool = True,
        smooth_update: bool = False,
        verbose=False,
    ):
        """
        This smooths, sums and interpolates gradients based on what is needed.

        :param smooth_individual: Smooth individual gradients?, defaults to False
        :type smooth_individual: bool, optional
        :param sum_gradients: Sum before smoothing?, defaults to True
        :type sum_gradients: bool, optional
        :param smooth_update: Smooth update rather than gradient?, defaults to False
        :type smooth_update: bool, optional
        :param verbose: Do you want to print the details?, defaults to False
        :type verbose: bool, optional
        """
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
            self.comm.lasif.move_gradient_to_cluster()
        self.adjoint_helper = helpers.AdjointHelper(
            comm=self.comm, events=self.comm.project.events_in_iteration
        )
        self.adjoint_helper.process_gradients(
            interpolate=interpolate,
            smooth_individual=smooth_individual,
            verbose=verbose,
        )
        assert self.adjoint_helper.assert_all_simulations_retrieved()

        self.smoothing_helper = helpers.SmoothingHelper(
            comm=self.comm, events=self.comm.project.events_in_iteration
        )
        if sum_gradients:
            gradient = (
                self.comm.lasif.lasif_comm.project.paths["gradients"]
                / f"ITERATION_{self.iteration_name}"
                / "summed_gradient.h5"
            )

            if not os.path.exists(gradient):
                if interpolate:
                    self.smoothing_helper.monitor_interpolations(
                        smooth_individual=smooth_individual, verbose=verbose
                    )
                self.smoothing_helper.sum_gradients()
            if smooth_update:
                return

        self.smoothing_helper.dispatch_smoothing_simulations(
            smooth_individual=smooth_individual, verbose=verbose
        )
        assert self.smoothing_helper.assert_all_simulations_dispatched(
            smooth_individual=smooth_individual
        )
        self.smoothing_helper.retrieve_smooth_gradients(
            smooth_individual=smooth_individual
        )
        assert self.smoothing_helper.assert_all_simulations_retrieved(
            smooth_individual=smooth_individual
        )

    def update_model(self):
        """
        Not yet implemented for the standard optimization.
        """
        pass

    def get_parameter_indices(self, filename):
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
            for param in self.parameters:
                indices.append(dim_labels.index(param))
        return indices

    def get_h5_data(self, filename):
        """
        Returns the relevant data in the form of ND_array with all the data.
        """
        indices = self.get_parameter_indices(filename)

        with h5py.File(filename, "r") as h5:
            data = h5["MODEL/data"][:, :, :].copy()
            return data[:, indices, :]

    def set_h5_data(self, filename, data):
        """Writes the data with shape [:, indices :]. Requires existing file."""
        if not os.path.exists(filename):
            raise Exception("only works on existing files.")

        indices = self.get_parameter_indices(filename)

        with h5py.File(filename, "r+") as h5:
            dat = h5["MODEL/data"]
            data_copy = dat[:, :, :].copy()
            # avoid writing the file many times. work on array in memory
            for i in range(len(indices)):
                data_copy[:, indices[i], :] = data[:, i, :]

            # writing only works in sorted order. This sort can only happen after
            # the above executed to preserve the ordering that data came in
            indices.sort()
            dat[:, indices, :] = data_copy[:, indices, :]
