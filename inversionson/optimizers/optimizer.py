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
from inversionson import autoinverter_helpers as helpers


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
        self.models_dir = self.opt_folder / "MODELS"
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
        return self.find_iteration_numbers()[0]

    def find_iteration_numbers(self):
        models = glob.glob(f"{self.models_dir}/*.h5")
        if len(models) == 0:
            return [0]
        iteration_numbers = []
        for model in models:
            iteration_numbers.append(int(model.split(".")[0].split("_")[-1]))
        iteration_numbers.sort(reverse=False)
        return iteration_numbers

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
        self, smooth_individual: bool = False, sum_gradients: bool = True, verbose=False
    ):
        """
        This smooths, sums and interpolates gradients based on what is needed.

        :param smooth_individual: Smooth individual gradients?, defaults to False
        :type smooth_individual: bool, optional
        :param sum_gradients: Sum gradients before smoothing?, defaults to True
        :type sum_gradients: bool, optional
        :param verbose: Do you want to print the details?, defaults to False
        :type verbose: bool, optional
        """
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
            self.comm.lasif.move_gradient_to_cluster()
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
            gradients = self.comm.lasif.lasif_comm.project.paths["gradients"]
            gradient = os.path.join(
                gradients,
                f"ITERATION_{self.comm.project.current_iteration}",
                "summed_gradient.h5",
            )
            if not os.path.exists(gradient):
                if interpolate:
                    self.smoothing_helper.monitor_interpolations(
                        smooth_individual=smooth_individual, verbose=verbose
                    )
                self.smoothing_helper.sum_gradients()
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
