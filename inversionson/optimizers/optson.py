import glob
import os
import toml
import numpy as np
import shutil
import inspect

from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset

from inversionson import InversionsonError
from inversionson.helpers.autoinverter_helpers import IterationListener
from inversionson.optimizers.optimizer import Optimize
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.utils import write_xdmf


from optson.base_classes.base_problem import StochasticBaseProblem
from optson.base_classes.model import ModelStochastic
from lasif.components.communicator import Communicator

from salvus.mesh.unstructured_mesh import UnstructuredMesh as um



def mesh_to_vector(mesh_filename, initial_model, gradient=True):
    """
    Maps a salvus mesh to a vector suitable for use with Optson.
    #TODO: add option for multiple parameters.
    """
    m = um.from_h5(mesh_filename)
    m_init = um.from_h5(initial_model)
    _, i = np.unique(m.connectivity, return_index=True)

    normalization = True
    # Normalization, still have to make the case with zero velocity work
    # TODO: fix case with velocities of zero
    if normalization:
        if gradient:
            vsv = m.element_nodal_fields["VSV"] * m_init.element_nodal_fields["VSV"]
        else:
            vsv = m.element_nodal_fields["VSV"] / m_init.element_nodal_fields["VSV"]
    else:
        vsv = m.element_nodal_fields["VSV"]

    # Gradient, this also implies the mesh filename is a gradient
    # and thus has the FemMassMatrix and Valence fields.
    if gradient:
        mm = m.element_nodal_fields["FemMassMatrix"]
        valence = m.element_nodal_fields["Valence"]
        vsv = vsv * mm * valence # multiply with valence to account for duplication.
    v = vsv.flatten()[i]
    return v


def vector_to_mesh(initial_model, to_mesh, m):
    """
    Maps an Optson vector to a salvus mesh.
    # TODO: add option for multiple parameters.
    """
    m_init = um.from_h5(initial_model)
    m_init.element_nodal_fields["VSV"][:] = m_init.element_nodal_fields["VSV"][:] * m.x[m_init.connectivity]
    m_init.write_h5(to_mesh)

class OptsonLink(Optimize):
    """
    A class that acts as a bridge to Optson
    """

    optimizer_name = "Optson"
    config_template_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
        ),
        "file_templates",
        "Optson.toml"
    )
    current_iteration_name = "x_00000"
    def __init__(self, comm):

        # Call the super init with all the common stuff
        super().__init__(comm)
        self.opt = None

    def _initialize_derived_class_folders(self):
        """These folder are needed only for Optson."""
        self.smooth_gradient_dir = self.opt_folder / "SMOOTHED_GRADIENTS"

    @property
    def smooth_gradient_path(self, cg=False, cg_prev=False):
        if cg and cg_prev:
            raise Exception("Only one control group can be set to true")
        if cg:
            return self.smooth_gradient_dir / f"smooth_g_cg_{self.iteration_number:05d}.h5"
        elif cg_prev:
            return self.smooth_gradient_dir / f"smooth_g_cg_prev_{self.iteration_number:05d}.h5"
        else:   
            return self.smooth_gradient_dir / f"smooth_g_{self.iteration_number:05d}.h5"

    @property
    def gradient_norm_path(self):
        return (
            self.gradient_norm_dir / f"gradient_norms_{self.iteration_number:05d}.toml"
        )

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

    def perform_task(self, verbose=False):
        """
        THIS is the key entry point for inversionson!!!!
        # TODO make this the entry
        Look at which task is the current one and call the function which does it.
        """
        problem = StochasticFWI(comm=self.comm, optlink=self,
                                batch_size=1)
        from optson.optimize import Optimize
        from optson.methods.trust_region_LBFGS import StochasticTrustRegionLBFGS
        from optson.methods.steepest_descent import StochasticSteepestDescent

        method = StochasticTrustRegionLBFGS(steepest_descent=StochasticSteepestDescent(initial_step_length=2e-2, verbose=True, step_length_as_percentage=True), verbose=True)
        x_0 = mesh_to_vector(self.initial_model, self.initial_model,
                             gradient=False)

        grdtest = self.gradient_test()

        self.opt = Optimize(x_0=x_0, problem=problem, method=method)
        self.opt.iterate(20)
        # raise NotImplementedError

    def gradient_test(self, h=None):
        """
        Function to perform a gradient test for the current project.
        from a notebook in the root directory of inversionson, you may call this
        like this in the following way:

            from inversionson.autoinverter import read_info_toml, AutoInverter
            info = read_info_toml("")
            auto = AutoInverter(info, manual_mode=True)
            auto.move_files_to_cluster()
            opt = auto.comm.project.get_optimizer()
            grd_test = opt.gradient_test()

        """
        if not h:
            h = np.logspace(-7, -1, 7)
        print("All h values that will be tested", h)
        problem = StochasticFWI(comm=self.comm, optlink=self,
                                batch_size=1, gradient_test=True)
        from optson.gradient_test import GradientTest

        x_0 = mesh_to_vector(self.initial_model, self.initial_model, gradient=False)
        grdtest = GradientTest(x_0=x_0, h=h,
                               problem=problem)
        return grdtest

    def find_iteration_numbers(self):
        models = glob.glob(f"{self.model_dir}/*.h5")
        if len(models) == 0:
            return [0]
        iteration_numbers = []
        for model in models:
            iteration_numbers.append(int(model.split("/")[-1].split(".")[0].split("_")[2]))
        return iteration_numbers

    @property
    def model_path(self, iteration_name=None):
        if not iteration_name:
            return self.model_dir / f"model_{self.current_iteration_name}.h5"
        else:
            return self.model_dir / f"model_{iteration_name}.h5"

    def _read_config(self):
        """Reads the config file."""
        if not os.path.exists(self.config_file):
            raise Exception("Can't read the SGDM config file")
        config = toml.load(self.config_file)

        self.initial_model = config["initial_model"]
        self.smoothing_timestep = config["smoothing_timestep"]
        self.gradient_smoothing_length = config["gradient_smoothing_length"]
        self.grad_scaling_fac = config["gradient_scaling_factor"]
        self.do_gradient_test = False

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
            self.average_model_dir,
            self.smooth_gradient_dir,
            self.raw_gradient_dir,
            self.raw_update_dir,
            self.task_dir,
            self.regularization_dir,
            self.gradient_norm_dir,
        ]

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)

        shutil.copy(self.initial_model, self.model_path)
        write_xdmf(self.model_path)

    def _finalize_iteration(self, verbose: bool):
        pass
        # """
        # Here we can do some documentation. Maybe just call a base function for that
        # """
        # super().delete_remote_files()
        # self.comm.storyteller.document_task(task="adam_documentation")

    def pick_data_for_iteration(self, batch_size, prev_control_group=[],
                                current_batch=[],
                                select_new_control_group=False,
                                control_group_size: int = None):

        all_events = self.comm.lasif.list_events()
        blocked_data = set(self.comm.project.validation_dataset +
                           self.comm.project.test_dataset
                           + prev_control_group)
        all_events = list(set(all_events) - blocked_data)
        n_events = batch_size - len(prev_control_group)
        events = []

        if select_new_control_group:
            if not current_batch:
                raise Exception("I need the current batch if you want"
                                "a control group to be selected.")
            all_events = current_batch
            if not control_group_size:
                control_group_size = int(np.ceil(0.5*len(current_batch)))
            n_events = control_group_size

        all_norms_path = self.gradient_norm_dir / "all_norms.toml"

        if n_events > 0: # for edge case batch size is same length as prev control group
            if os.path.exists(all_norms_path):
                norm_dict = toml.load(all_norms_path)
                unused_events = list(set(all_events).difference(set(norm_dict.keys())))
                list_of_vals = np.array(list(norm_dict.values()))
                # Set unused to 65%, so slightly above average
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

        if self.time_for_validation() and not select_new_control_group:
            print(
                "This is a validation iteration. Will add validation events to the iteration"
            )
            events += self.comm.project.validation_dataset
        events += prev_control_group

        return events

    def pick_control_group(self):
        pass

    def prepare_iteration(self, events, iteration_name=None):
        iteration_name = iteration_name if iteration_name else self.iteration_name
        if self.comm.lasif.has_iteration(iteration_name):
            self.comm.project.change_attribute("current_iteration",
                                               iteration_name)
            self.comm.project.get_iteration_attributes(iteration=iteration_name)
            return
        self.comm.project.change_attribute("current_iteration", iteration_name)
        self.print("Picking data for iteration")

        # this should become smart with control groups etc.
        super().prepare_iteration(it_name=iteration_name, events=events)

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
                    f"relative_perturbation_{self.iteration_name}",
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

    def compute_gradient(self, verbose):
        pass

class StochasticFWI(StochasticBaseProblem):
    def __init__(self, comm: Communicator,
                 optlink: OptsonLink,
                 batch_size=2,
                 gradient_test=False):
        self.comm = comm
        self.known_tags = []
        self.optlink = optlink
        self.gradient_test = gradient_test

        # All these things need to be cached
        self.mini_batch_dict = {}
        self.control_group_dict = {}
        self.batch_size = batch_size
        # list model names per iteration here to figure out the accepted and rejected models
        self.model_names = {}
        self.mass_matrix = None
        self.valence = None

    def create_iteration_if_needed(self, m: ModelStochastic):
        """
        Here, we create the iteration if needed. This also requires us
        to select the mini_batch already, but that is no problem.
        """
        previous_control_group = []
        events = []
        self.optlink.current_iteration_name = m.name

        if not os.path.exists(self.optlink.model_path):
            vector_to_mesh(self.optlink.initial_model,
                           self.optlink.model_path, m)

        if not self.comm.lasif.has_iteration(m.name):
            if m.iteration_number > 0:
                previous_control_group = \
                    self.control_group_dict[m.iteration_number-1]
            events = self.optlink.pick_data_for_iteration(
                batch_size=self.batch_size,
                prev_control_group=previous_control_group)

        # the below line will set the proper parameters in the project
        # component
        self.optlink.prepare_iteration(iteration_name=m.name, events=events)
        if m.iteration_number not in self.model_names.keys():
            self.model_names[m.iteration_number] = [m.name]
        else:
            if m.name not in self.model_names[m.iteration_number]:
                self.model_names[m.iteration_number].append(m.name)
        self.mini_batch_dict[m.iteration_number] = self.comm.project.events_in_iteration

    def select_batch(self, m: ModelStochastic):
        #
        self.create_iteration_if_needed(m=m)

    def _misfit(
        self, m: ModelStochastic, it_num: int, control_group: bool = False,
            misfit_only=True,
    ) -> float:
        self.create_iteration_if_needed(m=m)
        prev_control_group = []
        control_group_events = []
        misfit_only = misfit_only
        previous_iteration = None
        events = self.comm.project.events_in_iteration

        if control_group:
            # If we only want control group misfits, we don't need the gradients
            # and only ensure the control group events are simulated.
            events = self.control_group_dict[it_num]

        if m.iteration_number > 0 and m.iteration_number > it_num:
            # if it not the first model,
            # we need to consider the previous control group
            prev_control_group = self.control_group_dict[it_num]
            previous_iteration = self.model_names[it_num][-1]

        # Else we just take everything and immediately also compute the
        # gradient

        it_listen = IterationListener(
            comm=self.comm,
            events=events,
            control_group_events=control_group_events,
            prev_control_group_events=prev_control_group,
            misfit_only=misfit_only,
            prev_iteration=previous_iteration
        )
        it_listen.listen()

        # Now we need a way to actually collect the misfit for the events.
        # this involves the proper name for the mdoel and the set of events.
        self.comm.project.get_iteration_attributes(m.name)
        total_misfit = 0.0
        for event in events:
            total_misfit += self.comm.project.misfits[event]
        return total_misfit / len(events)

    def misfit(
        self, m: ModelStochastic, it_num: int, control_group: bool = False,
    ) -> float:
        """
        We may want some identifier to say which solution vector x is used.
        Things like model_00000_step_... or model_00000_TrRadius_....
        # TODO cache these results as well.
        """
        if control_group:
            return self._misfit(m=m, it_num=it_num, control_group=control_group,
                                misfit_only=True)
        else:
            return self._misfit(m=m, it_num=it_num, control_group=control_group,
                                misfit_only=False)

    def gradient(
        self, m: ModelStochastic, it_num: int, control_group: bool = False
    ) -> np.array:

        # Simply call the misfit function, but ensure we also compute the gradients.
        self._misfit(m=m, it_num=it_num, control_group=control_group,
                    misfit_only=False)

        # now we need to figure out how to sum the proper gradients.
        # for this we need the events
        if control_group:
            events = self.control_group_dict[it_num]
        else:
            events = set(self.mini_batch_dict[it_num]) - set(self.comm.project.validation_dataset)
            events = list(events)
        output_location = "tmp_gradient.h5"

        grad_summer = GradientSummer(comm=self.comm)
        store_norms = False if self.gradient_test else True
        grad_summer.sum_gradients(
            events=events,
            output_location=output_location,
            batch_average=True,
            sum_vpv_vph=True,
            store_norms=store_norms,
        )
        write_xdmf(output_location)

        return mesh_to_vector(output_location, self.optlink.initial_model)

    def select_control_group(self, m: ModelStochastic):
        current_batch = self.mini_batch_dict[m.iteration_number]

        control_group = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=current_batch,
            select_new_control_group=True
        )
        self.control_group_dict[m.iteration_number] = control_group

    def extend_control_group(self, m: ModelStochastic) -> bool:
        current_batch = self.mini_batch_dict[m.iteration_number]
        current_control_group = self.control_group_dict[m.iteration_number]
        non_control_events = set(current_batch) - set(current_control_group)

        if len(current_control_group) == len(current_batch):
            return False
        additional_controls = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=list(non_control_events),
            select_new_control_group=True
        )
        self.control_group_dict[m.iteration_number] = \
            current_control_group + additional_controls
        return True
