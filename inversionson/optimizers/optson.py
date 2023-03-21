import glob
import os
import sys

import toml
import numpy as np
import shutil
import inspect

from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset

from inversionson import InversionsonError
from optson.base_classes.vector import Vector
from inversionson.helpers.regularization_helper import RegularizationHelper
from inversionson.optimizers.optimizer import Optimize
from inversionson.utils import write_xdmf
from salvus.mesh.unstructured_mesh import UnstructuredMesh as um


class OptsonLink(Optimize):
    """
    A class that acts as a bridge to Optson.

    #TODO: Add smoothing
    #TODO: Clean up old iterations
    #TODO optimize calls to control group misfit/gradient.

    # Simply perform the mini-batch job first, and then call control group.
    # in this way adjoint jobs will be submitted already, so the next call will
    # be quick.
    #TODO: Add flag to iteration listerener that does misfit only, but still submits the adjoint jobs

    # Become smarter about the jobs. For example when gx_cg_prev is called, we know
    the model is accepted. we can run the iteration listener with all_events.
    # then we can for cg_prev, also already immediately smooth the mini-batch gradient,
    the new control group gradient and the previous gradient.
    #
    """

    optimizer_name = "Optson"
    config_template_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        ),
        "file_templates",
        "Optson.toml",
    )
    current_iteration_name = "x_00000"

    def __init__(self, comm):

        # Call the super init with all the common stuff
        super().__init__(comm)
        self.opt = None
        self.pt_idcs = None
        self.inv_pt_idcs = None

    def _initialize_derived_class_folders(self):
        """These folder are needed only for Optson."""
        self.smooth_gradient_dir = self.opt_folder / "SMOOTHED_GRADIENTS"

    def get_smooth_gradient_path(self, model_name, set_flag: str = "mb"):
        allowed_set_flags = ["mb", "cg", "cg_prev"]
        if set_flag not in allowed_set_flags:
            raise Exception(f"Only {allowed_set_flags} can be passed")
        return self.smooth_gradient_dir / f"smooth_g_{set_flag}_{model_name}.h5"

    def get_raw_gradient_path(self, model_name, set_flag: str = "mb"):
        allowed_set_flags = ["mb", "cg", "cg_prev"]
        if set_flag not in allowed_set_flags:
            raise Exception(f"Only {allowed_set_flags} can be passed")
        return self.raw_gradient_dir / f"raw_g_{set_flag}_{model_name}.h5"

    @property
    def status_file(self):
        return str(self.opt_folder / "optson_status_tracker.json")

    @property
    def task_file(self):
        return str(self.opt_folder / "optson_task_tracker.json")

    @property
    def cache_file(self):
        return str(self.opt_folder / "optson_cache.h5")

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

    # def vector_to_mesh(self, to_mesh, m):
    #     parameters = ["VPV", "VPH"]
    #     par_vals = np.array_split(m.x, len(parameters))
    #     m_init = um.from_h5(self.initial_model)
    #     m_grad = um.from_h5(os.path.join(self.raw_gradient_dir, "raw_g_cg_x_00000.h5"))
    #
    #     for idx, par in enumerate(parameters):
    #         normalization = True
    #         par_val = par_vals[idx].reshape(m_init.element_nodal_fields[par].shape)
    #         if normalization:
    #             par_val = (par_val-1) * m_grad.element_nodal_fields["Valence"]
    #             par_val += 1
    #             m_init.element_nodal_fields[par][:] = m_init.element_nodal_fields[par][:] * par_val
    #                 #m_init.connectivity]
    #         else:
    #             m_init.element_nodal_fields[par][:] = par_val
    #                 #m_init.connectivity]
    #     m_init.write_h5(to_mesh)
    #
    # def mesh_to_vector(self, mesh_filename, gradient=True, raw_grad_file=None):
    #     parameters = ["VPV", "VPH"]
    #     m = um.from_h5(mesh_filename)
    #     m_init = um.from_h5(self.initial_model)
    #
    #     if gradient:
    #         rg = um.from_h5(raw_grad_file)
    #         mm = rg.element_nodal_fields["FemMassMatrix"]
    #         # valence = rg.element_nodal_fields["Valence"]
    #
    #     # _, i = np.unique(m.connectivity, return_index=True)
    #
    #     normalization = True
    #     # par_dict = {}
    #     par_list = []
    #     for par in parameters:
    #         if normalization:
    #             if gradient:
    #                 par_val = m.element_nodal_fields[par] * \
    #                           m_init.element_nodal_fields[par]
    #             else:
    #                 par_val = m.element_nodal_fields[par] / \
    #                           m_init.element_nodal_fields[par]
    #         else:
    #             par_val = m.element_nodal_fields[par]
    #
    #         if gradient:
    #             par_val = par_val * mm# * valence  # multiply with valence to account for duplication.
    #         # par_dict[par] = par_val.flatten()[i]
    #         par_list.append(par_val.flatten())#[i])
    #
    #     v = np.concatenate(par_list)
    #     return v

    def vector_to_mesh_new(self, to_mesh, x):
        print("Writing vector to mesh started...")
        parameters = self.parameters
        if self.isotropic_vp:
            isotropic_pars = parameters.copy()
            isotropic_pars.remove("VPH")
        else:
            isotropic_pars = parameters
        to_mesh = str(to_mesh)# + "new_func"
        tmp_mesh_file = "tmp_mesh_file.h5"
        shutil.copy(self.initial_model, tmp_mesh_file)
        # we now split in isotropic pars
        par_vals = np.array_split(x, len(isotropic_pars))
        points = self.get_points(tmp_mesh_file)
        nelem, ngll, ndim = points.shape
        if self.pt_idcs is None:
            _, self.pt_idcs, self.inv_pt_idcs = np.unique(
                points.reshape(nelem * ngll, ndim),
                return_index=True,
                return_inverse=True,
                axis=0,
            )

        # here we get all parameters. That's good
        m_init = self.get_h5_data(self.initial_model, parameters)

        par_list = []
        for idx, val in enumerate(parameters):
            if self.isotropic_vp:
                if val == "VPH":
                    val = "VPV"
                opt_idx = isotropic_pars.index(val)
            else:
                opt_idx = idx
            par = par_vals[opt_idx]  # these are now flat with the same sorting.
            par = par[self.inv_pt_idcs]
            par = par.reshape((nelem, ngll))  # reshape into original form
            par = m_init[:, idx, :] * par  # here we have a mismatch perhaps. We nora
            par_list.append(par)

        data_in_original_shape = np.stack(par_list, axis=1)
        shutil.move(tmp_mesh_file, to_mesh)
        self.set_h5_data(to_mesh, data_in_original_shape, create_xdmf=True,
                         parameters=parameters)
        print("Writing vector to mesh completed.")

    def get_mm(self):
        # the below line is a bit slow
        if self.pt_idcs is None:
            points = self.get_points(self.initial_model)
            nelem, ngll, ndim = points.shape
            _, self.pt_idcs, self.inv_pt_idcs = np.unique(
                points.reshape(nelem * ngll, ndim),
                return_index=True,
                return_inverse=True,
                axis=0,
            )

        mm, valence = self.get_flat_non_duplicated_data(
            ["FemMassMatrix", "Valence"], self.mass_matrix_mesh, self.pt_idcs
        )
        mm_val = mm * valence

        parameters = self.parameters.copy()
        if self.isotropic_vp:
            parameters.remove("VPH")
        return np.tile(mm_val, len(parameters))

    def mesh_to_vector_new(self, mesh_filename, gradient=True, raw_grad_file=None):
        print("Writing mesh to vector started...")
        parameters = self.parameters.copy()
        # a simple thing we can do is only take VPV, but write it to both fields
        if self.isotropic_vp:
            parameters.remove("VPH")  # only do VPV

        # the below line is a bit slow
        if self.pt_idcs is None:
            points = self.get_points(mesh_filename)
            nelem, ngll, ndim = points.shape
            _, self.pt_idcs, self.inv_pt_idcs = np.unique(
                points.reshape(nelem * ngll, ndim),
                return_index=True, return_inverse=True, axis=0)
        mesh_data = self.get_flat_non_duplicated_data(
            parameters, mesh_filename, self.pt_idcs)

        initial_data = self.get_flat_non_duplicated_data(
            parameters, self.initial_model, self.pt_idcs
        )
        par_list = []
        for idx in range(len(parameters)):
            if gradient:
                par_val = mesh_data[idx] / initial_data[idx]
            else:
                par_val = mesh_data[idx] / initial_data[idx]

            par_list.append(par_val)
        v = np.concatenate(par_list)
        print("Writing mesh to vector completed.")
        return v

    def vector_to_mesh_debug(self):
        pass

    def perform_task(self, verbose=True):
        """
        Task manager calls this class. Main entry point. Here, all the magic will happen.
        """
        from optson.optimizer import Optimizer
        from optson.methods.trust_region_LBFGS import TrustRegionLBFGS
        from optson.methods.steepest_descent import SteepestDescent
        from inversionson.optimizers.StochasticFWI import StochasticFWI
        from optson.base_classes.stopping_criterion import BasicStoppingCriterion

        bsc = BasicStoppingCriterion(maxIterations=10000, tolerance=1e-30, divergenceTolerance=1e30)

        self.find_iteration_numbers()
        if self.do_gradient_test:
            self.gradient_test()
            sys.exit()

        problem = StochasticFWI(
            comm=self.comm,
            optlink=self,
            batch_size=self.comm.project.batch_size,
            status_file=self.status_file,
            task_file=self.task_file
        )

        steepest_descent = SteepestDescent(
            initial_step_length=self.initial_step_length,
            verbose=verbose,
            step_length_as_percentage=True)
        method = TrustRegionLBFGS(
            steepest_descent=steepest_descent, verbose=verbose
        )

        x_0 = self.mesh_to_vector_new(self.initial_model, gradient=False)
        self.opt = Optimizer(problem=problem, method=method, cache_file=self.cache_file, stopping_criterion=bsc, verbose=True)
        self.opt.iterate(x0=x_0, n_iter=self.max_iterations)

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
        from inversionson.optimizers.StochasticFWI import StochasticFWI
        from optson.gradient_test import GradientTest
        import matplotlib.pyplot as plt

        if not h:
            h = np.logspace(-7, -1, 7)
        print("All these h values that will be tested:", h)
        problem = StochasticFWI(
            comm=self.comm,
            optlink=self,
            batch_size=self.comm.project.batch_size,
            gradient_test=True,
            status_file=self.status_file,
            task_file=self.task_file
        )

        x_0 = self.mesh_to_vector_new(self.initial_model, gradient=False)
        grdtest = GradientTest(x0=x_0, h=h, problem=problem, verbose=True)

        # We need to enforce that the previous iteration has a control group
        grdtest.m.fx
        grdtest.m.gx
        grdtest.m.gx_cg
        grdtest()
        plt.loglog(grdtest.h, grdtest.relative_errors)
        plt.title(f"Minimum relative error: {min(grdtest.relative_errors)}")
        plt.xlabel("h")
        plt.ylabel("Relative error")
        plt.savefig("gradient_test.png")
        return grdtest

    def find_iteration_numbers(self):
        models = glob.glob(f"{self.model_dir}/*.h5")
        if len(models) == 0:
            return [0]
        iteration_numbers = []
        for model in models:
            it_number = int(model.split("/")[-1].split(".")[0].split("_")[2])
            if it_number not in iteration_numbers:
                iteration_numbers.append(it_number)
        iteration_numbers.sort()
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
        self.do_gradient_test = config["do_gradient_test"]
        self.max_iterations = config["max_iterations"]
        self.isotropic_vp = config["isotropic_vp"]
        self.speculative_forwards = config["speculative_forwards"]
        self.mass_matrix_mesh = config["mass_matrix_file"]
        self.initial_step_length = config["initial_step_length"]
        
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

    def pick_data_for_iteration(
        self,
        batch_size,
        prev_control_group=[],
        current_batch=[],
        select_new_control_group=False,
        control_group_size: int = None,
    ):
        print("Selecting data...")
        all_events = self.comm.lasif.list_events()
        blocked_data = set(
            self.comm.project.validation_dataset
            + self.comm.project.test_dataset
            + prev_control_group
        )
        all_events = list(set(all_events) - blocked_data)
        n_events = batch_size - len(prev_control_group)
        events = []

        if select_new_control_group:
            if not current_batch:
                raise Exception(
                    "I need the current batch if you want"
                    "a control group to be selected."
                )
            all_events = current_batch
            if not control_group_size:
                control_group_size = int(np.ceil(0.5 * len(current_batch)))
            n_events = control_group_size

        all_norms_path = self.gradient_norm_dir / "all_norms.toml"

        if (
            n_events > 0
        ):  # for edge case batch size is same length as prev control group
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
        print("Data selection completed.")
        return events

    def pick_control_group(self):
        pass

    def prepare_iteration(self, events, iteration_name=None):
        iteration_name = iteration_name if iteration_name else self.iteration_name
        if self.comm.lasif.has_iteration(iteration_name):
            self.comm.project.change_attribute("current_iteration", iteration_name)
            self.comm.project.get_iteration_attributes(iteration=iteration_name)
            return
        self.comm.project.change_attribute("current_iteration", iteration_name)
        self.print("Picking data for iteration")

        # this should become smart with control groups etc.
        super().prepare_iteration(it_name=iteration_name, events=events)

    def perform_smoothing(self, x: Vector, set_flag, file):
        """
        Writes the smoothing task only, does not monitor...
        """
        tasks = {}
        tag = file.name
        output_location = self.get_smooth_gradient_path(x.descriptor, set_flag=set_flag)
        if max(self.gradient_smoothing_length) > 0.0:
            tasks[tag] = {
                "reference_model": str(self.comm.lasif.get_master_model()),
                "model_to_smooth": str(file),
                "smoothing_lengths": self.gradient_smoothing_length,
                "smoothing_parameters": self.parameters,
                "output_location": str(output_location),
            }
        else:
            shutil.copy(file, output_location)

        if len(tasks.keys()) > 0:
            reg_helper = RegularizationHelper(
                comm=self.comm, iteration_name=x.descriptor, tasks=tasks, optimizer=self
            )
            if tag in reg_helper.tasks.keys() and not os.path.exists(output_location):
                # remove the completed tag if we need to redo smoothing.
                reg_helper.tasks[tag].update(reg_helper.base_dict)
                reg_helper._write_tasks(reg_helper.tasks)


    def compute_gradient(self, verbose):
        pass

    def get_mref(self):
        parameters = self.parameters.copy()
        # a simple thing we can do is only take VPV, but write it to both fields
        if self.isotropic_vp:
            parameters.remove("VPH")  # only do VPV
        initial_data = self.get_flat_non_duplicated_data(
            parameters, self.initial_model, self.pt_idcs
        )
        par_list = []
        for idx in range(len(parameters)):
            par_val = initial_data[idx]
            par_list.append(par_val)
        return np.concatenate(par_list)
        