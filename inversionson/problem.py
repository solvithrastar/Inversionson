"""
TODO: add option to smooth gradients.
TODO: Figure out what the previous iteration was. Probably this is best figured out from the iteration 

"""
from pathlib import Path
from typing import List, Optional
from optson.problem import AbstractProblem, CallCounter
from optson.vector import Vec
from inversionson.helpers.regularization_helper import RegularizationHelper
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.helpers.iteration_listener import IterationListener
from inversionson.utils import (
    get_list_hash,
    mesh_to_vector,
    vector_to_mesh,
    hash_vector,
)
from inversionson.project import Project
import numpy as np
from optson.problem import ModelProxy
from optson.preconditioner import AbstractUpdatePreconditioner


class InversionsonUpdatePrecondtioner(AbstractUpdatePreconditioner):
    def __init__(self, project: Project):
        self.project = project

    def __call__(self, x: Vec, model_descriptor: str) -> Vec:
        vec_hash = hash_vector(np.array(x))
        unsmoothed_filename = (
            self.project.paths.reg_dir / f"usm_p_{model_descriptor}_{vec_hash}.h5"
        )
        smoothed_filename = (
            self.project.paths.reg_dir / f"sm_p_{model_descriptor}_{vec_hash}.h5"
        )
        params = self.project.config.inversion.inversion_parameters
        if smoothed_filename.exists():
            return mesh_to_vector(smoothed_filename, params_to_invert=params)
        if not unsmoothed_filename.exists():
            m = vector_to_mesh(
                x, target_mesh=self.project.lasif.master_mesh, params_to_invert=params
            )
            m.write_h5(unsmoothed_filename)

        smoothing_lengths = self.project.config.inversion.smoothing_lengths

        if max(smoothing_lengths) <= 0.0:
            return x

        tag = unsmoothed_filename.name
        tasks = {
            tag: {
                "reference_model": str(self.project.lasif.get_master_model()),
                "model_to_smooth": str(unsmoothed_filename),
                "smoothing_lengths": smoothing_lengths,
                "smoothing_parameters": self.project.config.inversion.inversion_parameters,
                "output_location": str(smoothed_filename),
            }
        }
        reg = RegularizationHelper(
            project=self.project, iteration_name=model_descriptor, tasks=tasks
        )
        reg.monitor_tasks()
        return mesh_to_vector(smoothed_filename, params_to_invert=params)


class Problem(AbstractProblem):
    def __init__(self, project: Project, smooth_gradients: bool = False):
        super().__init__()
        self.project = project
        self.smooth_gradients = smooth_gradients  # TODO: implement this
        self.deleted_iters: List[str] = []
        self.completed_full_batches: List[str] = []

    def _clean_old_iters(self):
        all_tomls = sorted(self.project.paths.iteration_tomls.glob("*.toml"))
        for toml in all_tomls[:-1]:
            iter_name = toml.stem
            if iter_name in self.deleted_iters:
                continue
            self.project.flow.delete_remote_iteration(iteration=iter_name, verbose=True)
            self.deleted_iters.append(iter_name)

    @staticmethod
    def _get_tag(model: ModelProxy, indices: Optional[List[int]]) -> Optional[str]:
        if indices is None:
            return None
        hash_id = get_list_hash(indices)
        if indices == model.batch:
            return f"mb_{hash_id}"
        elif indices == model.control_group:
            return f"cg_{hash_id}"
        elif indices == model.control_group_previous:
            return f"cgp_{hash_id}"
        raise ValueError("This should not occur.")

    def _time_for_validation(self, iteration_number: int) -> bool:
        """Determine if this is a validation iteration."""
        val_itv = self.project.config.monitoring.iterations_between_validation_checks
        if val_itv == 0:
            return False
        return True if iteration_number == 0 else (iteration_number + 1) % val_itv == 0

    def _get_events_for_iteration(self, model: ModelProxy) -> List[str]:
        """Get all events for the iteration."""
        if model.batch is None:
            events = self.project.event_db.get_all_event_names(non_validation_only=True)
        else:
            events = self.project.event_db.get_event_names(model.batch)

        if self._time_for_validation(model.iteration):
            events += self.project.config.monitoring.validation_dataset
        return events

    def _get_events_from_indices(self, indices: Optional[List[int]]) -> List[str]:
        """Get events from a list of indices, returns everything when indices is None."""
        if indices is None:
            return self.project.non_val_events_in_iteration
        return self.project.event_db.get_event_names(indices)

    def _get_or_create_iteration(self, model: ModelProxy) -> None:
        """Create the iteration or load its attributes if it exists."""
        iteration = model.descriptor
        if self.project.paths.get_iteration_toml(iteration).exists():
            self.project.set_iteration_attributes(iteration=iteration)
            return

        previous_iter = model.previous.descriptor if model.previous else None
        self._clean_old_iters()

        events = self._get_events_for_iteration(model)
        self.project.lasif.set_up_iteration(iteration, events)
        self.project.create_iteration_toml(
            iteration, events, previous_iteration=previous_iter
        )

    def _prepare_iteration(self, model: ModelProxy) -> None:
        model_file = self.project.paths.get_model_path(model.descriptor)
        if not model_file.exists():
            vector_to_mesh(
                model.x,
                self.project.lasif.master_mesh,
                self.project.config.inversion.inversion_parameters,
            ).write_h5(model_file)
        self._get_or_create_iteration(model)

    def previous_control_group(self, model: ModelProxy) -> Optional[List[str]]:
        if model.batch is None and model.iteration != 0:  # Handle mono-batch case.
            return self.project.events_in_iteration
        previous_control_indices = model.control_group_previous
        if previous_control_indices is None:
            return None
        return self.project.event_db.get_event_names(previous_control_indices)

    @CallCounter
    def f(self, model: ModelProxy, indices: Optional[List[int]] = None) -> float:
        self._prepare_iteration(model)
        IterationListener(
            project=self.project,
            events=self.project.events_in_iteration,
            prev_control_group_events=self.previous_control_group(model),
            misfit_only=True,
            prev_iteration=self.project.previous_iteration,
            submit_adjoint=False,  # consider setting this to true.
        ).listen()

        train_events = self._get_events_from_indices(indices)
        total_misfit = 0.0
        for event in train_events:
            total_misfit += self.project.misfits[event]
        total_misfit /= len(train_events)
        return total_misfit

    def get_remaining_batch_indices(
        self, model: ModelProxy
    ) -> List[Optional[List[int]]]:
        """This returns a list of indices for all the required batches."""
        if model.control_group == [] or model.control_group is None:
            return []

        # The below line should trigger control group selection
        indices = [model.control_group]
        if model.control_group_previous != []:
            indices.append(model.control_group_previous)
        return indices

    def _write_smoothing_task(
        self, model: ModelProxy, indices: Optional[List[int]], file: Path
    ) -> None:
        """
        Writes the smoothing task only, does not monitor...
        """
        tasks = {}
        tag = file.name
        output_location = self.project.paths.get_smooth_gradient_path(
            model.descriptor, tag=self._get_tag(model, indices)
        )

        smoothing_lengths = self.project.config.inversion.smoothing_lengths
        if max(smoothing_lengths) > 0.0:
            tasks[tag] = {
                "reference_model": str(self.project.lasif.get_master_model()),
                "model_to_smooth": str(file),
                "smoothing_lengths": smoothing_lengths,
                "smoothing_parameters": self.project.config.inversion.inversion_parameters,
                "output_location": str(output_location),
            }

        if tasks:
            RegularizationHelper(
                project=self.project, iteration_name=model.descriptor, tasks=tasks
            )

    def _g(self, model: ModelProxy, indices: Optional[List[int]] = None) -> Vec:
        """I can probably implement a version that figures out if all need to be done.
        Then it first computes all gradients and then smoothes all of them here.

        3 cases:
        mono-batch, batch=None, control_group=[]
        overlapping_batch: has a batch and a control group
        non-overlapping_batch: only a mini batch and no control group
        3"""

        raw_grad_f = self.project.paths.get_raw_gradient_path(
            model.descriptor, self._get_tag(model, indices)
        )
        if not raw_grad_f.exists():
            self._prepare_iteration(model)
            events = self._get_events_from_indices(indices)
            IterationListener(
                project=self.project,
                events=events,
                prev_control_group_events=self.previous_control_group(model),
                misfit_only=False,
                prev_iteration=self.project.previous_iteration,
                submit_adjoint=True,
            ).listen()

            GradientSummer(project=self.project).sum_gradients(
                events=events,
                output_location=raw_grad_f,
                batch_average=True,
                sum_vpv_vph=False,
                store_norms=True,
            )
            if self.smooth_gradients:
                self._write_smoothing_task(model, indices, raw_grad_f)

        return mesh_to_vector(
            raw_grad_f, self.project.config.inversion.inversion_parameters
        )

    @CallCounter
    def g(self, model: ModelProxy, indices: Optional[List[int]] = None) -> Vec:
        """I can probably implement a version that figures out if all need to be done.
        Then it first computes all gradients and then smoothes all of them here.

        3 cases:
        mono-batch, batch=None, control_group=[]
        overlapping_batch: has a batch and a control group
        non-overlapping_batch: only a mini batch and no control group
        3"""

        # First ensure the batch gradient is there.
        if model.descriptor not in self.completed_full_batches:
            self._g(model, model.batch or None)

        # Collect all the relevant raw grads.
        for index_set in self.get_remaining_batch_indices(model):
            self._g(model, index_set)

        batch_tag = self._get_tag(model, indices)
        grad_f = self.project.paths.get_raw_gradient_path(model.descriptor, batch_tag)

        if self.smooth_gradients:
            reg_helper = RegularizationHelper(
                project=self.project,
                iteration_name=model.descriptor,
                tasks=False,
            )
            reg_helper.monitor_tasks()
            grad_f = self.project.paths.get_smooth_gradient_path(
                model_descriptor=model.descriptor, tag=batch_tag
            )
        scale_fac = 1e10
        return scale_fac * mesh_to_vector(
            grad_f, self.project.config.inversion.inversion_parameters
        )
