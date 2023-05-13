import os
from typing import List, Optional
import numpy as np
import json

from inversionson.helpers.autoinverter_helpers import IterationListener
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.helpers.regularization_helper import RegularizationHelper
from inversionson.utils import write_xdmf
from optson.problem import AbstractStochasticProblem
from lasif.components.communicator import Communicator
from inversionson import InversionsonError
from inversionson.optimizers.optson import OptsonLink
from optson.vector import OptsonVec
from numpy.typing import ArrayLike

from optson.preconditioner import AbstractPreconditioner


class InnerProductPrecondtioner(AbstractPreconditioner):
    def __init__(self, optlink, misfit_scaling_fac: float = 1):
        self.optlink = optlink
        self.misfit_scaling_fac = misfit_scaling_fac

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return (
            self.optlink.get_mm()
            / self.misfit_scaling_fac
            * self.optlink.get_mref() ** 2
            * x
        )


def get_set_flag(x: OptsonVec, it_num: int, control_group: bool):
    if not control_group:
        return "mb"
    elif it_num < x.iteration:
        return "cg_prev"
    else:
        return "cg"


# class AbstractProblem(metaclass=OptsonMeta):
#     """
#     Abstract baseproblem for regular problems.
#     """
#
#     def __init__(
#         self,
#         H: InstanceOrType[AbstractHessian] = LBFGSHessian,
#         preconditioner: InstanceOrType[AbstractPreconditioner] = IdentityPreconditioner,
#     ):
#         self.H = get_instance(H)
#         self.preconditioner = get_instance(preconditioner)
#         self.call_counters: Dict[str, int] = {}
#
#     @abstractmethod
#     @CallCounter
#     def f(self, x: OptsonVec) -> float:
#         raise NotImplementedError
#
#     @abstractmethod
#     @CallCounter
#     def g(self, x: OptsonVec) -> OptsonVec:
#         raise NotImplementedError
#
#     @CallCounter
#     def f_cg(self, x: OptsonVec) -> float:
#         raise NotImplementedError
#
#     @CallCounter
#     def g_cg(self, x: OptsonVec) -> OptsonVec:
#         raise NotImplementedError
#
#     @CallCounter
#     def f_cg_previous(self, x: OptsonVec) -> float:
#         raise NotImplementedError
#
#     @CallCounter
#     def g_cg_previous(self, x: OptsonVec) -> OptsonVec:
#         raise NotImplementedError


class StochasticFWI(AbstractStochasticProblem):
    def __init__(
        self,
        comm: Communicator,
        optlink: OptsonLink,
        batch_size=2,
        gradient_test=False,
        status_file="optson_status_tracker.json",
        task_file="optson_task_tracker.json",
    ):
        super().__init__(batchManager=self)  # Simply point tre batchManager to self
        self.comm = comm
        self.optlink = optlink
        self.gradient_test = gradient_test
        self.status_file = status_file
        self.task_file = task_file
        self.misfit_scaling_fac = 1e4
        self.preconditioner = InnerProductPrecondtioner(
            optlink=optlink, misfit_scaling_fac=self.misfit_scaling_fac
        )

        # All these things need to be cached
        self.mini_batch_dict = {}
        self.control_group_dict = {}
        self.batch_size = batch_size
        # list model names per iteration here to figure out the accepted and rejected models
        self.model_names = {}
        self.deleted_iterations = []
        self.performed_tasks = []
        self.read_status_json()
        self.speculative_adjoints = True
        self.speculative_forwards = True

    def read_status_json(self):
        if os.path.exists(self.task_file):
            with open(self.task_file, "r") as fh:
                task_dict = json.load(fh)
            self.performed_tasks = task_dict["performed_tasks"]
            if "deleted_iterations" in task_dict.keys():
                self.deleted_iterations = task_dict["deleted_iterations"]

        if os.path.exists(self.status_file):
            with open(self.status_file, "r") as fh:
                status_dict = json.load(fh)

            self.mini_batch_dict = status_dict["mini_batch_dict"]
            self.control_group_dict = status_dict["control_group_dict"]
            self.batch_size = status_dict["batch_size"]
            self.model_names = status_dict["model_names"]

    def clean_files(self):
        # Delete anything from the prior iterations
        all_its = self.optlink.find_iteration_numbers()
        for it in all_its[:-2]:
            for model in self.model_names[str(it)]:
                if model in self.deleted_iterations:
                    continue
                self.optlink.delete_remote_files(model)
                self.deleted_iterations.append(model)
                self.update_status_json()

        # Delete rejected models in most recent iterations.
        for it in all_its[-2:]:
            if str(it) in self.model_names.keys():
                for model in self.model_names[str(it)][:-1]:
                    if model in self.deleted_iterations:
                        continue
                    self.optlink.delete_remote_files(model)
                    self.deleted_iterations.append(model)
                    self.update_status_json()

    def update_status_json(self):
        """
        Store all job status in this mega file
        """
        task_dict = dict(
            performed_tasks=self.performed_tasks,
            deleted_iterations=self.deleted_iterations,
        )
        status_dict = dict(
            mini_batch_dict=self.mini_batch_dict,
            control_group_dict=self.control_group_dict,
            batch_size=self.batch_size,
            model_names=self.model_names,
        )
        with open(self.task_file, "w") as fh:
            json.dump(task_dict, fh)
        with open(self.status_file, "w") as fh:
            json.dump(status_dict, fh)

    def create_iteration_if_needed(self, x: OptsonVec):
        """
        Here, we create the iteration.
        A mini-batch is selected after which an iteration in LASIF
        is created and iteration_info is written to dik.
        """
        previous_control_group = []
        events = []
        self.optlink.current_iteration_name = x.descriptor

        if not os.path.exists(self.optlink.model_path):
            self.optlink.vector_to_mesh_new(self.optlink.model_path, x)
        # self.optlink.vector_to_mesh_new(self.optlink.model_path, m)
        if not self.comm.lasif.has_iteration(x.descriptor):
            if x.md.iteration > 0:
                previous_control_group = self.control_group_dict[
                    str(x.md.iteration - 1)
                ]
            events = self.optlink.pick_data_for_iteration(
                batch_size=self.batch_size, prev_control_group=previous_control_group
            )

        # the below line will set the proper parameters in the project
        # component
        self.optlink.prepare_iteration(iteration_name=x.descriptor, events=events)
        if x.md.iteration not in self.model_names:
            self.model_names[str(x.md.iteration)] = [x.descriptor]
        elif x.descriptor not in self.model_names[str(x.md.iteration)]:
            self.model_names[str(x.md.iteration)].append(x.descriptor)
        self.mini_batch_dict[
            str(x.md.iteration)
        ] = self.comm.project.events_in_iteration
        self.update_status_json()

    def select_batch(self, x: OptsonVec):
        self.create_iteration_if_needed(x=x)

    def _cancel_useless_adjoint_runs_if_needed(self, x: OptsonVec):
        if len(self.model_names[str(x.iteration)]) > 1:
            iter_name = self.model_names[str(x.iteration)][-2]
            # There was a failed iteration. Try cancelling jobs here
            for event in self.comm.project.events_in_iteration:
                if self.comm.comm.project.is_validation_event(event):
                    continue
                try:
                    job = self.comm.salvus_flow.get_job(
                        event=event, im_type="adjoint", iteration=iter_name
                    )
                    job.cancel()
                except (KeyError, InversionsonError):
                    continue

    def _f(
        self,
        x: OptsonVec,
        events: List[str],
        control_group_events: Optional[List[str]] = None,
        previous_control_group_events: Optional[List[str]] = None,
        previous_iteration: Optional[str] = None,
        compute_misfit_only: bool = True,
    ) -> float:
        self.create_iteration_if_needed(x=x)

        if self.speculative_adjoints:
            self._cancel_useless_adjoint_runs_if_needed(x)

        submit_adjoint = not compute_misfit_only or self.speculative_adjoints
        submission_events = (
            self.comm.project.events_in_iteration
            if self.speculative_forwards
            else events
        )

        it_listen = IterationListener(
            comm=self.comm,
            events=submission_events,
            control_group_events=control_group_events,
            prev_control_group_events=previous_control_group_events,
            misfit_only=compute_misfit_only,
            prev_iteration=previous_iteration,
            submit_adjoint=submit_adjoint,
        )
        it_listen.listen()
        # Get iteration attributes just in case.
        self.comm.project.get_iteration_attributes(x.descriptor)

        # Compute mmisfit for relevant events
        blocked_data = set(
            self.comm.project.validation_dataset + self.comm.project.test_dataset
        )
        misfit_events = set(events) - blocked_data
        total_misfit = 0.0
        for event in misfit_events:
            total_misfit += self.comm.project.misfits[event]
        self.update_status_json()
        return total_misfit / len(misfit_events)

    def f(self, x: OptsonVec) -> float:
        return self._f(x=x, control_group=False)

    def _g(self, x: OptsonVec, control_group: bool = False) -> np.array:
        self.clean_files()

        # Simply call the misfit function, but ensure we also compute the gradients.
        self._f(x=x, control_group=control_group, compute_misfit_only=False)

        set_flag = self.get_set_flag(x, it_num, control_group)
        raw_grad_file = self.optlink.get_raw_gradient_path(x.descriptor, set_flag)
        # For gradient we only track the mb_task!
        task_name = self.get_task_name(x, it_num, "gradient", control_group)

        if task_name not in self.performed_tasks:
            # now we need to figure out how to sum the proper gradients.
            # for this we need the events
            if control_group:
                events = self.control_group_dict[str(it_num)]
            else:
                events = set(self.mini_batch_dict[str(it_num)]) - set(
                    self.comm.project.validation_dataset
                )
                events = list(events)

            blocked_data = set(
                self.comm.project.validation_dataset + self.comm.project.test_dataset
            )
            gradient_events = set(events) - blocked_data

            grad_summer = GradientSummer(comm=self.comm)
            store_norms = not self.gradient_test
            sum_grads = bool(self.optlink.isotropic_vp)

            grad_summer.sum_gradients(
                events=gradient_events,
                output_location=raw_grad_file,
                batch_average=True,
                sum_vpv_vph=sum_grads,
                store_norms=store_norms,
            )
            write_xdmf(raw_grad_file)
            self.performed_tasks.append(task_name)
            # The below line only writes the task into the smoothing file
            self.optlink.perform_smoothing(x, set_flag, raw_grad_file)
            self.update_status_json()

    def gradient(self, x: OptsonVec) -> OptsonVec:
        return self.get_gradient(x=x, it_num=x.iteration)

    def gradient_cg(self, x: OptsonVec) -> OptsonVec:
        self.select_control_group(x=x)
        return self.get_gradient(x=x, it_num=x.iteration, control_group=True)

    def gradient_cg_previous(self, x: OptsonVec) -> OptsonVec:
        return self.get_gradient(x=x, it_num=x.iteration - 1, control_group=True)

    def document_iteration(self, x: OptsonVec):
        # We need to figure out the accepted iteration
        accepted_iteration_name = self.model_names[str(x.iteration - 1)][-1]
        # We need to load the correct iteration attributes (with the correct iteration name)
        if self.comm.lasif.has_iteration(accepted_iteration_name):
            self.comm.project.change_attribute(
                "current_iteration", accepted_iteration_name
            )
            self.comm.project.get_iteration_attributes(
                iteration=accepted_iteration_name
            )
        else:
            raise Exception("This should not occur")
        # Update usage of events
        self.comm.storyteller.document_task(task="adam_documentation")

    def get_gradient(
        self, x: OptsonVec, it_num: int, control_group: bool = False
    ) -> np.array:
        if control_group and it_num < x.iteration:
            # CG prev triggers MB, CG and CG prev
            # Here, we can also document the previous iteration
            self.document_iteration(x=x)
            self.select_control_group(x)
            self._gradient(x, x.iteration, False)
            self._gradient(x, x.iteration, control_group=True)
            self._gradient(x, it_num, control_group)
        elif (
            not control_group
            and it_num == x.iteration
            and str(it_num) in self.control_group_dict.keys()
            and len(self.control_group_dict[str(it_num)]) > 0
        ):
            # MB triggers CG and MB
            self._gradient(x, it_num, control_group)
            self._gradient(x, it_num, control_group=True)
        else:
            # only trigger what is asked for.
            self._gradient(x, it_num, control_group)

        # Now we monitor all gradients in one go.
        reg_helper = RegularizationHelper(
            comm=self.comm,
            iteration_name=x.descriptor,
            tasks=False,
            optimizer=self.optlink,
        )
        reg_helper.monitor_tasks()

        set_flag = self.get_set_flag(x, it_num, control_group)
        raw_grad_file = self.optlink.get_raw_gradient_path(x.descriptor, set_flag)
        smooth_grad = self.optlink.get_smooth_gradient_path(
            x.descriptor, set_flag=set_flag
        )
        return self.misfit_scaling_fac * self.optlink.mesh_to_vector_new(
            smooth_grad, gradient=True, raw_grad_file=raw_grad_file
        )

    def select_control_group(self, x: Vector):
        if str(x.iteration) in self.control_group_dict.keys():
            return

        current_batch = self.mini_batch_dict[str(x.iteration)]

        blocked_data = set(
            self.comm.project.validation_dataset + self.comm.project.test_dataset
        )

        all_events = set(current_batch) - blocked_data
        control_group = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=list(all_events),
            select_new_control_group=True,
        )
        self.control_group_dict[str(x.iteration)] = control_group
        self.update_status_json()

    def extend_control_group(self, x: Vector) -> bool:
        current_batch = self.mini_batch_dict[str(x.iteration)]
        current_control_group = self.control_group_dict[str(x.iteration)]
        non_control_events = set(current_batch) - set(current_control_group)
        blocked_data = set(
            self.comm.project.validation_dataset + self.comm.project.test_dataset
        )
        non_control_events = non_control_events - blocked_data

        non_validation_mb_events = set(current_batch) - blocked_data
        if len(current_control_group) == len(non_validation_mb_events):
            return False
        print("Extending Control group...")
        additional_controls = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=list(non_control_events),
            select_new_control_group=True,
        )
        self.control_group_dict[str(x.iteration)] = (
            current_control_group + additional_controls
        )

        all_events = self.comm.lasif.list_events()
        blocked_data = set(
            self.comm.project.validation_dataset + self.comm.project.test_dataset
        )
        all_events = list(set(all_events) - blocked_data)
        self.batch_size = min(
            2 * len(self.control_group_dict[str(x.iteration)]), len(all_events)
        )

        task_name_grad = self.get_task_name(
            x, x.iteration, "gradient", control_group=True
        )

        task_name_misfit = self.get_task_name(
            x, x.iteration, "misfit", control_group=True
        )

        # Ensure removal of all occurences. We should only have one occurence though.
        while task_name_misfit in self.performed_tasks:
            self.performed_tasks.remove(task_name_misfit)
        while task_name_grad in self.performed_tasks:
            self.performed_tasks.remove(task_name_grad)
        self.update_status_json()
        # Also delete gradients
        raw_grad = self.optlink.get_raw_gradient_path(x.descriptor, "cg")
        if os.path.exists(raw_grad):
            os.remove(raw_grad)
        smooth_grad = self.optlink.get_smooth_gradient_path(x.descriptor, "cg")
        if os.path.exists(smooth_grad):
            os.remove(smooth_grad)
        return True
