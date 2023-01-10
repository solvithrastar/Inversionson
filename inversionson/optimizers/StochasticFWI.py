import os
import numpy as np
import json

from inversionson.helpers.autoinverter_helpers import IterationListener
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.helpers.regularization_helper import RegularizationHelper
from inversionson.utils import write_xdmf
from optson.base_classes.base_problem import StochasticBaseProblem
from optson.base_classes.model import ModelStochastic
from lasif.components.communicator import Communicator
from inversionson import InversionsonError
from inversionson.optimizers.optson import OptsonLink


class StochasticFWI(StochasticBaseProblem):
    def __init__(self, comm: Communicator,
                 optlink: OptsonLink,
                 batch_size=2,
                 gradient_test=False,
                 status_file="optson_status_tracker.json"):
        self.comm = comm
        self.optlink = optlink
        self.gradient_test = gradient_test
        self.status_file = status_file

        # All these things need to be cached
        self.mini_batch_dict = {}
        self.control_group_dict = {}
        self.batch_size = batch_size
        # list model names per iteration here to figure out the accepted and rejected models
        self.model_names = {}
        self.deleted_iterations = []
        self.performed_tasks = [] # tasks will be a dict with as a key, the name of the iteration, the type of job,
        self.read_status_json()
        self.misfit_scaling_fac = 1e4

    @staticmethod
    def get_set_flag(m: ModelStochastic, it_num: int, control_group: bool):
        if not control_group:
            set_flag = "mb"
        elif it_num < m.iteration_number:
            set_flag = "cg_prev"
        else:
            set_flag = "cg"
        return set_flag

    def get_task_name(self, m: ModelStochastic, it_num: int, job_type: str,
                      control_group: bool):
        set_flag = self.get_set_flag(m, it_num, control_group)
        return f"{m.name}_{it_num}_{job_type}_{set_flag}_completed"

    def read_status_json(self):
        if os.path.exists(self.status_file):
            with open(self.status_file, "r") as fh:
                status_dict = json.load(fh)

            self.mini_batch_dict = status_dict["mini_batch_dict"]
            self.control_group_dict = status_dict["control_group_dict"]
            self.batch_size = status_dict["batch_size"]
            self.performed_tasks = status_dict["performed_tasks"]
            if "deleted_iterations" in status_dict.keys():
                self.deleted_iterations = status_dict["deleted_iterations"]
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
        status_dict = dict(mini_batch_dict=self.mini_batch_dict,
                           control_group_dict=self.control_group_dict,
                           batch_size=self.batch_size,
                           model_names=self.model_names,
                           performed_tasks=self.performed_tasks,
                           deleted_iterations=self.deleted_iterations)

        with open(self.status_file, "w") as fh:
            json.dump(status_dict, fh)

    def create_iteration_if_needed(self, m: ModelStochastic):
        """
        Here, we create the iteration if needed. This also requires us
        to select the mini_batch already, but that is no problem.
        """
        previous_control_group = []
        events = []
        self.optlink.current_iteration_name = m.name

        if not os.path.exists(self.optlink.model_path):
            self.optlink.vector_to_mesh_new(self.optlink.model_path, m)
        # self.optlink.vector_to_mesh_new(self.optlink.model_path, m)
        if not self.comm.lasif.has_iteration(m.name):
            if m.iteration_number > 0:
                previous_control_group = \
                    self.control_group_dict[str(m.iteration_number-1)]
            events = self.optlink.pick_data_for_iteration(
                batch_size=self.batch_size,
                prev_control_group=previous_control_group)

        # the below line will set the proper parameters in the project
        # component
        self.optlink.prepare_iteration(iteration_name=m.name, events=events)
        if m.iteration_number not in self.model_names:
            self.model_names[str(m.iteration_number)] = [m.name]
        else:
            if m.name not in self.model_names[str(m.iteration_number)]:
                self.model_names[str(m.iteration_number)].append(m.name)
        self.mini_batch_dict[str(m.iteration_number)] = self.comm.project.events_in_iteration
        self.update_status_json()

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

        job_type = "misfit" if misfit_only else "gradient"
        task_name = self.get_task_name(m, it_num, job_type, control_group)
        mb_task_name = self.get_task_name(m, m.iteration_number,
                                          job_type, control_group=False)

        submit_adjoint = (mb_task_name == task_name)
        mb_completed = (mb_task_name in self.performed_tasks)
        events = self.control_group_dict[str(it_num)] if control_group else self.comm.project.events_in_iteration

        # With the below option, we always just submit all events
        if self.optlink.speculative_forwards:
            submission_events = self.comm.project.events_in_iteration
        else:
            submission_events = events

        speculative_adjoints = True
        if speculative_adjoints and control_group and self.optlink.speculative_forwards:
            submit_adjoint = True
            if len(self.model_names[str(m.iteration_number)]) > 1:
                iter_name = self.model_names[str(m.iteration_number)][-2]
                # There was a failed iteration. Try cancelling jobs here
                for event in self.comm.project.events_in_iteration:
                    if self.comm.comm.project.is_validation_event(event):
                        continue
                    else:
                        try:
                            job = self.comm.salvus_flow.get_job(
                                event=event, im_type="adjoint", iteration=iter_name)
                            job.cancel()
                        except (KeyError, InversionsonError):
                            continue
                    

        blocked_data = \
            set(self.comm.project.validation_dataset + self.comm.project.test_dataset)
        misfit_events = set(events) - blocked_data

        if task_name not in self.performed_tasks and not mb_completed:
            if m.iteration_number > 0 and m.iteration_number > it_num:
                # if it not the first model,
                # we need to consider the previous control group
                prev_control_group = self.control_group_dict[str(it_num)]
                previous_iteration = self.model_names[str(it_num)][-1]

            # Else we just take everything and immediately also compute the
            # gradient
            it_listen = IterationListener(
                comm=self.comm,
                events=submission_events,
                control_group_events=control_group_events,
                prev_control_group_events=prev_control_group,
                misfit_only=misfit_only,
                prev_iteration=previous_iteration,
                submit_adjoint=submit_adjoint
            )
            it_listen.listen()
            if job_type == "misfit":
                self.performed_tasks.append(task_name) # only do this after summing for gradients

        # Now we need a way to actually collect the misfit for the events.
        # this involves the proper name for the mdoel and the set of events.
        self.comm.project.get_iteration_attributes(m.name)
        total_misfit = 0.0
        for event in misfit_events:
            total_misfit += self.comm.project.misfits[event]
        self.update_status_json()
        return total_misfit / len(misfit_events)

    def misfit(
        self, m: ModelStochastic, it_num: int, control_group: bool = False,
    ) -> float:
        """
        We may want some identifier to say which solution vector x is used.
        Things like model_00000_step_... or model_00000_TrRadius_....
        # TODO cache these results as well.
        """
        self.clean_files()
        if control_group:
            return self._misfit(m=m, it_num=it_num, control_group=control_group,
                                misfit_only=True)
        else:
            return self._misfit(m=m, it_num=it_num, control_group=control_group,
                                misfit_only=False)

    def _gradient(
        self, m: ModelStochastic, it_num: int, control_group: bool = False
    ) -> np.array:
        self.clean_files()

        # Simply call the misfit function, but ensure we also compute the gradients.
        self._misfit(m=m, it_num=it_num, control_group=control_group,
                    misfit_only=False)

        set_flag = self.get_set_flag(m, it_num, control_group)
        raw_grad_file = self.optlink.get_raw_gradient_path(m.name, set_flag)
        # For gradient we only track the mb_task!
        task_name = self.get_task_name(m, it_num, "gradient", control_group)

        sum_grads = True if self.optlink.isotropic_vp else False

        if task_name not in self.performed_tasks:
            # now we need to figure out how to sum the proper gradients.
            # for this we need the events
            if control_group:
                events = self.control_group_dict[str(it_num)]
            else:
                events = set(self.mini_batch_dict[str(it_num)]) - set(self.comm.project.validation_dataset)
                events = list(events)

            blocked_data = \
                set(
                    self.comm.project.validation_dataset + self.comm.project.test_dataset)
            gradient_events = set(events) - blocked_data

            grad_summer = GradientSummer(comm=self.comm)
            store_norms = False if self.gradient_test else True
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
            self.optlink.perform_smoothing(m, set_flag, raw_grad_file)
            self.update_status_json()

        
        # smooth_grad = self.optlink.get_smooth_gradient_path(m.name, set_flag=set_flag)
        # return self.optlink.mesh_to_vector_new(smooth_grad, gradient=True, raw_grad_file=raw_grad_file)


    def gradient(
            self, m: ModelStochastic, it_num: int, control_group: bool = False
        ) -> np.array:
        if control_group and it_num < m.iteration_number:
            # CG prev triggers MB, CG and CG prev
            self.select_control_group(m)
            self.update_status_json()
            self._gradient(m, m.iteration_number, False)
            self._gradient(m, m.iteration_number, control_group=True)
            self._gradient(m, it_num, control_group)
            
            if list(self.model_names.keys())[-1] == str(m.iteration_number):
                from optson.utils import angle_between
                set_flag_mb = self.get_set_flag(m, m.iteration_number, False)
                raw_grad_mb = self.optlink.get_raw_gradient_path(m.name, set_flag_mb)
                set_flag_cg = self.get_set_flag(m, m.iteration_number, True)
                raw_grad_cg = self.optlink.get_raw_gradient_path(m.name, set_flag_cg)
                v_mb = self.optlink.mesh_to_vector_new(mesh_filename=raw_grad_mb)
                v_cg = self.optlink.mesh_to_vector_new(mesh_filename=raw_grad_cg)
                angle = angle_between(v1=v_mb, v2=v_cg)
                print("angle between raw cg and mb gradients", angle)
                angle_threshold = 70.0

                if angle > angle_threshold:
                    self.extend_control_group(m=m)
                    self._gradient(m, m.iteration_number, control_group=True)
                if angle < 30.0:
                    latest_control_group_size = len(list(self.control_group_dict.values())[-1])
                    new_batch_size = int(1.5*latest_control_group_size)
                    if new_batch_size != self.batch_size:
                        print(f"Reducing batch size from {self.batch_size} to {new_batch_size}")
                        self.batch_size = new_batch_size
                    self.update_status_json()
            
        elif not control_group and it_num == m.iteration_number and \
                str(it_num) in self.control_group_dict.keys() and \
                len(self.control_group_dict[str(it_num)]) > 0:
            # MB triggers CG and MB
            self._gradient(m, it_num, control_group)
            self._gradient(m, it_num, control_group=True)
        else:
            # only trigger what is asked for.
            self._gradient(m, it_num, control_group)

        # Now we monitor all gradients in one go.
        reg_helper = RegularizationHelper(
            comm=self.comm, iteration_name=m.name, tasks=False,
            optimizer=self.optlink
        )
        reg_helper.monitor_tasks()

        set_flag = self.get_set_flag(m, it_num, control_group)
        raw_grad_file = self.optlink.get_raw_gradient_path(m.name, set_flag)
        smooth_grad = self.optlink.get_smooth_gradient_path(m.name,
                                                            set_flag=set_flag)
        # return what is asked for.
        return self.misfit_scaling_fac * self.optlink.mesh_to_vector_new(smooth_grad, gradient=True, raw_grad_file=raw_grad_file)

    def select_control_group(self, m: ModelStochastic):
        if str(m.iteration_number) in self.control_group_dict.keys():
            return

        current_batch = self.mini_batch_dict[str(m.iteration_number)]

        blocked_data = \
            set(self.comm.project.validation_dataset + self.comm.project.test_dataset)

        all_events = set(current_batch) - blocked_data
        control_group = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=list(all_events),
            select_new_control_group=True
        )
        self.control_group_dict[str(m.iteration_number)] = control_group
        self.update_status_json()

    def extend_control_group(self, m: ModelStochastic) -> bool:
        current_batch = self.mini_batch_dict[str(m.iteration_number)]
        current_control_group = self.control_group_dict[str(m.iteration_number)]
        non_control_events = set(current_batch) - set(current_control_group)
        blocked_data = \
            set(self.comm.project.validation_dataset + self.comm.project.test_dataset)
        non_control_events = non_control_events - blocked_data

        non_validation_mb_events = set(current_batch)- blocked_data
        if len(current_control_group) == len(non_validation_mb_events):
            return False
        print("Extending Control group...")
        additional_controls = self.optlink.pick_data_for_iteration(
            self.batch_size,
            current_batch=list(non_control_events),
            select_new_control_group=True
        )
        self.control_group_dict[str(m.iteration_number)] = \
            current_control_group + additional_controls

        all_events = self.comm.lasif.list_events()
        blocked_data = set(self.comm.project.validation_dataset +
                           self.comm.project.test_dataset)
        all_events = list(set(all_events) - blocked_data)
        self.batch_size = min(2 * len(self.control_group_dict[str(m.iteration_number)]), len(all_events))

        task_name_grad = self.get_task_name(m, m.iteration_number,
                                       "gradient", control_group=True)

        task_name_misfit = self.get_task_name(m, m.iteration_number,
                                       "misfit", control_group=True)

        # Ensure removal of all occurences. We should only have one occurence though.
        while task_name_misfit in self.performed_tasks:
            self.performed_tasks.remove(task_name_misfit)
        while task_name_grad in self.performed_tasks:
            self.performed_tasks.remove(task_name_grad)
        self.update_status_json()
        # Also delete gradients
        raw_grad = self.optlink.get_raw_gradient_path(m.name, "cg")
        if os.path.exists(raw_grad):
            os.remove(raw_grad)
        smooth_grad = self.optlink.get_smooth_gradient_path(m.name, "cg")
        if os.path.exists(smooth_grad):
            os.remove(smooth_grad)
        return True

    def get_mm(self):
        return self.optlink.get_mm() / self.misfit_scaling_fac * self.optlink.get_mref()**2
