"""
A class which takes care of all salvus opt communication.
Currently I have only used salvus opt in exodus mode so I'll
initialize it as such but change into hdf5 once I can.
"""

from .component import Component
import toml
import os
import subprocess
import shutil
import sys

from inversionson import InversionsonError
from inversionson import InversionsonOptError
import numpy as np


class SalvusOptComponent(Component):
    """
    Communications with Salvus Opt
    """

    def __init__(self, communicator, component_name):
        super(SalvusOptComponent, self).__init__(communicator, component_name)
        self.path = self.comm.project.paths["salvus_opt"]
        self.task_toml = os.path.join(self.path, "task.toml")
        self.models = os.path.join(self.path, "PHYSICAL_MODELS")
    
    def run_salvus_opt(self):
        """
        Run salvus opt to get next task. I think this should work well enough.
        """
        run_script = os.path.join(self.path, "run_salvus_opt.sh")
        if not os.path.exists(run_script):
            raise InversionsonError("Please create a shell script to run "
                                    "Salvus opt in your opt folder.")
        os.chdir(self.path)
        run_script = f"sh {run_script}"
        process = subprocess.Popen(
            run_script, shell=True, stdout=subprocess.PIPE, bufsize=1)
        for line in process.stdout:
            print(line, end="\n", flush=True)
            #sys.stdout.write(line)
        process.wait()
        print(process.returncode)
        os.chdir(self.comm.project.inversion_root)
        # subprocess.call([path_to_run_script])

    def read_salvus_opt(self) -> dict:
        """
        Read the task that salvus opt has issued into a dictionary

        :return: The information contained in the task toml file
        :rtype: dictionary
        """
        if os.path.exists(os.path.join(self.path, "task.toml")):
            task = toml.load(os.path.join(self.path, "task.toml"))
            return task
        else:
            raise InversionsonError("no_task_toml")
    
    def read_salvus_opt_task(self) -> str:
        """
        Read the task from salvus opt. See what to do next
        
        :return: task name
        :rtype: str
        """
        if os.path.exists(os.path.join(self.path, "task.toml")):
            task = toml.load(os.path.join(self.path, "task.toml"))
            task_type = task["task"][0]["type"]
            verbose = task["task"][0]["_meta"]["verbose"]
            return task_type, verbose
        else:
            raise InversionsonError("no task toml")

    def close_salvus_opt_task(self):
        """
        Label the salvus_opt task as closed when it has been performed
        """
        task = self.read_salvus_opt()
        task["task"][0]["status"]["open"] = False
        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    # def move_gradient_to_salvus_opt_folder(self, event=None):
    #     """
    #     Move the gradient into the folder where salvus opt operates.
    #     This can either be an individual gradient (not implemented yet),
    #     or a full iteration gradient. (Exodus gradients for now)

    #     :param event: Name of event if individual gradient, defaults to None
    #     :type event: string, optional
    #     """
    #     iteration = self.comm.project.current_iteration
    #     grad_name = "gradient_" + iteration + ".e"
    #     gradient = self.comm.lasif.find_gradient(
    #         iteration=iteration,
    #         event=
    #     )
    #     dest_path = os.path.join(self.models, grad_name)
    #     current_path = os.path.join(self.comm.project.lasif_root, "GRADIENTS",
    #                                 "ITERATION_" + iteration, "gradient.e")

    #     shutil.copy(current_path, dest_path)

    def get_model_from_salvus_opt(self):
        """
        Move new model from salvus_opt location to lasif location
        """
        iteration = self.comm.project.current_iteration
        model_path = os.path.join(self.models, iteration + ".e")
        lasif_path = os.path.join(self.comm.project.lasif_root, "MODELS", "ITERATION_" +
                                  iteration, iteration + ".e")

        shutil.copy(model_path, lasif_path)

    def get_model_path(self, gradient=False) -> str:
        """
        Get path of model related to iteration

        :param gradient: Is it a gradient?
        :type gradient: bool
        """
        iteration = self.comm.project.current_iteration
        if gradient:
            return os.path.join(self.models, "gradient_" + iteration + ".e")
        else:
            return os.path.join(self.models, iteration + ".e")

    def write_misfit_to_task_toml(self, events=None):
        """
        Report the correct misfit value to salvus opt.
        """
        iteration = self.comm.project.current_iteration
        if not events:
            events = self.comm.project.events_in_iteration
        misfits = toml.load(os.path.join(self.comm.project.lasif_root,
                                         f"ITERATION_{iteration}",
                                         "misfits.toml"))
        events_list = []
        task = self.read_salvus_opt()

        for event in events:
            misfit = misfits["event_misfits"][event]
            events_list.append({
                "misfit": float(misfit),
                "name": event
            })

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    def write_gradient_to_task_toml(self):
        """
        Give salvus opt the path to the iteration gradient.
        Currently only for a single summed gradient, might change.
        Make sure you move gradient to salvus opt directory first.
        """
        iteration = self.comm.project.current_iteration
        events_used = self.comm.project.events_in_iteration
        events_list = []
        task = self.read_salvus_opt()
        for event in events_used:
            grad_path = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=True
            )
            events_list.append({
                "gradient": grad_path,
                "name": event
            })
        task["task"][0]["output"]["event"] = events_list

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    def write_misfit_and_gradient_to_task_toml(self):
        """
        Write misfit and gradient to task toml
        """
        iteration = self.comm.project.current_iteration
        events_used = self.comm.project.events_in_iteration
        misfits = toml.load(os.path.join(self.comm.project.lasif_root,
                                         "ITERATIONS",
                                         f"ITERATION_{iteration}",
                                         "misfits.toml"))
        events_list = []
        task = self.read_salvus_opt()
        for event in events_used:
            grad_path = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=True
            )

            events_list.append({
                "gradient": grad_path,
                "misfit": float(misfits["event_misfits"][event]),
                "name": event
            })
        task["task"][0]["output"]["event"] = events_list

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)
    
    def write_control_group_to_task_toml(self, control_group: list):
        """
        Report the optimally selected control group to salvus opt
        by writing it into the task toml
        
        :param control_group: List of events selected for control group
        :type control_group: list
        """
        events_used = self.comm.project.events_in_iteration
        task = self.read_salvus_opt()
        print(f"Events used: {events_used}")
        
        events_list = []
        ctrl = False
        for event in events_used:
            if event in control_group:
                ctrl = True
            events_list.append({
                "control-group": ctrl,
                "name": event
            })
            ctrl = False
        task["task"][0]["output"]["event"] = events_list

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    def get_newest_iteration_name(self):
        """
        Get the name of the newest iteration given by salvus opt.
        We will look for the iteration with the highest number
        and if there are multiple we look at the one with the smallest
        trust region.
        """
        models = []
        for r, d, f in os.walk(self.models):
            for file in f:
                if 'gradient' not in file:
                    models.append(file)

        if len(models) == 0:
            raise InversionsonOptError("Please initialize inversion in Salvus Opt")
        iterations = self._parse_model_files(models)

        new_it_number = max(iterations)
        new_it_tr_region = min(iterations[new_it_number])

        return self._create_iteration_name(new_it_number, new_it_tr_region)
    
    def get_previous_iteration_name(self, tr_region=False):
        """
        Get the name of the previous iteration in order to find
        information needed from previous iteration.
        """
        models = []
        for r, d, f in os.walk(self.models):
            for file in f:
                if 'gradient' not in file:
                    models.append(file)

        if len(models) == 0:
            raise InversionsonError("Please initialize inversion in Salvus Opt")
        iterations = self._parse_model_files(models)

        if not tr_region:
            if max(iterations) == 0:
                return "it0000_model"
            old_it_number = max(iterations) - 1
            old_it_tr_region = min(iterations[old_it_number])
            return self._create_iteration_name(old_it_number, old_it_tr_region)
        else:
            it_number = max(iterations)
            # Currently I only use this to get old events, any old tr_region
            # will work for that task
            if len(iterations[it_number]) == 1:
                msg = f"Only one trust region for iteration {it_number}"
                raise InversionsonError(msg)
            tr_region = max(iterations[it_number])
            return self._create_iteration_name(it_number, tr_region)

    def first_trial_model_of_iteration(self) -> bool:
        """
        In order to distinguish between the first trust region test and the
        coming ones, this function returns true if there is only one model
        existing for the newest.
        """
        models = []
        for r, d, f in os.walk(self.models):
            for file in f:
                if 'gradient' not in file:
                    models.append(file)

        if len(models) == 0:
            raise InversionsonError("Please initialize inversion in Salvus Opt")
        iterations = self._parse_model_files(models)
        if len(iterations[max(iterations)]) == 1:
            return True
        else:
            if max(iterations) == 0:
                if len(iterations[max(iterations)]) == 2:
                    return True
            return False

    def get_batch_size(self) -> int:
        """
        Get the size of the batch to be used in an iteration.
        Batch size is double the control group
        
        :return: Number of sources to use.
        :rtype: int
        """
        task = self.read_salvus_opt()
        events = task["task"][0]["output"]["event"]
        return len(events) * 2

    def find_blocked_events(self):
        """
        Initially, in order to use all events. Each event which has been used
        once or more often is blocked until all events have been used.
        After that stage, the only blocked events become the ones which have
        been used in the previous iteration but were not in the control group.

        :return: Gives two lists, blocked events, and events to use. The
        suggested events to use is only given when there are fewer events
        available than needed. Otherwise the second output is None
        :rtype: list, list
        """
        blocked_events = []
        events_used = self.comm.storyteller.events_used  # Usage of all events
        needed_events = int(round(self.get_batch_size() / 2))
        for key, val in events_used.items():
            if val != 0:
                blocked_events.append(key)
        if abs(len(blocked_events) - len(events_used.keys())) >= needed_events:
            # We still have plenty of events to choose from.
            use_these = None
            return blocked_events, use_these
        # else:
        #     # We have fewer events to choose from
        #     prev_iter = self.get_previous_iteration_name()
        #     prev_it_toml = os.path.join(
        #         self.comm.project.paths["iteration_tomls"],
        #         f"{prev_iter}.toml"
        #     )
        #     prev_it_dict = toml.load(prev_it_toml)
        if len(blocked_events) == len(events_used.keys()):
            use_these = None
        else:
            use_these = list(set(events_used.keys()) - set(blocked_events))
        # The only blocked events are the events used in last iteration.
        # Lets remove that feature and keep them available for selection

        blocked_events = []
        # for key in prev_it_dict["events"]:
        #     if key not in prev_it_dict["new_control_group"]:
        #         blocked_events.append(key)
        return blocked_events, use_these

    def _parse_model_files(self, models: list) -> dict:
        """
        Read model file names from salvus opt and return a dict
        with iteration numbers and corresponding trust regions.

        :param models: A list of models inside the model directory
        :type models: list
        :return: Dictionary of iteration numbers and trust regions
        :rtype: dict
        """
        iterations = {}

        for model in models:
            if not model[-4:] == "xdmf":
                if len(model) < 17:
                    # The first iteration is shorter
                    iteration = int(model[2:6])
                    tr_region = 9.999999
                    iterations[iteration] = [tr_region]
                else:
                    iteration = int(model[2:6])
                    tr_region = float(model[-11:-3])
                    if iteration in iterations:
                        iterations[iteration].append(tr_region)
                    else:
                        iterations[iteration] = [tr_region]
        
        return iterations

    def _create_iteration_name(self, number: int, tr_region: float) -> str:
        """
        Create an iteration name in the salvus opt format based on
        number of iteration and it's trust region.
        
        :param number: Number of iteration
        :type number: int
        :param tr_region: Trust region length
        :type tr_region: float
        :return: Name of iteration
        :rtype: str
        """
        num_part = str(number)
        while len(num_part) < 4:
            num_part = '0' + num_part
        if tr_region == 9.999999:
            return "it" + num_part + "_model"
        
        tr_region_part = str(tr_region)

        return "it" + num_part + "_model_TrRadius_" + tr_region_part
