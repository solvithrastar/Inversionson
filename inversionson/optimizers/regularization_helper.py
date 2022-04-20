import os
import toml

from pathlib import Path
from salvus.flow import api as sapi
from salvus.opt.smoothing import get_smooth_model
from inversionson.utils import sleep_or_process


class RegularizationHelper(object):
    """
    This class can get a list of tasks that require smoothing,
    dispatch the tasks and monitor and retrieve them.
    """

    def __init__(self, comm, iteration_name, tasks):
        """
        Each tasks is a dict that has a reference model, a model that contains the fields
        that require smoothing, the smoothing lengths, the parameters that require
        smoothing and the output location to which the smoothed parameters
        should be retrieved

        tasks = {task_name: {"reference_model": str, "model_to_smooth": str,
                "smoothing_lengths": list, "smoothing_parameters": list,
                "output_location": str}, "..." : {...}}
        :param:
        """
        self.comm = comm
        self.site_name = self.comm.project.smoothing_site_name
        self.max_reposts = 3
        self.iteration_name = iteration_name
        self.regularization_folder = (
            Path(self.comm.project.paths["inversion_root"]) / "REGULARIZATION"
        )
        if not os.path.exists(self.regularization_folder):
            os.mkdir(self.regularization_folder)
        self.write_tasks(tasks)
        self.tasks = toml.load(self.get_iteration_toml_filename())

    def get_iteration_toml_filename(self):
        return self.regularization_folder / (self.iteration_name +
                                             "_regularization.toml")

    def write_tasks(self, tasks):
        """
        This function writes the tasks to file or updates the task file.

        :param tasks = Dictionary of dictionaries with the relevant info, where
        for the smoothing jobs. Each key in the outer dictionary represents a task.
        :type tasks: Dict
        """
        # Write initial toml if there is no task toml yet
        if not os.path.exists(self.get_iteration_toml_filename()):
            for task_dict in tasks.values():
                task_dict["job_name"] = ""
                task_dict["submitted"] = False
                task_dict["retrieved"] = False
                task_dict["reposts"] = 0
            with open(self.get_iteration_toml_filename(), "w") as fh:
                toml.dump(tasks, fh)

        else:  # We add the tasks to the existing tasks if needed
            existing_tasks = toml.load(self.get_iteration_toml_filename())
            for task_name, task in tasks.items():
                # Add the empty task if it does not exist
                if task_name not in existing_tasks.keys():
                    existing_tasks[task_name] = tasks[task_name]
                    existing_tasks[task_name]["job_name"] = ""
                    existing_tasks[task_name]["submitted"] = False
                    existing_tasks[task_name]["retrieved"] = False
                    existing_tasks[task_name]["reposts"] = 0
                else:  # Update existing tasks with passed tasks
                    existing_tasks[task_name] = tasks[task_name]
            with open(self.get_iteration_toml_filename(), "w") as fh:
                toml.dump(existing_tasks, fh)

    def dispatch_smoothing_tasks(self):
        """
        Dispatches tasks that are not yet dispatched.
        """
        for task_name, task_dict in self.tasks.items():
            if not task_dict["submitted"] and task_dict["reposts"] < self.max_reposts:
                sims = self.comm.smoother.run_remote_smoother_for_model(
                    reference_model=task_dict["reference_model"],
                    model_to_smooth=task_dict["model_to_smooth"],
                    smoothing_lengths=task_dict["smoothing_lengths"],
                    smoothing_parameters=task_dict["smoothing_parameters"])

                job = sapi.run_many_async(
                    input_files=sims,
                    site_name=self.comm.project.smoothing_site_name,
                    ranks_per_job=self.comm.project.smoothing_ranks,
                    wall_time_in_seconds_per_job=self.comm.project.smoothing_wall_time,
                )
                self.tasks[task_name]["submitted"] = True
                self.tasks[task_name]["job_name"] = job.job_array_name
                self.write_tasks(self.tasks)

    def update_tasks_and_retrieve(self):
        for task_dict in self.tasks.values():
            job = sapi.get_job_array(job_array_name=task_dict["job_name"],
                                     site_name=self.site_name)
            status = job.update_status(force_update=True)
            finished = True
            for _i, s in enumerate(status):
                if s.name != "finished":
                    finished = False
                elif s.name in ["unknown", "failed"]:
                    task_dict["reposts"] += 1
                    task_dict["submitted"] = False
                    self.write_tasks(self.tasks)
            if finished:
                smooth_gradient = get_smooth_model(
                    job=job,
                    model=task_dict["reference_model"],
                )
                smooth_gradient.write_h5(task_dict["output_location"])
                task_dict["retrieved"] = True
                self.write_tasks(self.tasks)

    def all_retrieved(self):
        for task_dict in self.tasks.values():
            if not task_dict["retrieved"]:
                return False
        return True

    def monitor_tasks(self):
        """
        This functions monitors the tasks, resubmits them if they fail
        or retrieves them
        """
        # Dispatch tasks that are not dispatched yet first.
        self.dispatch_smoothing_tasks()
        while not self.all_retrieved():
            print("Attempting to retrieve smoothing jobs.")
            self.update_tasks_and_retrieve()
            self.dispatch_smoothing_tasks()
            sleep_or_process(self.comm)
