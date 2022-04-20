import os
import time
import toml

from pathlib import Path
from salvus.flow import api as sapi
from salvus.opt.smoothing import get_smooth_model


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
        tasks is a dictionary of dictionaries, where each value represents
        a smoothing task and the key is the unique name.

        Each value or dictionary represents the inputs to the smoothing job
        which are the reference model, a model that needs smoothing,
        the smoothing lengths and parameters and the location to where
        the smoothed end result should be stored

        We then add their job name, number of reposts and submission, retrieved
        status.
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

        else: # We add the tasks to the old tasks if needed
            old_tasks = toml.load(self.get_iteration_toml_filename())
            for task_name, task in tasks.items():
                if task_name not in old_tasks.keys():
                    old_tasks[task_name] = tasks[task_name]
                    old_tasks[task_name]["job_name"] = ""
                    old_tasks[task_name]["submitted"] = False
                    old_tasks[task_name]["retrieved"] = False
                    old_tasks[task_name]["reposts"] = 0
            with open(self.get_iteration_toml_filename(), "w") as fh:
                toml.dump(old_tasks, fh)

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
            if finished:
                smooth_gradient = get_smooth_model(
                    job=task_dict["job_name"],
                    model=task_dict["reference_model"],
                )
                smooth_gradient.write_h5(task_dict["output_location"])
                task_dict["retrieved"] = True

    def all_retrieved(self):
        for task_dict in self.tasks.values():
            if not task_dict["retrieved"]:
                return False
        return True

    def monitor_tasks(self, sleep_time_in_s=30):
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
            print(f"Sleeping for {sleep_time_in_s} seconds before trying again.")
            time.sleep(sleep_time_in_s)
