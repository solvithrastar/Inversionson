from __future__ import annotations
import os
import toml
from typing import Dict, Union, List, TYPE_CHECKING

from salvus.flow import api as sapi
from salvus.opt.smoothing import get_smooth_model
from inversionson.utils import sleep_or_process

if TYPE_CHECKING:
    from inversionson.project import Project


class RegularizationHelper(object):
    """
    This class takes a list of tasks that require smoothing.
    It can then dispatch the jobs, monitor and retrieve them.
    """

    def __init__(
        self,
        project: Project,
        iteration_name: str,
        tasks: Union[Dict, bool],
        optimizer=None,
    ):
        """
        Each tasks is a dict that has a reference model, a model that contains the fields
        that require smoothing, the smoothing lengths, the parameters that require
        smoothing and the output location to which the smoothed parameters
        should be retrieved.

        :param tasks: a dict of dicts like this:
                {task_name: {"reference_model": str, "model_to_smooth": str,
                "smoothing_lengths": list, "smoothing_parameters": list,
                "output_location": str}, "task_name2" : {...}, etc.}
                If False, it does not add tasks but uses previously written tasks.
        :type tasks: Union[dict, bool]
        :param iteration_name: Name of the iteration.
        :type iteration_name: str
        """
        self.project = project
        self.site_name = self.project.smoothing_site_name
        self.iteration_name = iteration_name
        self.optimizer = optimizer or self.comm.project.get_optimizer()
        self.job_toml = (
            self.optimizer.regularization_dir / f"regularization_{iteration_name}.toml"
        )
        self._write_tasks(tasks)
        if os.path.exists(self.job_toml):
            self.tasks = toml.load(self.job_toml)
        else:
            self.tasks = tasks

    def print(
        self,
        message: str,
        color: str = "magenta",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = ":cop:",
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    @property
    def base_dict(self):
        return dict(job_name="", submitted=False, retrieved=False, reposts=0)

    def _write_tasks(self, tasks):
        """
        This function writes the tasks to file or updates the task file.
        """
        if os.path.exists(
            self.job_toml
        ):  # We add the tasks to the existing tasks if needed
            existing_tasks = toml.load(self.job_toml)
            if tasks:
                for task_name in tasks:
                    # Add the empty task if it does not exist
                    if task_name not in existing_tasks.keys():
                        existing_tasks[task_name] = tasks[task_name]
                        existing_tasks[task_name].update(self.base_dict)
                    else:  # Update existing tasks with passed tasks
                        existing_tasks[task_name].update(tasks[task_name])
                with open(self.job_toml, "w") as fh:
                    toml.dump(existing_tasks, fh)

        elif tasks:
            for task_dict in tasks.values():
                task_dict.update(self.base_dict)
            with open(self.job_toml, "w") as fh:
                toml.dump(tasks, fh)

    def dispatch_smoothing_tasks(self):
        dispatching_msg = True
        for task_name, task_dict in self.tasks.items():
            if (
                not task_dict["submitted"]
                and task_dict["reposts"] < self.comm.project.max_reposts
            ):
                if dispatching_msg:
                    self.print("Dispatching Smoothing Tasks")
                    dispatching_msg = False

                sims = self.comm.smoother.get_sims_for_smoothing_task(
                    reference_model=task_dict["reference_model"],
                    model_to_smooth=task_dict["model_to_smooth"],
                    smoothing_lengths=task_dict["smoothing_lengths"],
                    smoothing_parameters=task_dict["smoothing_parameters"],
                )

                job = sapi.run_many_async(
                    input_files=sims,
                    site_name=self.comm.project.smoothing_site_name,
                    ranks_per_job=self.comm.project.smoothing_ranks,
                    wall_time_in_seconds_per_job=self.comm.project.smoothing_wall_time,
                )
                self.tasks[task_name]["submitted"] = True
                self.tasks[task_name]["job_name"] = job.job_array_name
                self._write_tasks(self.tasks)
            elif task_dict["reposts"] >= self.comm.project.max_reposts:
                raise ValueError(
                    "Too many reposts in smoothing, "
                    "please check the time steps and the inputs."
                    "and reset the number of reposts in the toml file."
                )

    def update_task_status_and_retrieve(self):
        for task_dict in self.tasks.values():
            if task_dict["retrieved"]:
                continue
            job = sapi.get_job_array(
                job_array_name=task_dict["job_name"], site_name=self.site_name
            )
            status = job.update_status(force_update=True)
            finished = True
            for s in status:
                if s.name != "finished":
                    finished = False
                if s.name in ["unknown", "failed"]:
                    task_dict["reposts"] += 1
                    task_dict["submitted"] = False
                    self._write_tasks(self.tasks)
                    break
            if finished:
                smooth_gradient = get_smooth_model(
                    job=job,
                    model=task_dict["reference_model"],
                )
                smooth_gradient.write_h5(task_dict["output_location"])
                task_dict["retrieved"] = True
                self._write_tasks(self.tasks)

    def all_retrieved(self):
        return all(task_dict["retrieved"] for task_dict in self.tasks.values())

    def monitor_tasks(self):
        if not self.tasks:
            return
        self.dispatch_smoothing_tasks()
        first = True
        self.update_task_status_and_retrieve()  # Start with retrieval to skip loop
        while not self.all_retrieved():
            if first:
                self.print("Monitoring smoothing jobs...")
                first = False
            sleep_or_process(self.comm, color="magenta", emoji_alias=":cop:")
            self.dispatch_smoothing_tasks()
            self.update_task_status_and_retrieve()
