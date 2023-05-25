from __future__ import annotations
import os
import time
import toml
from typing import Dict, Optional, Union, List, TYPE_CHECKING

from salvus.flow import api as sapi  # type: ignore
from salvus.opt.smoothing import get_smooth_model  # type: ignore

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
        tasks: Optional[Dict] = None,
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
        self.site_name = self.project.config.hpc.sitename
        self.iteration_name = iteration_name
        self.job_toml = (
            self.project.paths.reg_dir / f"regularization_{iteration_name}.toml"
        )
        if tasks is not None:
            self._write_tasks(tasks)
        if os.path.exists(self.job_toml):
            self.tasks = toml.load(self.job_toml)
        else:
            assert tasks is not None
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
    def base_dict(self) -> Dict:
        return dict(job_name="", submitted=False, retrieved=False, reposts=0)

    def _write_tasks(self, tasks) -> None:
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
                and task_dict["reposts"] < self.project.config.hpc.max_reposts
            ):
                if dispatching_msg:
                    self.print("Dispatching Smoothing Tasks")
                    dispatching_msg = False

                sims = self.project.smoother.get_sims_for_smoothing_task(
                    reference_model=task_dict["reference_model"],
                    model_to_smooth=task_dict["model_to_smooth"],
                    smoothing_lengths=task_dict["smoothing_lengths"],
                    smoothing_parameters=task_dict["smoothing_parameters"],
                )

                job = sapi.run_many_async(
                    input_files=sims,
                    site_name=self.project.config.hpc.sitename,
                    ranks_per_job=self.project.config.hpc.n_diff_ranks,
                    wall_time_in_seconds_per_job=self.project.config.hpc.diff_wall_time,
                )
                self.tasks[task_name]["submitted"] = True
                self.tasks[task_name]["job_name"] = job.job_array_name
                self._write_tasks(self.tasks)
            elif task_dict["reposts"] >= self.project.config.hpc.max_reposts:
                raise ValueError(
                    "Too many reposts in smoothing, "
                    "please check the time steps and the inputs."
                    "and reset the number of reposts in the toml file."
                )

    def update_task_status_and_retrieve(self) -> None:
        task_str = ""
        for task_name, task_dict in self.tasks.items():
            if task_dict["retrieved"]:
                continue
            job = sapi.get_job_array(
                job_array_name=task_dict["job_name"], site_name=self.site_name
            )
            status = job.update_status(force_update=True)
            finished = True
            i = 0
            for s in status:
                task_str += f" {task_name}_{i}: {s.name} \n"
                i += 1
                if s.name != "finished":
                    finished = False
                if s.name in ["unknown", "failed"]:
                    self.project.flow._delete_remote_job(task_dict["job_name"])
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
        print(task_str)

    def all_retrieved(self) -> bool:
        return all(task_dict["retrieved"] for task_dict in self.tasks.values())

    def monitor_tasks(self) -> None:
        if not self.tasks:
            return
        self.dispatch_smoothing_tasks()
        first = True
        sleep_time = self.project.config.hpc.sleep_time_in_seconds
        self.update_task_status_and_retrieve()  # Start with retrieval to skip loop
        while not self.all_retrieved():
            if first:
                self.print("Monitoring smoothing jobs...")
                first = False
            time.sleep(self.project.config.hpc.sleep_time_in_seconds)
            print(
                f"Waiting for smoothing jobs, will check again in {sleep_time} seconds."
            )
            self.dispatch_smoothing_tasks()
            self.update_task_status_and_retrieve()
