import os
import toml
from typing import Union, List
from pathlib import Path
import lasif
import numpy as np

from inversionson.autoinverter import _find_project_comm, read_info_toml


class HandyMan(object):
    """
    A helper class which can perform all sorts of tasks which might be useful
    while the inversion is ongoing, or when problems arise.
    Call it from the Inversionson root directory or initialize it with a path
    to the root
    """

    def __init__(self, root=None):
        info = read_info_toml(root=root)
        self.comm = _find_project_comm(info)
        self.lasif_comm = self.comm.lasif.lasif_comm
        self.optimizer = self.comm.project.get_optimizer()
        self._print("Inversionson HandyMan... How can I help you?")

    def _print(
        self,
        message: str,
        color: str = "yellow",
        emoji_alias: Union[str, List[str]] = [":axe:", ":hatched_chick:", ":toolbox:"],
        line_above=False,
        line_below=False,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            emoji_alias=emoji_alias,
            line_above=line_above,
            line_below=line_below,
        )

    @property
    def task_iteration(self):
        return int(Path(self.optimizer.task_path).stem.split("_")[-2])

    @property
    def task_number(self):
        return int(Path(self.optimizer.task_path).stem.split("_")[-1])

    def _move_to_initial_task_of_iteration(self):
        if self.comm.lasif.has_iteration(it_name=self.optimizer.iteration_name):
            lasif.api.set_up_iteration(
                lasif_root=self.lasif_comm,
                iteration=self.optimizer.iteration_name,
                remove_dirs=True,
            )
        iteration_toml = (
            self.comm.project.paths["documentation"]
            / "ITERATIONS"
            / f"{self.optimizer.iteration_name}.toml"
        )
        if os.path.exists(iteration_toml):
            os.remove(iteration_toml)
        while (
            self.task_iteration == self.optimizer.iteration_number
            and self.task_number != 0
        ):
            os.remove(self.optimizer.task_path)

        task_dict = toml.load(self.optimizer.task_path)
        assert (
            task_dict["task"] == "prepare_iteration"
        ), "The task should be prepare_iteration"
        task_dict["finished"] = False
        with open(self.optimizer.task_path, "w") as fh:
            toml.dump(task_dict, fh)

    def _remove_current_iteration(self):
        if self.comm.lasif.has_iteration(it_name=self.optimizer.iteration_name):
            lasif.api.set_up_iteration(
                lasif_root=self.lasif_comm,
                iteration=self.optimizer.iteration_name,
                remove_dirs=True,
            )
        iteration_toml = (
            self.comm.project.paths["documentation"]
            / "ITERATIONS"
            / f"{self.optimizer.iteration_name}.toml"
        )
        if os.path.exists(iteration_toml):
            os.remove(iteration_toml)
        while self.task_iteration == self.optimizer.iteration_number:
            os.remove(self.optimizer.task_path)

        if self.comm.project.optimizer.lower() == "adam":
            files_to_delete = [
                self.optimizer.raw_gradient_path,
                self.optimizer._get_path_for_iteration(
                    self.optimizer.iteration_number + 1,
                    self.optimizer.first_moment_path,
                ),
                self.optimizer._get_path_for_iteration(
                    self.optimizer.iteration_number + 1,
                    self.optimizer.second_moment_path,
                ),
                self.optimizer.smoothed_model_path,
                self.optimizer.raw_update_path,
                self.optimizer.smooth_update_path,
                self.optimizer.regularization_job_toml,
                self.optimizer.relative_perturbation_path,
                self.optimizer.gradient_norm_path,
            ]
        for filename in files_to_delete:
            if os.path.exists(filename):
                os.remove(filename)
        os.remove(self.optimizer.model_path)

    # I'll write down functions that I think it should do before implementing them
    def go_back_one_iteration(self, verbose=True):
        if verbose:
            self._print(
                f"You are asking me to delete iteration number {self.optimizer.iteration_number}: {self.optimizer.iteration_name}.",
                line_above=True,
            )
            if self.task_number == 0:
                self._print(
                    f"This means that I will move to the beginning of iteration number {self.optimizer.iteration_number - 1}",
                    line_below=True,
                )
            else:
                self._print(
                    f"This means that I will move to the beginning of iteration number {self.optimizer.iteration_number}",
                    line_below=True,
                )
        if self.task_number == 0:
            delete_iteration = True
        else:
            delete_iteration = False
        if not verbose:
            reply = "y"
        else:
            reply = input("Are you sure? (y/n) ")
        if reply.lower() == "y":
            if verbose:
                self._print(
                    "Ok, I'm on it. Can you make me some coffee while I work on it?",
                    line_above=True,
                    line_below=True,
                )
            if delete_iteration:
                self._remove_current_iteration()
            self._move_to_initial_task_of_iteration()
            if verbose:
                self._print(
                    f"Ok, you're all set here. Your newest iteration is now {self.optimizer.iteration_name}. \n"
                    "There might be files on the remote machine for you to delete. "
                    "That's above my paygrade right now. Thanks for the coffee!",
                    emoji_alias=[":coffee:", ":hatched_chick:", ":toolbox:"],
                    line_above=True,
                    line_below=True,
                )
        else:
            self._print(
                "Ok, better safe than sorry. Anything else I can help you with?",
                line_above=True,
            )

    def go_back_one_task(self, verbose=True):
        if verbose:
            self._print(
                f"You are asking me to delete task {self.optimizer.task_path.stem}.",
                line_above=True,
            )
            self._print(
                f"I don't really want to do that to be honest. Maybe some other time",
            )
        pass

    def go_back_to_iteration(self, iteration: str):
        self._print(
            f"You are asking me to delete all iterations after {iteration}. This will move you to the beginning of that iteration.",
            line_above=True,
            line_below=False,
        )
        if not self.comm.lasif.has_iteration(it_name=iteration):
            self._print(
                f"The iteration needs to exist! Be careful! This could have deleted your entire project. You're lucky I'm good at my job!",
                line_above=True,
                line_below=True,
            )
            return
        reply = input("Are you sure? (y/n) ")
        if reply.lower() == "y":
            self._print(
                f"Ok, that's quite the job. But nothing I can't handle. Can you bring me a doughnut please?",
                line_above=True,
                line_below=True,
            )
            while self.optimizer.iteration_name != iteration:
                self._print(f"Deleting iteration {self.optimizer.iteration_name}")
                self.go_back_one_iteration(verbose=False)
            if self.optimizer.iteration_name == iteration and self.task_number != 0:
                self.go_back_one_iteration(verbose=False)
            self._print(
                f"Ok, you're at the beginning of iteration {self.optimizer.iteration_name}. \n"
                "There might be files on the remote machine for you to delete. "
                "That's above my paygrade right now. Thanks for the doughnut!",
                emoji_alias=[":doughnut:", ":hatched_chick:", ":toolbox:"],
                line_above=True,
                line_below=True,
            )

        else:
            self._print(
                "Ok, better safe than sorry. Anything else I can help you with?",
                line_above=True,
            )

    def change_value_in_iteration_toml(
        self,
        iteration: str,
        entry: str,
        new_value: Union[bool, str, int],
        job_type: str = None,
        event_name: str = None,
    ):
        """
        Change specific values in the iteration toml files.

        :param iteration: The name of the iteration
        :type iteration: str
        :param entry: which parameter to change. e.g. name, reposts, submitted
        :type entry: str
        :param new_value: The new value into the entry
        :type new_value: Union[bool, str, int]
        :param job_type: If you want to look at a specific job type, None makes you loop through
            all job types, defaults to None
        :type job_type: str, optional
        :param event_name: Name of event in question, None makes you loop through
            all of them, defaults to None
        :type event_name: str, optional
        """
        toml_path = self.comm.project.paths["iteration_tomls"] / f"{iteration}.toml"
        info = toml.load(toml_path)
        loop_through_events = True if event_name is None else False
        loop_through_job_types = True if job_type is None else False

        # Informing
        if loop_through_events and loop_through_events:
            self._print(
                f"I'll change {entry} to {new_value} for all events and job types.",
                line_above=True,
                line_below=True,
            )
        elif loop_through_events:
            self._print(
                f"I'll change {entry} to {new_value} for {job_type} for all events.",
                line_above=True,
                line_below=True,
            )
        elif loop_through_job_types:
            self._print(
                f"I'll change {entry} to {new_value} for {event_name} for all job_types.",
                line_above=True,
                line_below=True,
            )
        else:
            self._print(
                f"I'll change {entry} to {new_value} for {event_name} for all {job_type}.",
                line_above=True,
                line_below=True,
            )
        reply = input("Are you sure? (y/n) ")
        if reply.lower() != "y":
            self._print(
                "Ok, better safe than sorry. Anything else I can help you with?",
                line_above=True,
            )
            return
        else:
            self._print("No problem at all, this will take no time!", line_above=True)

        if event_name is not None:
            for event in info["events"].keys():
                if info["events"][event]["name"] == event_name:
                    self._print(
                        f"I have identified the {event_name} as event number {int(event)}",
                        line_above=True,
                    )
                    event_name = event

        if loop_through_events and loop_through_job_types:
            for event in info["events"].keys():
                for field in info["events"][event]["job_info"].keys():
                    info["events"][event]["job_info"][field][entry] = new_value
        elif loop_through_events:
            for event in info["events"].keys():
                info["events"][event]["job_info"][job_type][entry] = new_value
        elif loop_through_job_types:
            for job in info["events"][event]["job_info"].keys():
                info["events"][event]["job_info"][job][entry] = new_value
        else:
            info["events"][event]["job_info"][job_type][entry] = new_value

        with open(toml_path, "w") as fh:
            toml.dump(info, fh)
        self._print(
            "You're all set here. Anything else I can help you with?",
            line_above=True,
            line_below=True,
        )

    def plot_validation_misfit_curve(self, normalized=True, save_path=None):
        import matplotlib.pyplot as plt

        file_path = self.comm.project.paths["documentation"] / "validation.toml"
        info = toml.load(file_path)

        misfits = []
        iterations = []
        for iteration in info.keys():
            iterations.append(iteration)
            misfits.append(float(info[iteration]["total"]))

        misfits = np.array(misfits)
        if normalized:
            misfits /= misfits.max()

        it_numbers = [int(x.split("_")[-1]) for x in iterations]
        fig = plt.figure(figsize=(10, 8))
        plt.plot(it_numbers, misfits)
        plt.xlabel("Iterations")
        plt.ylabel("Misfits")
        plt.title("Validation Misfits")
        if save_path is not None:
            plt.savefig(save_path, dpi=250)
        else:
            plt.show()
