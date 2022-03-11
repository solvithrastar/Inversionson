"""
A class with a collection of tasks that need to be done as a part of an inversion.
Created to have an easily accessible collection of methods which different optimizers have in common.
"""
from typing import List, Dict
import inversionson.autoinverter_helpers as helper
from inversionson import InversionsonError
from inversionson.optimizers.adam_opt import AdamOpt


class TaskManager(object):
    def __init__(self, optimization_method: str, comm):
        self.comm = comm
        self.optimization = self._get_optimizer(optimization_method, self.comm)

    def _get_optimizer(self, optimization_method: str, comm: object):
        """
        This creates an instance of the optimization class which is
        picked by the user.
        """
        if optimization_method.lower() == "adam":
            return AdamOpt(comm=comm)
        else:
            raise InversionsonError(
                f"Optimization method {optimization_method} not defined"
            )

    def _time_for_validation(self) -> bool:
        validation = False
        if self.comm.project.when_to_validate == 0:
            return False
        if self.optimization.iteration_number == 0:
            validation = True
        if (
            self.optimization.iteration_number + 1
        ) % self.comm.project.when_to_validate == 0:
            validation = True

        if validation:
            validation = self.optimization.ready_for_validation()

        return validation

    def perform_task(self, verbose=False):
        if self._time_for_validation():
            self.optimization.do_validation_iteration()
        self.optimization.perform_task(verbose=verbose)

    def finish_task(self):
        self.optimization.finish_task()

    def get_new_task(self):
        self.optimization.get_new_task()

    def get_n_tasks(self):
        return len(self.optimization.available_tasks)


"""
I think this will mostly include basic information.
There can be an inform task object method that gives info
And then there can be the actual tasks in there.
Whether it actually needs the info, I'm not sure. We'll see.
It would be good if the status can somehow be saved on the go.
"""
