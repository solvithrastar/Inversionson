"""
A class with a collection of tasks that need to be done as a part of an inversion.
Created to have an easily accessible collection of methods which different optimizers have in common.
"""


class TaskManager(object):
    def __init__(self, comm):
        self.comm = comm
        self.optimization = self.comm.project.get_optimizer()

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

