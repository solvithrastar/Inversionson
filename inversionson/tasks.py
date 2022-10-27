import sys


class TaskManager(object):
    """
    A class with a collection of tasks that need to be done as a part of an inversion.
    Created to have an easily accessible collection of methods which different optimizers have in common.
    """
    def __init__(self, comm):
        self.comm = comm
        self.optimization = self.comm.project.get_optimizer()

    def perform_task(self, verbose=False):
        self.optimization.perform_task(verbose=verbose)

    def finish_task(self):
        self.optimization.finish_task()

    def get_new_task(self):
        self.optimization.get_new_task()

    def get_n_tasks(self):
        return len(self.optimization.available_tasks)

