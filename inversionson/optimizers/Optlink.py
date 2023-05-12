from optson.preconditioner import AbstractPreconditioner
from numpy.typing import ArrayLike


class Optlink:
    """This class will create the Optson Problem
    and define the settings."""

    def __init__(self, comm):
        self.comm = comm  #  sdas

    def define_problem(self):
        pass

    def call_optson(self):
        pass
