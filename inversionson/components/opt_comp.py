"""
A class which takes care of all salvus opt communication.
Currently I have only used salvus opt in exodus mode so I'll
initialize it as such but change into hdf5 once I can.
"""

from .component import Component
import toml
import os
import shutil
from inversionson import InversionsonError


class SalvusOptComponent(Component):
    """
    Communications with Salvus Opt

    :param infodict: Information related to inversion project
    :type infodict: Dictionary
    """

    def __init__(self, communicator, component_name):
        super(SalvusOptComponent, self).__init__(communicator, component_name)
        self.path = self.comm.project.paths["salvus_opt_dir"]
        # self.lasif_root = self.comm.project.paths["lasif_root"]
        self.task_toml = os.path.join(self.path, "task.toml")
        self.models = os.path.join(self.path, "PHYSICAL_MODELS")

    def read_salvus_opt(self) -> dict:
        """
        Read the task that salvus opt has issued into a dictionary

        :return: The information contained in the task toml file
        :rtype: dictionary
        """
        task = toml.load(os.path.join(self.path, "task.toml"))
        return task

    def close_salvus_opt_task(self):
        """
        Label the salvus_opt task as closed when it has been performed
        """
        task = self.read_salvus_opt()
        task["task"][0]["status"]["open"] = False
        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    def move_gradient_to_salvus_opt_folder(self, event=None):
        """
        Move the gradient into the folder where salvus opt operates.
        This can either be an individual gradient (not implemented yet),
        or a full iteration gradient. (Exodus gradients for now)

        :param event: Name of event if individual gradient, defaults to None
        :type event: string, optional
        """
        iteration = self.comm.project.current_iteration
        grad_name = "gradient_" + iteration + ".e"
        dest_path = os.path.join(self.models, grad_name)
        current_path = os.path.join(self.comm.project.lasif_root, "GRADIENTS",
                                    "ITERATION_" + iteration, "gradient.e")

        shutil.copy(current_path, dest_path)

    def get_model_from_salvus_opt(self):
        """
        Move new model from salvus_opt location to lasif location
        """
        iteration = self.comm.project.current_iteration
        model_path = os.path.join(self.models, iteration + ".e")
        lasif_path = os.path.join(self.comm.project.lasif_root, "MODELS", "ITERATION_" +
                                  iteration, iteration + ".e")

        shutil.copy(model_path, lasif_path)

    def get_model_path(self, iteration: str, gradient=False) -> str:
        """
        Get path of model related to iteration

        :param iteration: Name of iteration
        :type iteration: string
        :param gradient: Is it a gradient?
        :type gradient: bool
        """
        if gradient:
            return os.path.join(self.models, "gradient_" + iteration + ".e")
        else:
            return os.path.join(self.models, iteration + ".e")

    def write_misfit_to_task_toml(self, iteration: str):
        """
        Report the correct misfit value to salvus opt.
        ** Still have to find a consistant place to read/write misfit **
        ** Maybe this should be done on an individual level aswell **

        :param iteration: Name of iteration
        :type iteration: string
        """
        misfits = toml.load(os.path.join(self.lasif_root, "something"))
        misfit = misfits["blabla"][1]  # This needs to be fixed
        task = self.read_salvus_opt()
        task["task"][0]["output"]["misfit"] = float(misfit)

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)

    def write_gradient_path_to_task_toml(self, iteration: str):
        """
        Give salvus opt the path to the iteration gradient.
        Currently only for a single summed gradient, might change.
        Make sure you move gradient to salvus opt directory first.

        :param iteration: Name of iteration
        :type iteration: string
        """
        grad_path = os.path.join(self.models, "gradient_", iteration + ".e")
        task = self.read_salvus_opt()
        task["task"][0]["output"]["gradient"] = grad_path

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
            raise InversionsonError("Please initialize inversion in Salvus Opt")
        iterations = self._parse_model_files(models)

        new_it_number = max(iterations)
        new_it_tr_region = min(iterations[new_it_number])

        return self._create_iteration_name(new_it_number, new_it_tr_region)

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
            if len(model) < 16:
                # The first iteration is shorter
                iteration = int(model[2:6])
                tr_region = 9.999999
                iterations[iteration] = [tr_region]
            iteration = int(model[2:6])
            tr_region = float(model[-10:-2])
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
