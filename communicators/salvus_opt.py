"""
A class which takes care of all salvus opt communication.
Currently I have only used salvus opt in exodus mode so I'll 
initialize it as such but change into hdf5 once I can.
"""
import toml
import os
import shutil


class salvus_opt_comm(object):
    """
    Communications with Salvus Opt

    :param infodict: Information related to inversion project
    :type infodict: Dictionary
    """

    def __init__(self, infodict: dict):
        self.path = infodict["salvus_opt_dir"]
        self.lasif_root = infodict["lasif_project"]
        self.task_toml = os.path.join(self.path, "task.toml")

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

    def move_gradient_to_salvus_opt_folder(self, iteration: str, event=None):
        """
        Move the gradient into the folder where salvus opt operates.
        This can either be an individual gradient (not implemented yet),
        or a full iteration gradient. (Exodus gradients for now)

        :param iteration: Name of iteration (Should be enough to find gradient)
        :type iteration: string
        :param event: Name of event if individual gradient, defaults to None
        :type event: string, optional
        """
        grad_name = "gradient_" + iteration + ".e"
        dest_path = os.path.join(self.path, "PHYSICAL_MODELS", grad_name)
        current_path = os.path.join(self.lasif_root, "GRADIENTS",
                                    "ITERATION_" + iteration, "gradient.e")

        shutil.copy(current_path, dest_path)

    def get_model_from_salvus_opt(self, iteration: str):
        """
        Move new model from salvus_opt location to lasif location

        :param iteration: Name of iteration
        :type iteration: string
        """
        model_path = os.path.join(self.path, "PHYSICAL_MODELS",
                                  iteration + ".e")
        lasif_path = os.path.join(self.lasif_root, "MODELS", "ITERATION_" +
                                  iteration, iteration + ".e")

        shutil.copy(model_path, lasif_path)

    def get_model_path(self, iteration: str) -> str:
        """
        Get path of model related to iteration

        :param iteration: Name of iteration
        :type iteration: string
        """
        return os.path.join(self.path, "PHYSICAL_MODELS",
                            iteration + ".e")

    def write_misfit_to_task_toml(self, iteration: str):
        """
        Report the correct misfit value to salvus opt.
        ** Still have to find a consistant place to read/write misfit **
        ** Maybe this should be done on an individual level aswell **
        
        :param iteration: Name of iteration
        :type iteration: string
        """
        misfits = toml.load(os.path.join(self.lasif_root,"something"))
        misfit = misfits["blabla"][1] # This needs to be fixed
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
        grad_path = os.path.join(self.path, "PHYSICAL MODELS", 
        "gradient_", iteration + ".e")
        task = self.read_salvus_opt()
        task["task"][0]["output"]["gradient"] = grad_path

        with open(os.path.join(self.path, "task.toml"), "w") as fh:
            toml.dump(task, fh)