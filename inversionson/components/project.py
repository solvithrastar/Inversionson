"""
A class which includes information regarding inversion
and sets up all the components that are needed inside
the inversion itself.
"""
# from __future__ import absolute_import

import os
import toml
from inversionson import InversionsonError

from .communicator import Communicator
from .component import Component
from .lasif_comp import LasifComponent
from .multimesh_comp import MultiMeshComponent
from .flow_comp import SalvusFlowComponent
from .mesh_comp import SalvusMeshComponent
from .opt_comp import SalvusOptComponent


class ProjectComponent(Component):

    def __init__(self, information_dict: dict):
        """
        Initiate everything to make it work correctly. Make sure that
        a running inversion can be restarted although this is done.
        """
        self.info = information_dict
        self.__comm = Communicator()
        super(ProjectComponent, self).__init__(self.__comm, "project")

        self.__setup_components()
        self.simulation_dict = self._read_config_file()

    def _read_config_file(self) -> dict:
        """
        Parse the Lasif config file to use it in the inversion.
        I might set this up to just be some parameters in the class

        :return: Simulation dictionary
        :rtype: dict
        """
        with open(os.path.join(self.info["lasif_project"], "lasif_config.toml"), "r") as fh:
            config_dict = toml.load(fh)

        simulation_info = {}
        solver_settings = config_dict["solver_settings"]
        simulation_info["time_step"] = -solver_settings["time_increment"]
        simulation_info["number_of_time_steps"] = int(round(
            (solver_settings["end_time"] - solver_settings["start_time"]) / solver_settings["time_increment"]))
        simulation_info["end_time"] = solver_settings["end_time"]
        simulation_info["start_time"] = solver_settings["start_time"]

        simulation_info["period_low"] = config_dict["data_processing"]["highpass_period"]
        simulation_info["period_high"] = config_dict["data_processing"]["lowpass_period"]

        return simulation_info

    def get_communicator(self):
        return self.__comm()

    def __setup_components(self):
        """
        Setup the different components that need to be used in the inversion.
        These are wrappers around the main libraries used in the inversion.
        """
        LasifComponent(communicator=self.comm, component_name="lasif")
        MultiMeshComponent(communicator=self.comm, component_name="multi_mesh")
        SalvusFlowComponent(communicator=self.comm,
                            component_name="salvus_flow")
        SalvusMeshComponent(communicator=self.comm,
                            component_name="salvus_mesher")
        SalvusOptComponent(communicator=self.comm, component_name="salvus_opt")

    def __get_inversion_attributes(self, simulation_info: dict):
        """
        Read crucial components into memory to keep them easily accessible.

        :param simulation_info: Information regarding numerical simulations,
        read from lasif
        :type simulation_info: dict
        """
        # Simulation attributes
        self.time_step = simulation_info["time_step"]
        self.start_time = simulation_info["start_time"]
        self.end_time = simulation_info["end_time"]
        self.period_low = simulation_info["period_low"]
        self.period_high = simulation_info["period_high"]

        # Inversion attributes
        self.inversion_root = self.info["inversion_path"]
        self.lasif_root = self.info["lasif_project"]
        self.inversion_id = self.info["inversion_id"]
        self.model_interpolation_mode = self.info["model_interpolation_mode"]
        self.gradient_interpolation_mode = self.info["gradient_interpolation_mode"]
        self.site_name = self.info["site_name"]
        self.ranks = self.info["ranks"]
        self.wall_time = self.info["wall_time"]
        self.current_iteration = self.comm.salvus_opt.get_newest_iteration_name()

        # Some useful paths
        self.paths = {}
        self.paths["inversion_root"] = self.inversion_root
        self.paths["lasif_root"] = self.lasif_root
        self.paths["salvus_opt"] = os.path.join(
            self.inversion_root, "SALVUS_OPT")
        if not os.path.exists(self.paths["salvus_opt"]):
            raise InversionsonError(
                "Please make a folder for Salvus opt and initialize it in there")

        self.paths["documentation"] = os.path.join(
            self.inversion_root, "DOCUMENTATION")
        if not os.path.exists(self.paths["documentation"]):
            os.makedirs(self.paths["documentation"])

        self.paths["iteration_tomls"] = os.path.join(
            self.paths["documentation"], "ITERATIONS")
        if not os.path.exists(self.paths["iteration_tomls"]):
            os.makedirs(self.paths["iteration_tomls"])

    def create_iteration_toml(self, iteration: str):
        """
        Create the toml file for an iteration. This toml file is then updated.
        To create the toml, we need the events and the control group

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml")
        if os.path.exists(iteration_toml):
            raise InversionsonError(
                f"Iteration toml for iteration: {iteration} already exists.")

        it_dict = {}
        it_dict["name"] = iteration
        it_dict["events"] = self.comm.lasif.list_events(iteration=iteration)
        # I need a way to figure out what the controlgroup is
        it_dict["control_group"] = []
        for event in it_dict["events"]:
            it_dict["events"][event]["misfit"] = 0.0
            it_dict["events"][event]["jobs"]["forward"] = {
                "name": "",
                "submitted": False,
                "retrieved": False
            }
            it_dict["events"][event]["jobs"]["adjoint"] = {
                "name": "",
                "submitted": False,
                "retrieved": False
            }

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def update_iteration_toml(self, iteration="current"):
        """
        Use iteration parameters to update iteration toml file

        :param iteration: Name of iteration
        :type iteration: str
        """
        if iteration == "current":
            iteration = self.current_iteration
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml")
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"Iteration toml for iteration: {iteration} does not exists")

        it_dict = {}
        it_dict["name"] = iteration
        it_dict["events"] = self.events_in_iteration
        # I need a way to figure out what the controlgroup is
        it_dict["control_group"] = self.control_group
        for event in it_dict["events"]:
            it_dict["events"][event]["misfit"] = self.misfits[event]
            it_dict["events"][event]["jobs"]["forward"] = self.forward_job
            it_dict["events"][event]["jobs"]["adjoint"] = self.adjoint_job

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def get_iteration_attributes(self, iteration: str):
        """
        Save the attributes of the current iteration into memory

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml")
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"No toml file eists for iteration: {iteration}")

        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)

        self.iteration_name = it_dict["name"]
        self.events_in_iteration = it_dict["events"]
        self.control_group = it_dict["control_group"]
        self.misfits = {}
        self.forward_job = {}
        self.adjoint_job = {}
        # Not sure if it's worth it to include station misfits
        for event in self.events_in_iteration:
            self.misfits[event] = it_dict["events"][event]["misfit"]
            self.forward_job[event] = it_dict["events"][event]["jobs"]["forward"]
            self.adjoint_job[event] = it_dict["events"][event]["jobs"]["adjoint"]
    
    def get_old_iteration_info(self, iteration: str) -> dict:
        """
        For getting information about something else than current iteration
        
        :param iteration: Name of iteration
        :type iteration: str
        :return: Information regarding that iteration
        :rtype: dict
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml")
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"No toml file eists for iteration: {iteration}")
        
        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)
        return it_dict
