"""
A class which includes information regarding inversion
and sets up all the components that are needed inside
the inversion itself.
"""

import os
import toml
import shutil
import pathlib
from inversionson import InversionsonError, InversionsonWarning
import warnings
from inversionson.optimizers.adam_opt import AdamOpt
from inversionson.optimizers.sgd_with_momentum import SGDM
from typing import Union, List

from lasif.components.communicator import Communicator
from lasif.components.component import Component
from .lasif_comp import LasifComponent
from .multimesh_comp import MultiMeshComponent
from .flow_comp import SalvusFlowComponent
from .mesh_comp import SalvusMeshComponent
from .storyteller import StoryTellerComponent
from .smooth_comp import SalvusSmoothComponent
import salvus.flow.api as sapi


class ProjectComponent(Component):
    def __init__(self, information_dict: dict):
        """
        Initiate everything to make it work correctly. Make sure that
        a running inversion can be restarted although this is done.
        """
        self.info = information_dict
        self.random_event_processing = False
        self.ad_src_type = "tf_phase_misfit"
        self.__comm = Communicator()
        super(ProjectComponent, self).__init__(self.__comm, "project")
        self.simulation_dict = self._read_config_file()
        self.get_inversion_attributes(first=True)
        self.__setup_components()
        self.get_inversion_attributes(first=False)
        self._validate_inversion_project()
        self.remote_gradient_processing = True
        self.simulation_time_step = None
        # Attempt to get the simulation timestep immediately if it exists.
        self.get_simulation_time_step()

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def _read_config_file(self) -> dict:
        """
        Parse the Lasif config file to use it in the inversion.
        I might set this up to just be some parameters in the class

        :return: Simulation dictionary
        :rtype: dict
        """
        with open(
            os.path.join(self.info["lasif_root"], "lasif_config.toml"), "r"
        ) as fh:
            config_dict = toml.load(fh)

        simulation_info = {}
        solver_settings = config_dict["simulation_settings"]
        simulation_info["start_time"] = solver_settings["start_time_in_s"]
        simulation_info["number_of_time_steps"] = int(
            round(
                (solver_settings["end_time_in_s"] - simulation_info["start_time"])
                / solver_settings["time_step_in_s"]
            )
            + 1
        )
        simulation_info["end_time"] = solver_settings["end_time_in_s"]
        simulation_info["time_step"] = solver_settings["time_step_in_s"]
        simulation_info["min_period"] = solver_settings["minimum_period_in_s"]
        simulation_info["max_period"] = solver_settings["maximum_period_in_s"]
        simulation_info["attenuation"] = config_dict["salvus_settings"]["attenuation"]
        simulation_info["ocean_loading"] = config_dict["salvus_settings"][
            "ocean_loading"
        ]
        simulation_info["absorbing_boundaries_length"] = config_dict["salvus_settings"][
            "absorbing_boundaries_in_km"
        ]
        simulation_info["domain_file"] = config_dict["lasif_project"][
            "domain_settings"
        ]["domain_file"]

        return simulation_info

    def get_communicator(self):
        return self.__comm

    def get_optimizer(self):
        """
        This creates an instance of the optimization class which is
        picked by the user.
        """
        if self.optimizer == "adam":
            return AdamOpt(comm=self.comm)
        if self.optimizer == "sgdm":
            return SGDM(comm=self.comm)
        else:
            raise InversionsonError(f"Optimization method {self.optimizer} not defined")

    def _validate_inversion_project(self):
        """
        Make sure everything is correctly set up in order to perform inversion.

        :param info_dict: Information needed
        :type info_dict: dict
        :param simulation_dict: Information regarding simulations
        :type simulation_dict: dict
        """
        import pathlib

        if "inversion_path" not in self.info.keys():
            raise InversionsonError(
                "We need a given path for the inversion root directory."
                " Key: inversion_path"
            )

        if "meshes" not in self.info.keys():
            raise InversionsonError(
                "We need information on which sorts of meshes you use. "
                "Options are multi-mesh or mono-mesh. "
                "Key: meshes"
            )

        if "HPC" not in self.info.keys():
            raise InversionsonError(
                "We need information regarding your computational resources."
                " run create_dummy_info_file.py for an example"
            )

        if "wave_propagation" not in self.info["HPC"].keys():
            raise InversionsonError(
                "We need specific computational info on wave_propagation"
            )

        if "diffusion_equation" not in self.info["HPC"].keys():
            raise InversionsonError(
                "We need specific computational info on diffusion_equation"
            )
        if "interpolation" not in self.info["HPC"].keys():
            raise InversionsonError(
                "We need to know some info on your remote interpolations"
            )

        if "model_wall_time" not in self.info["HPC"]["interpolation"].keys():
            raise InversionsonError(
                "We need to know the wall time of your model "
                " interpolations. Key: HPC.interpolation.model_wall_time"
            )

        if "gradient_wall_time" not in self.info["HPC"]["interpolation"].keys():
            raise InversionsonError(
                "We need to know the wall time of your model "
                " interpolations. Key: HPC.interpolation.gradient_wall_time"
            )
        if "wall_time" not in self.info["HPC"]["processing"].keys():
            raise InversionsonError(
                "We need to know the wall time for the remote processing. "
                "Key: HPC.processing.wall_time"
            )
        if "remote_mesh_directory" not in self.info["HPC"].keys():
            raise InversionsonError(
                "We need to know the location where the meshes are stored"
                ". Key: HPC.remote_mesh_directory"
            )
        if not self.info["HPC"]["remote_mesh_directory"].startswith("/"):
            raise InversionsonError(
                "The remote mesh/model dir needs to be a legit absolute path"
            )
        if "inversionson_fast_dir" not in self.info["HPC"].keys():
            raise InversionsonError(
                "We need to know the location where to quickly access files used in "
                "simualations"
                ". Key: HPC.inversionson_fast_dir"
            )
        if not self.info["HPC"]["inversionson_fast_dir"].startswith("/"):
            raise InversionsonError(
                "The inversionson fast directory needs to be a legit absolute path"
            )
        if "site_name" not in self.info["HPC"]["wave_propagation"].keys():
            raise InversionsonError(
                "We need information on the site where jobs are submitted. "
                "Key: HPC.wave_propagation.site_name"
            )

        if "wall_time" not in self.info["HPC"]["wave_propagation"].keys():
            raise InversionsonError(
                "We need information on the site where jobs are submitted. "
                "Key: HPC.wave_propagation.site_name"
            )

        if "ranks" not in self.info["HPC"]["wave_propagation"].keys():
            raise InversionsonError(
                "We need information on the amount of ranks you want to "
                "run your simulations. Key: HPC.wave_propagation.ranks"
            )

        if "wall_time" not in self.info["HPC"]["diffusion_equation"].keys():
            raise InversionsonError(
                "We need information on the site where jobs are submitted. "
                "Key: HPC.diffusion_equation.site_name"
            )

        if "ranks" not in self.info["HPC"]["diffusion_equation"].keys():
            raise InversionsonError(
                "We need information on the amount of ranks you want to "
                "run your simulations. Key: HPC.diffusion_equation.ranks"
            )

        if "inversion_parameters" not in self.info.keys():
            raise InversionsonError(
                "We need information on the parameters you want to invert for."
                " Key: inversion_parameters"
            )

        if "modelling_parameters" not in self.info.keys():
            raise InversionsonError(
                "We need information on the parameters you keep in your mesh "
                "for forward modelling. Key: modelling_parameters"
            )

        if "meshes" not in self.info.keys():
            raise InversionsonError(
                "We need to know what sorts of meshes you use. "
                "Either mono-mesh for simulation mesh = inversion mesh "
                "or multi-mesh for wavefield adapted meshes. "
                "Key: meshes"
            )

        if self.info["meshes"] not in ["mono-mesh", "multi-mesh"]:
            raise InversionsonError("We only accept 'mono-mesh' or 'multi-mesh'")

        if "optimizer" not in self.info.keys():
            raise InversionsonError(
                "We need to know what type of optimization you want. "
                "The available ones are 'Adam' and 'SGDM'"
                "Key: optimizer"
            )

        if self.info["optimizer"].lower() not in ["adam", "sgdm"]:
            raise InversionsonError("We only accept 'adam' and 'sgdm'")

        if "Meshing" not in self.info.keys() and self.info["meshes"] == "multi-mesh":
            raise InversionsonError(
                "We need some information regarding your meshes. "
                "We need to know how many elements you want per azimuthal "
                "quarter. Key: Meshing"
            )
        if self.info["meshes"] == "multi-mesh":
            if "elements_per_azimuthal_quarter" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need to know how many elements you need per azimuthal "
                    "quarter. Key: Meshing.elements_per_azimuthal_quarter"
                )

            if not isinstance(
                self.info["Meshing"]["elements_per_azimuthal_quarter"], int
            ):
                raise InversionsonError(
                    "Elements per azimuthal quarter need to be an integer."
                )

            if "elements_per_wavelength" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need to know how many elements you need per wavelength "
                    "Key: Meshing.elements_per_wavelength"
                )
            if "ellipticity" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need a boolean value regarding ellipticity "
                    "of your meshes. \n"
                    "Key: Meshing.ellipticity"
                )
            if "topography" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need information on whether you use topography " "in your mesh."
                )
            else:
                if "use" not in self.info["Meshing"]["topography"].keys():
                    raise InversionsonError(
                        "We need a boolean value telling us if you use "
                        "topography in your mesh. \n"
                        "If True, we need file and variable name"
                    )
                if self.info["Meshing"]["topography"]["use"]:
                    if len(self.info["Meshing"]["topography"]["file"]) == 0:
                        raise InversionsonError(
                            "Please specify path to your topography file.\n"
                            "Key: Meshing.topography.file"
                        )
                    if len(self.info["Meshing"]["topography"]["variable"]) == 0:
                        raise InversionsonError(
                            "Please specify path to your topography variable "
                            "name. You can find it by opening the file in "
                            "ParaView \n"
                            "Key: Meshing.topography.variable"
                        )
        if "use" not in self.info["Meshing"]["ocean_loading"].keys():
            raise InversionsonError(
                "We need a boolean value telling us if you use "
                "ocean_loading in your mesh. \n"
                "If True, we need file and variable name"
            )
        if self.info["Meshing"]["ocean_loading"]["use"]:
            if len(self.info["Meshing"]["ocean_loading"]["file"]) == 0:
                if self.info["meshes"] == "multi-mesh":
                    raise InversionsonError(
                        "Please specify path to your bathymetry file.\n"
                        "Key: Meshing.ocean_loading.file"
                    )
            if len(self.info["Meshing"]["ocean_loading"]["variable"]) == 0:
                if self.info["meshes"] == "multi-mesh":
                    raise InversionsonError(
                        "Please specify path to your bathymetry variable "
                        "name. You can find it by opening the file in "
                        "ParaView \n"
                        "Key: Meshing.ocean_loading.variable"
                    )

        # Lasif
        if "lasif_root" not in self.info.keys():
            raise InversionsonError(
                "Information on lasif_project is missing from information. "
                "Key: lasif_root"
            )
        else:
            folder = pathlib.Path(self.info["lasif_root"])
            if not (folder / "lasif_config.toml").exists():
                raise InversionsonError("Lasif project not initialized")

        # Simulation parameters:
        if "end_time" not in self.simulation_dict.keys():
            raise InversionsonError(
                "Information regarding end time of simulation missing"
            )

        if "time_step" not in self.simulation_dict.keys():
            raise InversionsonError(
                "Information regarding time step of simulation missing"
            )

        if "start_time" not in self.simulation_dict.keys():
            raise InversionsonError(
                "Information regarding start time of simulation missing"
            )

        if "inversion_monitoring" not in self.info.keys():
            raise InversionsonError(
                "Information regarding inversion monitoring is missing"
            )
        if (
            self.info["inversion_monitoring"]["iterations_between_validation_checks"]
            != 0
        ):
            if len(self.info["inversion_monitoring"]["validation_dataset"]) == 0:
                raise InversionsonError(
                    "You need to specify a validation dataset if you want"
                    " to check it regularly."
                )

    def __setup_components(self):
        """
        Setup the different components that need to be used in the inversion.
        These are wrappers around the main libraries used in the inversion.
        """
        LasifComponent(communicator=self.comm, component_name="lasif")
        MultiMeshComponent(communicator=self.comm, component_name="multi_mesh")
        SalvusFlowComponent(communicator=self.comm, component_name="salvus_flow")
        SalvusMeshComponent(communicator=self.comm, component_name="salvus_mesher")
        StoryTellerComponent(communicator=self.comm, component_name="storyteller")
        SalvusSmoothComponent(communicator=self.comm, component_name="smoother")

    def arrange_params(self, parameters: list) -> list:
        """
        Re-arrange list of parameters in order to have
        them conveniently aranged when called upon. This can be an annoying
        problem when working with hdf5 files.
        This method can only handle a few cases. If it doesn't
        recognize the case it will return an unmodified list.

        :param parameters: parameters to be arranged
        :type parameters: list
        """
        case_tti_inv = set(["VSV", "VSH", "VPV", "VPH", "RHO"])
        case_tti_mod = set(["VSV", "VSH", "VPV", "VPH", "RHO", "QKAPPA", "QMU", "ETA"])
        case_iso_mod = set(["QKAPPA", "QMU", "VP", "VS", "RHO"])
        case_iso_inv = set(["VP", "VS"])
        case_iso_inv_dens = set(["VP", "VS", "RHO"])
        case_tti_inv_norho = set(["VSV", "VSH", "VPV", "VPH"])

        if set(parameters) == case_tti_inv:
            parameters = ["VPV", "VPH", "VSV", "VSH", "RHO"]
        elif set(parameters) == case_tti_inv_norho:
            parameters = ["VPV", "VPH", "VSV", "VSH"]
        elif set(parameters) == case_tti_mod:
            parameters = [
                "VPV",
                "VPH",
                "VSV",
                "VSH",
                "RHO",
                "QKAPPA",
                "QMU",
                "ETA",
            ]
        elif set(parameters) == case_iso_inv:
            parameters = ["VP", "VS"]
        elif set(parameters) == case_iso_inv_dens:
            parameters = ["RHO", "VP", "VS"]
        elif set(parameters) == case_iso_mod:
            parameters = ["QKAPPA", "QMU", "RHO", "VP", "VS"]
        else:
            raise InversionsonError(
                f"Parameter list {parameters} not " f"a recognized set of parameters"
            )
        return parameters

    def get_inversion_attributes(self, first=False):
        """
        Read crucial components into memory to keep them easily accessible.

        :param first: Befor components are set up, defaults to False
        :type first: bool, optional
        """
        # Simulation attributes
        self.time_step = self.simulation_dict["time_step"]
        self.start_time = self.simulation_dict["start_time"]
        self.end_time = self.simulation_dict["end_time"]
        self.min_period = self.simulation_dict["min_period"]
        self.max_period = self.simulation_dict["max_period"]
        self.attenuation = self.simulation_dict["attenuation"]
        self.abs_bound_length = self.simulation_dict["absorbing_boundaries_length"]
        self.absorbing_boundaries = self.info["absorbing_boundaries"]
        self.domain_file = self.simulation_dict["domain_file"]

        # Inversion attributes
        self.inversion_root = pathlib.Path(self.info["inversion_path"])
        self.lasif_root = pathlib.Path(self.info["lasif_root"])
        self.inversion_mode = "mini-batch"
        self.meshes = self.info["meshes"]
        self.remote_interp = True if self.meshes == "multi-mesh" else False
        self.optimizer = self.info["optimizer"].lower()
        self.elem_per_quarter = self.info["Meshing"]["elements_per_azimuthal_quarter"]
        self.elem_per_wavelength = self.info["Meshing"]["elements_per_wavelength"]
        self.topography = self.info["Meshing"]["topography"]
        self.ellipticity = self.info["Meshing"]["ellipticity"]
        self.ocean_loading = self.info["Meshing"]["ocean_loading"]
        self.interpolation_mode = "remote"
        self.cut_source_radius = self.info["cut_source_region_from_gradient_in_km"]
        self.clip_gradient = self.info["clip_gradient"]
        self.site_name = self.info["HPC"]["wave_propagation"]["site_name"]
        self.ranks = self.info["HPC"]["wave_propagation"]["ranks"]
        self.wall_time = self.info["HPC"]["wave_propagation"]["wall_time"]
        self.model_interp_wall_time = self.info["HPC"]["interpolation"][
            "model_wall_time"
        ]
        self.grad_interp_wall_time = self.info["HPC"]["interpolation"][
            "gradient_wall_time"
        ]
        self.interpolation_site = self.site_name
        self.remote_data_processing = self.info["HPC"]["remote_data_processing"]["use"]
        self.remote_data_proc_wall_time = self.info["HPC"]["remote_data_processing"][
            "wall_time"
        ]
        self.remote_raw_data_dir = self.info["HPC"]["remote_data_processing"][
            "remote_raw_data_directory"
        ]
        self.smoothing_site_name = self.site_name
        self.smoothing_ranks = self.info["HPC"]["diffusion_equation"]["ranks"]
        self.smoothing_wall_time = self.info["HPC"]["diffusion_equation"]["wall_time"]
        self.remote_mesh_dir = pathlib.Path(self.info["HPC"]["remote_mesh_directory"])
        self.sleep_time_in_s = self.info["HPC"]["sleep_time_in_seconds"]
        self.max_reposts = self.info["HPC"]["max_reposts"]
        self.remote_inversionson_dir = pathlib.Path(
            self.info["HPC"]["inversionson_fast_dir"]
        )
        self.hpc_processing = self.info["HPC"]["processing"]["use"]
        self.hpc_processing_wall_time = self.info["HPC"]["processing"]["wall_time"]
        self.remote_conda_env = self.info["HPC"]["remote_conda_environment"]
        self.remote_conda_source_location = self.info["HPC"]["remote_conda_source_location"]
        self.remote_diff_model_dir = self.remote_inversionson_dir / "DIFFUSION_MODELS"
        self.fast_mesh_dir = self.remote_inversionson_dir / "MESHES"
        self.batch_size = self.info["batch_size"]
        self.val_it_interval = self.info["inversion_monitoring"][
            "iterations_between_validation_checks"
        ]
        self.use_model_averaging = self.info["inversion_monitoring"]["use_model_averaging"]
        self.validation_dataset = self.info["inversion_monitoring"][
            "validation_dataset"
        ]
        self.test_dataset = self.info["inversion_monitoring"]["test_dataset"]

        self.inversion_params = self.arrange_params(self.info["inversion_parameters"])
        self.modelling_params = self.arrange_params(self.info["modelling_parameters"])

        # Some useful paths
        self.paths = {}
        self.paths["inversion_root"] = pathlib.Path(self.inversion_root)
        self.paths["lasif_root"] = pathlib.Path(self.lasif_root)
        self.paths["documentation"] = self.inversion_root / "DOCUMENTATION"
        if not os.path.exists(self.paths["documentation"]):
            os.makedirs(self.paths["documentation"])
            os.mkdir(self.paths["documentation"] / "BACKUP")

        optimizer = self.get_optimizer()
        self.smoothing_tensor_order = optimizer.get_tensor_order(
            optimizer.initial_model
        )
        self.smoothing_timestep = optimizer.smoothing_timestep

        if self.meshes == "multi-mesh" or self.remote_data_processing:
            self.prepare_forward = True
        else:
            self.prepare_forward = False

        if not first:
            self.current_iteration = optimizer.iteration_name
            self.print(
                f"Current Iteration: {self.current_iteration}",
                line_above=True,
                line_below=True,
                emoji_alias=":date:",
            )

        self.paths["iteration_tomls"] = self.paths["documentation"] / "ITERATIONS"

        if not os.path.exists(self.paths["iteration_tomls"]):
            os.makedirs(self.paths["iteration_tomls"])

    def create_iteration_toml(self, iteration: str):
        """
        Create the toml file for an iteration. This toml file is then updated.
        To create the toml, we need the events

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )

        if os.path.exists(iteration_toml):
            warnings.warn(
                f"Iteration toml for iteration: {iteration} already exists. backed it up",
                InversionsonWarning,
            )
            backup = os.path.join(
                self.paths["iteration_tomls"], f"backup_{iteration}.toml"
            )
            shutil.copyfile(iteration_toml, backup)

        it_dict = dict(name=iteration, events={})
        if self.meshes == "mono-mesh":
            it_dict["remote_simulation_mesh"] = None

        job_dict = dict(name="", submitted=False, retrieved=False, reposts=0)

        for _i, event in enumerate(self.comm.lasif.list_events(iteration=iteration)):
            if self.is_validation_event(event):
                jobs = {"forward": job_dict}
                if self.prepare_forward:
                    jobs["prepare_forward"] = job_dict
            if not self.is_validation_event(event):
                jobs = {
                    "forward": job_dict,
                    "adjoint": job_dict,
                }
                if self.prepare_forward:
                    jobs["prepare_forward"] = job_dict
                if self.remote_interp:
                    jobs["gradient_interp"] = job_dict
                if self.hpc_processing and not self.is_validation_event(event):
                    jobs["hpc_processing"] = job_dict
            it_dict["events"][str(_i)] = {
                "name": event,
                "job_info": jobs,
            }
            if not self.is_validation_event(event):
                it_dict["events"][str(_i)]["misfit"] = float(0.0)
                it_dict["events"][str(_i)]["usage_updated"] = False

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def change_attribute(self, attribute: str, new_value):
        """
        Not possible to change attributes from another class.
        This method should take care of it

        :param attribute: Name of attribute
        :type attribute: str
        :param new_value: The new value to assign to the attribute
        :type new_value: whatever the attribure needs
        """
        if isinstance(new_value, str):
            command = f'self.{attribute} = "{new_value}"'
        elif isinstance(new_value, list):
            command = f"self.{attribute} = {new_value}"
        elif isinstance(new_value, bool):
            command = f"self.{attribute} = {new_value}"
        elif isinstance(new_value, dict):
            command = f"self.{attribute} = {new_value}"
        elif isinstance(new_value, float):
            command = f"self.{attribute} = {new_value}"
        elif isinstance(new_value, int):
            command = f"self.{attribute} = {new_value}"
        else:
            raise InversionsonError(f"Method not implemented for type {new_value.type}")
        exec(command)

    def update_iteration_toml(self, iteration="current", validation=False):
        """
        Use iteration parameters to update iteration toml file

        :param iteration: Name of iteration
        :type iteration: str
        """
        if iteration == "current":
            iteration = self.current_iteration

        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"Iteration toml for iteration: {iteration} does not exists"
            )

        it_dict = toml.load(iteration_toml)

        # This definitely needs improvement
        for _i, event in enumerate(self.comm.lasif.list_events(iteration=iteration)):
            jobs = {"forward": self.forward_job[event]}
            if not self.is_validation_event(event):
                jobs["adjoint"] = self.adjoint_job[event]
            if self.prepare_forward:
                jobs["prepare_forward"] = self.prepare_forward_job[event]

            if self.remote_interp and not self.is_validation_event(event):
                    jobs["gradient_interp"] = self.gradient_interp_job[event]
            if self.hpc_processing and not self.is_validation_event(event):
                jobs["hpc_processing"] = self.hpc_processing_job[event]

            it_dict["events"][str(_i)] = {
                "name": event,
                "job_info": jobs,
            }
            if not self.is_validation_event(event):
                it_dict["events"][str(_i)]["misfit"] = float(self.misfits[event])
                it_dict["events"][str(_i)]["usage_updated"] = self.updated[event]

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def get_iteration_attributes(self):
        """
        Save the attributes of the current iteration into memory

        :param iteration: Name of iteration
        :type iteration: str
        """
        optimizer = self.get_optimizer()
        iteration = optimizer.iteration_name

        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        if not os.path.exists(iteration_toml):
            raise InversionsonError(f"No toml file exists for iteration: {iteration}")

        it_dict = toml.load(iteration_toml)

        self.iteration_name = it_dict["name"]
        self.current_iteration = self.iteration_name
        self.events_in_iteration = self.comm.lasif.list_events(iteration=iteration)
        self.non_val_events_in_iteration = list(set(self.events_in_iteration) -
                                                 set(self.validation_dataset))
        self.adjoint_job = {}
        self.misfits = {}
        self.updated = {}

        self.prepare_forward_job = {}
        self.forward_job = {}
        self.hpc_processing_job = {}
        self.gradient_interp_job = {}

        # Not sure if it's worth it to include station misfits
        for _i, event in enumerate(self.events_in_iteration):
            if not self.is_validation_event(event):
                self.updated[event] = it_dict["events"][str(_i)]["usage_updated"]
                self.misfits[event] = it_dict["events"][str(_i)]["misfit"]

                self.adjoint_job[event] = it_dict["events"][str(_i)]["job_info"][
                    "adjoint"
                ]
            self.forward_job[event] = it_dict["events"][str(_i)]["job_info"]["forward"]
            if self.prepare_forward:
                self.prepare_forward_job[event] = it_dict["events"][str(_i)]["job_info"][
                    "prepare_forward"
                ]
            if self.remote_interp:
                if not self.is_validation_event(event):
                    self.gradient_interp_job[event] = it_dict["events"][str(_i)][
                        "job_info"
                    ]["gradient_interp"]
            if self.hpc_processing and not self.is_validation_event(event):
                self.hpc_processing_job[event] = it_dict["events"][str(_i)]["job_info"][
                    "hpc_processing"
                ]

    def get_old_iteration_info(self, iteration: str) -> dict:
        """
        For getting information about something else than current iteration

        :param iteration: Name of iteration
        :type iteration: str
        :return: Information regarding that iteration
        :rtype: dict
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        if not os.path.exists(iteration_toml):
            raise InversionsonError(f"No toml file exists for iteration: {iteration}")

        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)
        return it_dict

    def is_validation_event(self, event: str):
        if event in self.validation_dataset:
            return True
        else:
            return False

    def get_key_number_for_event(self, event: str, iteration: str = "current") -> str:
        """
        Due to an annoying problem with toml. We can not use event names
        as keys in the iteration dictionaries. This is a function to find
        the index.
        Lasif returns a sorted list which should always be the same.

        :param event: Name of event
        :type event: str
        :return: The correct key for the event
        :rtype: str
        """
        if iteration == "current":
            iteration = self.current_iteration
        events = self.comm.lasif.list_events(iteration=iteration)
        return str(events.index(event))

    def get_simulation_time_step(self, event=None):
        """
        Get the timestep from a forward job if it does not exist yet.

        Returns the timestep if it is there and managed to do so.
        If an event is passed and it does not exist it, it will get it
        from an stdout file.
        """

        misc_folder = os.path.join(
            self.comm.project.paths["documentation"], "MISC")
        if not os.path.exists(misc_folder):
            os.mkdir(misc_folder)

        timestep_file = os.path.join(misc_folder, "simulation_timestep.toml")

        if not os.path.exists(timestep_file) and event is not None:
            local_stdout = os.path.join(
                self.comm.project.paths["documentation"], "stdout_for_timestep_test")
            hpc_cluster = sapi.get_site(self.comm.project.site_name)
            forward_job = self.comm.salvus_flow.get_job(event=event, sim_type="forward")
            stdout = forward_job.path / "stdout"

            hpc_cluster.remote_get(stdout, local_stdout)

            with open(local_stdout, "r") as fh:
                stdout_str = fh.read()
            stdout_str_split = stdout_str.split()
            if os.path.exists(local_stdout):
                os.remove(local_stdout)
            if "(CFL)" in stdout_str_split:
                time_step_idx = stdout_str_split.index("(CFL)") + 1
                try:
                    time_step = float(stdout_str_split[time_step_idx])
                    # basic check to see if timestep makes some sense
                    if time_step > 0.00001 and time_step < 1000:
                        time_step_dict = dict(time_step=time_step)
                        with open(timestep_file, "w") as fh:
                            toml.dump(time_step_dict, fh)
                    self.simulation_time_step = time_step
                    simulation_dict_folder = (
                            self.comm.lasif.lasif_comm.project.paths["salvus_files"]
                            / f"SIMULATION_DICTS")
                    # Clear cache of simulation dicts with the old checkpointing settings.
                    shutil.rmtree(simulation_dict_folder)
                except Exception as e:
                    print(e)
                self.simulation_time_step = None
        else:
            if os.path.exists(timestep_file):
                time_dict = toml.load(timestep_file)
                self.simulation_time_step = time_dict["time_step"]
            else:
                self.simulation_time_step = None
