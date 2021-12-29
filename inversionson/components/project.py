"""
A class which includes information regarding inversion
and sets up all the components that are needed inside
the inversion itself.
"""
# from __future__ import absolute_import

import os
import toml
import shutil
from inversionson import InversionsonError, InversionsonWarning
import warnings
from inversionson.optimizers.adam_optimizer import BOOL_ADAM, AdamOptimizer

from lasif.components.communicator import Communicator
from lasif.components.component import Component
from .lasif_comp import LasifComponent
from .multimesh_comp import MultiMeshComponent
from .flow_comp import SalvusFlowComponent
from .mesh_comp import SalvusMeshComponent
from .opt_comp import SalvusOptComponent
from .storyteller import StoryTellerComponent
from .batch_comp import BatchComponent
from .smooth_comp import SalvusSmoothComponent


class ProjectComponent(Component):
    def __init__(self, information_dict: dict):
        """
        Initiate everything to make it work correctly. Make sure that
        a running inversion can be restarted although this is done.
        """
        self.info = information_dict
        self.__comm = Communicator()
        super(ProjectComponent, self).__init__(self.__comm, "project")
        self.simulation_dict = self._read_config_file()
        self.get_inversion_attributes(first=True)
        self.__setup_components()
        self.get_inversion_attributes(first=False)
        self._validate_inversion_project()
        self.remote_gradient_processing = True

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
                (
                    solver_settings["end_time_in_s"]
                    - simulation_info["start_time"]
                )
                / solver_settings["time_step_in_s"]
            )
        )
        simulation_info["end_time"] = solver_settings["end_time_in_s"]
        simulation_info["time_step"] = solver_settings["time_step_in_s"]
        simulation_info["min_period"] = solver_settings["minimum_period_in_s"]
        simulation_info["max_period"] = solver_settings["maximum_period_in_s"]
        simulation_info["attenuation"] = config_dict["salvus_settings"][
            "attenuation"
        ]
        simulation_info["ocean_loading"] = config_dict["salvus_settings"][
            "ocean_loading"
        ]
        simulation_info["absorbing_boundaries_length"] = config_dict[
            "salvus_settings"
        ]["absorbing_boundaries_in_km"]
        simulation_info["domain_file"] = config_dict["lasif_project"][
            "domain_settings"
        ]["domain_file"]

        return simulation_info

    def get_communicator(self):
        return self.__comm

    def _validate_inversion_project(self):
        """
        Make sure everything is correctly set up in order to perform inversion.

        :param info_dict: Information needed
        :type info_dict: dict
        :param simulation_dict: Information regarding simulations
        :type simulation_dict: dict
        """
        import pathlib

        allowed_interp_modes = ["local", "remote"]
        if "inversion_id" not in self.info.keys():
            raise ValueError("The inversion needs a name, Key: inversion_id")

        if "inversion_path" not in self.info.keys():
            raise InversionsonError(
                "We need a given path for the inversion root directory."
                " Key: inversion_path"
            )
        if self.info["dropout_probability"] >= 1.0:
            raise InversionsonError(
                "The dropout probability should be between 0.0 and 0.99."
            )

        if "meshes" not in self.info.keys():
            raise InversionsonError(
                "We need information on which sorts of meshes you use. "
                "Options are multi-mesh or mono-mesh. "
                "Key: meshes"
            )

        if "interpolation_mode" not in self.info.keys():
            raise InversionsonError(
                "We need information on how you want to interpolate "
                "between meshes, local or remote. "
                "If you use mono-mesh, just put 'local'"
                "Key: interpolation_mode "
            )

        if self.info["interpolation_mode"] not in allowed_interp_modes:
            raise InversionsonError(
                f"The allowable model_interpolation_modes are: "
                f" {allowed_interp_modes}"
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
        if self.info["interpolation_mode"] == "remote":
            if "interpolation" not in self.info["HPC"].keys():
                raise InversionsonError(
                    "We need to know some info on your remote interpolations"
                )

            if (
                "model_wall_time"
                not in self.info["HPC"]["interpolation"].keys()
            ):
                raise InversionsonError(
                    "We need to know the wall time of your model "
                    " interpolations. Key: HPC.interpolation.model_wall_time"
                )

            if (
                "gradient_wall_time"
                not in self.info["HPC"]["interpolation"].keys()
            ):
                raise InversionsonError(
                    "We need to know the wall time of your model "
                    " interpolations. Key: HPC.interpolation.gradient_wall_time"
                )
            if (
                "remote_mesh_directory"
                not in self.info["HPC"]["interpolation"].keys()
            ):
                raise InversionsonError(
                    "We need to know the location where the meshes are stored"
                    ". Key: HPC.interpolation.remote_mesh_directory"
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

        # Removed because it's not fully supported
        # if "site_name" not in self.info["HPC"]["diffusion_equation"].keys():
        #     raise InversionsonError(
        #         "We need information on the site where jobs are submitted. "
        #         "Key: HPC.diffusion_equation.site_name"
        #     )

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

        if (
            "diff_model_directory"
            not in self.info["HPC"]["diffusion_equation"].keys()
        ):
            raise InversionsonError(
                "We need information on where diffusion models will be stored"
                " in order to be reused. "
                "Key: HPC.diffusion_equation.diff_model_directory"
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

        if "random_event_fraction" not in self.info.keys():
            raise InversionsonError(
                "We need information regarding how many events should be "
                "randomly picked when all events have been used. "
                "Key: random_event_fraction"
            )

        if (
            self.info["random_event_fraction"] > 1
            or self.info["random_event_fraction"] < 0
            or not isinstance(self.info["random_event_fraction"], float)
        ):
            raise InversionsonError(
                "random_event_fraction should be a float"
                "and lie between 0.0 and 1.0"
            )

        if "min_ctrl_group_size" not in self.info.keys():
            raise InversionsonError(
                "We need information regarding minimum control group size."
                " Key: min_ctrl_group_size"
            )

        if "inversion_mode" not in self.info.keys():
            raise InversionsonError(
                "We need information on inversion mode. mini-batch or normal"
            )

        if self.info["inversion_mode"] not in ["mini-batch", "mono-batch"]:
            raise InversionsonError(
                "Only implemented inversion modes are mini-batch or mono-batch"
            )

        if "meshes" not in self.info.keys():
            raise InversionsonError(
                "We need to know what sorts of meshes you use. "
                "Either mono-mesh for simulation mesh = inversion mesh "
                "or multi-mesh for wavefield adapted meshes. "
                "Key: meshes"
            )

        if self.info["meshes"] not in ["mono-mesh", "multi-mesh"]:
            raise InversionsonError(
                "We only accept 'mono-mesh' or 'multi-mesh'"
            )

        # Smoothing
        if "Smoothing" not in self.info.keys():
            raise InversionsonError(
                "Please specify smoothing parameters in info file. "
                "Key: Smoothing"
            )

        if "smoothing_mode" not in self.info["Smoothing"].keys():
            raise InversionsonError(
                "Please specify smoothing mode under Smoothing in info file. "
                "Key: Smoothing.smoothing_mode"
            )

        if self.info["Smoothing"]["smoothing_mode"] not in [
            "anisotropic",
            "isotropic",
            "none",
        ]:
            raise InversionsonError(
                "Only implemented smoothing modes are 'anisotropic', "
                "'isotropic' and 'none'"
            )

        if "timestep" not in self.info["Smoothing"].keys():
            raise InversionsonError(
                "Please specify the timestep you want for your smoothing "
                "The total time is 1 second so it needs to be a fraction of "
                "that. Key: Smoothing.timestep"
            )
        if self.info["Smoothing"]["timestep"] > 0.5:
            raise InversionsonError(
                "Smoothing timestep can not be larger than 0.5 seconds"
            )
        if not self.info["Smoothing"]["smoothing_mode"] == "none":
            if "smoothing_lengths" not in self.info["Smoothing"].keys():
                raise InversionsonError(
                    "Please specify smoothing lengths under Smoothing in info "
                    "file. Key: Smoothing.smoothing_lengths"
                )

        if self.info["Smoothing"]["smoothing_mode"] == "anisotropic":
            if not isinstance(
                self.info["Smoothing"]["smoothing_lengths"], list
            ):
                raise InversionsonError(
                    "Make sure you input a list as smoothing_lengths if you "
                    "want to smooth anisotropically. List of length 3. "
                    "Order: r, theta, phi."
                )
            if not len(self.info["Smoothing"]["smoothing_lengths"]) == 3:
                raise InversionsonError(
                    "Make sure your smoothing_lengths are a list of length 3."
                    "Order: r, theta, phi."
                )

        if self.info["Smoothing"]["smoothing_mode"] == "isotropic":
            if isinstance(self.info["Smoothing"]["smoothing_lengths"], list):
                if len(self.info["Smoothing"]["smoothing_lengths"]) == 1:
                    self.info["Smoothing"]["smoothing_lengths"] = self.info[
                        "Smoothing"
                    ]["smoothing_lengths"][0]
                else:
                    raise InversionsonError(
                        "If you give a list of isotropic lengths, you can only"
                        " give a list of length one, as all dimensions will "
                        "be smoothed with equally many wavelengths. You can "
                        "also just give a number."
                    )
        if (
            "Meshing" not in self.info.keys()
            and self.info["meshes"] == "multi-mesh"
        ):
            raise InversionsonError(
                "We need some information regarding your meshes. "
                "We need to know how many elements you want per azimuthal "
                "quarter. Key: Meshing"
            )
        if self.info["meshes"] == "multi-mesh":
            if (
                "elements_per_azimuthal_quarter"
                not in self.info["Meshing"].keys()
            ):
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

            if "ellipticity" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need a boolean value regarding ellipticity "
                    "of your meshes. \n"
                    "Key: Meshing.ellipticity"
                )
            if "topography" not in self.info["Meshing"].keys():
                raise InversionsonError(
                    "We need information on whether you use topography "
                    "in your mesh."
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
                    if (
                        len(self.info["Meshing"]["topography"]["variable"])
                        == 0
                    ):
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
            self.info["inversion_monitoring"][
                "iterations_between_validation_checks"
            ]
            != 0
        ):
            if (
                len(self.info["inversion_monitoring"]["validation_dataset"])
                == 0
            ):
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
        SalvusOptComponent(communicator=self.comm, component_name="salvus_opt")
        MultiMeshComponent(communicator=self.comm, component_name="multi_mesh")
        SalvusFlowComponent(
            communicator=self.comm, component_name="salvus_flow"
        )
        SalvusMeshComponent(
            communicator=self.comm, component_name="salvus_mesher"
        )
        StoryTellerComponent(
            communicator=self.comm, component_name="storyteller"
        )
        BatchComponent(communicator=self.comm, component_name="minibatch")
        SalvusSmoothComponent(
            communicator=self.comm, component_name="smoother"
        )

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
        case_tti_mod = set(
            ["VSV", "VSH", "VPV", "VPH", "RHO", "QKAPPA", "QMU", "ETA"]
        )
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
                f"Parameter list {parameters} not "
                f"a recognized set of parameters"
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
        self.abs_bound_length = self.simulation_dict[
            "absorbing_boundaries_length"
        ]
        self.absorbing_boundaries = self.info["absorbing_boundaries"]
        self.domain_file = self.simulation_dict["domain_file"]

        # Inversion attributes
        self.inversion_root = self.info["inversion_path"]
        self.lasif_root = self.info["lasif_root"]
        self.inversion_id = self.info["inversion_id"]
        self.inversion_mode = self.info["inversion_mode"]
        self.meshes = self.info["meshes"]
        if self.meshes == "multi-mesh":
            self.elem_per_quarter = self.info["Meshing"][
                "elements_per_azimuthal_quarter"
            ]
            self.topography = self.info["Meshing"]["topography"]
            self.ellipticity = self.info["Meshing"]["ellipticity"]
        self.ocean_loading = self.info["Meshing"]["ocean_loading"]
        self.interpolation_mode = self.info["interpolation_mode"]
        self.cut_source_radius = self.info[
            "cut_source_region_from_gradient_in_km"
        ]
        self.cut_receiver_radius = self.info[
            "cut_receiver_region_from_gradient_in_km"
        ]
        self.clip_gradient = self.info["clip_gradient"]
        self.site_name = self.info["HPC"]["wave_propagation"]["site_name"]
        self.ranks = self.info["HPC"]["wave_propagation"]["ranks"]
        self.wall_time = self.info["HPC"]["wave_propagation"]["wall_time"]
        if self.interpolation_mode == "remote":
            self.model_interp_wall_time = self.info["HPC"]["interpolation"][
                "model_wall_time"
            ]
            self.grad_interp_wall_time = self.info["HPC"]["interpolation"][
                "gradient_wall_time"
            ]
            self.interpolation_site = self.info["HPC"]["interpolation"][
                "site_name"
            ]
            self.remote_mesh_dir = self.info["HPC"]["interpolation"][
                "remote_mesh_directory"
            ]
        # self.smoothing_site_name = self.info["HPC"]["diffusion_equation"][
        #     "site_name"
        # ]
        # We currently assume smoothing site to be simulation site.
        self.smoothing_site_name = self.site_name
        self.smoothing_ranks = self.info["HPC"]["diffusion_equation"]["ranks"]
        self.smoothing_wall_time = self.info["HPC"]["diffusion_equation"][
            "wall_time"
        ]
        self.smoothing_mode = self.info["Smoothing"]["smoothing_mode"]
        self.smoothing_lengths = self.info["Smoothing"]["smoothing_lengths"]
        self.smoothing_timestep = self.info["Smoothing"]["timestep"]
        self.remote_diff_model_dir = self.info["HPC"]["diffusion_equation"][
            "diff_model_directory"
        ]

        self.initial_batch_size = self.info["initial_batch_size"]
        self.random_event_fraction = self.info["random_event_fraction"]
        self.min_ctrl_group_size = self.info["min_ctrl_group_size"]
        self.maximum_grad_divergence_angle = self.info["max_angular_change"]
        self.dropout_probability = self.info["dropout_probability"]
        self.when_to_validate = self.info["inversion_monitoring"][
            "iterations_between_validation_checks"
        ]
        self.validation_dataset = self.info["inversion_monitoring"][
            "validation_dataset"
        ]
        self.test_dataset = self.info["inversion_monitoring"]["test_dataset"]
        if not first:
            if BOOL_ADAM:
                adam_opt = AdamOptimizer(inversion_root=
                                         self.comm.project.
                                         paths["inversion_root"])
                adam_opt.get_iteration_name()
            else:
                self.current_iteration = (
                    self.comm.salvus_opt.get_newest_iteration_name()
                )
            print(f"Current Iteration: {self.current_iteration}")
            self.event_quality = toml.load(
                self.comm.storyteller.events_quality_toml
            )
        self.inversion_params = self.arrange_params(
            self.info["inversion_parameters"]
        )
        self.modelling_params = self.arrange_params(
            self.info["modelling_parameters"]
        )

        # Some useful paths
        self.paths = {}
        self.paths["inversion_root"] = self.inversion_root
        self.paths["lasif_root"] = self.lasif_root
        self.paths["salvus_opt"] = os.path.join(
            self.inversion_root, "SALVUS_OPT"
        )
        if not os.path.exists(self.paths["salvus_opt"]):
            raise InversionsonError(
                "Please make a folder for Salvus opt and initialize it in there"
            )

        self.paths["documentation"] = os.path.join(
            self.inversion_root, "DOCUMENTATION"
        )
        if not os.path.exists(self.paths["documentation"]):
            os.makedirs(self.paths["documentation"])
            os.mkdir(os.path.join(self.paths["documentation"], "BACKUP"))

        self.paths["iteration_tomls"] = os.path.join(
            self.paths["documentation"], "ITERATIONS"
        )
        if not os.path.exists(self.paths["iteration_tomls"]):
            os.makedirs(self.paths["iteration_tomls"])
        # self.paths["salvus_smoother"] = self.info["salvus_smoother"]

        self.paths["control_group_toml"] = os.path.join(
            self.paths["documentation"], "control_groups.toml"
        )

    def create_iteration_toml(self, iteration: str):
        """
        Create the toml file for an iteration. This toml file is then updated.
        To create the toml, we need the events and the control group

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        validation = False
        if "validation" in iteration:
            validation = True

        remote_interp = False
        if self.meshes == "multi-mesh" and self.interpolation_mode == "remote":
            remote_interp = True

        if os.path.exists(iteration_toml):
            warnings.warn(
                f"Iteration toml for iteration: {iteration} already exists. backed it up",
                InversionsonWarning,
            )
            backup = os.path.join(
                self.paths["iteration_tomls"], f"backup_{iteration}.toml"
            )
            shutil.copyfile(iteration_toml, backup)

        it_dict = {}
        it_dict["name"] = iteration
        it_dict["events"] = {}

        if self.meshes == "mono-mesh":
            it_dict["remote_simulation_mesh"] = None

        last_control_group = []
        if (
            iteration != "it0000_model"
            and not validation and not BOOL_ADAM
            and self.inversion_mode == "mini-batch"
        ):
            ctrl_grps = toml.load(
                self.comm.project.paths["control_group_toml"]
            )
            prev_iter = self.comm.salvus_opt.get_previous_iteration_name()
            last_control_group = ctrl_grps[prev_iter]["new"]

        if not validation and self.inversion_mode == "mini-batch":
            it_dict["last_control_group"] = last_control_group
            it_dict["new_control_group"] = []
        f_job_dict = {
            "name": "",
            "submitted": False,
            "retrieved": False,
            "reposts": 0,
        }
        if remote_interp:
            i_job_dict = {
                "name": "",
                "submitted": False,
                "retrieved": False,
                "reposts": 0,
            }
        if validation:
            f_job_dict["windows_selected"] = False
        if not validation:
            a_job_dict = {
                "name": "",
                "submitted": False,
                "retrieved": False,
                "reposts": 0,
            }
            s_job_dict = {
                "name": "",
                "submitted": False,
                "retrieved": False,
                "reposts": 0,
            }
        if self.meshes == "multi-mesh":
            f_job_dict["interpolated"] = False
            if not validation:
                a_job_dict["interpolated"] = False

        for _i, event in enumerate(
            self.comm.lasif.list_events(iteration=iteration)
        ):
            if validation:
                jobs = {"forward": f_job_dict}
                if remote_interp:
                    jobs["model_interp"] = i_job_dict
            if self.inversion_mode == "mini-batch" and not BOOL_ADAM:
                if not validation:
                    jobs = {
                        "forward": f_job_dict,
                        "adjoint": a_job_dict,
                        "smoothing": s_job_dict,
                    }
                    if remote_interp:
                        jobs["model_interp"] = i_job_dict
                        jobs["gradient_interp"] = i_job_dict
                it_dict["events"][str(_i)] = {
                    "name": event,
                    "job_info": jobs,
                }
                # it_dict["events"][event] = {
                #     "job_info": jobs,
                # }
            else:
                if not validation:
                    jobs = {
                        "forward": f_job_dict,
                        "adjoint": a_job_dict,
                    }
                    if remote_interp:
                        jobs["model_interp"] = i_job_dict
                        jobs["gradient_interp"] = i_job_dict
                it_dict["events"][str(_i)] = {
                    "name": event,
                    "job_info": jobs,
                }
            if not validation:
                it_dict["events"][str(_i)]["misfit"] = 0.0
                it_dict["events"][str(_i)]["usage_updated"] = False
        if (self.inversion_mode == "mono-batch" or BOOL_ADAM) and not validation:
            it_dict["smoothing"] = s_job_dict

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
            raise InversionsonError(
                f"Method not implemented for type {new_value.type}"
            )
        exec(command)

    def update_control_group_toml(self, new=False, first=False):
        """
        A toml file for monitoring which control group is used in each
        iteration.
        Structure: dict[iteration] = {old: [], new: []}
        :param new: Should the new control group be updated?
        :type new: bool, optional
        :param first: Does the toml need to be created?
        :type first: bool, optional
        """
        iteration = self.current_iteration
        print(f"Iteration: {iteration}")
        if first:
            cg_dict = {}
            cg_dict[iteration] = {"old": [], "new": []}
            with open(self.paths["control_group_toml"], "w") as fh:
                toml.dump(cg_dict, fh)
                return
        else:
            cg_dict = toml.load(self.paths["control_group_toml"])
            if not new:
                prev_iter = self.comm.salvus_opt.get_previous_iteration_name()
                cg_dict[iteration] = {}
                cg_dict[iteration]["old"] = cg_dict[prev_iter]["new"]
                if new not in cg_dict[iteration].keys():
                    cg_dict[iteration]["new"] = []
            if new:
                if iteration not in cg_dict.keys():
                    cg_dict[iteration] = {}
                cg_dict[iteration]["new"] = self.new_control_group

        with open(self.paths["control_group_toml"], "w") as fh:
            toml.dump(cg_dict, fh)

    def update_iteration_toml(self, iteration="current", validation=False):
        """
        Use iteration parameters to update iteration toml file

        :param iteration: Name of iteration
        :type iteration: str
        """
        if iteration == "current":
            iteration = self.current_iteration
        if "validation" in iteration:
            validation = True
        if validation and "validation" not in iteration:
            iteration = f"validation_{iteration}"
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        remote_interp = False
        if self.meshes == "multi-mesh" and self.interpolation_mode == "remote":
            remote_interp = True
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"Iteration toml for iteration: {iteration} does not exists"
            )
        if os.path.exists(self.paths["control_group_toml"]) and not validation:
            control_group_dict = toml.load(self.paths["control_group_toml"])
            control_group_dict = control_group_dict[iteration]
        elif self.inversion_mode == "mini-batch" and not BOOL_ADAM:
            control_group_dict = {"old": [], "new": []}
        it_dict = {}
        it_dict["name"] = iteration
        it_dict["events"] = {}

        if self.meshes == "mono-mesh":
            it_dict["remote_simulation_mesh"] = self.remote_mesh

        # I need a way to figure out what the controlgroup is
        # This definitely needs improvement
        if not (validation or BOOL_ADAM) and self.inversion_mode == "mini-batch":
            it_dict["last_control_group"] = control_group_dict["old"]
            it_dict["new_control_group"] = control_group_dict["new"]
        for _i, event in enumerate(
            self.comm.lasif.list_events(iteration=iteration)
        ):
            jobs = {"forward": self.forward_job[event]}
            if not validation:
                jobs["adjoint"] = self.adjoint_job[event]
            if remote_interp:
                jobs["model_interp"] = self.model_interp_job[event]
                if not validation:
                    jobs["gradient_interp"] = self.gradient_interp_job[event]
            if self.inversion_mode == "mini-batch" and not BOOL_ADAM:
                if not validation:
                    jobs["smoothing"] = self.smoothing_job[event]
                it_dict["events"][str(_i)] = {
                    "name": event,
                    "job_info": jobs,
                }
            else:
                it_dict["events"][str(_i)] = {
                    "job_info": jobs,
                }
            if not validation:
                it_dict["events"][str(_i)]["misfit"] = self.misfits[event]
                it_dict["events"][str(_i)]["usage_updated"] = self.updated[
                    event
                ]
        if (self.inversion_mode == "mono-batch" or BOOL_ADAM) and not validation:
            it_dict["smoothing"] = self.smoothing_job

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def get_iteration_attributes(self, validation=False):
        """
        Save the attributes of the current iteration into memory

        :param iteration: Name of iteration
        :type iteration: str
        """
        if BOOL_ADAM:
            adam_opt = AdamOptimizer(
                inversion_root=self.comm.project.paths["inversion_root"])
            iteration = adam_opt.get_iteration_name()
        else:
            iteration = self.comm.salvus_opt.get_newest_iteration_name()
        if validation:
            iteration = f"validation_{iteration}"
        remote_interp = False
        if self.meshes == "multi-mesh" and self.interpolation_mode == "remote":
            remote_interp = True
        iteration_toml = os.path.join(
            self.paths["iteration_tomls"], iteration + ".toml"
        )
        if not os.path.exists(iteration_toml):
            raise InversionsonError(
                f"No toml file exists for iteration: {iteration}"
            )

        it_dict = toml.load(iteration_toml)

        self.iteration_name = it_dict["name"]
        self.current_iteration = self.iteration_name
        self.events_in_iteration = self.comm.lasif.list_events(
            iteration=iteration
        )
        if not validation:
            if self.inversion_mode == "mini-batch":
                self.old_control_group = it_dict["last_control_group"]
                self.new_control_group = it_dict["new_control_group"]
            self.adjoint_job = {}
            self.smoothing_job = {}
            self.misfits = {}
            self.updated = {}
        self.forward_job = {}
        if remote_interp:
            self.model_interp_job = {}
            self.gradient_interp_job = {}

        if self.meshes == "mono-mesh":
            if "remote_simulation_mesh" not in it_dict.keys():
                self.remote_mesh = None
            else:
                self.remote_mesh = it_dict["remote_simulation_mesh"]
        else:
            self.remote_mesh = None

        # Not sure if it's worth it to include station misfits
        for _i, event in enumerate(self.events_in_iteration):
            if not validation:
                self.updated[event] = it_dict["events"][str(_i)][
                    "usage_updated"
                ]
                self.misfits[event] = it_dict["events"][str(_i)]["misfit"]

                self.adjoint_job[event] = it_dict["events"][str(_i)][
                    "job_info"
                ]["adjoint"]
                if self.inversion_mode == "mini-batch" and not BOOL_ADAM:
                    self.smoothing_job[event] = it_dict["events"][str(_i)][
                        "job_info"
                    ]["smoothing"]
            self.forward_job[event] = it_dict["events"][str(_i)]["job_info"][
                "forward"
            ]
            if remote_interp:
                self.model_interp_job[event] = it_dict["events"][str(_i)][
                    "job_info"
                ]["model_interp"]
                if not validation:
                    self.gradient_interp_job[event] = it_dict["events"][
                        str(_i)
                    ]["job_info"]["gradient_interp"]
        if self.inversion_mode == "mono-batch" and not validation:
            self.smoothing_job = it_dict["smoothing"]

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
            raise InversionsonError(
                f"No toml file exists for iteration: {iteration}"
            )

        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)
        return it_dict

    def get_key_number_for_event(
        self, event: str, iteration: str = "current"
    ) -> str:
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
