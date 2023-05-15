from __future__ import annotations, absolute_import
import os
import toml
import shutil
from pathlib import Path
from inversionson import InversionsonWarning
import warnings
from typing import Optional, Union, List
from inversionson.file_templates.inversion_info_template import InversionsonConfig
from inversionson.utils import get_tensor_order


class RemotePaths:
    def __init__(self, project: Project):
        # Directories
        self.project = project
        self.root = self.project.config.hpc.inversionson_folder
        self.diff_dir = self.root / "DIFFUSION_MODELS"
        self.stf_dir = self.root / "SOURCE_TIME_FUNCTIONS"
        self.interp_weights_dir = self.root / "INTERPOLATION_WEIGHTS"
        self.multi_mesh_dir = self.root / "MULTI_MESHES"
        self.window_dir = self.root / "WINDOWS"
        self.misfit_dir = self.root / "MISFITS"
        self.model_dir = self.root / "MODELS"
        self.adj_src_dir = self.root / "ADJOINT_SOURCES"
        self.receiver_dir = self.root / "RECEIVERS"
        self.script_dir = self.root / "SCRIPTS"
        self.proc_data_dir = self.root / "PROCESSED_DATA"

        # Remote files
        self.ocean_loading_f = self.root / "ocean_loading_file"
        self.topography_f = self.root / "topography_file"
        self.interp_script = self.script_dir / "interpolation.py"

    @property
    def master_gradient(self):
        """This is the target for interpolating gradients to"""
        return self.multi_mesh_dir / "standard_gradient.h5"

    def get_master_model_path(self, iteration: Optional[str] = None):
        """This is the path to the master or mono-mesh model"""
        iteration = iteration or self.project.current_iteration
        return self.model_dir / f"{iteration}.h5"

    def get_event_specific_model(self, event: str):
        """Get the already interpolated event-specific model/mesh combination used for forward runs."""
        job = self.project.flow.get_job(
            event=event,
            sim_type="prepare_forward",
            iteration=self.project.current_iteration,
        )
        return job.stdout_path.parent / "output" / "mesh.h5"

    def get_event_specific_gradient(self, event: str) -> Path:
        output = self.project.flow.get_job_file_paths(event=event, sim_type="adjoint")
        return output[0][("adjoint", "gradient", "output_filename")]

    def get_event_specific_mesh_path(self, event: str):
        """This is the path to the cached event-specific mesh without the model interpolated"""
        return self.multi_mesh_dir / f"{event}.h5"

    def create_remote_directories(self, hpc_cluster):
        all_directories = [
            d for d in self.__dict__.values() if isinstance(d, Path) and d.is_dir()
        ]
        for directory in all_directories:
            if not hpc_cluster.remote_exists(directory):
                hpc_cluster.remote_mkdir(directory)


class ProjectPaths:
    def __init__(self, project: Project):
        self.project = project
        self.root = self.project.config.inversion_path
        self.lasif_root = self.project.config.lasif_root
        assert self.root.is_dir()
        assert self.lasif_root.is_dir()
        self.doc_dir = self.root / "DOCUMENTATION"
        self.iteration_tomls = self.doc_dir / "ITERATIONS"
        self.misc_dir = self.doc_dir / "MISC"
        self.diff_model_dir = self.root / "DIFFUSION_MODELS"

        self.opt_dir = self.root / "OPTIMIZATION"
        self.reg_dir = self.opt_dir / "REGULARIZATION"
        self.gradient_norm_dir = self.root / "GRADIENT_NORMS"
        self.all_gradient_norms_toml = self.gradient_norm_dir / "all_gradients_norms.toml"
        self.model_dir = self.opt_dir / "MODELS"

    def _initialize_dirs(self):
        all_directories = [
            d for d in self.__dict__.values() if isinstance(d, Path) and d.is_dir()
        ]
        for directory in all_directories:
            if not directory.is_dir():
                directory.mkdir()

    def get_model_path(self, model_descriptor: str) -> Path:
        return self.model_dir / f"{model_descriptor}.h5"

    def get_iteration_toml(self, iteration: str) -> Path:
        return self.iteration_tomls / f"{iteration}.toml"

    def gradient_norms_path(self, iteration: Optional[str] = None) -> Path:
        iteration = iteration or self.project.current_iteration
        return self.gradient_norm_dir / f"gradient_norms_{iteration}.toml"


class LASIFSimulationSettings:
    def __init__(self, lasif_config_path: Union[Path, str]):
        with open(lasif_config_path, "r") as fh:
            config_dict = toml.load(fh)

        self.simulation_settings = config_dict["simulation_settings"]
        self.start_time: float = self.simulation_settings["start_time_in_s"]
        self.end_time: float = self.simulation_settings["end_time_in_s"]
        self.time_step: float = self.simulation_settings["time_step_in_s"]

        self.number_of_time_steps: int = int(
            round((self.end_time - self.start_time) / self.time_step) + 1
        )

        self.min_period: float = self.simulation_settings["minimum_period_in_s"]
        self.max_period: float = self.simulation_settings["maximum_period_in_s"]
        self.attenuation: bool = config_dict["salvus_settings"]["attenuation"]
        self.salvus_settings = config_dict["salvus_settings"]
        self.ocean_loading: bool = self.salvus_settings["ocean_loading"]
        self.absorbing_boundaries_length: float = self.salvus_settings[
            "absorbing_boundaries_in_km"
        ]


class Project(object):
    # The below stuff is hardcoded for now
    ad_src_type = "tf_phase_misfit"

    def __init__(self, config: InversionsonConfig):
        """
        Initiate everything to make it work correctly. Make sure that
        a running inversion can be restarted although this is done.
        """
        self.config = config
        self.paths = ProjectPaths(self.config)
        self.remote_paths = RemotePaths(self)

        self.lasif_settings = LASIFSimulationSettings(
            self.config.lasif_root / "lasif_config.toml"
        )

        self._validate_inversion_project()
        self._initialize_components()
        self.simulation_time_step: Union[float, None] = None

        self.tensor_order = get_tensor_order(self.config.inversion.initial_model)
        self.find_simulation_time_step()

    def _initialize_components(self):
        from inversionson.components.lasif_comp import LASIF
        from .components.multimesh_comp import MultiMesh
        from .components.flow_comp import SalvusFlow
        from .components.mesh_comp import Mesh
        from .components.storyteller import StoryTeller
        from .components.smooth_comp import Smoother
        from .components.event_db import EventDataBase

        # Project acts as a communicator that contains a bunch of components.
        self.lasif = LASIF(project=self)
        self.multi_mesh = MultiMesh(project=self)
        self.flow = SalvusFlow(project=self)
        self.salvus_mesher = Mesh(project=self)
        self.storyteller = StoryTeller(project=self)
        self.smoother = Smoother(project=self)
        self.event_db = EventDataBase(project=self)

    def print(
        self,
        message: str,
        color: Optional[str] = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = None,
    ):
        self.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def _validate_inversion_project(self):
        """
        This used to check the inputs in inversion_info.toml.
        TODO: reimplement this.
        """
        assert self.config.inversion.initial_model.name.endswith(
            ".h5"
        ), "Provide a valid initial model."

    def create_iteration_toml(self, iteration: str, events: List[str]):
        """
        Create the toml file for an iteration. This toml file is then updated.
        To create the toml, we need the events

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = self.paths.get_iteration_toml(iteration)

        if os.path.exists(iteration_toml):
            warnings.warn(
                f"Iteration toml for iteration: {iteration} already exists. backed it up",
                InversionsonWarning,
            )
            backup = self.paths.iteration_tomls / f"backup_{iteration}.toml"
            shutil.copyfile(iteration_toml, backup)

        it_dict = {"name": iteration, "events": {}}
        if not self.config.meshing.multi_mesh:
            it_dict["remote_simulation_mesh"] = ""

        job_dict = dict(name="", submitted=False, retrieved=False, reposts=0)

        for event in events:
            str_idx = str(self.event_db.get_event_idx(event))
            if self.is_validation_event(event):
                jobs = {"prepare_forward": job_dict, "forward": job_dict}

            if not self.is_validation_event(event):
                jobs.update(
                    {
                        "hpc_processing": job_dict,
                        "adjoint": job_dict,
                    }
                )
                if self.config.meshing.multi_mesh:
                    jobs.update({"gradient_interp": job_dict})

            it_dict["events"][str_idx] = {
                "name": event,
                "job_info": jobs,
            }
            if not self.is_validation_event(event):
                it_dict["events"][str_idx]["misfit"] = 0.0
                it_dict["events"][str_idx]["usage_updated"] = False

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def change_attribute(
        self, attribute: str, new_value: Union[list, bool, dict, float, int, str]
    ):
        assert isinstance(attribute, str)
        if type(new_value) not in [list, bool, dict, float, int, str]:
            raise ValueError(f"Method not implemented for type {type(new_value)}.")
        setattr(self, attribute, new_value)
        self.update_iteration_toml()

    def update_iteration_toml(self, iteration: Optional[str] = None):
        """Update iteration the iteration toml file."""
        iteration = iteration or self.current_iteration
        iteration_toml = self.paths.get_iteration_toml(iteration)
        assert iteration_toml.exists(), f"Iteration toml {iteration_toml} not found."

        it_dict = toml.load(iteration_toml)
        for ev in it_dict["events"].values():
            event = ev["name"]
            str_idx = str(self.event_db.get_event_idx(event))

            jobs = {
                "prepare_forward": self.prepare_forward_job[event],
                "forward": self.forward_job[event],
            }

            if not self.is_validation_event(event):
                jobs["hpc_processing"] = self.hpc_processing_job[event]
                jobs["adjoint"] = self.adjoint_job[event]
                if self.config.meshing.multi_mesh:
                    jobs["gradient_interp"] = self.gradient_interp_job[event]

            it_dict["events"][str_idx] = {
                "name": event,
                "job_info": jobs,
            }
            if not self.is_validation_event(event):
                it_dict["events"][str_idx]["misfit"] = float(self.misfits[event])
                it_dict["events"][str_idx]["usage_updated"] = self.updated[event]

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def set_iteration_attributes(self, iteration: str):
        """Set iteration attributes from iterationt toml."""
        iteration_toml = self.paths.get_iteration_toml(iteration)
        assert iteration_toml.exists()

        it_dict = toml.load(iteration_toml)
        self.current_iteration = it_dict["name"]
        self.events_in_iteration = self.event_db.get_event_names(
            list(it_dict["events"].keys())
        )
        self.non_val_events_in_iteration = list(
            set(self.events_in_iteration)
            - set(self.config.monitoring.validation_dataset)
        )
        self.adjoint_job = {}
        self.misfits = {}
        self.updated = {}
        self.prepare_forward_job = {}
        self.forward_job = {}
        self.hpc_processing_job = {}
        self.gradient_interp_job = {}

        for event in self.events_in_iteration:
            ev_idx = str(self.event_db.get_event_idx(event))
            self.prepare_forward_job[event] = it_dict["events"][ev_idx]["job_info"][
                "prepare_forward"
            ]
            self.forward_job[event] = it_dict["events"][ev_idx]["job_info"]["forward"]
            if not self.is_validation_event(event):
                self.hpc_processing_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "hpc_processing"
                ]
                self.misfits[event] = it_dict["events"][ev_idx]["misfit"]
                self.adjoint_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "adjoint"
                ]
                self.gradient_interp_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "gradient_interp"
                ]
                self.updated[event] = it_dict["events"][ev_idx]["usage_updated"]

    def get_old_iteration_info(self, iteration: str) -> dict:
        """
        For getting information about something else than current iteration.

        :param iteration: Name of iteration
        :type iteration: str
        :return: Information regarding that iteration
        :rtype: dict
        """
        iteration_toml = self.paths.get_iteration_toml(iteration)
        assert iteration_toml.exists(), f"Iteration toml {iteration_toml} not found."

        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)
        return it_dict

    def is_validation_event(self, event: str) -> bool:
        return event in self.config.monitoring.validation_dataset

    def find_simulation_time_step(self, event: Optional[str] = None):
        """
        Get the timestep from a forward job if it does not exist yet.

        Returns the timestep if it is there and managed to do so.
        If an event is passed and it does not exist it, it will get it
        from an stdout file.
        """
        timestep_file = self.paths.misc_dir / "simulation_timestep.toml"

        if event and not timestep_file.exists():
            self._get_timestep_from_stdout(event, timestep_file)
        elif timestep_file.exists():
            time_dict = toml.load(timestep_file)
            self.simulation_time_step = time_dict["time_step"]

    def _get_timestep_from_stdout(self, event, timestep_file):
        """Read the timestep value from a stdout produced by salvus run."""
        local_stdout = self.paths.doc_dir / "stdout_to_find_tinestep"
        hpc_cluster = self.flow.hpc_cluster
        forward_job = self.flow.get_job(event=event, sim_type="forward")
        stdout = forward_job.path / "stdout"
        hpc_cluster.remote_get(stdout, local_stdout)

        with open(local_stdout, "r") as fh:
            stdout_str = fh.read()

        stdout_str_split = stdout_str.split()
        if local_stdout.exists():
            local_stdout.unlink()  # remove file
        if "(CFL)" in stdout_str_split:
            time_step_idx = stdout_str_split.index("(CFL)") + 1
            try:
                time_step = float(stdout_str_split[time_step_idx])
                # basic check to see if timestep makes some sense
                if time_step > 0.0 and time_step < 1000.0:
                    time_step_dict = dict(time_step=time_step)
                    with open(timestep_file, "w") as fh:
                        toml.dump(time_step_dict, fh)
                self.simulation_time_step = time_step
                simulation_dict_folder = (
                    self.lasif.lasif_comm.project.paths["salvus_files"]
                    / "SIMULATION_DICTS"
                )
                # Clear cache of simulation dicts with the old checkpointing settings.
                shutil.rmtree(simulation_dict_folder)
            except Exception as e:
                print(e)
