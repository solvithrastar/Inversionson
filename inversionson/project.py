from __future__ import annotations, absolute_import

import os
import toml
import shutil
from pathlib import Path
from inversionson import InversionsonWarning
import warnings
from typing import Optional, Union, List
import salvus.flow.api as sapi

from inversionson.file_templates.inversion_info_template import InversionsonConfig


class RemotePaths:
    def __init__(self, config: InversionsonConfig):
        self.root = config.hpc.inversionson_folder

        self.diff_dir = self.root / "DIFFUSION_MODELS"
        self.stf_dir = self.root / "SOURCE_TIME_FUNCTIONS"
        self.interp_weights_dir = self.root / "INTERPOLATION_WEIGHTS"
        self.mesh_dir = self.root / "MESHES"
        self.window_dir = self.root / "WINDOWS"
        self.misfit_dir = self.root / "MISFITS"
        self.adj_src_dir = self.root / "ADJOINT_SOURCES"
        self.receiver_dir = self.root / "RECEIVERS"
        self.script_dir = self.root / "SCRIPTS"
        self.proc_data_dir = self.root / "PROCESSED_DATA"

        self.ocean_loading_f = self.root / "ocean_loading_file"
        self.topography_f = self.root / "topography_file"
        self.interp_script = self.script_dir / "interpolation.py"

    def create_remote_directories(self, hpc_cluster):
        if not hpc_cluster.remote_exists(self.root):
            hpc_cluster.remote_mkdir(self.root)

        for directory in [
            self.diff_dir,
            self.stf_dir,
            self.interp_weights_dir,
            self.mesh_dir,
            self.window_dir,
            self.misfit_dir,
            self.adj_src_dir,
            self.receiver_dir,
            self.script_dir,
            self.proc_data_dir,
        ]:
            if not hpc_cluster.remote_exists(self.root / directory):
                hpc_cluster.remote_mkdir(self.root / directory)


class ProjectPaths:
    def __init__(self, config: InversionsonConfig):
        self.root = config.inversion_path
        self.lasif_root = config.lasif_root
        self.documentation = self.root / "DOCUMENTATION"
        self.documentation.mkdir(exist_ok=True)
        self.iteration_tomls = self.documentation / "ITERATIONS"
        self.iteration_tomls.mkdir(exist_ok=True)
        self.misc_folder = self.documentation / "MISC"
        self.misc_folder.mkdir(exist_ok=True)

        self.remote_ocean_loading_f = (
            config.hpc.inversionson_folder / "ocean_loading_file"
        )
        self.remote_topography_f = config.hpc.inversionson_folder / "topography_file"

    def get_iteration_toml(self, iteration: str) -> Path:
        return self.iteration_tomls / f"{iteration}.toml"


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
        self.domain_file: str = config_dict["lasif_project"]["domain_settings"][
            "domain_file"
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
        self.remote_paths = RemotePaths(self.config)

        self.simulation_settings = LASIFSimulationSettings(
            self.config.lasif_root / "lasif_config.toml"
        )
        self.__setup_components()
        self.simulation_time_step: Union[float, None] = None
        # Attempt to get the simulation timestep immediately if it exists.
        self.find_simulation_time_step()

    def __setup_components(self):
        from inversionson.components.lasif_comp import Lasif
        from .components.multimesh_comp import MultiMesh
        from .components.flow_comp import SalvusFlow
        from .components.mesh_comp import Mesh
        from .components.storyteller import StoryTeller
        from .components.smooth_comp import Smoother
        from .components.event_db import EventDataBase

        # Project acts as a communicator that contains a bunch of components.
        self.lasif = Lasif(project=self)
        self.multi_mesh = MultiMesh(project=self)
        self.flow = SalvusFlow(project=self)
        self.salvus_mesher = Mesh(project=self)
        self.storyteller = StoryTeller(project=self)
        self.smoother = Smoother(project=self)
        self.event_db = EventDataBase(project=self)

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
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
        return

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

        it_dict = dict(name=iteration, events={})
        if self.meshes == "mono-mesh":
            it_dict["remote_simulation_mesh"] = None

        job_dict = dict(name="", submitted=False, retrieved=False, reposts=0)

        for event in events:
            str_idx = str(self.event_db.get_event_idx(event))
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
        """
        Store iteration attributes in iteration_toml file.

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration = iteration or self.current_iteration
        iteration_toml = self.paths.get_iteration_toml(iteration)

        if not iteration_toml.exists():
            raise FileNotFoundError(
                f"Iteration toml for iteration: {iteration} does not exists"
            )

        it_dict = toml.load(iteration_toml)

        for ev in it_dict["events"].values():
            event = ev["name"]
            str_idx = str(self.event_db.get_event_idx(event))

            jobs = {"forward": self.forward_job[event]}
            if not self.is_validation_event(event):
                jobs["adjoint"] = self.adjoint_job[event]
            if self.prepare_forward:
                jobs["prepare_forward"] = self.prepare_forward_job[event]

            if self.remote_interp and not self.is_validation_event(event):
                jobs["gradient_interp"] = self.gradient_interp_job[event]
            if self.hpc_processing and not self.is_validation_event(event):
                jobs["hpc_processing"] = self.hpc_processing_job[event]

            it_dict["events"][str_idx] = {
                "name": event,
                "job_info": jobs,
            }
            if not self.is_validation_event(event):
                it_dict["events"][str_idx]["misfit"] = float(self.misfits[event])
                it_dict["events"][str_idx]["usage_updated"] = self.updated[event]

        with open(iteration_toml, "w") as fh:
            toml.dump(it_dict, fh)

    def get_iteration_attributes(self, iteration: str):
        """
        Save the attributes of the current iteration into memory

        :param iteration: Name of iteration
        :type iteration: str
        """
        iteration_toml = self.paths.get_iteration_toml(iteration)
        if not iteration_toml.exists():
            raise FileNotFoundError(f"No toml file exists for iteration: {iteration}")

        it_dict = toml.load(iteration_toml)

        self.current_iteration = it_dict["name"]
        self.events_in_iteration = self.event_db.get_event_names(
            list(it_dict["events"].keys())
        )
        self.non_val_events_in_iteration = list(
            set(self.events_in_iteration) - set(self.validation_dataset)
        )
        self.adjoint_job = {}
        self.misfits = {}
        self.updated = {}
        self.prepare_forward_job = {}
        self.forward_job = {}
        self.hpc_processing_job = {}
        self.gradient_interp_job = {}

        # Not sure if it's worth it to include station misfits
        for event in self.events_in_iteration:
            ev_idx = str(self.event_db.get_event_idx(event))
            if not self.is_validation_event(event):
                self.updated[event] = it_dict["events"][ev_idx]["usage_updated"]
                self.misfits[event] = it_dict["events"][ev_idx]["misfit"]

                self.adjoint_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "adjoint"
                ]
            self.forward_job[event] = it_dict["events"][ev_idx]["job_info"]["forward"]
            if self.prepare_forward:
                self.prepare_forward_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "prepare_forward"
                ]
            if self.remote_interp and not self.is_validation_event(event):
                self.gradient_interp_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "gradient_interp"
                ]
            if self.hpc_processing and not self.is_validation_event(event):
                self.hpc_processing_job[event] = it_dict["events"][ev_idx]["job_info"][
                    "hpc_processing"
                ]

    def get_old_iteration_info(self, iteration: str) -> dict:
        """
        For getting information about something else than current iteration.

        :param iteration: Name of iteration
        :type iteration: str
        :return: Information regarding that iteration
        :rtype: dict
        """
        iteration_toml = self.paths.get_iteration_toml(iteration)
        if not iteration_toml.exists():
            raise FileNotFoundError(f"No toml file exists for iteration: {iteration}")

        with open(iteration_toml, "r") as fh:
            it_dict = toml.load(fh)
        return it_dict

    def is_validation_event(self, event: str) -> bool:
        return event in self.validation_dataset

    def find_simulation_time_step(self, event: Optional[str] = None):
        """
        Get the timestep from a forward job if it does not exist yet.

        Returns the timestep if it is there and managed to do so.
        If an event is passed and it does not exist it, it will get it
        from an stdout file.
        """
        timestep_file = self.paths.misc_folder / "simulation_timestep.toml"

        if not timestep_file.exists() and event is not None:
            self._get_timestep_from_stdout(event, timestep_file)
        elif timestep_file.exists():
            time_dict = toml.load(timestep_file)
            self.simulation_time_step = time_dict["time_step"]

    def _get_timestep_from_stdout(self, event, timestep_file):
        """Read the timestep value from a stdout produced by salvus run."""
        local_stdout = self.paths.documentation / "stdout_to_find_tinestep"
        hpc_cluster = sapi.get_site(self.site_name)
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
                if time_step > 0.00001 and time_step < 1000:
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
