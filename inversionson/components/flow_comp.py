from __future__ import annotations
import shutil
import salvus.flow.api as sapi  # type: ignore
import os
import time
import pathlib
import toml
import random
from typing import Optional, Union, List, TYPE_CHECKING

from .component import Component
from salvus.flow import schema_validator  # type: ignore
import typing
from salvus.flow.simple_config.simulation import Waveform  # type: ignore
from salvus.flow.sites import job as s_job  # type: ignore
from salvus.flow.api import get_site  # type: ignore
from inversionson import InversionsonError
from salvus.mesh.unstructured_mesh import UnstructuredMesh  # type: ignore
from salvus.flow.db import SalvusFlowDoesNotExistDBException  # type: ignore

if TYPE_CHECKING:
    from inversionson.project import Project


class SalvusFlow(Component):
    """
    A class which handles all dealings with salvus flow.
    """

    def __init__(self, project: Project):
        super().__init__(project=project)
        self.__hpc_cluster = None

    def print(
        self,
        message: str,
        color: Optional[str] = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = None,
    ):
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    @property
    def hpc_cluster(self):
        if not self.__hpc_cluster:
            self.__hpc_cluster = get_site(self.project.config.hpc.sitename)
        return self.__hpc_cluster

    def safe_put(
        self,
        local_file: Union[pathlib.Path, str],
        remote_file: Union[pathlib.Path, str],
    ):
        tmp_remote_file = f"{remote_file}_tmp"
        self.hpc_cluster.remote_put(local_file, tmp_remote_file)
        self.hpc_cluster.run_ssh_command(f"mv {tmp_remote_file} {remote_file}")

    def safe_get(
        self,
        remote_file: Union[pathlib.Path, str],
        local_file: Union[pathlib.Path, str],
    ):
        tmp_local_path = f"{local_file}_tmp"
        self.hpc_cluster.remote_get(remote_file, tmp_local_path)
        shutil.move(tmp_local_path, local_file)

    def _get_job_name(
        self, event: str, sim_type: str, new=True, iteration="current"
    ) -> str:
        """
        We need to relate iteration and event to job name. Here you can find
        it. Currently not used. Removed it from the workflow

        :param event: Name of event
        :type event: str
        :param sim_type: Options are "forward" and "adjoint"
        :type sim_type: str
        :param new: If we need a new job name. Otherwise we look for an
        existing one
        :type new: bool
        :param iteration: Name of iteration: defaults to "current"
        :type iteration: str
        :return: Job name
        :rtype: str
        """
        sim_types = {
            "prepare_forward",
            "forward",
            "hpc_processing",
            "adjoint",
            "gradient_interp",
        }
        assert sim_type in sim_types

        if iteration == "current":
            iteration = self.project.current_iteration
        old_iter = iteration != self.project.current_iteration

        if new:
            unique_id = "".join(random.choice("0123456789ABCDEF") for _ in range(8))
            job = f"{iteration}_{sim_type}_{unique_id}"
            if sim_type == "adjoint":
                self.project.adjoint_job[event]["name"] = job
            elif sim_type == "forward":
                self.project.forward_job[event]["name"] = job
            else:
                raise InversionsonError("This isn't even used anyway")
        elif old_iter:
            iteration_info = self.project.get_old_iteration_info(iteration)
            event_index = str(self.project.event_db.get_event_idx(event))
            job = iteration_info["events"][event_index]["job_info"][sim_type]["name"]
        elif sim_type == "adjoint":
            job = self.project.adjoint_job[event]["name"]
        elif sim_type == "forward":
            job = self.project.forward_job[event]["name"]
        elif sim_type == "gradient_interp":
            job = self.project.gradient_interp_job[event]["name"]
        elif sim_type == "hpc_processing":
            job = self.project.hpc_processing_job[event]["name"]
        elif sim_type == "prepare_forward":
            job = self.project.prepare_forward_job[event]["name"]

        self.project.update_iteration_toml()
        return job

    def get_job_name(self, event: str, sim_type: str, iteration="current"):
        return self._get_job_name(
            event=event, sim_type=sim_type, new=False, iteration=iteration
        )

    def get_job(self, event: str, sim_type: str, iteration: Optional[str] = None):
        """
        Get Salvus.flow.job.Job Object
        """
        if not iteration or iteration == "current":
            iteration = self.project.current_iteration
        assert isinstance(iteration, str)

        site_name = self.project.config.hpc.sitename
        custom_job = sim_type in {
            "gradient_interp",
            "prepare_forward",
            "hpc_processing",
        }
        if custom_job:
            return self.__get_custom_job(
                event=event, sim_type=sim_type, iteration=iteration
            )

        # Note: DP: the below if seems optional, but may have performance benefits.
        if iteration == self.project.current_iteration:
            if sim_type == "adjoint":
                assert self.project.adjoint_job[event]["submitted"]
                job_name = self.project.adjoint_job[event]["name"]
            elif sim_type == "forward":
                assert self.project.forward_job[event]["submitted"]
                job_name = self.project.forward_job[event]["name"]

        else:  # get it from an old iteration
            it_dict = self.project.get_old_iteration_info(iteration)
            event_index = str(self.project.event_db.get_event_idx(event))
            assert it_dict["events"][event_index]["job_info"][sim_type]["submitted"]
            job_name = it_dict["events"][event_index]["job_info"][sim_type]["name"]
        return sapi.get_job(job_name=job_name, site_name=site_name)

    def __get_custom_job(self, event: str, sim_type: str, iteration=None):
        """
        A get_job function which handles job types which are not of type
        salvus.flow.sites.salvus_job.SalvusJob

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation
        :type sim_type: str
        """
        gradient = False
        if not iteration:
            iteration = self.project.current_iteration
        if iteration != self.project.current_iteration:
            self.project.change_attribute(
                "current_iteration", self.project.current_iteration
            )
            self.project.get_iteration_attributes(iteration=iteration)

        if sim_type == "gradient_interp":
            gradient = True
            if self.project.gradient_interp_job[event]["submitted"]:
                job_name = self.project.gradient_interp_job[event]["name"]
            else:
                raise InversionsonError(
                    f"Gradient interpolation job for event: {event} "
                    "has not been submitted"
                )
        elif sim_type == "hpc_processing":
            if self.project.hpc_processing_job[event]["submitted"]:
                job_name = self.project.hpc_processing_job[event]["name"]
            else:
                raise InversionsonError(
                    f"HPC processing job for event: {event} " "has not been submitted"
                )
        elif sim_type == "prepare_forward":
            if self.project.prepare_forward_job[event]["submitted"]:
                job_name = self.project.prepare_forward_job[event]["name"]
            else:
                raise InversionsonError(
                    f"Model interpolation job for event: {event} "
                    "has not been submitted"
                )
        site_name = self.project.config.hpc.sitename
        db_job = sapi._get_config()["db"].get_jobs(
            limit=1,
            site_name=site_name,
            job_name=job_name,
        )[0]

        return s_job.Job(
            site=sapi.get_site(site_name=db_job.site.site_name),
            commands=self.project.multi_mesh.get_interp_commands(event, gradient),
            job_type=db_job.job_type,
            job_info=db_job.info,
            jobname=db_job.job_name,
            job_description=db_job.description,
            wall_time_in_seconds=db_job.wall_time_in_seconds,
            working_dir=pathlib.Path(db_job.working_directory),
            tmpdir_root=pathlib.Path(db_job.temp_directory_root)
            if db_job.temp_directory_root
            else None,
            rundir_root=pathlib.Path(db_job.run_directory_root)
            if db_job.run_directory_root
            else None,
            job_groups=[i.group_name for i in db_job.groups],
            initialize_on_site=False,
        )

    def retrieve_outputs(self, event_name: str, sim_type: str):
        """
        Currently we need to use command line salvus opt to
        retrieve the seismograms. There must be some better way
        though.

        :param event_name: Name of event
        :type event_name: str
        :param sim_type: Type of simulation, forward, adjoint
        :type sim_type: str
        """

        job_name = self._get_job_name(event=event_name, sim_type=sim_type, new=False)
        salvus_job = sapi.get_job(
            site_name=self.project.config.hpc.sitename, job_name=job_name
        )
        if sim_type == "forward":
            destination = self.project.lasif.find_seismograms(
                event=event_name, iteration=self.project.current_iteration
            )

        elif sim_type == "adjoint":
            destination = self.project.lasif.find_gradient(
                iteration=self.project.current_iteration,
                event=event_name,
                smooth=False,
                inversion_grid=False,
                just_give_path=True,
            )

        else:
            raise InversionsonError(
                f"Simulation type {sim_type} not supported in this function"
            )
        salvus_job.copy_output(
            destination=os.path.dirname(destination),
            allow_existing_destination_folder=True,
        )

    def construct_simulation_from_dict(self, event: str):
        """
        Download a dictionary with the simulation object and use it to create a local simulation object
        without having any of the relevant data locally.
        Only used to submit a job to the remote without having to store anything locally.

        :param event: Name of event
        :type event: str
        """

        # Always write events to the same folder
        destination = (
            self.project.lasif.lasif_comm.project.paths["salvus_files"]
            / "SIMULATION_DICTS"
            / event
            / "simulation_dict.toml"
        )
        if not os.path.exists(destination.parent):
            os.makedirs(destination.parent)

        if not os.path.exists(destination):
            interp_job = self.get_job(event, sim_type="prepare_forward")
            remote_dict = (
                interp_job.stdout_path.parent / "output" / "simulation_dict.toml"
            )
            self.safe_get(remote_file=remote_dict, local_file=destination)

        sim_dict = toml.load(destination)

        local_dummy_mesh_path = self.project.lasif.get_master_model()
        local_dummy_mesh = self.project.lasif.get_master_mesh()
        for key in ["mesh", "model", "geometry"]:
            sim_dict["domain"][key]["filename"] = local_dummy_mesh_path

        return self.simulation_from_dict(sim_dict, local_dummy_mesh)

    @classmethod
    def simulation_from_dict(
        cls, dictionary: typing.Dict, mesh_object: UnstructuredMesh
    ) -> "Waveform":
        """
        Custom version of the from Waveform.from_dict method
        to allow passing of mesh objects and prevent reading mesh files
        again and again.

        Must be initialized with a locally existing mesh. but can
        be modified to use a remote mesh after creation

        Args:
            dictionary: Dictionary
            mesh_object: salvus mesh object
        """

        w = Waveform()

        # make sure the dictionary is compatible
        schema_validator.validate(value=dictionary, schema=w._schema, pretty_error=True)

        # ensure the same mesh file is given for mesh, model and geometry
        filenames = [
            dictionary["domain"]["mesh"]["filename"],
            dictionary["domain"]["model"]["filename"],
            dictionary["domain"]["geometry"]["filename"],
        ]

        if len(set(filenames)) != 1:
            msg = (
                "This method currently only supports unique file names "
                "for mesh, model and geometry."
            )
            raise ValueError(msg)

        mesh = dictionary["domain"]["mesh"]["filename"]
        if mesh == "__SALVUS_FLOW_SPECIAL_TEMP__":
            msg = "The dictionary does not contain a path to a mesh file."
            raise ValueError(msg)

        w.set_mesh(mesh_object)
        w.apply(dictionary)
        w.validate()

        return w

    def construct_adjoint_simulation_from_dict(self, event: str):
        """
        Download a dictionary with the simulation object and use it to create a local simulation object
        without having any of the relevant data locally.
        Only used to submit a job to the remote without having to store anything locally.

        :param event: Name of event
        :type event: str
        """

        hpc_cluster = sapi.get_site(self.project.config.hpc.sitename)
        hpc_proc_job = self.get_job(event, sim_type="hpc_processing")

        # Always write events to the same folder
        destination = (
            (
                self.project.lasif.lasif_comm.project.paths["salvus_files"]
                / "SIMULATION_DICTS"
            )
            / event
        ) / "adjoint_simulation_dict.toml"
        if not os.path.exists(destination.parent):
            os.makedirs(destination.parent)

        if os.path.exists(destination):
            os.remove(destination)
        remote_dict = (
            hpc_proc_job.stdout_path.parent / "output" / "adjoint_simulation_dict.toml"
        )
        hpc_cluster.remote_get(remotepath=remote_dict, localpath=destination)

        adjoint_sim_dict = toml.load(destination)
        remote_mesh = adjoint_sim_dict["domain"]["mesh"]["filename"]
        local_dummy_mesh_path = self.project.lasif.get_master_model()
        local_dummy_mesh = self.project.lasif.get_master_mesh()
        for key in ["mesh", "model", "geometry"]:
            adjoint_sim_dict["domain"][key]["filename"] = local_dummy_mesh_path

        w = self.simulation_from_dict(adjoint_sim_dict, local_dummy_mesh)
        w.set_mesh(f"REMOTE:{str(remote_mesh)}")
        return w

    def submit_job(
        self,
        event: str,
        simulation: object,
        sim_type: str,
        site="daint",
        ranks=1024,
    ):
        """
        Submit a job with some information. Salvus flow returns an object
        which can be used to interact with job.

        :param event: Name of event
        :type event: str
        :param simulation: Simulation object constructed beforehand
        :type simulation: object
        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        :param site: Name of site in salvus flow config file, defaults
        to "daint"
        :type site: str, optional
        :param ranks: How many cores to run on. (A multiple of 12 on daint),
        defaults to 1024
        :type ranks: int, optional
        """
        # Adjoint simulation takes longer and seems to be less predictable
        # we thus give it a longer wall time.

        wall_time = self.project.config.hpc.wave_wall_time
        if sim_type == "adjoint":
            wall_time = wall_time * 1.5

        start_submit = time.time()
        job = sapi.run_async(
            site_name=site,
            input_file=simulation,
            ranks=ranks,
            wall_time_in_seconds=wall_time,
        )
        end_submit = time.time()
        self.print(
            f"Submitting took {end_submit - start_submit:.3f} seconds",
            emoji_alias=":hourglass:",
            color="magenta",
        )
        hpc_cluster = sapi.get_site(self.project.config.hpc.sitename)

        if sim_type == "forward":
            self.project.change_attribute(
                f'forward_job["{event}"]["name"]', job.job_name
            )
            self.project.change_attribute(f'forward_job["{event}"]["submitted"]', True)

        elif sim_type == "adjoint":
            self.project.change_attribute(
                f'adjoint_job["{event}"]["name"]', job.job_name
            )
            self.project.change_attribute(f'adjoint_job["{event}"]["submitted"]', True)
        self.project.update_iteration_toml()
        if hpc_cluster.config["site_type"] == "local":
            self.print(f"Running {sim_type} simulation...")
            job.wait(
                poll_interval_in_seconds=self.project.config.hpc.sleep_time_in_seconds
            )

    def get_job_status(self, event: str, sim_type: str, iteration="current") -> str:
        """
        Check the status of a salvus opt job

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation: forward, adjoint or smoothing
        :type sim_type: str
        :param iteration: Name of iteration. "current" if current iteration
        :type iteration: str
        :return: status of job
        :rtype: str
        """

        job = self.get_job(
            event=event,
            sim_type=sim_type,
            iteration=iteration,
        )
        return job.update_status(force_update=True)

    def get_job_file_paths(self, event: str, sim_type: str) -> dict:
        """
        Get the output folder for an event

        :param event: Name of event
        :type event: str
        :param sim_type: Forward or adjoint simulation
        :type sim_type: str
        """
        if sim_type == "forward":
            job_name = self.project.forward_job[event]["name"]
        elif sim_type == "adjoint":
            job_name = self.project.adjoint_job[event]["name"]
        else:
            raise InversionsonError(f"Don't recognize sim_type {sim_type}")

        job = sapi.get_job(
            job_name=job_name, site_name=self.project.config.hpc.sitename
        )

        return job.get_output_files()

    def delete_stored_wavefields(
        self, iteration: str, sim_type: str, event_name: Optional[str] = None
    ):
        """
        Delete all stored jobs for a certain simulation type of an iteration

        :param iteration: Name of iteration
        :type iteration: str
        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        :param event_name: Name of an event if only for a single event, this can
            for example be used after a job fails and needs to be reposted.
            Defaults to None
        :type event_name: str, optional
        """

        if event_name is not None:
            try:
                job = self.get_job(event=event_name, sim_type=sim_type)
                job.delete(verbosity=0)
            except (SalvusFlowDoesNotExistDBException, InversionsonError) as e:
                self.print(
                    f"Could not delete job {sim_type} for event {event_name}",
                    emoji_alias=":hankey:",
                )
                print(e)
            return
        events_in_iteration = self.project.lasif.list_events(iteration=iteration)
        non_val_tasks = ["gradient_interp", "hpc_processing"]
        for event in events_in_iteration:
            if self.project.is_validation_event(event) and sim_type in non_val_tasks:
                continue
            try:
                try:
                    job = self.get_job(
                        event=event, sim_type=sim_type, iteration=iteration
                    )
                    job.delete(verbosity=0)
                except (KeyError, InversionsonError):
                    continue
            except SalvusFlowDoesNotExistDBException as e:
                self.print(
                    f"Could not delete job {sim_type} for event {event}",
                    emoji_alias=":hankey:",
                )
                print(e)
