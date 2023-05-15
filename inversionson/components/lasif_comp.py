from __future__ import annotations
import shutil

from .component import Component
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from inversionson.project import Project
import lasif.api as lapi  # type: ignore
from lasif.utils import write_custom_stf  # type: ignore
import os
from inversionson import InversionsonError, InversionsonWarning
import warnings
import toml
import pathlib

from lasif.components.project import Project as LASIFProject  # type: ignore

from typing import List, Dict, Union
from salvus.mesh.unstructured_mesh import UnstructuredMesh  # type: ignore


class Lasif(Component):
    """
    Communication with Lasif
    """

    def __init__(self, project: Project):
        super().__init__(project=project)
        self.lasif_root = self.project.config.lasif_root
        self.lasif_comm = self._find_project_comm()

        # Store if some event might not processing
        self.everything_processed = False
        self.validation_data_processed = False
        self._master_mesh = None

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

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        folder = pathlib.Path(self.lasif_root).absolute()
        max_folder_depth = 4

        for _ in range(max_folder_depth):
            if (folder / "lasif_config.toml").exists():
                return LASIFProject(folder).get_communicator()
            folder = folder.parent
        raise ValueError(f"Path {self.lasif_root} is not a LASIF project")

    def has_iteration(self, it_name: str) -> bool:
        """
        See if lasif project has the iteration already

        :param it_name: name of iteration
        :type name: str
        :return: True if lasif has the iteration
        """
        iterations = lapi.list_iterations(self.lasif_comm, output=True, verbose=False)
        if it_name.startswith("ITERATION_"):
            it_name = it_name.replace("ITERATION_", "")
        return isinstance(iterations, list) and it_name in iterations

    def move_gradient_to_cluster(self):
        """
        Empty gradient moved to a dedicated directory on cluster

        :param hpc_cluster: A Salvus site object, defaults to None
        :type hpc_cluster: salvus.flow.Site, optional
        """
        hpc_cluster = self.project.flow.hpc_cluster

        remote_gradient = self.project.remote_paths.master_gradient
        if hpc_cluster.remote_exists(remote_gradient):
            self.print(
                "Empty gradient already on cluster", emoji_alias=":white_check_mark:"
            )
            return

        local_grad = (
            pathlib.Path(self.lasif_comm.project.paths["models"])
            / "GRADIENT"
            / "mesh.h5"
        )
        local_grad.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.get_master_model(), local_grad)
        self.project.salvus_mesher.fill_inversion_params_with_zeroes(local_grad)
        self.project.flow.safe_put(local_grad, remote_gradient)

    def set_up_iteration(self, name: str, events: Optional[List[str]] = None):
        """
        Create a new iteration in the lasif project

        :param name: Name of iteration
        :type name: str
        :param events: list of events used in iteration, defaults to []
        :type events: list, optional
        """
        if events is None:
            events = []
        iterations = lapi.list_iterations(self.lasif_comm, output=True, verbose=False)
        if isinstance(iterations, list) and name in iterations:
            warnings.warn(f"Iteration {name} already exists", InversionsonWarning)
        event_specific = self.project.config.meshing.multi_mesh
        lapi.set_up_iteration(
            self.lasif_root,
            iteration=name,
            events=events,
            event_specific=event_specific,
        )

    def list_events(self, iteration: Optional[str] = None) -> List[str]:
        """
        Make lasif list events, supposed to be used when all events
        are used per iteration. IF only for an iteration, pass
        an iteration value.

        :param iteration: Name of iteration, defaults to None
        :type iteration: str
        """
        return lapi.list_events(
            self.lasif_root, just_list=True, iteration=iteration, output=True
        )

    def find_stf(self, iteration: str) -> pathlib.Path:
        """
        Get path to source time function file

        :param iteration: Name of iteration
        :type iteration: str
        """
        long_iter = self.lasif_comm.iterations.get_long_iteration_name(iteration)
        stfs = pathlib.Path(self.lasif_comm.project.paths["salvus_files"])
        return stfs / long_iter / "stf.h5"

    def upload_stf(self, iteration: str) -> None:
        """
        Upload the source time function to the remote machine

        :param iteration: Name of iteration
        :type iteration: str
        """
        local_stf = self.find_stf(iteration=iteration)
        if not os.path.exists(local_stf):
            write_custom_stf(stf_path=local_stf, comm=self.lasif_comm)

        hpc_cluster = self.project.flow.hpc_cluster

        if not hpc_cluster.remote_exists(self.project.remote_paths.stf_dir / iteration):
            hpc_cluster.remote_mkdir(self.project.remote_paths.stf_dir / iteration)
        if not hpc_cluster.remote_exists(
            self.project.remote_paths.stf_dir / iteration / "stf.h5"
        ):
            self.project.flow.safe_put(
                local_stf, self.project.remote_paths.stf_dir / iteration / "stf.h5"
            )

    def get_master_model(self) -> str:
        """
        Get the path to the inversion grid used in inversion

        :return: Path to inversion grid
        :rtype: str
        """
        return self.project.config.inversion.initial_model

    @property
    def master_mesh(self):
        """
        Get the salvus mesh object.
        This function is there to keep the mesh object in memory.

        :return: Mesh of inversion grid
        :rtype: UnstructuredMesh
        """
        # We assume the lasif domain is the inversion grid
        if self._master_mesh is None:
            path = self.project.config.inversion.initial_model
            self.master_mesh = UnstructuredMesh.from_h5(path)
        return self._master_mesh

    def get_source(self, event_name: str) -> dict:
        """
        Get information regarding source used in simulation

        :param event_name: Name of source
        :type event_name: str
        :return: Dictionary with source information
        :rtype: dict
        """
        return lapi.get_source(
            self.lasif_comm, event_name, self.project.current_iteration
        )

    def get_receivers(self, event_name: str) -> List[Dict]:
        """
        Locate receivers and get them in a format that salvus flow
        can use

        :param event_name: Name of event
        :type event_name: str
        :return: A list of receiver dictionaries
        :rtype: dict
        """
        return lapi.get_receivers(
            lasif_root=self.lasif_comm, event=event_name, load_from_file=True
        )

    def calculate_station_weights(self, event: str) -> None:
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        # Name weight set after event to know it
        weight_set_name = event
        # If set exists, we don't recalculate it
        if self.lasif_comm.weights.has_weight_set(weight_set_name):
            self.print(
                f"Weight set already exists for event {weight_set_name}",
                emoji_alias=":white_check_mark:",
            )
            return

        lapi.compute_station_weights(
            self.lasif_comm, weight_set=weight_set_name, events=[weight_set_name]
        )

    def misfit_quantification(
        self, event: str, validation: bool = False, window_set: Optional[str] = None
    ):
        """
        Quantify misfit and calculate adjoint sources.

        :param event: Name of event
        :type event: str
        :param n: How many ranks to run on
        :type n: int
        :param validation: Whether this is for a validation set, default False
        :type validation: bool, optional
        :param window_set: Name of a window set, if None will select a logical
            one, default None
        :type window: str, optional
        """

        iteration = self.project.current_iteration
        if window_set is None:
            if self.project.config.inversion.mini_batch:
                window_set = f"{iteration}_{event}"
            else:
                window_set = event
        # Check if adjoint sources exist:
        adjoint_path = os.path.join(
            self.lasif_root,
            "ADJOINT_SOURCES",
            f"ITERATION_{iteration}",
            event,
            "stf.h5",
        )
        if os.path.exists(adjoint_path) and not validation:
            self.print(
                f"Adjoint source exists for event: {event} ",
                emoji_alias=":white_check_mark:",
            )
            self.print(
                "Will not be recalculated. If you want them "
                f"calculated, delete file: {adjoint_path}"
            )
        elif validation:
            misfit = self.lasif_comm.adj_sources.calculate_validation_misfits_multiprocessing(
                event, iteration
            )
        else:
            lapi.calculate_adjoint_sources_multiprocessing(
                self.lasif_comm,
                iteration=iteration,
                window_set=window_set,
                weight_set=event,
                events=[event],
                num_processes=12,
            )

        misfit_toml_path = (
            self.lasif_comm.project.paths["iterations"]
            / f"ITERATION_{iteration}"
            / "misfits.toml"
        )
        if validation:  # We just return some random value as it is not used
            if os.path.exists(misfit_toml_path):
                misfits = toml.load(misfit_toml_path)
            else:
                misfits = {}
            if event not in misfits.keys():
                misfits[event] = {}
            misfits[event]["event_misfit"] = misfit
            with open(misfit_toml_path, mode="w") as fh:
                toml.dump(misfits, fh)
            return 1.1
        # See if misfit has already been written into iteration toml
        if self.project.misfits[event] == 0.0:
            misfit_toml_path = (
                self.lasif_comm.project.paths["iterations"]
                / f"ITERATION_{iteration}"
                / "misfits.toml"
            )
            misfit = toml.load(misfit_toml_path)[event]["event_misfit"]
        else:
            misfit = self.project.misfits[event]
            self.print(
                f"Misfit for {event} has already been computed.",
                emoji_alias=":white_check_mark:",
            )
        return misfit

    def _already_processed(self, event: str) -> bool:
        """
        Looks for processed data for a certain event

        :param event: Name of event
        :type event: str
        :return: True/False regarding the alreadyness of the processed data.
        :rtype: bool
        """
        high_period = self.project.lasif_settings.max_period
        low_period = self.project.lasif_settings.min_period
        processed_filename = (
            f"preprocessed_{int(low_period)}s_to_{int(high_period)}s.h5"
        )
        processed_data_folder = self.lasif_comm.project.paths["preproc_eq_data"]

        return os.path.exists(
            os.path.join(processed_data_folder, event, processed_filename)
        )

    def process_data(self, event: str):
        """
        Process the data for the periods specified in Lasif project.

        :param event: Name of event to be processed
        :type event: str
        """
        if self._already_processed(event):
            return

        # Get local proc filename
        proc_filename = (
            f"preprocessed_{int(self.project.lasif_settings.min_period)}s_"
            f"to_{int(self.project.lasif_settings.max_period)}s.h5"
        )

        local_proc_folder = (
            self.project.config.lasif_root / "PROCESSED_DATA" / "EARTHQUAKES" / event
        )
        local_proc_folder.mkdir(exist_ok=True)
        local_proc_file = local_proc_folder / proc_filename

        remote_proc_file_name = f"{event}_{proc_filename}"
        hpc_cluster = self.project.flow.hpc_cluster

        remote_proc_path = (
            self.project.remote_paths.proc_data_dir / remote_proc_file_name
        )
        if hpc_cluster.remote_exists(remote_proc_path):
            self.project.flow.safe_get(remote_proc_path, local_proc_file)

    def find_seismograms(self, event: str, iteration: str) -> pathlib.Path:
        """
        Find path to seismograms

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :return: str
        """
        if not iteration.startswith("ITERATION_"):
            iteration = f"ITERATION_{iteration}"

        folder = self.lasif_root / "SYNTHETICS" / "EARTHQUAKES" / iteration / event
        folder.parent.mkdir(exist_ok=True, parents=True)
        return folder / "receivers.h5"
