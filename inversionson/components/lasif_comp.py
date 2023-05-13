from __future__ import annotations
import shutil

from .component import Component
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from inversionson.project import Project
import lasif.api as lapi
from lasif.utils import write_custom_stf
import os
from inversionson import InversionsonError, InversionsonWarning
import warnings
import toml
import pathlib

from salvus.flow.api import get_site
from typing import List, Dict, Union
from salvus.mesh.unstructured_mesh import UnstructuredMesh


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
        self.master_mesh = None

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
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

        from lasif.components.project import Project

        folder = pathlib.Path(self.lasif_root).absolute()
        max_folder_depth = 4

        for _ in range(max_folder_depth):
            if (folder / "lasif_config.toml").exists():
                return Project(folder).get_communicator()
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
        if isinstance(iterations, list):
            if it_name in iterations:
                return True
        else:
            return False

    def has_remote_mesh(
        self,
        event: str,
        gradient: bool,
        interpolate_to: bool = True,
        iteration: str = None,
        validation: bool = False,
    ):
        """
        Just to check if remote mesh exists

        :param event: Name of event
        :type event: str
        :param gradient: Is it a gradient?
        :type gradient: bool
        :param interpolate_to: Mesh to interpolate to?, defaults to True
        :type interpolate_to: bool, optional
        :param hpc_cluster: you can pass the site object. Defaults to None
        :type hpc_cluster: salvus.flow.Site, optional
        :param iteration: Name of an iteration, defaults to None
        :type iteration: str, optional
        """

        hpc_cluster = self.project.flow.hpc_cluster
        mesh = self.find_remote_mesh(
            event=event,
            check_if_exists=False,
            iteration=iteration,
            validation=validation,
            interpolate_to=interpolate_to,
            gradient=gradient,
        )

        return hpc_cluster.remote_exists(mesh), mesh

    def find_remote_mesh(
        self,
        event: str,
        gradient: bool = False,
        interpolate_to: bool = True,
        check_if_exists: bool = False,
        iteration: str = None,
        already_interpolated: bool = False,
        validation: bool = False,
    ) -> pathlib.Path:
        """
        Find the path to the relevant mesh on the hpc cluster

        :param event: Name of event
        :type event: str
        :param gradient: Is it a gradient? If not, it's a model,
            defaults to False
        :type gradient: bool, optional
        :param interpolate_to: Mesh to interpolate to?, defaults to True
        :type interpolate_to: bool, optional
        :param check_if_exists: Check if the file exists?, defaults to False
        :type check_if_exists: bool, optional
        :param hpc_cluster: you can pass the site object. Defaults to None
        :type hpc_cluster: salvus.flow.Site, optional
        :param iteration: Name of an iteration, defaults to None
        :type iteration: str, optional
        :param already_interpolated: If mesh has been interpolated,
            we find it in the interpolation job folder, defaults to False
        :type already_interpolated: bool, optional
        :return: The path to the correct mesh
        :rtype: pathlib.Path
        """
        hpc_cluster = self.project.flow.hpc_cluster
        remote_mesh_dir = self.project.remote_paths.mesh_dir
        fast_dir = self.project.config.hpc.inversionson_folder
        if iteration is None:
            iteration = self.project.current_iteration

        if gradient:
            if interpolate_to:
                mesh = (
                    self.project.remote_paths.mesh_dir / "standard_gradient" / "mesh.h5"
                )
            else:
                output = self.project.salvus_flow.get_job_file_paths(
                    event=event, sim_type="adjoint"
                )
                mesh = output[0][("adjoint", "gradient", "output_filename")]
        elif already_interpolated:
            job = self.project.salvus_flow.get_job(
                event=event,
                sim_type="prepare_forward",
                iteration=iteration,
            )
            mesh = job.stdout_path.parent / "output" / "mesh.h5"
        elif interpolate_to:
            mesh = remote_mesh_dir / "meshes" / event / "mesh.h5"
        else:
            mesh = (
                (fast_dir / "AVERAGE_MODELS" / iteration / "mesh.h5")
                if validation
                else fast_dir / "MODELS" / iteration / "mesh.h5"
            )
        if check_if_exists and not hpc_cluster.remote_exists(mesh):
            if gradient and interpolate_to:
                self._move_mesh_to_cluster(
                    event=None, gradient=gradient, hpc_cluster=hpc_cluster
                )
            raise InversionsonError("Mesh for event {event} does not exist")
        return mesh

    def has_mesh(self, event: str, hpc_cluster=None) -> bool:
        """
        Check whether mesh has been constructed for respective event

        :param event: Name of event
        :type event: str
        :return: Answer whether mesh exists
        :rtype: bool
        """
        has, _ = self.has_remote_mesh(event, gradient=False, hpc_cluster=hpc_cluster)
        return has

    def find_event_mesh(self, event: str) -> pathlib.Path:
        """
        Find the path for an event mesh

        :param event: Name of event
        :type event: str
        :return: Path to where the mesh is stored.
        :rtype: Pathlib.Path
        """
        if not self.project.config.meshing.multi_mesh:
            return self.project.config.inversion.initial_model
        has, mesh = lapi.find_event_mesh(self.lasif_comm, event)
        if not has:
            raise InversionsonError(f"Mesh for event: {event} can not be found.")
        return pathlib.Path(mesh)

    def _move_mesh_to_cluster(
        self, event: str, gradient: bool = False, hpc_cluster=None
    ):
        """
        Move the mesh to the cluster for interpolation

        :param event: Name of event
        :type event: str
        """
        if event is None:
            if gradient:
                self.print(
                    "Moving example gradient to cluster", emoji_alias=":package:"
                )
                self.move_gradient_to_cluster(hpc_cluster)
            else:
                # This happens when we want to move the model to the cluster
                self.print("Moving model to cluster", emoji_alias=":package:")
                self._move_model_to_cluster(hpc_cluster)
            return
        has, event_mesh = lapi.find_event_mesh(self.lasif_comm, event)

        if not has:
            raise InversionsonError(f"Mesh for event {event} does not exist.")
        # Get remote connection
        hpc_cluster = self.project.flow.hpc_cluster

        path_to_mesh = self.find_remote_mesh(
            event=event,
            interpolate_to=True,
            check_if_exists=False,
            hpc_cluster=hpc_cluster,
        )
        if not hpc_cluster.remote_exists(path_to_mesh.parent):
            hpc_cluster.remote_mkdir(path_to_mesh.parent)
        if not hpc_cluster.remote_exists(path_to_mesh):
            self.print(
                f"Moving mesh for event {event} to cluster", emoji_alias=":package:"
            )
            hpc_cluster.remote_put(event_mesh, path_to_mesh)

    def _move_model_to_cluster(
        self,
        overwrite: bool = False,
        validation: bool = False,
    ):
        """
        The model is moved to a dedicated directory on cluster

        :param hpc_cluster: A Salvus site object, defaults to None
        :type hpc_cluster: salvus.flow.Site, optional
        :param overwrite: Overwrite mesh already there?, defaults to False
        :type overwrite: bool, optional
        """
        hpc_cluster = self.project.flow.hpc_cluster

        optimizer = self.project.get_optimizer()
        iteration = optimizer.iteration_name
        if validation:
            print("It's validation!")
            iteration = f"validation_{iteration}"
            local_model = self.project.multi_mesh.find_model_file(iteration)
        else:
            local_model = optimizer.model_path
        has, path_to_mesh = self.has_remote_mesh(
            event=None,
            interpolate_to=False,
            gradient=False,
            hpc_cluster=hpc_cluster,
            iteration=iteration,
            validation=validation,
        )
        if has:
            if overwrite:
                hpc_cluster.remote_put(local_model, path_to_mesh)
            else:
                self.print(
                    f"Model for iteration {iteration} already on cluster",
                    emoji_alias=":white_check_mark:",
                )
                return
        else:
            if not hpc_cluster.remote_exists(path_to_mesh.parent):
                self.print("Making the directory")
                self.print(f"Directory is: {path_to_mesh.parent}")
                hpc_cluster.remote_mkdir(path_to_mesh.parent)
            self.print(f"Path to mesh is: {path_to_mesh}")
            hpc_cluster.remote_put(local_model, path_to_mesh)
            self.print("Did it")

    def move_gradient_to_cluster(self, overwrite: bool = False):
        """
        Empty gradient moved to a dedicated directory on cluster

        :param hpc_cluster: A Salvus site object, defaults to None
        :type hpc_cluster: salvus.flow.Site, optional
        """
        hpc_cluster = self.project.flow.hpc_cluster

        has, path_to_mesh = self.has_remote_mesh(
            event=None,
            interpolate_to=True,
            gradient=True,
            hpc_cluster=hpc_cluster,
            iteration=None,
            validation=False,
        )

        if has and not overwrite:
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

        inversion_grid = self.get_master_model()
        shutil.copy(inversion_grid, local_grad)
        self.project.salvus_mesher.fill_inversion_params_with_zeroes(local_grad)

        if not hpc_cluster.remote_exists(path_to_mesh.parent):
            hpc_cluster.remote_mkdir(path_to_mesh.parent)
        hpc_cluster.remote_put(local_grad, path_to_mesh)

    def move_mesh(self, event: str, iteration: str, hpc_cluster=None, validation=False):
        """
        Move mesh to simulation mesh path, where model will be added to it

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        """
        # If we use mono-mesh we copy the mesh here.
        if not self.project.config.meshing.multi_mesh:
            optimizer = self.project.get_optimizer()
            model = optimizer.model_path
            # copy to lasif project and also move to cluster
            simulation_mesh = self.project.lasif.get_simulation_mesh(event_name=None)
            shutil.copy(model, simulation_mesh)
            self._move_model_to_cluster(
                hpc_cluster=hpc_cluster, overwrite=False, validation=validation
            )
            return
        else:  # multi-mesh case
            if event is None:
                self._move_model_to_cluster(
                    hpc_cluster=hpc_cluster,
                    overwrite=False,
                    validation=validation,
                )
            else:
                self._move_mesh_to_cluster(event=event, hpc_cluster=hpc_cluster)

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
        return str(stfs / long_iter / "stf.h5")

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

        if not hpc_cluster.remote_exists(
            self.project.remote_inversionson_dir / "SOURCE_TIME_FUNCTIONS" / iteration
        ):
            hpc_cluster.remote_mkdir(
                self.project.remote_inversionson_dir
                / "SOURCE_TIME_FUNCTIONS"
                / iteration
            )
        if not hpc_cluster.remote_exists(
            self.project.remote_inversionson_dir
            / "SOURCE_TIME_FUNCTIONS"
            / iteration
            / "stf.h5"
        ):
            hpc_cluster.remote_put(
                local_stf,
                self.project.remote_inversionson_dir
                / "SOURCE_TIME_FUNCTIONS"
                / iteration
                / "stf.h5",
            )

    def get_master_model(self) -> str:
        """
        Get the path to the inversion grid used in inversion

        :return: Path to inversion grid
        :rtype: str
        """
        return self.project.config.inversion.initial_model

    def get_master_mesh(self) -> str:
        """
        Get the salvus mesh object.

        This is function is there to keep the mesh object in memory.
        This is useful, because reading it from the file
        is slow and happens often.

        :return: Mesh of inversion grid
        :rtype: UnstructuredMesh
        """
        # We assume the lasif domain is the inversion grid
        if self.master_mesh is None:
            path = self.project.config.inversion.initial_model
            self.master_mesh = UnstructuredMesh.from_h5(path)
        return self.master_mesh

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

    def get_simulation_mesh(self, event_name: str, iteration="current") -> str:
        """
        Get path to correct simulation mesh for a simulation

        :param event_name: Name of event
        :type event_name: str
        :return: Path to a mesh
        :rtype: str
        """
        if iteration == "current":
            iteration = self.project.current_iteration
        if self.project.meshes == "multi-mesh":
            return (
                str(
                    self.find_remote_mesh(
                        event=event_name,
                        iteration=iteration,
                        already_interpolated=True,
                    )
                )
                if self.project.interpolation_mode == "remote"
                else lapi.get_simulation_mesh(
                    self.lasif_comm,
                    event_name,
                    iteration,
                )
            )
        optimizer = self.project.get_optimizer()
        return (
            optimizer.get_average_model_name()
            if (
                self.project.is_validation_event(event_name)
                and self.project.use_model_averaging
                and "00000" not in self.project.current_iteration
            )
            else optimizer.model_path
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
            if self.project.inversion_mode == "mini-batch":
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

    def get_adjoint_source_file(self, event: str, iteration: str) -> str:
        """
        Find the path to the correct asdf file containing the adjoint sources

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        :return: Path to adjoint source file
        :rtype: str
        """
        adjoint_filename = "stf.h5"
        adj_sources = self.lasif_comm.project.paths["adjoint_sources"]
        it_name = self.lasif_comm.iterations.get_long_iteration_name(iteration)
        return os.path.join(adj_sources, it_name, event, adjoint_filename)

    def _already_processed(self, event: str) -> bool:
        """
        Looks for processed data for a certain event

        :param event: Name of event
        :type event: str
        :return: True/False regarding the alreadyness of the processed data.
        :rtype: bool
        """
        high_period = self.project.simulation_settings.max_period
        low_period = self.project.simulation_settings.min_period
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
        lasif_root = self.project.config.lasif_root
        proc_filename = (
            f"preprocessed_{int(self.project.simulation_settings.min_period)}s_"
            f"to_{int(self.project.simulation_settings.max_period)}s.h5"
        )
        local_proc_folder = os.path.join(
            lasif_root, "PROCESSED_DATA", "EARTHQUAKES", event
        )
        local_proc_file = os.path.join(local_proc_folder, proc_filename)

        if not os.path.exists(local_proc_folder):
            os.mkdir(local_proc_folder)

        remote_proc_file_name = f"{event}_{proc_filename}"
        hpc_cluster = self.project.flow.hpc_cluster

        remote_processed_dir = self.project.remote_paths.proc_data_dir

        remote_proc_path = remote_processed_dir / remote_proc_file_name
        if hpc_cluster.remote_exists(remote_proc_path):
            self.project.flow.safe_get(remote_proc_path, local_proc_file)
            return  # Return if it got it and got it there.

    def process_random_unprocessed_event(self) -> bool:
        """
        Instead of sleeping when we queue for the HPC, we can also process a
        random unprocessed event. That is what this function does.

        it first tries to process a high priority event, from
        the validation dataset or the current iteration, otherwise
        it tries to process any other event that may be used in the future.

        Leaves the function as soon as one event was processed or
        if there was nothing to process.

        :return: Returns True if an event was processed, otherwise False
        :rtype: bool
        """

        events_in_iteration = self.project.events_in_iteration
        events = self.project.lasif.list_events()
        validation_events = self.project.config.monitoring.validation_dataset
        if not self.everything_processed:
            self.everything_processed = True
            msg = "Seems like there is nothing to do now. I might as well process some random event."
            # First give the most urgent events a try.
            if not self.validation_data_processed:
                self.validation_data_processed = True
                for event in validation_events:
                    if self._already_processed(event):
                        continue
                    self.print(msg)
                    self.print(f"Processing validation {event}...")
                    self.validation_data_processed = False
                    lapi.process_data(self.lasif_comm, events=[event])
                    return True
            for event in events_in_iteration:
                if self._already_processed(event):
                    continue
                self.print(msg)
                self.print(f"Processing {event} from current iteration...")
                self.everything_processed = False
                lapi.process_data(self.lasif_comm, events=[event])
                return True
            for event in events:
                if self._already_processed(event):
                    continue
                self.print(msg)
                self.print(f"Processing random other {event}...")
                self.everything_processed = False
                lapi.process_data(self.lasif_comm, events=[event])
                return True
        return False

    def select_windows(
        self,
        window_set_name: str,
        event: str,
        validation: bool = False,
    ):
        """
        Select window for a certain event in an iteration.

        :param window_set_name: Name of window set
        :type window_set_name: str
        :param event: Name of event to pick windows on
        :type event: str
        """
        # Check if window set exists:
        path = os.path.join(
            self.lasif_root, "SETS", "WINDOWS", f"{window_set_name}.sqlite"
        )

        if os.path.exists(path) and not validation:
            self.print(
                f"Window set for event {event} exists.",
                emoji_alias=":white_check_mark:",
            )
            return

        lapi.select_windows_multiprocessing(
            self.lasif_comm,
            iteration=self.project.current_iteration,
            window_set=window_set_name,
            events=[event],
            num_processes=8,
        )

    def find_seismograms(self, event: str, iteration: str) -> str:
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

        event_folder = os.path.join(
            self.lasif_root, "SYNTHETICS", "EARTHQUAKES", iteration, event
        )
        if not os.path.exists(event_folder):
            os.mkdir(event_folder)

        return os.path.join(event_folder, "receivers.h5")
