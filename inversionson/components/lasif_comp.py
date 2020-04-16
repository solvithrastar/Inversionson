from __future__ import absolute_import

from .component import Component
import lasif.api as lapi
import os
from inversionson import InversionsonError, InversionsonWarning
import warnings
import subprocess
import sys
import toml
from typing import Union
import pathlib


class LasifComponent(Component):
    """
    Communication with Lasif
    """

    def __init__(self, communicator, component_name):
        super(LasifComponent, self).__init__(communicator, component_name)
        self.lasif_root = self.comm.project.lasif_root
        self.lasif_comm = self._find_project_comm()

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
        iterations = lapi.list_iterations(self.lasif_comm, output=True)
        if it_name.startswith("ITERATION_"):
            it_name = it_name.replace("ITERATION_", "")
        if isinstance(iterations, list):
            if it_name in iterations:
                return True
        else:
            return False

    def set_up_iteration(self, name: str, events=[]):
        """
        Create a new iteration in the lasif project

        :param name: Name of iteration
        :type name: str
        :param events: list of events used in iteration, defaults to []
        :type events: list, optional
        """
        iterations = lapi.list_iterations(self.lasif_comm, output=True)
        if isinstance(iterations, list):
            if name in iterations:
                warnings.warn(
                    f"Iteration {name} already exists", InversionsonWarning
                )
        event_specific = False
        if self.comm.project.meshes == "multi-mesh":
            event_specific = True
        lapi.set_up_iteration(
            self.lasif_root,
            iteration=name,
            events=events,
            event_specific=event_specific,
        )

    def get_minibatch(self, first=False) -> list:
        """
        Get a batch of events to use in the coming iteration.
        This is still under development

        :param first: First batch of inversion?
        :type first: bool
        :return: A fresh batch of earthquakes
        :rtype: list
        """
        # If this is the first time ever that a batch is selected
        if first:
            blocked_events = list(
                set(
                    self.comm.project.validation_dataset
                    + self.comm.project.test_dataset
                )
            )
            use_these = None
            count = self.comm.project.initial_batch_size
            events = self.list_events()
            avail_events = list(set(events) - set(blocked_events))
            batch = lapi.get_subset(
                self.lasif_comm,
                count=count,
                events=avail_events,
                existing_events=None,
            )
            return batch
        else:
            (
                blocked_events,
                use_these,
            ) = self.comm.salvus_opt.find_blocked_events()
            count = self.comm.salvus_opt.get_batch_size()
        events = self.list_events()
        prev_iter = self.comm.salvus_opt.get_previous_iteration_name()
        prev_iter_info = self.comm.project.get_old_iteration_info(prev_iter)
        existing = prev_iter_info["new_control_group"]
        self.comm.project.change_attribute(
            attribute="old_control_group", new_value=existing
        )
        count -= len(existing)
        if use_these:
            count -= len(use_these)
            batch = list(set(use_these + existing))
            avail_events = list(
                set(events) - set(blocked_events) - set(use_these)
            )
            existing = list(set(existing + use_these))
            avail_events = list(set(avail_events) - set(existing))
        else:
            batch = existing
            if len(blocked_events) == 0:
                rand_batch = self.comm.minibatch.get_random_event(
                    n=self.comm.project.n_random_events_picked,
                    existing=existing,
                )
                batch = list(batch + rand_batch)
                existing = batch
            avail_events = list(
                set(events) - set(blocked_events) - set(existing)
            )
        # TODO: existing events should only be the control group.
        # events should exclude the blocked events because that's what
        # are the options to choose from. The existing go into the poisson disc
        print(f"count: {count}")
        add_batch = lapi.get_subset(
            self.lasif_comm,
            count=count,
            events=avail_events,
            existing_events=existing,
        )
        batch = list(set(batch + add_batch))
        print(f"Picked batch: {batch}")
        return batch

    def list_events(self, iteration=None):
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

    def has_mesh(self, event: str) -> bool:
        """
        Check whether mesh has been constructed for respective event

        :param event: Name of event
        :type event: str
        :return: Answer whether mesh exists
        :rtype: bool
        """
        has, _ = lapi.find_event_mesh(self.lasif_comm, event)

        return has

    def find_event_mesh(self, event: str) -> pathlib.Path:
        """
        Find the path for an event mesh
        
        :param event: Name of event
        :type event: str
        :return: Path to where the mesh is stored.
        :rtype: Pathlib.Path
        """
        has, mesh = lapi.find_event_mesh(self.lasif_comm, event)
        if not has:
            raise InversionsonError(
                f"Mesh for event: {event} can not be found."
            )
        return pathlib.Path(mesh)

    def move_mesh(self, event: str, iteration: str):
        """
        Move mesh to simulation mesh path, where model will be added to it

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration
        :type iteration: str
        """
        import shutil

        has, event_mesh = lapi.find_event_mesh(self.lasif_comm, event)

        if not has:
            raise ValueError(f"{event_mesh} does not exist")
        else:
            event_iteration_mesh = lapi.get_simulation_mesh(
                self.lasif_comm, event, iteration
            )
            if not os.path.exists(event_iteration_mesh):
                if not os.path.exists(os.path.dirname(event_iteration_mesh)):
                    os.makedirs(os.path.dirname(event_iteration_mesh))
                shutil.copy(event_mesh, event_iteration_mesh)
                event_xdmf = event_mesh[:-2] + "xdmf"
                event_iteration_xdmf = event_iteration_mesh[:-2] + "xdmf"
                shutil.copy(event_xdmf, event_iteration_xdmf)
                print(
                    f"Mesh for event: {event} has been moved to correct path for "
                    f"iteration: {iteration} and is ready for interpolation."
                )
            else:
                print(
                    f"Mesh for event: {event} already exists in the "
                    f"correct path for iteration {iteration}. "
                    f"Will not move new one."
                )

    def find_stf(self, iteration: str) -> pathlib.Path:
        """
        Get path to source time function file
        
        :param iteration: Name of iteration
        :type iteration: str
        """
        long_iter = self.lasif_comm.iterations.get_long_iteration_name(
            iteration
        )
        stfs = pathlib.Path(self.lasif_comm.project.paths["salvus_files"])
        stf = str(stfs / long_iter / "stf.h5")
        return stf

    # TODO: Write find_gradient for Pathlib
    def find_gradient(
        self,
        iteration: str,
        event: str,
        summed=False,
        smooth=False,
        inversion_grid=False,
        just_give_path=False,
    ) -> str:
        """
        Find the path to a gradient produced by an adjoint simulation.

        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event, None if mono-batch
        :type event: str
        :param summed: Do you want it to be a sum of many gradients,
        defaults to False
        :type summed: bool
        :param smooth: Do you want the smoothed gradient, defaults to False
        :type smooth: bool
        :param inversion_grid: Do you want the gradient on inversion 
            discretization?, defaults to False
        :type inversion_grid: bool
        :param just_give_path: If True, the gradient does not have to exist,
        defaults to False
        :type just_give_path: bool
        :return: Path to a gradient
        :rtype: str
        """
        gradients = self.lasif_comm.project.paths["gradients"]
        if self.comm.project.meshes == "multi-mesh":
            if smooth:
                gradient = os.path.join(
                    gradients,
                    f"ITERATION_{iteration}",
                    event,
                    "smooth_gradient.h5",
                )
                if inversion_grid:
                    gradient = os.path.join(
                        gradients,
                        f"ITERATION_{iteration}",
                        event,
                        "smooth_grad_master.h5",
                    )
        elif self.comm.project.meshes == "mono-mesh":
            if summed:
                if smooth:
                    gradient = os.path.join(
                        gradients,
                        f"ITERATION_{iteration}",
                        "smooth_gradient.h5",
                    )
                else:
                    gradient = os.path.join(
                        gradients,
                        f"ITERATION_{iteration}",
                        "summed_gradient.h5",
                    )
            else:
                gradient = os.path.join(
                    gradients, f"ITERATION_{iteration}", event, "gradient.h5",
                )
        if not smooth:
            if not os.path.exists(
                os.path.join(gradients, f"ITERATION_{iteration}", event)
            ):
                os.makedirs(
                    os.path.join(gradients, f"ITERATION_{iteration}", event)
                )

            gradient = os.path.join(
                gradients, f"ITERATION_{iteration}", event, "gradient.h5"
            )
        if os.path.exists(gradient):
            return gradient
        if just_give_path:
            return gradient
        else:
            raise ValueError(f"File: {gradient} does not exist.")

    def plot_iteration_events(self) -> str:
        """
        Return the path to a file containing an illustration of
        event distribution for the current iteration

        :return: Path to figure
        :rtype: str
        """
        lapi.plot_events(
            self.lasif_comm,
            type_of_plot="map",
            iteration=self.comm.project.current_iteration,
            save=True,
        )
        filename = os.path.join(
            self.lasif_root,
            "OUTPUT",
            "event_plots",
            "events",
            f"events_{self.comm.project.current_iteration}.png",
        )
        return filename

    def plot_event_misfits(
        self, event: str, iteration: str = "current"
    ) -> str:
        """
        Make a plot where stations are color coded by their respective misfits
        
        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration, defaults to "current"
        :type iteration: str, optional
        :return: Path to figure
        :rtype: str
        """
        if iteration == "current":
            iteration = self.comm.project.current_iteration

        lapi.plot_station_misfits(
            self.lasif_comm, event=event, iteration=iteration, save=True,
        )
        filename = os.path.join(
            self.lasif_root,
            "OUTPUT",
            "event_plots",
            "events",
            f"misfit_{event}_{iteration}.png",
        )
        return filename

    def plot_iteration_raydensity(self) -> str:
        """
        Return the path to a file containing an illustration of
        event distribution for the current iteration

        :return: Path to figure
        :rtype: str
        """
        lapi.plot_raydensity(
            self.lasif_comm,
            iteration=self.comm.project.current_iteration,
            plot_stations=True,
            save=True,
        )
        filename = os.path.join(
            self.lasif_root,
            "OUTPUT",
            "raydensity_plots",
            f"ITERATION_{self.comm.project.current_iteration}",
            "raydensity.png",
        )
        return filename

    def get_master_model(self) -> str:
        """
        Get the path to the inversion grid used in inversion
        
        :return: Path to inversion grid
        :rtype: str
        """
        # We assume the lasif domain is the inversion grid
        path = self.lasif_comm.project.lasif_config["domain_settings"][
            "domain_file"
        ]

        return path

    def get_source(self, event_name: str) -> dict:
        """
        Get information regarding source used in simulation

        :param event_name: Name of source
        :type event_name: str
        :return: Dictionary with source information
        :rtype: dict
        """
        return lapi.get_source(
            self.lasif_comm, event_name, self.comm.project.current_iteration
        )

    def get_receivers(self, event_name: str) -> dict:
        """
        Locate receivers and get them in a format that salvus flow
        can use

        :param event_name: Name of event
        :type event_name: str
        :return: A list of receiver dictionaries
        :rtype: dict
        """
        return lapi.get_receivers(self.lasif_comm, event_name)

    def get_simulation_mesh(self, event_name: str) -> str:
        """
        Get path to correct simulation mesh for a simulation

        :param event_name: Name of event
        :type event_name: str
        :return: Path to a mesh
        :rtype: str
        """
        if self.comm.project.info["meshes"] == "multi-mesh":
            return lapi.get_simulation_mesh(
                self.lasif_comm,
                event_name,
                self.comm.project.current_iteration,
            )
        else:
            iteration = self.comm.project.current_iteration
            return os.path.join(
                self.comm.project.lasif_root,
                "MODELS",
                f"ITERATION_{iteration}",
                "mesh.h5",
            )
            # return os.path.join(
            #    self.comm.project.paths["salvus_opt"],
            #    "PHYSICAL_MODELS",
            #    f"{iteration}.h5"
            # )

    def calculate_station_weights(self, event: str):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        # Name weight set after event to know it
        weight_set_name = event
        # If set exists, we don't recalculate it
        if self.lasif_comm.weights.has_weight_set(weight_set_name):
            print(f"Weight set already exists for event {event}")
            return

        lapi.compute_station_weights(
            self.lasif_comm, weight_set=event, events=[event]
        )

    def misfit_quantification(self, event: str, mpi=True, n=8):
        """
        Quantify misfit and calculate adjoint sources.

        :param event: Name of event
        :type event: str
        :param mpi: If you want to run with MPI, default True
        :type mpi: bool
        :param n: How many ranks to run on
        :type n: int
        """
        iteration = self.comm.project.current_iteration
        if self.comm.project.inversion_mode == "mini-batch":
            window_set = iteration + "_" + event
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
        if os.path.exists(adjoint_path):
            print(f"Adjoint source exists for event: {event} ")
            print(
                "Will not be recalculated. If you want them "
                f"calculated, delete file: {adjoint_path}"
            )
        elif mpi:
            os.chdir(self.comm.project.lasif_root)
            command = f"mpirun -n {n} lasif calculate_adjoint_sources "
            command += f"{iteration} "
            command += f"{window_set} {event} --weight_set {event}"
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, bufsize=1
            )
            for line in process.stdout:
                print(line, end="\n", flush=True)
            process.wait()
            print(process.returncode)
            os.chdir(self.comm.project.inversion_root)

        else:
            lapi.calculate_adjoint_sources(
                self.lasif_comm,
                iteration=iteration,
                window_set=window_set,
                weight_set=event,
                events=[event],
            )
        # See if misfit has already been written into iteration toml
        if self.comm.project.misfits[event] == 0.0:
            misfit_toml_path = (
                self.lasif_comm.project.paths["iterations"]
                / f"ITERATION_{iteration}"
                / "misfits.toml"
            )
            misfit = toml.load(misfit_toml_path)[event]["event_misfit"]
            # misfit = self.lasif_comm.adj_sources.get_misfit_for_event(
            #     event=event, weight_set_name=event, iteration=iteration
            # )
        else:
            misfit = self.comm.project.misfits[event]
            print(f"Misfit for {event} has already been computed. ")
            print(
                "If you want it recomputed, change it to 0.0 in the iteration toml file"
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

    def write_misfit(self, events=None, details=None):  # Not used currently
        """
        Write the iteration's misfit into a toml file.
        TODO: I might want to add this to make it do more statistics
        """
        print("Writing Misfit")
        iteration = self.comm.project.current_iteration
        misfit_path = os.path.join(
            self.lasif_root,
            "ITERATIONS",
            f"ITERATION_{iteration}",
            "misfits.toml",
        )
        if os.path.exists(misfit_path):
            if details and "compute additional" in details:
                # Reason for this that I have to append to path in this
                # specific case.
                print("Misfit file exists, will append additional events")
                # Need to see if the misfit is already in there or not
                misfits = toml.load(misfit_path)
                append = False
                for event in events:
                    if event in misfits.keys():
                        if misfits[event]["event_misfit"] == 0.0:
                            append = True
                    else:
                        append = True
                if not append:
                    print(
                        "Misfit already exists. If you want it rewritten, "
                        "delete the misfit toml in the lasif_project"
                    )
                    return
                print("Misfit file exists, will append additional events")
            else:
                print(
                    "Misfit already exists. If you want it rewritten, "
                    "delete the misfit toml in the lasif_project"
                )
                return
        lapi.write_misfit(self.lasif_comm, iteration=iteration, events=events)

    def _already_processed(self, event: str) -> bool:
        """
        Looks for processed data for a certain event

        :param event: Name of event
        :type event: str
        :return: True/False regarding the alreadyness of the processed data.
        :rtype: bool
        """
        low_period = self.comm.project.min_period
        high_period = self.comm.project.max_period
        processed_filename = (
            "preprocessed_"
            + str(int(low_period))
            + "s_to_"
            + str(int(high_period))
            + "s.h5"
        )
        processed_data_folder = self.lasif_comm.project.paths[
            "preproc_eq_data"
        ]

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

        lapi.process_data(self.lasif_comm, events=[event])

    def select_windows(self, window_set_name: str, event: str, mpi=True, n=12):
        """
        Select window for a certain event in an iteration.

        :param window_set_name: Name of window set
        :type window_set_name: str
        :param event: Name of event to pick windows on
        :type event: str
        :param mpi: Switch on/off for running with MPI
        :type mpi: bool
        :param n: How many ranks to use
        :type n: int
        """
        # Check if window set exists:
        path = os.path.join(
            self.lasif_root, "SETS", "WINDOWS", f"{window_set_name}.sqlite"
        )
        if os.path.exists(path):
            print(f"Window set for event {event} exists.")
            return

        if mpi:
            os.chdir(self.comm.project.lasif_root)
            command = f"mpirun -n {n} lasif select_windows "
            command += f"{self.comm.project.current_iteration} "
            command += f"{window_set_name} {event}"
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, bufsize=1
            )
            for line in process.stdout:
                print(line, end=" \n", flush=True)
            process.wait()
            print(process.returncode)
            os.chdir(self.comm.project.inversion_root)
        else:
            lapi.select_windows(
                self.lasif_comm,
                iteration=self.comm.project.current_iteration,
                window_set=window_set_name,
                events=[event],
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
