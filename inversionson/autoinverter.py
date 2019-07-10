"""
A class which takes care of a Full-Waveform Inversion using multiple meshes.
It uses Salvus, Lasif and Multimesh to perform most of its actions.
This is a class which wraps the three packages together to perform an
automatic Full-Waveform Inversion
"""
import numpy as np
import time
import os
import pyasdf
import multi_mesh.api as mapi
from storyteller import Storyteller
from inversionson import InversionsonError, InversionsonWarning


class autoinverter(object):
    """
    Ok lets do this.
    We need something that reads Salvus opt
    Something that talks to salvus flow
    Something that creates meshes
    Something that talks to lasif
    Something that talks to multimesh
    Something interacting with dp-s random selection (No need now)
    Something talking to the smoother.
    So let's create a few files:
    salvus_opt communicator
    salvus_flow communicator
    salvus_mesher (not necessary now)
    lasif communicator
    multimesh communicator
    I can regularly save the inversion_dict as a toml file and reload it
    """

    def __init__(self, info_dict: dict, simulation_dict: dict,
                 inversion_dict: dict):
        self.info = info_dict
        self.comm = self._find_project_comm()
        self.storyteller = Storyteller()
        self.iteration_dict = {}

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        from inversionson.components.project import ProjectComponent

        return ProjectComponent(self.info).get_communicator()

    def _validate_inversion_project(self):
        """
        Make sure everything is correctly set up in order to perform inversion.

        :param info_dict: Information needed
        :type info_dict: dict
        :param simulation_dict: Information regarding simulations
        :type simulation_dict: dict
        """
        import pathlib

        if "inversion_name" not in self.info.keys():
            raise ValueError(
                "The inversion needs a name")

        # Salvus Opt
        if "salvus_opt_dir" not in self.info.keys():
            raise ValueError(
                "Information on salvus_opt_dir is missing from information")
        else:
            folder = pathlib.Path(self.info["salvus_opt_dir"])
            if not (folder / "inversion.toml").exists():
                raise ValueError("Salvus opt inversion not initiated")

        # Lasif
        if "lasif_project" not in self.info.keys():
            raise ValueError(
                "Information on lasif_project is missing from information")
        else:
            folder = pathlib.Path(self.info["lasif_project"])
            if not (folder / "lasif_config.toml").exists():
                raise ValueError("Lasif project not initialized")

        # Simulation parameters:
        if "end_time_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding end time of simulation missing")

        if "time_step_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding time step of simulation missing")

        if "start_time_in_seconds" not in self.sim_info.keys():
            raise ValueError(
                "Information regarding start time of simulation missing")

    def initialize_inversion(self):
        """
        Set up everything regarding the inversion. Make sure everything
        is correct and that information is there.
        Make this check status of salvus opt, the inversion does not have
        to be new to call this method.
        """
        # Check status of inversion. If no task file or if task is closed,
        # run salvus opt. Otherwise just read task and start working.
        # Will do later.
        task = self.comm.salvus_opt.read_salvus_opt()
        if task == "no_task_toml":
            print("Salvus Opt has not been fully configured yet."
                  "Please supply it with an initial model and "
                  "your parameter settings. Run the binary, and"
                  " there should be a 'task.toml' file in the Opt "
                  "directory. Do this and start Inversionson.")
            # Do something regarding initializing inversion
        else:
            self.perform_task(task)

    def prepare_iteration(self):
        """
        Prepare iteration.
        Get iteration name from salvus opt
        Modify name in inversion status
        Create iteration
        Pick events
        Make meshes if needed
        Update information in iteration dictionary.
        """
        it_name = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.current_iteration = it_name
        it_toml = os.path.join(self.comm.project.paths["iteration_tomls"], iteration + ".toml")
        if os.path.exists(it_toml):
            self.comm.project.get_iteration_attributes(it_name)
            # If the iteration toml was just created but
            # not the iteration, we finish making the iteration
            if len(self.comm.project.events_in_iteration) != 0:
                return
        events = self.comm.lasif.get_minibatch(it_name)  # Sort this out.
        self.comm.lasif.set_up_iteration(it_name, events)

        for event in events:
            if not self.comm.lasif.has_mesh(event):
                self.comm.salvus_mesher.create_mesh(event)
                self.comm.lasif.move_mesh(event, it_name)
            else:
                self.comm.lasif.move_mesh(event, it_name)

        self.comm.project.create_iteration_toml(it_name, events)
        self.comm.project.get_iteration_attributes(it_name)
        # Storyteller

    def interpolate_model(self, event: str):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        """
        self.comm.multi_mesh.interpolate_to_simulation_mesh(event)

    def interpolate_gradient(self, event: str, first: bool):
        """
        Interpolate gradient to master mesh

        :param event: Name of event
        :type event: str
        :param first: First iteration gradient to be interolated?
        :type first: bool
        """
        self.comm.multi_mesh.interpolate_gradient_to_model(event, first=first)

    def run_forward_simulation(self, event: str):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        receivers = self.comm.salvus_flow.get_receivers(event)
        source = self.comm.salvus_flow.get_source_object(event)
        w = self.comm.salvus_flow.construct_simulation(
            event, source, receivers)
        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks
        )
        self.comm.project.forward_job[event]["submitted"] = True
        self.comm.project.update_iteration_toml()

    def calculate_station_weights(self, event: str):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        self.comm.lasif.calculate_station_weights(event)

    def run_adjoint_simulation(self, event: str):
        """
        Submit adjoint simulation to daint and possibly monitor

        :param event: Name of event
        :type event: str
        """
        adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
        w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
            event, adj_src)

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks
        )
        self.comm.project.adjoint_job[event]["submitted"] = True
        self.comm.project.update_iteration_toml()

    def misfit_quantification(self, event: str):
        """
        Compute misfit and adjoint source for iteration

        :param event: Name of event
        :type event: str
        """
        misfit = self.comm.lasif.misfit_quantification(event)
        self.comm.project.misfits[event] = misfit
        self.comm.project.update_iteration_toml()

    def process_data(self, event: str):
        """
        Process data for an event in the currently considered
        period range. If the processed data already exists,
        this does not do anything. The processing parameters
        are defined in the Lasif project. Make sure they are
        consistant with what is defined in Inversionson.

        :param event: Name of event
        :type event: str
        """
        self.comm.lasif.process_data(event)

    def select_windows(self, event: str):
        """
        Select windows for an event in this iteration.
        If event is in the control group, new windows will
        not be picked.

        :param event: [description]
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        window_set_name = iteration + "_" + event

        # If event is in control group, we look for newest window set for event
        if event in self.comm.project.control_group:
            import glob
            import shutil
            windows = self.comm.lasif.lasif_comm.project.paths["windows"]
            window_sets = glob.glob(os.path.join(windows, "*"+event+"*"))
            latest_windows = max(window_sets, key=os.path.getctime)

            shutil.copy(latest_windows, os.path.join(
                windows, window_set_name + ".sqlite"))
        else:
            self.comm.lasif.select_windows(
                window_set_name=window_set_name,
                event=event
            )

    def monitor_jobs(self, sim_type: str):
        """
        Takes events in iteration and monitors its job stati

        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str

        Can return a list of events which have been retrieved.
        If none... call itself again.
        """
        import time
        events_retrieved_now = []
        events_already_retrieved = []
        for event in self.comm.project.events_used:
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    time.sleep(30)
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type)  # This thing might time out

                    if status == JobStatus.finished:
                        self.comm.project.forward_job[event]["retrieved"] = True
                        self.comm.project.update_iteration_toml()
                        events_retrieved_now.append(event)
                    elif status == JobStatus.pending:
                        continue
                    elif status == JobStatus.running:
                        continue
                    elif status == JobStatus.failed:
                        print("Job failed. Need to implement something here")
                        print("Probably resubmit or something like that")
                    elif status == JobStatus.unknown:
                        print("Job unknown. Need to implement something here")
                    elif status == JobStatus.cancelled:
                        print("What to do here?")
                    else:
                        raise InversionsonWarning(
                            f"Inversionson does not recognise job status: "
                            f"{status}")

            elif sim_type == "adjoint":
                if self.comm.project.forward_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    time.sleep(30)
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type)  # This thing might time out

                    if status == JobStatus.finished:
                        self.comm.project.adjoint_job[event]["retrieved"] = True
                        self.comm.project.update_iteration_toml()
                        events_retrieved_now.append(event)
                    elif status == JobStatus.pending:
                        continue
                    elif status == JobStatus.running:
                        continue
                    elif status == JobStatus.failed:
                        print("Job failed. Need to implement something here")
                        print("Probably resubmit or something like that")
                    elif status == JobStatus.unknown:
                        print("Job unknown. Need to implement something here")
                    elif status == JobStatus.cancelled:
                        print("What to do here?")
                    else:
                        raise InversionsonWarning(
                            f"Inversionson does not recognise job status: "
                            f"{status}")
            else:
                raise ValueError(f"Sim type {sim_type} not supported")

        # If no events have been retrieved, we call the function again.
        if len(events_already_retrieved) == len(self.comm.project.events_used):
            return "All retrieved"
        if len(events_retrieved_now) == 0:
            self.monitor_jobs(sim_type)
        return events_retrieved_now

    def wait_for_all_jobs_to_finish(self, sim_type: str):
        """
        Just a way to make the algorithm wait until all jobs are done.

        :param sim_type: Simulation type forward or adjoint
        :type sim_type: str
        """
        if sim_type == "forward":
            jobs = self.comm.project.forward_job
        elif sim_type == "adjoint":
            jobs = self.comm.project.adjoint_job

        done = np.zeros(len(self.comm.project.events_used), dtype=bool)
        for _i, event in enumerate(self.comm.project.events_used):
            if jobs[event]["retrieved"]:
                done[_i] = True
            else:
                status = self.comm.salvus_flow.get_job_status(event, sim_type)
                if status == JobStatus.finished:
                    jobs[event]["retrieved"] = True

        if sim_type == "forward":
            self.comm.project.forward_job = jobs
        elif sim_type == "adjoint":
            self.comm.project.adjoint_job = jobs
        self.comm.project.update_iteration_toml()

        if not np.all(done):  # If not all done, keep monitoring
            self.wait_for_all_jobs_to_finish(sim_type)

    def perform_task(self, task: str):
        """
        Input a task and send to correct function

        :param task: task issued by salvus opt
        :type task: str
        """
        if task == "compute_misfit_and_gradient":
            self.prepare_iteration()
            # Figure out a way to do this on a per event basis.
            for event in self.comm.project.events_used:
                self.interpolate_model(event)
                self.run_forward_simulation(event)
                self.calculate_station_weights(event)

            events_retrieved = []
            while events_retrieved != "All retrieved":
                time.sleep(30)
                events_retrieved = self.monitor_jobs("forward")
                if events_retrieved == "All retrieved":
                    break
                else:
                    for event in events_retrieved:
                        self.process_data(event)
                        self.select_windows(event)
                        self.misfit_quantification(event)
                        self.run_adjoint_simulation(event)

            self.wait_for_all_jobs_to_finish("forward")
            self.wait_for_all_jobs_to_finish("adjoint")
            self.comm.salvus_opt.write_misfit_to_task_toml()
            self.storyteller.document_task(task)
            self.comm.project.update_iteration_toml()
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.salvus_opt.run_salvus_opt()
            task = self.comm.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "compute_misfit":
            self.prepare_iteration()
            for event in self.comm.project.events_used:
                self.interpolate_model(event)
                self.run_forward_simulation(event)
                self.calculate_station_weights(event)
            events_retrieved = []
            while events_retrieved != "All retrieved":
                time.sleep(30)
                events_retrieved = self.monitor_jobs("forward")
                if events_retrieved == "All retrieved":
                    break
                else:
                    for event in events_retrieved:
                        self.process_data(event)
                        self.select_windows(event)
                        self.misfit_quantification(event)

            self.comm.salvus_opt.write_misfit_to_task_toml()
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.project.update_iteration_toml()
            self.comm.salvus_opt.run_salvus_opt()
            task = self.comm.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "compute_gradient":
            iteration = self.comm.project.get_newest_iteration_name()
            self.comm.project.get_iteration_attributes(iteration)
            for event in self.comm.project.events_used:
                self.run_adjoint_simulation(event)
            # Cut sources and receivers?
            events_retrieved = []
            first = True  # Maybe this will not be needed later.
            while events_retrieved != "All retrieved":
                time.sleep(30)
                events_retrieved = self.monitor_jobs("adjoint")
                if events_retrieved == "All retrieved":
                    break
                else:
                    for event in events_retrieved:
                        self.interpolate_gradient(event, first)
                        first = False
            # Smooth gradients
            self.comm.salvus_opt.move_gradient_to_salvus_opt_folder(event)
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.salvus_opt.run_salvus_opt()
            task = self.comm.salvus_opt.read_salvus_opt()
            self.perform_task(task)

        elif task == "finalize_iteration":
            iteration = self.comm.project.get_newest_iteration_name()
            self.comm.project.get_iteration_attributes(iteration)
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.project.update_iteration_toml()
            self.comm.salvus_opt.run_salvus_opt()
            task = self.comm.salvus_opt.read_salvus_opt()
            self.perform_task(task)
            # Possibly delete wavefields

        else:
            raise ValueError(f"Salvus Opt task {task} not known")

    def run_inversion(self):
        """
        Workflow:
                Read Salvus opt,
                Perform task,
                Document it
                Close task, repeat.
        """
        # Always do this as a first thing, Might write a different function for checking status
        self.initialize_inversion()

        task = self.comm.salvus_opt.read_salvus_opt()

        self.perform_task(task)
