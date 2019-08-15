"""
A class which takes care of a Full-Waveform Inversion using multiple meshes.
It uses Salvus, Lasif and Multimesh to perform most of its actions.
This is a class which wraps the three packages together to perform an
automatic Full-Waveform Inversion
"""
import numpy as np
import time
import os
import sys
from inversionson import InversionsonError, InversionsonWarning
import lasif.api


class AutoInverter(object):
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

    def __init__(self, info_dict: dict):
        self.info = info_dict
        print("Will make communicator now")
        self.comm = self._find_project_comm()
        print("Now I want to start running the inversion")
        self.run_inversion()

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        from inversionson.components.project import ProjectComponent

        return ProjectComponent(self.info).get_communicator()

    # def initialize_inversion(self):
    #     """
    #     Set up everything regarding the inversion. Make sure everything
    #     is correct and that information is there.
    #     Make this check status of salvus opt, the inversion does not have
    #     to be new to call this method.
    #     """
    #     # Check status of inversion. If no task file or if task is closed,
    #     # run salvus opt. Otherwise just read task and start working.
    #     # Will do later.
    #     task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
    #     self.comm.project.get_inversion_attributes(
    #         simulation_info=self.comm.project.simulation_dict
    #     )
    #     self.comm.project.create_control_group_toml()
    #     self.perform_task(task, verbose)

    def prepare_iteration(self, first=False):
        """
        Prepare iteration.
        Get iteration name from salvus opt
        Modify name in inversion status
        Pick events
        Create iteration
        Make meshes if needed
        Update information in iteration dictionary.
        """
        it_name = self.comm.salvus_opt.get_newest_iteration_name()
        first_try = self.comm.salvus_opt.first_trial_model_of_iteration()
        self.comm.project.change_attribute("current_iteration", it_name)
        it_toml = os.path.join(
            self.comm.project.paths["iteration_tomls"], it_name + ".toml")
        if self.comm.lasif.has_iteration(it_name):
            if not os.path.exists(it_toml):
                self.comm.project.create_iteration_toml(it_name)
            self.comm.project.get_iteration_attributes()
            # If the iteration toml was just created but
            # not the iteration, we finish making the iteration
            # Should never happen though
            if len(self.comm.project.events_in_iteration) != 0:
                for event in self.comm.project.events_in_iteration:
                    if not self.comm.lasif.has_mesh(event):
                        self.comm.salvus_mesher.create_mesh(event)
                        self.comm.lasif.move_mesh(event, it_name)
                    else:
                        self.comm.lasif.move_mesh(event, it_name)
                return
        if first_try:
            events = self.comm.lasif.get_minibatch(first)
        else:
            prev_try = self.comm.salvus_opt.get_previous_iteration_name(
                tr_region=True)
            prev_try = self.comm.project.get_old_iteration_info(prev_try)
            events = list(prev_try["events"].keys())
        self.comm.project.change_attribute("current_iteration", it_name)
        self.comm.lasif.set_up_iteration(it_name, events)

        for event in events:
            if not self.comm.lasif.has_mesh(event):
                self.comm.salvus_mesher.create_mesh(event)
                self.comm.lasif.move_mesh(event, it_name)
            else:
                self.comm.lasif.move_mesh(event, it_name)

        self.comm.project.update_control_group_toml(first=first)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes(it_name)

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
        :param first: First iteration gradient to be interpolated?
        :type first: bool
        """
        self.comm.multi_mesh.interpolate_gradient_to_model(event,
                                                           smooth=True)

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

    def get_first_batch_of_events(self) -> list:
        """
        Get the initial batch of events to compute misfits and gradients for
        
        :return: list of events to use
        :rtype: list
        """
        events = self.comm.lasif.get_minibatch(first=True)
        self.comm.project.events_used = events

    def smooth_gradient(self, event: str):
        """
        Send a gradient for an event to the Salvus smoother

        :param event: name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        gradient = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event,
            smooth=False
        )
        self.comm.smoother.generate_input_toml(gradient)
        self.comm.smoother.run_smoother(gradient)

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

    def perform_task(self, task: str, verbose: str):
        """
        Input a task and send to correct function

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        if task == "compute_misfit_and_gradient":
            print("Will prepare iteration")
            self.prepare_iteration(first=True)
            print("Iteration prepared")
            print("Will select first event batch")
            # self.get_first_batch_of_events() Already have the batch from
            # The iteration preparation
            print("Initial batch selected")
            for event in self.comm.project.events_in_iteration:
                print(f"{event} interpolation")
                self.interpolate_model(event)
                print("Run forward simulation")
                #self.run_forward_simulation(event)
                print("Calculate station weights")
                self.calculate_station_weights(event)
            
            sys.exit("Don't want to run simulations")
            print("Waiting for jobs")
            events_retrieved = []
            while events_retrieved != "All retrieved":
                time.sleep(30)
                events_retrieved = self.monitor_jobs("forward")
                if events_retrieved == "All retrieved":
                    break
                else:
                    for event in events_retrieved:
                        print(f"{event} retrieved")
                        self.process_data(event)
                        self.select_windows(event)
                        self.misfit_quantification(event)
                        self.run_adjoint_simulation(event)

            self.wait_for_all_jobs_to_finish("forward")
            self.wait_for_all_jobs_to_finish("adjoint")

            for event in self.comm.project.events_used:
                self.smooth_gradient(event)

            self.comm.salvus_opt.write_misfit_and_gradient_to_task_toml()
            self.comm.project.update_iteration_toml()
            self.comm.storyteller.document_task(task)
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.salvus_opt.run_salvus_opt()
            task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
            self.perform_task(task, verbose)

        elif task == "compute_misfit":
            self.prepare_iteration()
            if "compute misfit for" in verbose:
                events_to_use = self.comm.project.old_control_group
            else:
                events_to_use = list(
                    set(self.comm.project.events_in_iteration) - set(
                        self.comm.project.old_control_group))
            for event in events_to_use:
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

            self.comm.storyteller.document_task(task, verbose)
            self.comm.salvus_opt.write_misfit_to_task_toml()
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.project.update_iteration_toml()
            self.comm.salvus_opt.run_salvus_opt()
            task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
            self.perform_task(task, verbose)

        elif task == "compute_gradient":
            iteration = self.comm.salvus_opt.get_newest_iteration_name()
            self.comm.project.current_iteration = iteration
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
                        self.smooth_gradient(event)
                        self.interpolate_gradient(event, first)
                        first = False
            # Smooth gradients
            self.comm.salvus_opt.move_gradient_to_salvus_opt_folder(event)
            self.comm.salvus_opt.get_new_control_group()
            self.comm.storyteller.document_task(task)
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.salvus_opt.run_salvus_opt()
            task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
            self.perform_task(task, verbose)

        elif task == "finalize_iteration":
            iteration = self.comm.salvus_opt.get_newest_iteration_name()
            self.comm.project.current_iteration = iteration
            self.comm.project.get_iteration_attributes(iteration)
            self.comm.salvus_opt.close_salvus_opt_task()
            self.comm.project.update_iteration_toml()
            self.comm.salvus_opt.run_salvus_opt()
            task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
            self.perform_task(task, verbose)
            # Possibly delete wavefields
        
        elif task == "select_control_batch":
            iteration = self.comm.salvus_opt.get_newest_iteration_name()
            self.comm.project.current_iteration = iteration
            self.comm.project.get_iteration_attributes()
            # Need to implement these below
            control_group = self.comm.minibatch.select_optimal_control_group()
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=control_group)
            self.comm.project.new_control_group = control_group
            self.comm.project.update_control_group_toml(new=True)

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
        # Always do this as a first thing, Might write a different function
        # for checking status
        # self.initialize_inversion()

        task, verbose = self.comm.salvus_opt.read_salvus_opt_task()

        self.perform_task(task, verbose)


def read_information_toml(info_toml_path: str):
    """
    Read a toml file with inversion information into a dictionary

    :param info_toml_path: Path to the toml file
    :type info_toml_path: str
    """
    import toml

    return toml.load(info_toml_path)


if __name__ == "__main__":
    info_toml = input("Give me a path to your information_toml \n")
    if not info_toml.startswith("/"):
        import os
        cwd = os.getcwd()
        if info_toml.startswith("./"):
            info_toml = os.path.join(cwd, info_toml[2:])
        elif info_toml.startswith("."):
            info_toml = os.path.join(cwd, info_toml[1:])
        else:
            info_toml = os.path.join(cwd, info_toml)
    info = read_information_toml(info_toml)
    invert = AutoInverter(info)
