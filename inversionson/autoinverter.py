import numpy as np
import time
import os
import shutil
import sys
import warnings
import emoji
from inversionson import InversionsonError, InversionsonWarning
import toml
from colorama import init
from colorama import Fore, Style

init()


class AutoInverter(object):
    """
    A class which takes care of a Full-Waveform Inversion using multiple 
    meshes.
    It uses Salvus, Lasif and Multimesh to perform most of its actions.
    This is a class which wraps the three packages together to perform an
    automatic Full-Waveform Inversion
    """

    def __init__(self, info_dict: dict):
        self.info = info_dict
        print(Fore.RED + "Will make communicator now")
        self.comm = self._find_project_comm()
        print(Fore.GREEN + "Now I want to start running the inversion")
        print(Style.RESET_ALL)
        self.run_inversion()

    def _find_project_comm(self):
        """
        Get lasif communicator.
        """
        from inversionson.components.project import ProjectComponent

        return ProjectComponent(self.info).get_communicator()

    def _send_whatsapp_announcement(self):
        """
        Send a quick announcement via whatsapp when an iteration is done
        You need to have a twilio account, a twilio phone number
        and enviroment variables:
        MY_NUMBER
        TWILIO_NUMBER
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        """
        from twilio.rest import Client

        client = Client()
        from_whatsapp = f"whatsapp:{os.environ['TWILIO_NUMBER']}"
        to_whatsapp = f"whatsapp:{os.environ['MY_NUMBER']}"
        iteration = self.comm.project.current_iteration
        string = f"Your Inversionson code is DONE_WITH_{iteration}"
        client.messages.create(body=string, from_=from_whatsapp, to=to_whatsapp)

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
            self.comm.project.paths["iteration_tomls"], it_name + ".toml"
        )
        if self.comm.lasif.has_iteration(it_name):
            if not os.path.exists(it_toml):
                self.comm.project.create_iteration_toml(it_name)
            self.comm.project.get_iteration_attributes()
            # If the iteration toml was just created but
            # not the iteration, we finish making the iteration
            # Should never happen though
            if len(self.comm.project.events_in_iteration) != 0:
                if self.comm.project.meshes == "multi-mesh":
                    for event in self.comm.project.events_in_iteration:
                        if not self.comm.lasif.has_mesh(event):
                            self.comm.salvus_mesher.create_mesh(event)
                            self.comm.lasif.move_mesh(event, it_name)
                        else:
                            self.comm.lasif.move_mesh(event, it_name)
                return
        if first_try:
            if self.comm.project.inversion_mode == "mini-batch":
                events = self.comm.lasif.get_minibatch(first)
            else:
                events = self.comm.lasif.list_events()
        else:
            prev_try = self.comm.salvus_opt.get_previous_iteration_name(tr_region=True)
            prev_try = self.comm.project.get_old_iteration_info(prev_try)
            events = list(prev_try["events"].keys())
        self.comm.project.change_attribute("current_iteration", it_name)
        self.comm.lasif.set_up_iteration(it_name, events)
        if self.comm.project.meshes == "multi-mesh":
            for event in events:
                if not self.comm.lasif.has_mesh(event):
                    self.comm.salvus_mesher.create_mesh(event)
                    self.comm.lasif.move_mesh(event, it_name)
                else:
                    self.comm.lasif.move_mesh(event, it_name)

        self.comm.project.update_control_group_toml(first=first)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes()
        # mixa inn control group fra gamalli vinkonu
        # Get control group info into iteration attributes
        # ctrl_groups = toml.load(
        #     self.comm.project.paths["control_group_toml"])
        # if it_name in ctrl_groups.keys():
        #     self.comm.project.change_attribute(
        #         "old_control_group", ctrl_groups[it_name]["old"])

    def interpolate_model(self, event: str):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        """
        if self.comm.project.forward_job[event]["submitted"]:
            print(
                f"Event {event} has already been submitted. "
                "Will not do interpolation."
            )
            return
        if self.comm.project.forward_job[event]["interpolated"]:
            print(
                f"Mesh for {event} has already been interpolated. "
                "Will not do interpolation."
            )
            return
        interp_folder = os.path.join(
            self.comm.project.inversion_root,
            "INTERPOLATION",
            event,
            "model"
        )
        if not os.path.exists(interp_folder):
            os.makedirs(interp_folder)
        self.comm.multi_mesh.interpolate_to_simulation_mesh(event, interp_folder=interp_folder)
        self.comm.project.change_attribute(
            attribute=f'forward_job["{event}"]["interpolated"]', new_value=True
        )
        self.comm.project.update_iteration_toml()

    def interpolate_gradient(self, event: str):
        """
        Interpolate gradient to master mesh

        :param event: Name of event
        :type event: str
        """
        if self.comm.project.adjoint_job[event]["interpolated"]:
            print(
                f"Gradient for {event} has already been interpolated. "
                f"Will not do interpolation."
            )
            return
        interp_folder = os.path.join(
            self.comm.project.inversion_root,
            "INTERPOLATION",
            event,
            "gradient"
        )
        if not os.path.exists(interp_folder):
            os.makedirs(interp_folder)
        self.comm.multi_mesh.interpolate_gradient_to_model(
            event,
            smooth=True,
            interp_folder=interp_folder)
        self.comm.project.change_attribute(
            attribute=f'adjoint_job["{event}"]["interpolated"]', new_value=True
        )
        self.comm.project.update_iteration_toml()

    def run_forward_simulation(self, event: str):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        # Check status of simulation
        job_info = self.comm.project.forward_job[event]
        if job_info["submitted"]:
            if job_info["retrieved"]:
                print(f"Simulation for event {event} already done.")
                print("If you want it redone, change its status in iteration toml")
                return
            else:
                status = str(
                    self.comm.salvus_flow.get_job_status(
                        event=event, sim_type="forward"
                    )
                )
                if status == "JobStatus.running":
                    print(f"Forward job for event {event} is running ")
                    print("Will not resubmit. ")
                    print("You can work with jobs using salvus-flow")
                    return
                elif status == "JobStatus.pending":
                    print(f"Forward job for event {event} is pending ")
                    print("Will not resubmit. ")
                    return
                elif status == "JobStatus.unknown":
                    print(f"Status of job for event {event} is unknown")
                    print(f"Will resubmit")
                elif status == "JobStatus.cancelled":
                    print(f"Status of job for event {event} is cancelled")
                    print(f"Will resubmit")
                elif status == "JobStatus.finished":
                    print(f"Status of job for event {event} is finished")
                    # print("Will retrieve and update toml")
                    # self.comm.project.change_attribute(
                    #         attribute=f"forward_job[\"{event}\"][\"retrieved\"]",
                    #         new_value=True)
                    # self.comm.project.update_iteration_toml()
                    return
                else:
                    print("Jobstatus unknown for event {event}")
                    print("Will resubmit")

        receivers = self.comm.salvus_flow.get_receivers(event)
        source = self.comm.salvus_flow.get_source_object(event)
        # print("Adding correct fields to mesh, in a test phase currently")
        # self.comm.salvus_mesher.add_fluid_and_roi_from_lasif_mesh()
        w = self.comm.salvus_flow.construct_simulation(event, source, receivers)
        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks,
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
        job_info = self.comm.project.adjoint_job[event]
        if job_info["submitted"]:
            if job_info["retrieved"]:
                print(f"Simulation for event {event} already done.")
                print("If you want it redone, change its status in iteration toml")
                return
            else:
                status = str(
                    self.comm.salvus_flow.get_job_status(
                        event=event, sim_type="adjoint"
                    )
                )
                if status == "JobStatus.running":
                    print(f"Adjoint job for event {event} is running ")
                    print("Will not resubmit. ")
                    print("You can work with jobs using salvus-flow")
                    return
                elif status == "JobStatus.pending":
                    print(f"Adjoint job for event {event} is pending ")
                    print("Will not resubmit. ")
                    return
                elif status == "JobStatus.unknown":
                    print(f"Status of job for event {event} is unknown")
                    print(f"Will resubmit")
                elif status == "JobStatus.cancelled":
                    print(f"Status of job for event {event} is cancelled")
                    print(f"Will resubmit")
                elif status == "JobStatus.finished":
                    print(f"Status of job for event {event} is finished")
                    print("Will retrieve and update toml")
                    self.retrieve_gradient(event)
                    self.comm.project.change_attribute(
                        attribute=f'adjoint_job["{event}"]["retrieved"]', new_value=True
                    )
                    self.comm.project.update_iteration_toml()
                    return
                else:
                    print("Jobstatus unknown for event {event}")
                    print("Will resubmit")
        adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
        w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(event, adj_src)

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks,
        )
        self.comm.project.change_attribute(
            attribute=f'adjoint_job["{event}"]["submitted"]', new_value=True
        )
        self.comm.project.update_iteration_toml()

    def misfit_quantification(self, event: str):
        """
        Compute misfit and adjoint source for iteration

        :param event: Name of event
        :type event: str
        """
        mpi = True
        if self.comm.project.site_name == "swp":
            mpi = False
        misfit = self.comm.lasif.misfit_quantification(event, mpi=mpi)
        self.comm.project.change_attribute(
            attribute=f'misfits["{event}"]', new_value=misfit
        )
        # self.comm.project.misfits[event] = misfit
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

    def retrieve_seismograms(self, event: str):
        """
        Move seismograms from salvus_flow folder to output folder

        :param event: Name of event
        :type event: str
        """
        self.comm.salvus_flow.retrieve_outputs(event_name=event, sim_type="forward")
        # job_paths = self.comm.salvus_flow.get_job_file_paths(
        #         event=event,
        #         sim_type="forward")

        # seismograms = job_paths[('output', 'point-data', 'filename')]
        # lasif_seismograms = self.comm.lasif.find_seismograms(
        #         event=event,
        #         iteration=self.comm.project.current_iteration)

        # if self.comm.project.site_name == "swp":
        #     shutil.copyfile(seismograms, lasif_seismograms)
        # elif self.comm.project.site_name == "daint":
        #     string = f"scp daint:{seismograms} {lasif_seismograms}"
        #     os.system(string)
        #     print("Security_sleep")
        #     time.sleep(10)
        # else:
        #     raise InversionsonError("Retrieval only works for swp & daint")
        print(f"Copied seismograms for event {event} to lasif folder")

    def retrieve_gradient(self, event: str, smooth=False, par=None):
        """
        Move gradient from salvus_flow folder to lasif folder
        
        :param event: Name of event
        :type event: str
        :param smooth: Am I returning gradient from smoothing? Default: False
        :type smooth: bool
        """
        # TODO: Do I want this?
        if self.comm.project.adjoint_job[event]["retrieved"] and not smooth:
            print(
                f"Gradient for event {event} has already been retrieved. "
                f"Will not retrieve it again."
            )
            return
        if smooth:
            sim_type = "smoothing"
            self.comm.salvus_flow.retrieve_outputs(
                event_name=event, sim_type=sim_type, par=par
            )
        else:
            sim_type = "adjoint"
            self.comm.salvus_flow.retrieve_outputs(event_name=event, sim_type=sim_type)
        print(f"Gradient for event {event} has been retrieved.")

    def select_windows(self, event: str):
        """
        Select windows for an event in this iteration.
        If event is in the control group, new windows will
        not be picked.

        :param event: [description]
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        if self.comm.project.inversion_mode == "mini-batch":
            window_set_name = iteration + "_" + event
        else:
            window_set_name = event
        mpi = True
        if self.comm.project.site_name == "swp":
            mpi = False

        # If event is in control group, we look for newest window set for event
        if event in self.comm.project.old_control_group:
            import glob

            windows = self.comm.lasif.lasif_comm.project.paths["windows"]
            window_sets = glob.glob(os.path.join(windows, "*" + event + "*"))
            latest_windows = max(window_sets, key=os.path.getctime)
            if not os.path.exists(os.path.join(windows, window_set_name + ".sqlite")):
                shutil.copy(
                    latest_windows, os.path.join(windows, window_set_name + ".sqlite")
                )
        else:
            self.comm.lasif.select_windows(
                window_set_name=window_set_name, event=event, mpi=mpi
            )

    def get_first_batch_of_events(self) -> list:
        """
        Get the initial batch of events to compute misfits and gradients for
        
        :return: list of events to use
        :rtype: list
        """
        events = self.comm.lasif.get_minibatch(first=True)
        self.comm.project.events_used = events

    def preprocess_gradient(self, event: str):
        """
        Cut sources and receivers from gradient before smoothing.
        We also clip the gradient to some percentile
        This can all be configured in information toml.

        :param event: Name of event
        :type event: str
        """
        # print("Don't want to preprocess gradient now.")
        # return
        from .utils import cut_source_region_from_gradient
        from .utils import cut_receiver_regions_from_gradient
        from .utils import clip_gradient

        gradient = self.comm.lasif.find_gradient(
            iteration=self.comm.project.current_iteration,
            event=event,
            smooth=False,
            inversion_grid=False,
        )
        if self.comm.project.smoothing_job[event][self.comm.project.inversion_params[0]]["submitted"]:
            print(f"Already preprocessed gradient for {event}, will not redo")
            return

        if self.comm.project.cut_source_radius > 0.0:
            print("Cutting source region")
            source_location = self.comm.lasif.get_source(event_name=event)
            cut_source_region_from_gradient(
                mesh=gradient,
                source_location=source_location,
                radius_to_cut=self.comm.project.cut_source_radius,
            )

        if self.comm.project.cut_receiver_radius > 0.0:
            print("Cutting receiver regions")
            receiver_info = self.comm.lasif.get_receivers(event_name=event)
            cut_receiver_regions_from_gradient(
                mesh=gradient,
                receivers=receiver_info,
                radius_to_cut=self.comm.project.cut_receiver_radius,
            )
        if self.comm.project.clip_gradient != 1.0:
            print("Clipping gradient")
            clip_gradient(mesh=gradient, percentile=self.comm.project.clip_gradient)

    def sum_gradients(self):
        """
        Sum the computed gradients onto one mesh
        """
        from inversionson.utils import sum_gradients
        events = self.comm.projects.events_in_iteration
        gradients = []
        for event in events:
            gradients.append(
                self.comm.lasif.find_gradient(
                    iteration=self.comm.project.current_iteration,
                    event=event,
                    summed=False,
                    smooth=False,
                    just_give_path=False
                )
            )
        grad_mesh = self.comm.lasif.find_gradient(
            iteration=self.comm.project.current_iteration,
            event=None,
            summed=True,
            smooth=False,
            just_give_path=True
        )
        sum_gradients(
            mesh=grad_mesh,
            gradients=gradients
        )

    def prepare_gradient_for_smoothing(self, event) -> object:
        """
        Add smoothing fields to the relevant mesh.
        
        :param event: Name of event
        :type event: str
        """
        return self.comm.salvus_mesher.add_smoothing_fields(event)

    def smooth_gradient(self, event: str):
        """
        Send a gradient for an event to the Salvus smoother

        :param event: name of event
        :type event: str
        """
        job = self.comm.project.smoothing_job
        for param in self.comm.project.inversion_params:
            if self.comm.project.meshes == "mono-mesh":
                condition = job[param]["retrieved"]
                message = "Summed gradient already smoothed."
            else:
                condition = job[event][param]["retrived"]
                message = f"Gradient for event {event} already smoothed."
            if condition:
                print(
                    message +
                    f" Will not repeat. Change its status in iteration toml "
                    f"if you want to smooth gradient again"
                )
                return
        iteration = self.comm.project.current_iteration
        
        if self.comm.project.meshes == "mono-mesh":
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=None,
                smooth=False,
                summed=True
            )
        else:
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=False
            )
        for _i, par in enumerate(self.comm.project.inversion_params):
            if job[event][par]["submitted"]:
                if job[event][par]["retrieved"]:
                    print(f"Simulation for event {event} already done.")
                    print("If you want it redone, change its status in iteration toml")
                    continue
                else:
                    status = str(
                        self.comm.salvus_flow.get_job_status(
                            event=event, sim_type="smoothing", par=par
                        )
                    )
                    if status == "JobStatus.running":
                        print(f"Adjoint job for event {event} is running ")
                        print("Will not resubmit. ")
                        print("You can work with jobs using salvus-flow")
                        return
                    elif status == "JobStatus.unknown":
                        print(f"Status of job for event {event} is unknown")
                        print(f"Will resubmit")
                    elif status == "JobStatus.cancelled":
                        print(f"Status of job for event {event} is cancelled")
                        print(f"Will resubmit")
                    elif status == "JobStatus.finished":
                        print(f"Status of job for event {event} is finished")
                        print("Will retrieve and update toml")
                        self.retrieve_gradient(event, smooth=True, par=par)
                        self.comm.project.change_attribute(
                            attribute=f'smoothing_job["{event}"]["{par}"]["retrieved"]',
                            new_value=True,
                        )
                        self.comm.project.update_iteration_toml()
                        continue
                    else:
                        print("Jobstatus unknown for event {event}")
                        print("Will resubmit")
                    print(
                        f"Gradient for event {event} already smooth."
                        f" Will not repeat. Change its status in iteration toml "
                        f"if you want to smooth gradient again"
                    )
                    continue
            if _i == 0:
                mesh, smoothed_gradient = self.prepare_gradient_for_smoothing(event)
            simulation = self.comm.smoother.generate_diffusion_object(
                gradient=gradient, par=par, mesh=mesh
            )
            self.comm.salvus_flow.submit_smoothing_job(event, simulation, par)
            self.comm.project.update_iteration_toml()

    def monitor_jobs(self, sim_type: str, events=None, reposts=None):
        """
        Takes events in iteration and monitors its job statuses
        Can return a list of events which have been retrieved.
        If none... call itself again.

        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        :param events: list of events used in task, if nothing given,
            will take all events in iteration
        :type events: list
        """
        import time
        if not reposts:
            reposts = {}
        events_retrieved_now = []
        events_already_retrieved = []
        if not events:
            events = self.comm.project.events_in_iteration
        for event in events:
            if event not in reposts.keys():
                reposts[event] = 0
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    # TODO: Improve this
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type
                    )  # This thing might time out
                    # print(f"Status = {status}")
                    status = str(status)

                    if status == "JobStatus.finished":
                        self.retrieve_seismograms(event)
                        events_retrieved_now.append(event)
                        print(f"Events retrieved now: {events_retrieved_now}")
                    elif status == "JobStatus.pending":
                        print(f"Status = {status}, event: {event}")
                    elif status == "JobStatus.running":
                        print(f"Status = {status}, event: {event}")
                    elif status == "JobStatus.failed":
                        print(f"Status = {status}, event: {event}")
                        print("Job failed. Will resubmit")
                        if reposts[event] >=3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_forward_simulation(event)
                        reposts[event] += 1
                        print("Probably resubmit or something like that")
                    elif status == "JobStatus.unknown":
                        print(f"Status = {status}, event: {event}")
                        print("Job unknown. Will resubmit")
                        if reposts[event] >=3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_forward_simulation(event)
                        reposts[event] += 1
                    elif status == "JobStatus.cancelled":
                        print("What to do here?")
                    else:
                        warnings.warn(
                            f"Inversionson does not recognise job status: " f"{status}",
                            InversionsonWarning,
                        )

            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    status = str(
                        self.comm.salvus_flow.get_job_status(event, sim_type)
                    )  # This thing might time out

                    if status == "JobStatus.finished":
                        self.retrieve_gradient(event)
                        events_retrieved_now.append(event)
                    elif status == "JobStatus.pending":
                        print(f"Status = {status}, event: {event}")
                    elif status == "JobStatus.running":
                        print(f"Status = {status}, event: {event}")
                    elif status == "JobStatus.failed":
                        print(f"Status = {status}, event: {event}")
                        print("Job failed. Will be resubmitted")
                        if reposts[event] >=3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_adjoint_simulation(event)
                        reposts[event] += 1
                    elif status == "JobStatus.unknown":
                        print(f"Status = {status}, event: {event}")
                        print("Job unknown. Will be resubmitted")
                        if reposts[event] >=3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_adjoint_simulation(event)
                        reposts[event] += 1
                    elif status == "JobStatus.cancelled":
                        print(f"Status = {status}, event: {event}")
                        print("What to do here?")
                    else:
                        warnings.warn(
                            f"Inversionson does not recognise job status: " f"{status}",
                            InversionsonWarning,
                        )

            # Smoothing is done on a one job per parameter basis,
            # So the complexity of the monitoring increases.
            elif sim_type == "smoothing":
                params_retrieved_now = []
                params_already_retrieved = []
                if self.comm.project.meshes == "multi-mesh":
                    smoothing_job = self.comm.project.smoothing_job[event]
                else:
                    smoothing_job = self.comm.project.smoothing_job
                for param in self.comm.project.inversion_params:
                    if smoothing_job[param]["retrieved"]:
                        params_already_retrieved.append(param)
                if len(params_already_retrieved) == len(
                    self.comm.project.inversion_params
                ):
                    if self.comm.project.meshes == "multi-mesh":
                        events_already_retrieved.append(event)
                    else:
                        events_already_retrived.append("master")
                    continue
                for param in self.comm.project.inversion_params:
                    if smoothing_job[param]["retrieved"]:
                        continue
                    else:
                        status = str(
                            self.comm.salvus_flow.get_job_status(event, sim_type)
                        )  # This thing might time out

                        if status == "JobStatus.finished":
                            self.retrieve_gradient(event, smooth=True, par=param)
                            params_retrieved_now.append(param)
                            if self.comm.project.meshes = "multi-mesh":
                                self.comm.project.change_attribute(
                                    attribute=f'smoothing_job["{event}"]["{param}"]["retrieved"]',
                                    new_value=True,
                                )
                            else:
                                self.comm.project.change_attribute(
                                    attribute=f'smoothing_job["{param}"]["retrieved"]',
                                    new_value=True,
                                )
                        elif status == "JobStatus.pending":
                            print(f"Status = {status}, event: {event}, param: {param}")
                        elif status == "JobStatus.running":
                            print(f"Status = {status}, event: {event}, param: {param}")
                        elif status == "JobStatus.failed":
                            print(f"Status = {status}, event: {event}, param: {param}")
                            print("Job failed. Need to implement something here")
                            print("Probably resubmit or something like that")
                        elif status == "JobStatus.unknown":
                            print(f"Status = {status}, event: {event}, param: {param}")
                            print("Job unknown. Need to implement something here")
                        elif status == "JobStatus.cancelled":
                            print(f"Status = {status}, event: {event}, param: {param}")
                            print("What to do here?")
                        else:
                            warnings.warn(
                                f"Inversionson does not recognise job status: "
                                f"{status}",
                                InversionsonWarning,
                            )
                params_retrieved = params_retrieved_now + params_already_retrieved
                if len(params_retrieved) == len(self.comm.project.inversion_params):
                    events_retrieved_now.append(event)
            else:
                raise ValueError(f"Sim type {sim_type} not supported")

        # If no events have been retrieved, we call the function again.
        if len(events_retrieved_now) == 0:
            if len(events_already_retrieved) == len(events):
                return "All retrieved"
            if self.comm.project.meshes == "mono-mesh":
                if sim_type == "smoothing":
                    if "master" in events_already_retrieved:
                        return "All retrieved"
            else:
                if not sim_type == "smoothing":
                    print(
                        f"Recovered {len(events_already_retrieved)} out of "
                        f"{len(events)} events."
                    )
                else:
                    if self.comm.project.meshes == "mono-mesh":
                        print(f"Still waiting to recover the smoothing job")
                    else:
                        print(
                            f"Recovered {len(events_already_retrieved)} out of "
                            f"{len(events)} events."
                        )

                time.sleep(60)
                return self.monitor_jobs(sim_type, events=events,
                                         reposts=reposts)

        for event in events_retrieved_now:
            if sim_type == "smoothing":
                continue
            self.comm.project.change_attribute(
                attribute=f'{sim_type}_job["{event}"]["retrieved"]', new_value=True
            )
            self.comm.project.update_iteration_toml()
        return events_retrieved_now

    def wait_for_all_jobs_to_finish(self, sim_type: str, events=None):
        """
        Just a way to make the algorithm wait until all jobs are done.

        :param sim_type: Simulation type forward or adjoint
        :type sim_type: str
        :param events: list of events used in task, if nothing given,
            will take all events in iteration
        :type events: list
        """
        if sim_type == "forward":
            jobs = self.comm.project.forward_job
        elif sim_type == "adjoint":
            jobs = self.comm.project.adjoint_job
        if not events:
            events = self.comm.project.events_in_iteration
        done = np.zeros(len(events), dtype=bool)
        for _i, event in enumerate(events):
            if jobs[event]["retrieved"]:
                done[_i] = True
            else:
                status = str(self.comm.salvus_flow.get_job_status(event, sim_type))
                print(f"Status = {status}, event: {event}, sim_type: {sim_type}")
                if status == "JobStatus.finished":
                    if sim_type == "forward":
                        self.retrieve_seismograms(event)
                    elif sim_type == "adjoint":
                        self.retrieve_gradient(event)
                    else:
                        self.retrieve_gradient(event, smoothed=True)
                    jobs[event]["retrieved"] = True

        if sim_type == "forward":
            self.comm.project.change_attribute(attribute="forward_job", new_value=jobs)
        elif sim_type == "adjoint":
            self.comm.project.change_attribute(attribute="adjoint_job", new_value=jobs)
        self.comm.project.update_iteration_toml()

        if not np.all(done):  # If not all done, keep monitoring
            time.sleep(20)
            self.wait_for_all_jobs_to_finish(sim_type, events=events)

    def compute_misfit_on_validation_data(self):  # Not functional
        """
        We define a validation dataset and compute misfits on it for an average
        model of the past few iterations.
        Currently not sure whether we should recompute windows. As the
        increasing window size increases the misfit. But at the same time, we
        are fitting an increasingly large window sets on the inverted
        dataset so maybe that should also apply to the validation set.
        """
        print(Fore.YELLOW + "\n ================== \n")
        print("Computing misfit on validation dataset")

        events = self.comm.project.validation_dataset
        for event in events:
            if self.comm.project.meshes == "multi-mesh":
               print(Fore.CYAN + "\n ============================= \n")
               print(
                    emoji.emojize(
                        ":globe_with_meridians: :point_right: "
                        ":globe_with_meridians: | Interpolation Stage",
                        use_aliases=True,
                    )
                )
                print(f"{event} interpolation")

                self.interpolate_model(event) 
        # Figure out current iteration.
        # Look at how many iterations are between checking this.
        # Take that many iterations back
        # Average the model for those iterations
        # Maybe there is no need to average the models.
        # Now is maybe the time to submit a job array.
        # Job arrays might be the go to thing later on in the
        # inversion as the interpolations get really quick.
        # Enough about that
        # Do the interpolations
        # submit a job array.
        # wait for job to finish
        # compute misfit
        # write into documentation

    def compute_misfit_and_gradient(self, task: str, verbose: str):
        """
        A task associated with the initial iteration of FWI
        
        :param task: Task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print("Will prepare iteration")

        self.prepare_iteration(first=True)

        print(emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True))

        print(f"Current Iteration: {self.comm.project.current_iteration}")

        for event in self.comm.project.events_in_iteration:
            if self.comm.project.meshes == "multi-mesh":
                print(Fore.CYAN + "\n ============================= \n")
                print(
                    emoji.emojize(
                        ":globe_with_meridians: :point_right: "
                        ":globe_with_meridians: | Interpolation Stage",
                        use_aliases=True,
                    )
                )
                print(f"{event} interpolation")

                self.interpolate_model(event)

            print(Fore.YELLOW + "\n ============================ \n")
            print(emoji.emojize(":rocket: | Run forward simulation", use_aliases=True))

            self.run_forward_simulation(event)

            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(":trident: | Calculate station weights", use_aliases=True)
            )

            self.calculate_station_weights(event)

        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            i += 1
            # time.sleep(5)
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for forward jobs", use_aliases=True
                )
            )
            events_retrieved_now = self.monitor_jobs("forward")
            print(f"Events retrieved: {events_retrieved_now}")

            if len(events_retrieved_now) == 0:
                print(f"Events retrieved: {events_retrieved_now}")
                print("No new events retrieved, lets wait")
                continue
                # Should not happen
            if events_retrieved_now == "All retrieved":
                events_retrieved_now = self.comm.project.events_in_iteration
                events_retrieved = "All retrieved"
            for event in events_retrieved_now:
                print(f"{event} retrieved")
                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Process data if needed", use_aliases=True
                    )
                )

                self.process_data(event)

                print(Fore.WHITE + "\n ===================== \n")
                print(emoji.emojize(":foggy: | Select windows", use_aliases=True))

                self.select_windows(event)

                print(Fore.MAGENTA + "\n ==================== \n")
                print(emoji.emojize(":zap: | Quantify Misfit", use_aliases=True))

                self.misfit_quantification(event)

                print(Fore.YELLOW + "\n ==================== \n")
                print(
                    emoji.emojize(":rocket: | Run adjoint simulation", use_aliases=True)
                )
                # if "NEAR" in event:
                self.run_adjoint_simulation(event)
                # elif "REYKJANES" in event:
                # self.run_adjoint_simulation(event)

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        self.wait_for_all_jobs_to_finish("forward")
        self.comm.lasif.write_misfit()

        events_retrieved_adjoint = "None retrieved"
        while events_retrieved_adjoint != "All retrieved":
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for adjoint jobs", use_aliases=True
                )
            )
            # time.sleep(15)
            events_retrieved_adjoint_now = self.monitor_jobs("adjoint")
            # if events_retrieved_adjoint_now == "All retrieved" and i != 1:
            #     break
            # else:
            if len(events_retrieved_adjoint_now) == 0:
                print("No new events retrieved, lets wait")
                continue
                # Should not happen
            if events_retrieved_adjoint_now == "All retrieved":
                events_retrieved_adjoint_now = self.comm.project.events_in_iteration
                events_retrieved_adjoint = "All retrieved"
            for event in events_retrieved_adjoint_now:
                print(f"{event} retrieved")

                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Cut sources and " "receivers from gradient",
                        use_aliases=True,
                    )
                )
                self.preprocess_gradient(event)
                if self.comm.project.meshes == "multi-mesh":
                    print(Fore.YELLOW + "\n ==================== \n")
                    print(
                        emoji.emojize(":rocket: | Run Diffusion equation", use_aliases=True)
                    )
                    print(f"Event: {event} gradient will be smoothed")

                    self.smooth_gradient(event)
        if self.comm.project.meshes == "mono-mesh":
            print(Fore.GREEN + "\n ===================== \n")
            print(
                emoji.emojize(
                    ":floppy_disk: | Summing gradients",
                    use_aliases=True,
                )
                )

            self.sum_gradients()

            print(Fore.YELLOW + "\n ==================== \n")
            print(
                emoji.emojize(":rocket: | Run Diffusion equation", use_aliases=True)
            )
            print(f"Gradient will be smoothed")

            self.smooth_gradient(event=None)

        events_retrieved_smoothing = "None retrieved"
        while events_retrieved_smoothing != "All retrieved":

            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for smoothing jobs", use_aliases=True
                )
            )

            events_retrieved_smoothing_now = self.monitor_jobs("smoothing")
            if len(events_retrieved_smoothing_now) == 0:
                print("No new events retrieved, lets wait")
                continue
            if events_retrieved_smoothing_now == "All retrieved":
                events_retrieved_smoothing_now = self.comm.project.events_in_iteration
                events_retrieved_smoothing = "All retrieved"

            if self.comm.project.meshes == "multi-mesh":
                for event in events_retrieved_smoothing_now:
                    print(f"{event} retrieved")
                    print(Fore.CYAN + "\n ============================= \n")
                    print(
                        emoji.emojize(
                            ":globe_with_meridians: :point_right: "
                            ":globe_with_meridians: | Interpolation Stage",
                            use_aliases=True,
                        )
                    )
                    print(f"{event} interpolation")

                    self.interpolate_gradient(event)

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all adjoint jobs are " "done",
                use_aliases=True,
            )
        )
        self.wait_for_all_jobs_to_finish("smoothing")

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )

        self.comm.salvus_opt.write_misfit_and_gradient_to_task_toml()
        self.comm.project.update_iteration_toml()
        self.comm.storyteller.document_task(task)
        # Try to make ray-density plot work
        self.comm.salvus_opt.close_salvus_opt_task()
        # Bypass a Salvus Opt Error
        self.comm.salvus_opt.quickfix_delete_old_gradient_files()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                f":grinning: | Iteration "
                f"{self.comm.project.current_iteration} done",
                use_aliases=True,
            )
        )

        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        self.assign_task_to_function(task_2, verbose_2)

    def compute_misfit(self, task: str, verbose: str):
        """
        Compute misfit for a test model

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print("Will prepare iteration")
        self.prepare_iteration()

        print(emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True))

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        print(f"More specifically: {verbose}")

        if "compute misfit for" in verbose:
            events_to_use = self.comm.project.old_control_group
        else:
            events_to_use = list(
                set(self.comm.project.events_in_iteration)
                - set(self.comm.project.old_control_group)
            )
        for event in events_to_use:

            if self.comm.project.meshes == "multi-mesh":
                print(Fore.CYAN + "\n ============================= \n")
                print(
                    emoji.emojize(
                        ":globe_with_meridians: :point_right: "
                        ":globe_with_meridians: | Interpolation Stage",
                        use_aliases=True,
                    )
                )
                print(f"{event} interpolation")

                self.interpolate_model(event)

            print(Fore.YELLOW + "\n ============================ \n")
            print(emoji.emojize(":rocket: | Run forward simulation", use_aliases=True))

            self.run_forward_simulation(event)

            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(":trident: | Calculate station weights", use_aliases=True)
            )

            self.calculate_station_weights(event)

        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            print(Fore.BLUE)
            print(emoji.emojize(":hourglass: | Waiting for jobs", use_aliases=True))
            i += 1
            events_retrieved_now = self.monitor_jobs(
                sim_type="forward", events=events_to_use
            )

            if events_retrieved_now == "All retrieved":
                events_retrieved_now = events_to_use
                events_retrieved = "All retrieved"
            for event in events_retrieved_now:
                print(f"{event} retrieved")
                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Process data if " "needed", use_aliases=True
                    )
                )

                self.process_data(event)

                print(Fore.WHITE + "\n ===================== \n")
                print(emoji.emojize(":foggy: | Select windows", use_aliases=True))

                self.select_windows(event)

                print(Fore.MAGENTA + "\n ==================== \n")
                print(emoji.emojize(":zap: | Quantify Misfit", use_aliases=True))

                self.misfit_quantification(event)

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        self.wait_for_all_jobs_to_finish(sim_type="forward", events=events_to_use)
        self.comm.lasif.write_misfit(events=events_to_use, details=verbose)

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )
        self.comm.storyteller.document_task(task, verbose)
        if "compute additional" in verbose:
            events_to_use = None
        self.comm.salvus_opt.write_misfit_to_task_toml(events=events_to_use)
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        self.assign_task_to_function(task_2, verbose_2)

    def compute_gradient(self, task: str, verbose: str):
        """
        Compute gradient for accepted trial model

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        iteration = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.change_attribute(
            attribute="current_iteration", new_value=iteration
        )
        self.comm.project.get_iteration_attributes()

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")

        for event in self.comm.project.events_in_iteration:
            print(Fore.YELLOW + "\n ==================== \n")
            print(emoji.emojize(":rocket: | Run adjoint simulation", use_aliases=True))
            print(f"Event: {event} \n")
            self.run_adjoint_simulation(event)

        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            print(Fore.BLUE)
            print(emoji.emojize(":hourglass: | Waiting for jobs", use_aliases=True))
            i += 1
            events_retrieved_now = self.monitor_jobs(sim_type="adjoint")
            # if events_retrieved_now == "All retrieved" and i != 1:
            #     break
            # else:
            if len(events_retrieved_now) == 0:
                print("No new events retrieved, lets wait")
                continue
            if events_retrieved_now == "All retrieved":
                events_retrieved_now = self.comm.project.events_in_iteration
                events_retrieved = "All retrieved"
            for event in events_retrieved_now:
                print(f"{event} retrieved")

                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Cut sources and " "receivers from gradient",
                        use_aliases=True,
                    )
                )
                self.preprocess_gradient(event)

                print(Fore.YELLOW + "\n ==================== \n")
                print(
                    emoji.emojize(":rocket: | Run Diffusion equation", use_aliases=True)
                )
                print(f"Event: {event} gradient will be smoothed")
                if self.comm.project.meshes == "multi-mesh":
                    self.smooth_gradient(event)

                # TODO: Add something to check status of smoother here
                # to go to interpolating immediately
        if self.comm.project.meshes == "mono-mesh":
            self.sum_gradients()
            self.smooth_gradient(event=None)
        events_retrieved_smoothing = "None retrieved"
        while events_retrieved_smoothing != "All retrieved":

            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for smoothing jobs", use_aliases=True
                )
            )

            events_retrieved_smoothing_now = self.monitor_jobs("smoothing")
            if len(events_retrieved_smoothing_now) == 0:
                print("No new events retrieved, lets wait")
                continue
            if events_retrieved_smoothing_now == "All retrieved":
                events_retrieved_smoothing_now = self.comm.project.events_in_iteration
                events_retrieved_smoothing = "All retrieved"
            if self.comm.project.meshes == "multi-mesh":
                for event in events_retrieved_smoothing_now:
                    print(f"{event} retrieved")

                    print(Fore.CYAN + "\n ============================= \n")
                    print(
                        emoji.emojize(
                            ":globe_with_meridians: :point_right: "
                            ":globe_with_meridians: | Interpolation Stage",
                            use_aliases=True,
                        )
                    )
                    print(f"{event} interpolation")

                    self.interpolate_gradient(event)

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all jobs are done", use_aliases=True
            )
        )
        self.wait_for_all_jobs_to_finish("adjoint")

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )

        # Smooth gradients
        self.comm.salvus_opt.write_gradient_to_task_toml()
        self.comm.storyteller.document_task(task)
        # bypassing an opt bug
        self.comm.salvus_opt.quickfix_delete_old_gradient_files()
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        self.assign_task_to_function(task_2, verbose_2)

    def select_control_batch(self, task, verbose):
        """
        Select events that will carry on to the next iteration

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task} \n")
        print("Selection of Dynamic Mini-Batch Control Group:")
        print(f"More specifically: {verbose}")
        self.comm.project.get_iteration_attributes()
        if "increase control group size" in verbose:
            self.comm.minibatch.increase_control_group_size()
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=self.comm.project.new_control_group
            )
            self.comm.storyteller.document_task(task, verbose)

        else:
            self.comm.minibatch.print_dp()

            control_group = self.comm.minibatch.select_optimal_control_group()
            print(f"Selected Control group: {control_group}")
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=control_group
            )
            self.comm.project.change_attribute(
                attribute="new_control_group", new_value=control_group
            )
            self.comm.project.update_control_group_toml(new=True)
            self.comm.storyteller.document_task(task)
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        self.assign_task_to_function(task_2, verbose_2)

    def finalize_iteration(self, task, verbose):
        """
        Wrap up loose ends and get on to next iteration.

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        iteration = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.get_iteration_attributes()
        self.comm.storyteller.document_task(task)
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()
        # self.comm.salvus_flow.delete_stored_wavefields(iteration, "forward")
        # self.comm.salvus_flow.delete_stored_wavefields(iteration, "adjoint")
        self._send_whatsapp_announcement()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        self.assign_task_to_function(task_2, verbose_2)

    def assign_task_to_function(self, task: str, verbose: str):
        """
        Input a salvus opt task and send to correct function

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(f"\nNext salvus opt task is: {task}\n")
        print(f"More specifically: {verbose}")

        if task == "compute_misfit_and_gradient":
            self.compute_misfit_and_gradient(task, verbose)
        elif task == "compute_misfit":
            self.compute_misfit(task, verbose)
        elif task == "compute_gradient":
            self.compute_gradient(task, verbose)
        elif task == "select_control_batch":
            self.select_control_batch(task, verbose)
        elif task == "finalize_iteration":
            self.finalize_iteration(task, verbose)
        else:
            raise InversionsonError(f"Don't know task: {task}")

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

        self.assign_task_to_function(task, verbose)


def read_information_toml(info_toml_path: str):
    """
    Read a toml file with inversion information into a dictionary

    :param info_toml_path: Path to the toml file
    :type info_toml_path: str
    """
    return toml.load(info_toml_path)


if __name__ == "__main__":
    print(
        emoji.emojize(
            "\n :flag_for_Iceland: | Welcome to Inversionson | :flag_for_Iceland: \n",
            use_aliases=True,
        )
    )
    info_toml = input("Give me a path to your information_toml \n\n")
    # Tired of writing it in, I'll do this quick mix for now
    # print("Give me a path to your information_toml \n\n")
    # time.sleep(1)
    print("Just kidding, I know it")
    info_toml = "inversion_info.toml"
    if not info_toml.startswith("/"):
        cwd = os.getcwd()
        if info_toml.startswith("./"):
            info_toml = os.path.join(cwd, info_toml[2:])
        elif info_toml.startswith("."):
            info_toml = os.path.join(cwd, info_toml[1:])
        else:
            info_toml = os.path.join(cwd, info_toml)
    info = read_information_toml(info_toml)
    invert = AutoInverter(info)
