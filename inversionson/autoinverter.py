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
from typing import Union, List
from inversionson.remote_helpers.helpers import preprocess_remote_gradient
from salvus.flow import api

init()


def _find_project_comm(info):
    """
    Get Inversionson communicator.
    """
    from inversionson.components.project import ProjectComponent

    return ProjectComponent(info).get_communicator()


class AutoInverter(object):
    """
    A class which takes care of automating a Full-Waveform Inversion.

    It works for mono-batch or mini-batch, mono-mesh or multi-mesh
    or any combination of the two.
    It uses Salvus, Lasif and Multimesh to perform most of its actions.
    This is a class which wraps the three packages together to perform an
    automatic Full-Waveform Inversion
    """

    def __init__(self, info_dict: dict):
        self.info = info_dict
        print(Fore.RED + "Will make communicator now")
        self.comm = _find_project_comm(self.info)
        print(Fore.GREEN + "Now I want to start running the inversion")
        print(Style.RESET_ALL)
        self.task = None
        self.run_inversion()

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
        client.messages.create(
            body=string, from_=from_whatsapp, to=to_whatsapp
        )
        
    def jobs_on_remote_site(self, site_name: str, job_number: int) -> bool:
        """
        Check how many jobs are running or pending on the remote site
        
        :param site_name: Name of remote site
        :type site_name: str
        :param job_numer: Number of submitting jobs
        :type job_number: int
        """
        max_submitted_jobs_per_user = 30
        
        if job_number == 1:
            job_list = api.get_jobs(
                limit=100,
                site_name = site_name, 
                job_status = ["running", "pending"],
                update_jobs = True,
            )
            
            finished_job = 0
            for job in job_list:
                status = job.update_status(force_update=False)
                if status.name == "finished":
                    finished_job += 1
            remaining_job = len(job_list) - finished_job
                    
        elif job_number > 1:
            job_array_list = api.get_job_arrays(
                limit=100,
                site_name = site_name, 
                job_array_status = ["running", "pending"], 
                update_job_arrays = True,
            )
            
            remaining_job = 0
            for job_array in job_array_list:
                status = job_array.update_status()
                for job_status in status:
                    if job_status.name == "running" or job_status.name == "pending":
                        remaining_job += 1

        else:
            raise InversionsonError(f"Don't accept {job_number}")

        
        if max_submitted_jobs_per_user - remaining_job >= job_number:
            boolean = True
        else:
            boolean = False
                
        if not boolean:
            print(
                f"{remaining_job} jobs are on {site_name}: " 
                "Sleeping for a minute before checking again"
            )
            time.sleep(60)
            return self.jobs_on_remote_site(site_name, job_number)
        
        return boolean

    def prepare_iteration(self, first=False, validation=False):
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
        if validation:
            it_name = f"validation_{it_name}"
        first_try = self.comm.salvus_opt.first_trial_model_of_iteration()
        self.comm.project.change_attribute("current_iteration", it_name)
        it_toml = os.path.join(
            self.comm.project.paths["iteration_tomls"], it_name + ".toml"
        )
        if self.comm.lasif.has_iteration(it_name):
            if not os.path.exists(it_toml):
                self.comm.project.create_iteration_toml(it_name)
            self.comm.project.get_iteration_attributes(validation=validation)
            # If the iteration toml was just created but
            # not the iteration, we finish making the iteration
            # Should never happen though
            if len(self.comm.project.events_in_iteration) != 0:
                if self.comm.project.meshes == "multi-mesh":
                    for event in self.comm.project.events_in_iteration:
                        if not self.comm.lasif.has_mesh(event):
                            self.comm.salvus_mesher.create_mesh(event=event,)
                            self.comm.salvus_mesher.add_region_of_interest(
                                event=event
                            )
                            self.comm.lasif.move_mesh(event, it_name)
                        else:
                            self.comm.lasif.move_mesh(event, it_name)
                else:
                    self.comm.lasif.move_mesh(event=None, iteration=it_name)
                return
        if first_try and not validation:
            if self.comm.project.inversion_mode == "mini-batch":
                print("Getting minibatch")
                events = self.comm.lasif.get_minibatch(first)
            else:
                events = self.comm.lasif.list_events()
        elif validation:
            events = self.comm.project.validation_dataset
        else:
            prev_try = self.comm.salvus_opt.get_previous_iteration_name(
                tr_region=True
            )
            events = self.comm.lasif.list_events(iteration=prev_try)
        self.comm.project.change_attribute("current_iteration", it_name)
        self.comm.lasif.set_up_iteration(it_name, events)
        if self.comm.project.meshes == "multi-mesh":
            for event in events:
                if not self.comm.lasif.has_mesh(event):
                    self.comm.salvus_mesher.create_mesh(event=event)
                    self.comm.lasif.move_mesh(event, it_name)
                else:
                    self.comm.lasif.move_mesh(event, it_name)
        else:
            self.comm.lasif.move_mesh(event=None, iteration=it_name)

        if not validation and self.comm.project.inversion_mode == "mini-batch":
            self.comm.project.update_control_group_toml(first=first)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes(validation)

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
            self.comm.project.inversion_root, "INTERPOLATION", event, "model"
        )
        if not os.path.exists(interp_folder):
            os.makedirs(interp_folder)
        self.comm.multi_mesh.interpolate_to_simulation_mesh(
            event, interp_folder=interp_folder
        )
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
            "gradient",
        )
        if not os.path.exists(interp_folder):
            os.makedirs(interp_folder)
        self.comm.multi_mesh.interpolate_gradient_to_model(
            event, smooth=True, interp_folder=interp_folder
        )
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
                print(
                    "If you want it redone, change its status in iteration toml"
                )
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
        w = self.comm.salvus_flow.construct_simulation(
            event, source, receivers
        )

        if (
            self.comm.project.remote_mesh is not None
            and self.comm.project.meshes == "mono-mesh"
        ):
            w.set_mesh(self.comm.project.remote_mesh)
            
        boolean = self.jobs_on_remote_site(self.comm.project.site_name, 1)
        if boolean:
            print(f"Submit forward simulation to {self.comm.project.site_name}")
            pass
        
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
                print(
                    "If you want it redone, change its status in iteration toml"
                )
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
                    if self.comm.project.remote_gradient_processing:
                        job = self.comm.salvus_flow.get_job(event, "adjoint")
                        output_files = job.get_output_files()
                        grad = output_files[0][
                            ("adjoint", "gradient", "output_filename")
                        ]
                        print("calling preprocess")
                        preprocess_remote_gradient(self.comm, grad, event)
                    else:
                        self.retrieve_gradient(event)
                    self.comm.project.change_attribute(
                        attribute=f'adjoint_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
                    self.comm.project.update_iteration_toml()
                    return
                else:
                    print("Jobstatus unknown for event {event}")
                    print("Will resubmit")
        adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
        w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
            event, adj_src
        )

        if (
            self.comm.project.remote_mesh is not None
            and self.comm.project.meshes == "mono-mesh"
        ):
            w_adjoint.set_mesh(self.comm.project.remote_mesh)
            
        boolean = self.jobs_on_remote_site(self.comm.project.site_name, 1)
        if boolean:
            print(f"Submit adjoint simulation to {self.comm.project.site_name}")
            pass 
        
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

    def misfit_quantification(
        self, event: str, validation=False, window_set=None
    ):
        """
        Compute misfit and adjoint source for iteration

        :param event: Name of event
        :type event: str
        """
        mpi = False
        if self.comm.project.site_name == "swp":
            mpi = False
        misfit = self.comm.lasif.misfit_quantification(
            event, mpi=mpi, validation=validation, window_set=window_set
        )
        if not validation:
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
        self.comm.salvus_flow.retrieve_outputs(
            event_name=event, sim_type="forward"
        )
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

    def retrieve_gradient(self, event: str, smooth=False):
        """
        Move gradient from salvus_flow folder to lasif folder
        
        :param event: Name of event
        :type event: str
        :param smooth: Am I returning gradient from smoothing? Default: False
        :type smooth: bool
        """
        if event is not None and not smooth:
            if self.comm.project.adjoint_job[event]["retrieved"]:
                print(
                    f"Gradient for event {event} has already been retrieved. "
                    f"Will not retrieve it again."
                )
                return
        if smooth:
            sim_type = "smoothing"
            self.comm.smoother.retrieve_smooth_gradient(event_name=event,)
        else:
            sim_type = "adjoint"
            self.comm.salvus_flow.retrieve_outputs(
                event_name=event, sim_type=sim_type
            )
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

        mpi = False
        if self.comm.project.site_name == "swp":
            mpi = False
        if self.comm.project.inversion_mode == "mono-batch":
            if self.task == "compute_misfit_and_gradient":
                self.comm.lasif.select_windows(
                    window_set_name=window_set_name, event=event, mpi=mpi
                )
                return
            else:
                print("Windows were selected in a previous iteration.")
                print(" ... On we go")
                return

        if "validation_" in iteration:
            window_set_name = iteration
            if self.comm.project.forward_job[event]["windows_selected"]:
                print(f"Windows already selected for event {event}")
                return
            self.comm.lasif.select_windows(
                window_set_name=window_set_name,
                event=event,
                mpi=mpi,
                validation=True,
            )
            self.comm.project.change_attribute(
                attribute=f"forward_job['{event}']['windows_selected']",
                new_value=True,
            )
            self.comm.project.update_iteration_toml(validation=True)
            return
        # If event is in control group, we look for newest window set for event
        if (
            iteration != "it0000_model"
            and event in self.comm.project.old_control_group
        ):
            import glob

            windows = self.comm.lasif.lasif_comm.project.paths["windows"]
            window_sets = glob.glob(os.path.join(windows, "*" + event + "*"))
            latest_windows = max(window_sets, key=os.path.getctime)
            if not os.path.exists(
                os.path.join(windows, window_set_name + ".sqlite")
            ):
                shutil.copy(
                    latest_windows,
                    os.path.join(windows, window_set_name + ".sqlite"),
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
        if self.comm.project.inversion_mode == "mini-batch":
            job_submitted = self.comm.project.smoothing_job[event]["submitted"]
        elif self.comm.project.inversion_mode == "mono-batch":
            job_submitted = self.comm.project.smoothing_job["submitted"]
        if job_submitted:
            print(
                f"Already Submitted job for gradient smoothing "
                f" so gradient should be preprocessed for event {event}"
            )
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
            clip_gradient(
                mesh=gradient,
                percentile=self.comm.project.clip_gradient,
                parameters=self.comm.project.inversion_params,
            )

    def sum_gradients(self):
        """
        Sum the computed gradients onto one mesh
        """
        from inversionson.utils import sum_gradients

        events = self.comm.project.events_in_iteration
        grad_mesh = self.comm.lasif.find_gradient(
            iteration=self.comm.project.current_iteration,
            event=None,
            summed=True,
            smooth=False,
            just_give_path=True,
        )
        if os.path.exists(grad_mesh):
            print("Gradient has already been summed. Moving on...")
            return
        gradients = []
        for event in events:
            gradients.append(
                self.comm.lasif.find_gradient(
                    iteration=self.comm.project.current_iteration,
                    event=event,
                    summed=False,
                    smooth=False,
                    just_give_path=False,
                )
            )
        shutil.copy(gradients[0], grad_mesh)
        sum_gradients(mesh=grad_mesh, gradients=gradients)

    def smooth_gradient(self, event: str):
        """
        Send a gradient for an event to the Salvus smoother

        :param event: name of event
        :type event: str
        """
        if self.comm.project.inversion_mode == "mini-batch":
            job = self.comm.project.smoothing_job[event]
            message = "Summed gradient already smoothed."
        elif self.comm.project.inversion_mode == "mono-batch":
            job = self.comm.project.smoothing_job
            message = f"Gradient for event {event} already smoothed."

        if job["retrieved"]:
            print(
                message
                + f" Will not repeat. Change its status in iteration toml "
                f"if you want to smooth gradient again"
            )
            return
        need_resubmit = False
        if job["submitted"]:
            if job["retrieved"]:
                if self.comm.project.inversion_mode == "mini-batch":
                    print(f"Simulation for event {event} already done.")
                else:
                    print("Smoothing simulation already done.")
                print(
                    "If you want it redone, change its status in iteration toml"
                )
            else:
                status = self.comm.salvus_flow.get_job_status(
                    event, "smoothing"
                )
                # Make sure this works
                if all(x.name == "finished" for x in status):
                    print(
                        f"All parameters have been smoothed for event {event}."
                        " Will retrieve gradient."
                    )
                    self.retrieve_gradient(event, smooth=True)
                    if self.comm.project.inversion_mode == "mini-batch":
                        self.comm.project.change_attribute(
                            attribute=f'smoothing_job["{event}"]["retrieved"]',
                            new_value=True,
                        )
                    else:
                        self.comm.project.change_attribute(
                            attribute=f'smoothing_job["retrieved"]',
                            new_value=True,
                        )
                    self.comm.project.update_iteration_toml()
                    return
                # If they are not all finished we check to see what's going on
                params = []
                need_resubmit = False
                for _i, s in enumerate(status):
                    if s.name == "finished":
                        params.append(s)
                    else:
                        print(
                            f"Status = {s.name}, event: {event} "
                            f"for smoothing job {_i}/{len(status)}"
                        )
                        if s.name == "pending" or s.name == "running":
                            continue
                        else:
                            print("Job failed. Will resubmit")
                            need_resubmit = True

                if len(params) == len(status):
                    print(
                        f"All parameters for event {event} have "
                        "now been smoothed"
                    )
                    self.retrieve_gradient(event, smooth=True)
                    self.comm.project.change_attribute(
                        attribute=f'smoothing_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
                    return

                if not need_resubmit:
                    print(
                        "Some smoothing events have not been finished"
                        "but they are still pending so we wait."
                    )
                    return

        if need_resubmit:
            job_array = self.comm.salvus_flow.get_job(
                event=event, sim_type="smoothing", iteration="current"
            )
            try:
                job_array.cancel()
            except:
                pass

        if self.comm.project.remote_gradient_processing:
            self.comm.smoother.run_remote_smoother(event=event)
        else:
            boolean = self.jobs_on_remote_site(
                self.comm.project.smoothing_site_name, 
                len(self.comm.project.inversion_params),
            )
            if boolean:
                print(f"Submit smooth simulation to {self.comm.project.site_name}")
                pass
            
            smoothing_config = self.comm.smoother.\
                generate_smoothing_config(event=event
            )
            self.comm.smoother.run_smoother(
                smoothing_config=smoothing_config, event=event
            )

        self.comm.project.update_iteration_toml()

    def monitor_jobs(self, sim_type: str, events=None):
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

        events_retrieved_now = []
        events_already_retrieved = []
        if not events:
            events = self.comm.project.events_in_iteration
        for event in events:
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    # TODO: Improve this
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type
                    ).name  # This thing might time out
                    # print(f"Status = {status}")
                    reposts = self.comm.project.forward_job[event]["reposts"]
                    if status == "finished":
                        self.retrieve_seismograms(event)
                        events_retrieved_now.append(event)
                        print(f"Events retrieved now: {events_retrieved_now}")
                    elif status == "pending":
                        print(f"Status = {status}, event: {event}")
                    elif status == "running":
                        print(f"Status = {status}, event: {event}")
                    elif status == "failed":
                        print(f"Status = {status}, event: {event}")
                        print("Job failed. Will resubmit")
                        if reposts >= 3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_forward_simulation(event)
                        reposts += 1
                    elif status == "unknown":
                        print(f"Status = {status}, event: {event}")
                        print("Job unknown. Will resubmit")
                        if reposts >= 3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_forward_simulation(event)
                        reposts += 1
                    elif status == "cancelled":
                        print("What to do here?")
                    else:
                        warnings.warn(
                            f"Inversionson does not recognise job status: "
                            f"{status}",
                            InversionsonWarning,
                        )

            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["retrieved"]:
                    events_already_retrieved.append(event)
                    continue
                else:
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type
                    ).name
                    reposts = self.comm.project.adjoint_job[event]["reposts"]

                    if status == "finished":
                        # Potentially add preprocess_remote here
                        if self.comm.project.remote_gradient_processing:
                            job = self.comm.salvus_flow.get_job(
                                event, "adjoint"
                            )
                            output_files = job.get_output_files()
                            grad = output_files[0][
                                ("adjoint", "gradient", "output_filename")
                            ]
                            preprocess_remote_gradient(self.comm, grad, event)
                        # retrieve job, then path. write toml and call process
                        else:
                            self.retrieve_gradient(event)
                        events_retrieved_now.append(event)
                    elif status == "pending":
                        print(f"Status = {status}, event: {event}")
                    elif status == "running":
                        print(f"Status = {status}, event: {event}")
                    elif status == "failed":
                        print(f"Status = {status}, event: {event}")
                        print("Job failed. Will be resubmitted")
                        if reposts >= 3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_adjoint_simulation(event)
                        reposts += 1
                    elif status == "unknown":
                        print(f"Status = {status}, event: {event}")
                        print("Job unknown. Will be resubmitted")
                        if reposts >= 3:
                            print("No, I've actually reposted this too often")
                            print("Something must be wrong")
                            raise InversionsonError("Too many reposts")
                        self.run_adjoint_simulation(event)
                        reposts += 1
                    elif status == "cancelled":
                        print(f"Status = {status}, event: {event}")
                        print("What to do here?")
                    else:
                        warnings.warn(
                            f"Inversionson does not recognise job status: "
                            f"{status}",
                            InversionsonWarning,
                        )

            else:
                raise ValueError(f"Sim type {sim_type} not supported")

            self.comm.project.change_attribute(
                attribute=f'{sim_type}_job["{event}"]["reposts"]',
                new_value=reposts,
            )

        # If no events have been retrieved, we call the function again.
        if len(events_retrieved_now) == 0:
            if len(events_already_retrieved) == len(events):
                return "All retrieved"
            if self.comm.project.inversion_mode == "mono-batch":
                if sim_type == "smoothing":
                    if "master" in events_already_retrieved:
                        return "All retrieved"
                    return self.monitor_jobs(sim_type, events=events,)
            else:
                print(
                    f"Recovered {len(events_already_retrieved)} out of "
                    f"{len(events)} events."
                )

                time.sleep(60)
                return self.monitor_jobs(sim_type, events=events,)

        for event in events_retrieved_now:
            self.comm.project.change_attribute(
                attribute=f'{sim_type}_job["{event}"]["retrieved"]',
                new_value=True,
            )
        self.comm.project.update_iteration_toml()
        return events_retrieved_now

    def monitor_job_arrays(
        self, sim_type: str, events: list = None,
    ) -> Union[List[str], str]:
        """
        Monitor the progress of a SalvusJobArray. It only returns when all the
        jobs in the array are done. Currently this is designed for a smoothing
        job where the function only returns once all the parameters are
        smoothed.
        In the future this will be more developed for mono-batch mono-mesh
        inversions where it makes sense to use job arrays, especially for
        the adjoint simulations. Note that this does not currently work though.
        
        :param sim_type: forward, adjoint, smoothing
        :type sim_type: str
        :param events: List of events, if none, all events in iteration used, 
            defaults to None
        :type events: list, optional
        :param reposts: How often each array has been reposted, defaults to {}
        :type reposts: dict, optional
        :return: Either returns a list of finished events or 'All finished'
        :rtype: Union[List[str], str]
        """
        import time

        events_retrieved_now = []
        events_already_retrieved = []
        if not events:
            events = self.comm.project.events_in_iteration

        if sim_type == "smoothing":
            if self.comm.project.inversion_mode == "mono-batch":
                smoothing_job = self.comm.project.smoothing_job
                reposts = smoothing_job["reposts"]
                if smoothing_job["retrieved"]:
                    return "All retrieved"
                else:
                    status = self.comm.salvus_flow.get_job_status(
                        event=None, sim_type=sim_type
                    )
                    params = []
                    for _i, s in enumerate(status):
                        if s.name == "finished":
                            params.append(s.name)
                        else:
                            print(
                                f"Status = {s.name} for smoothing job "
                                f"{_i+1}/{len(status)}"
                            )
                            if s.name == "pending" or s.name == "running":
                                continue
                            else:
                                print("Job failed. Will resubmit")
                                if reposts >= 3:
                                    print(
                                        "No, I've actually reposted "
                                        "this too often Something must "
                                        "be wrong"
                                    )

                                    raise InversionsonError("Too many reposts")
                                # TODO: Implement cancelling of the old job array
                                self.smooth_gradient(event=None)
                                reposts += 1
                                self.comm.project.change_attribute(
                                    attribute=f'smoothing_job["reposts"]',
                                    new_value=reposts,
                                )
                                print("I'll wait for a minute and check again")
                                time.sleep(60)
                                return self.monitor_job_arrays(
                                    sim_type=sim_type, events=events,
                                )

                        print(
                            f"{len(params)} out of {len(status)} have "
                            f"been smoothed"
                        )
                    if len(params) == len(status):
                        print(f"All parameters have now been smoothed")
                        self.retrieve_gradient(event=None, smooth=True)
                        self.comm.project.change_attribute(
                            attribute=f'smoothing_job["retrieved"]',
                            new_value=True,
                        )
                        return "All retrieved"

            else:
                for event in events:

                    smoothing_job = self.comm.project.smoothing_job[event]
                    reposts = smoothing_job["reposts"]
                    if smoothing_job["retrieved"]:
                        events_already_retrieved.append(event)
                        continue
                    else:
                        status = self.comm.salvus_flow.get_job_status(
                            event, sim_type
                        )
                        params = []
                        for _i, s in enumerate(status):
                            if s.name == "finished":
                                params.append(s.name)
                            else:
                                print(
                                    f"Status = {s.name}, event: {event} "
                                    f"for smoothing job {_i + 1}/{len(status)}"
                                )
                                if s.name == "pending" or s.name == "running":
                                    continue
                                else:
                                    print("Job failed. Will resubmit")
                                    # TODO: Print reposts to file so it doesn't zero out in every run.
                                    if reposts >= 3:
                                        print(
                                            "No, I've actually reposted "
                                            "this too often Something must "
                                            "be wrong"
                                        )

                                        raise InversionsonError(
                                            f"Too many reposts for smoothing "
                                            f"event {event}"
                                        )
                                    self.smooth_gradient(event)
                                    reposts += 1
                            print(
                                f"{len(params)} out of {len(status)} have "
                                f"been smoothed for event: {event}"
                            )
                        if len(params) == len(status):
                            print(
                                f"All parameters for event {event} have "
                                "now been smoothed"
                            )
                            self.retrieve_gradient(event, smooth=True)
                            self.comm.project.change_attribute(
                                attribute=f'smoothing_job["{event}"]["retrieved"]',
                                new_value=True,
                            )
                            events_retrieved_now.append(event)
                    self.comm.project.change_attribute(
                        attribute=f'smoothing_job["{event}"]["reposts"]',
                        new_value=reposts,
                    )

                if len(events_already_retrieved) == len(events):
                    self.comm.project.update_iteration_toml()
                    return "All retrieved"

        if len(events_retrieved_now) == 0:
            if len(events_already_retrieved) == len(events):
                return "All retrieved"
            print("Sleeping for a minute before checking again")
            time.sleep(60)
            return self.monitor_job_arrays(sim_type=sim_type, events=events)
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
        elif sim_type == "smoothing":
            jobs = self.comm.project.smoothing_job
        else:
            raise InversionsonError(f"We do not recognise sim type {sim_type}")
        if not events:
            events = self.comm.project.events_in_iteration

        if (
            self.comm.project.inversion_mode == "mono-batch"
            and sim_type == "smoothing"
        ):
            if jobs["retrieved"]:
                return
            else:
                status = self.comm.salvus_flow.get_job_status(events, sim_type)
                for _i, s in enumerate(status):
                    print(
                        f"Status = {s.name}, "
                        f"smoothing parameter {_i+1}/{len(status)}"
                    )
                if all(x.name == "finished" for x in status):
                    self.retrieve_gradient(event=None, smooth=True)
                    jobs["retrieved"] = True
        else:
            done = np.zeros(len(events), dtype=bool)
            for _i, event in enumerate(events):
                if jobs[event]["retrieved"]:
                    done[_i] = True
                else:
                    status = self.comm.salvus_flow.get_job_status(
                        event, sim_type
                    )
                    if not sim_type == "smoothing":
                        print(
                            f"Status = {status.name}, event: {event}, sim_type: {sim_type}"
                        )
                        if status.name == "finished":
                            if sim_type == "forward":
                                self.retrieve_seismograms(event)
                            elif sim_type == "adjoint":
                                if (
                                    self.comm.project.remote_gradient_processing
                                ):
                                    job = self.comm.salvus_flow.get_job(
                                        event, "adjoint"
                                    )
                                    output_files = job.get_output_files()
                                    grad = output_files[0][
                                        (
                                            "adjoint",
                                            "gradient",
                                            "output_filename",
                                        )
                                    ]
                                    preprocess_remote_gradient(
                                        self.comm, grad, event
                                    )
                                else:
                                    self.retrieve_gradient(event)
                            jobs[event]["retrieved"] = True
                    else:
                        for _i, s in enumerate(status):
                            print(
                                f"Status = {s.name}, event: {event}, "
                                f"smoothing parameter {_i+1}/{len(status)}"
                            )
                        if all(x.name == "finished" for x in status):
                            self.retrieve_gradient(event, smooth=True)
                            jobs[event]["retrieved"] = True

        if sim_type == "forward":
            self.comm.project.change_attribute(
                attribute="forward_job", new_value=jobs
            )
        elif sim_type == "adjoint":
            self.comm.project.change_attribute(
                attribute="adjoint_job", new_value=jobs
            )
        else:
            self.comm.project.change_attribute(
                attribute="smoothing_job", new_value=jobs,
            )
        self.comm.project.update_iteration_toml()

        if not np.all(done):  # If not all done, keep monitoring
            time.sleep(20)
            self.wait_for_all_jobs_to_finish(sim_type, events=events)

    def need_misfit_quantification(
        self, iteration: str, event: str, window_set: str
    ) -> bool:
        """
        Check whether validation misfit needs to be computed or not
        
        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event
        :type event: str
        :param window_set: Name of window set
        :type window_set: str
        """
        validation = self.comm.storyteller.validation_dict

        quantify_misfit = True
        if iteration in validation.keys():
            if event in validation[iteration]["events"].keys():
                if window_set in validation[iteration]["events"][event].keys():
                    if (
                        validation[iteration]["events"][event][window_set]
                        != 0.0
                    ):
                        quantify_misfit = False

        if not quantify_misfit:
            message = (
                f"Will not quantify misfit for event {event}, "
                f"iteration {iteration} "
                f"window set {window_set}. If you want it computed, "
                f"change value in validation toml to 0.0"
            )
            print(message)

        return quantify_misfit

    def compute_misfit_on_validation_data(self):
        """
        We define a validation dataset and compute misfits on it for an average
        model of the past few iterations. *Currently we do not average*
        We will both compute the misfit on the initial window set and two
        newer sets.

        Probably a good idea to only run this after a model has been accepted
        and on the first one of course.
        """
        if self.comm.project.when_to_validate == 0:
            return
        run_function = False

        iteration_number = (
            self.comm.salvus_opt.get_number_of_newest_iteration()
        )
        # We execute the validation check if iteration is either:
        # a) The initial iteration or
        # b) When #Iteration + 1 mod when_to_validate = 0
        if self.comm.project.current_iteration == "it0000_model":
            run_function = True
        if (iteration_number + 1) % self.comm.project.when_to_validate == 0:
            run_function = True
        # I could always call this at the beginning of iterations and
        # this function figures our whether it is the correct thing to do.

        if not run_function:
            return
        print(Fore.GREEN + "\n ================== \n")
        print(
            emoji.emojize(
                ":white_check_mark: | Computing misfit on validation dataset",
                use_aliases=True,
            )
        )
        print("\n\n")
        # Prepare validation iteration.
        # Simple enough, just create an iteration with the validation events
        print("Preparing iteration for validation set")
        self.prepare_iteration(validation=True)

        # Prepare simulations and submit them
        # I need something to monitor them.
        # There are definitely complications there
        print(Fore.YELLOW + "\n ============================ \n")
        print(
            emoji.emojize(
                ":rocket: | Run forward simulations", use_aliases=True
            )
        )
        if (
            self.comm.project.when_to_validate > 1
            and "it0000_model" not in self.comm.project.current_iteration
        ):
            # Find iteration range
            to_it = iteration_number
            from_it = iteration_number - self.comm.project.when_to_validate + 1
            self.comm.salvus_mesher.get_average_model(
                iteration_range=(from_it, to_it)
            )
        for event in self.comm.project.validation_dataset:
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

            print(f"Preparing validation simulation for event {event}")
            self.run_forward_simulation(event=event)

            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(
                    ":trident: | Calculate station weights", use_aliases=True
                )
            )

            self.calculate_station_weights(event)

        # Retrieve waveforms
        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for jobs", use_aliases=True
                )
            )
            i += 1
            events_retrieved_now = self.monitor_jobs(
                sim_type="forward", events=self.comm.project.validation_dataset
            )

            if events_retrieved_now == "All retrieved":
                events_retrieved_now = self.comm.project.validation_dataset
                events_retrieved = "All retrieved"
            for event in events_retrieved_now:
                print(f"{event} retrieved")
                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Process data if " "needed",
                        use_aliases=True,
                    )
                )

                self.process_data(event)

                print(Fore.WHITE + "\n ===================== \n")
                print(
                    emoji.emojize(":foggy: | Select windows", use_aliases=True)
                )

                self.select_windows(event)

                print(Fore.MAGENTA + "\n ==================== \n")
                print(
                    emoji.emojize(":zap: | Quantify Misfit", use_aliases=True)
                )
                validation_iterations = (
                    self.comm.lasif.get_validation_iteration_numbers()
                )
                iteration = self.comm.project.current_iteration
                if self.need_misfit_quantification(
                    iteration=iteration, event=event, window_set=iteration
                ):

                    self.misfit_quantification(
                        event, validation=True, window_set=iteration,
                    )
                    # Use storyteller to report recently computed misfit
                    self.comm.storyteller.report_validation_misfit(
                        iteration=iteration,
                        window_set=iteration,
                        event=event,
                        total_sum=False,
                    )

                if len(validation_iterations.keys()) > 1:
                    # We have at least two window sets to compute misfit on
                    last_number = (
                        max(validation_iterations.keys())
                        - self.comm.project.when_to_validate
                    )
                    last_iteration = validation_iterations[last_number]
                    # Figure out what the last validation iteration was.
                    if self.need_misfit_quantification(
                        iteration=iteration,
                        event=event,
                        window_set=last_iteration,
                    ):
                        self.misfit_quantification(
                            event, validation=True, window_set=last_iteration,
                        )
                        self.comm.storyteller.report_validation_misfit(
                            iteration=iteration,
                            window_set=last_iteration,
                            event=event,
                            total_sum=False,
                        )

                    if last_number != -1:
                        # We have three window sets to compute misfits on
                        if self.need_misfit_quantification(
                            iteration=iteration,
                            event=event,
                            window_set=validation_iterations[-1],
                        ):
                            self.misfit_quantification(
                                event,
                                validation=True,
                                window_set=validation_iterations[-1],
                            )
                            self.comm.storyteller.report_validation_misfit(
                                iteration=self.comm.project.current_iteration,
                                window_set=validation_iterations[-1],
                                event=event,
                                total_sum=False,
                            )

                self.comm.storyteller.report_validation_misfit(
                    iteration=self.comm.project.current_iteration,
                    window_set=self.comm.project.current_iteration,
                    event=event,
                    total_sum=True,
                )
        # leave the validation iteration.
        iteration = iteration[11:]
        self.comm.project.change_attribute(
            attribute="current_iteration", new_value=iteration,
        )
        self.comm.project.get_iteration_attributes()

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

        print(
            emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True)
        )

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
            print(
                emoji.emojize(
                    ":rocket: | Run forward simulation", use_aliases=True
                )
            )

            print(f"Event: {event} \n")         
            self.run_forward_simulation(event)

            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(
                    ":trident: | Calculate station weights", use_aliases=True
                )
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
                        ":floppy_disk: | Process data if needed",
                        use_aliases=True,
                    )
                )

                self.process_data(event)

                print(Fore.WHITE + "\n ===================== \n")
                print(
                    emoji.emojize(":foggy: | Select windows", use_aliases=True)
                )

                self.select_windows(event)

                print(Fore.MAGENTA + "\n ==================== \n")
                print(
                    emoji.emojize(":zap: | Quantify Misfit", use_aliases=True)
                )

                self.misfit_quantification(event)

                print(Fore.YELLOW + "\n ==================== \n")
                print(
                    emoji.emojize(
                        ":rocket: | Run adjoint simulation", use_aliases=True
                    )
                )
                
                print(f"Event: {event} \n")
                self.run_adjoint_simulation(event)

        self.compute_misfit_on_validation_data()

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        self.wait_for_all_jobs_to_finish("forward")

        events_retrieved_adjoint = "None retrieved"
        while events_retrieved_adjoint != "All retrieved":
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for adjoint jobs", use_aliases=True
                )
            )
            # time.sleep(15)
            # TODO: Do a check to see if I can send any events in for smoothing first.
            events_retrieved_adjoint_now = self.monitor_jobs("adjoint")
            # if events_retrieved_adjoint_now == "All retrieved" and i != 1:
            #     break
            # else:
            if len(events_retrieved_adjoint_now) == 0:
                print("No new events retrieved, lets wait")
                continue
                # Should not happen
            if events_retrieved_adjoint_now == "All retrieved":
                events_retrieved_adjoint_now = (
                    self.comm.project.events_in_iteration
                )
                events_retrieved_adjoint = "All retrieved"
            for event in events_retrieved_adjoint_now:
                print(f"{event} retrieved")

                print(Fore.GREEN + "\n ===================== \n")
                print(
                    emoji.emojize(
                        ":floppy_disk: | Cut sources and "
                        "receivers from gradient",
                        use_aliases=True,
                    )
                )
                if not self.comm.project.remote_gradient_processing:
                    self.preprocess_gradient(event)
                if self.comm.project.inversion_mode == "mini-batch":
                    print(Fore.YELLOW + "\n ==================== \n")
                    print(
                        emoji.emojize(
                            ":rocket: | Run Diffusion equation",
                            use_aliases=True,
                        )
                    )
                    print(f"Event: {event} gradient will be smoothed")

                    self.smooth_gradient(event)

        if self.comm.project.inversion_mode == "mono-batch":
            print(Fore.GREEN + "\n ===================== \n")
            print(
                emoji.emojize(
                    ":floppy_disk: | Summing gradients", use_aliases=True,
                )
            )

            self.sum_gradients()

            print(Fore.YELLOW + "\n ==================== \n")
            print(
                emoji.emojize(
                    ":rocket: | Run Diffusion equation", use_aliases=True
                )
            )
            print(f"Gradient will be smoothed")

            self.smooth_gradient(event=None)

        events_retrieved_smoothing = "None retrieved"
        while events_retrieved_smoothing != "All retrieved":

            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for smoothing jobs",
                    use_aliases=True,
                )
            )

            events_retrieved_smoothing_now = self.monitor_job_arrays(
                "smoothing"
            )
            if len(events_retrieved_smoothing_now) == 0:
                print("No new events retrieved, lets wait")
                continue
            if events_retrieved_smoothing_now == "All retrieved":
                events_retrieved_smoothing_now = (
                    self.comm.project.events_in_iteration
                )
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
        return task_2, verbose_2

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

        print(
            emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True)
        )

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        print(f"More specifically: {verbose}")

        if (
            "compute misfit for" in verbose
            and self.comm.project.inversion_mode == "mini-batch"
        ):
            events_to_use = self.comm.project.old_control_group
        elif self.comm.project.inversion_mode == "mini-batch":
            # If model is accepted we consider looking into validation data.
            self.compute_misfit_on_validation_data()
            events_to_use = list(
                set(self.comm.project.events_in_iteration)
                - set(self.comm.project.old_control_group)
            )
        else:
            events_to_use = self.comm.project.events_in_iteration
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
            print(
                emoji.emojize(
                    ":rocket: | Run forward simulation", use_aliases=True
                )
            )
            
            print(f"Event: {event} \n")
            self.run_forward_simulation(event)

            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(
                    ":trident: | Calculate station weights", use_aliases=True
                )
            )

            self.calculate_station_weights(event)

        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for jobs", use_aliases=True
                )
            )
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
                        ":floppy_disk: | Process data if " "needed",
                        use_aliases=True,
                    )
                )

                self.process_data(event)

                print(Fore.WHITE + "\n ===================== \n")
                print(
                    emoji.emojize(":foggy: | Select windows", use_aliases=True)
                )

                self.select_windows(event)

                print(Fore.MAGENTA + "\n ==================== \n")
                print(
                    emoji.emojize(":zap: | Quantify Misfit", use_aliases=True)
                )

                self.misfit_quantification(event)

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        self.wait_for_all_jobs_to_finish(
            sim_type="forward", events=events_to_use
        )
        # self.comm.lasif.write_misfit(events=events_to_use, details=verbose)

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
        return task_2, verbose_2

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
        self.compute_misfit_on_validation_data()

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")

        for event in self.comm.project.events_in_iteration:
            print(Fore.YELLOW + "\n ==================== \n")
            print(
                emoji.emojize(
                    ":rocket: | Run adjoint simulation", use_aliases=True
                )
            )
            print(f"Event: {event} \n")
            self.run_adjoint_simulation(event)

        print(Fore.BLUE + "\n ========================= \n")

        events_retrieved = "None retrieved"
        i = 0
        while events_retrieved != "All retrieved":
            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for jobs", use_aliases=True
                )
            )
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
                        ":floppy_disk: | Cut sources and "
                        "receivers from gradient",
                        use_aliases=True,
                    )
                )
                if not self.comm.project.remote_gradient_processing:
                    self.preprocess_gradient(event)

                print(Fore.YELLOW + "\n ==================== \n")
                print(
                    emoji.emojize(
                        ":rocket: | Run Diffusion equation", use_aliases=True
                    )
                )
                print(f"Event: {event} gradient will be smoothed")
                if self.comm.project.inversion_mode == "mini-batch":
                    self.smooth_gradient(event)

                # TODO: Add something to check status of smoother here
                # to go to interpolating immediately. Yeah not a bad idea.
        if self.comm.project.inversion_mode == "mono-batch":
            self.sum_gradients()
            self.smooth_gradient(event=None)
        events_retrieved_smoothing = "None retrieved"
        while events_retrieved_smoothing != "All retrieved":

            print(Fore.BLUE)
            print(
                emoji.emojize(
                    ":hourglass: | Waiting for smoothing jobs",
                    use_aliases=True,
                )
            )

            events_retrieved_smoothing_now = self.monitor_job_arrays(
                "smoothing"
            )
            if len(events_retrieved_smoothing_now) == 0:
                print("No new events retrieved, lets wait")
                continue
            if events_retrieved_smoothing_now == "All retrieved":
                events_retrieved_smoothing_now = (
                    self.comm.project.events_in_iteration
                )
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
        return task_2, verbose_2

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
            if self.comm.project.current_iteration == "it0000_model":
                self.comm.project.update_control_group_toml(first=True)
            self.comm.minibatch.print_dp()

            control_group = self.comm.minibatch.select_optimal_control_group()
            # print(f"Selected Control group: {control_group}")
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=control_group
            )
            self.comm.project.change_attribute(
                attribute="new_control_group", new_value=control_group
            )
            first = False

            self.comm.project.update_control_group_toml(new=True, first=first)
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
        return task_2, verbose_2

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

        self.comm.salvus_flow.delete_stored_wavefields(iteration, "forward")
        self.comm.salvus_flow.delete_stored_wavefields(iteration, "adjoint")
        try:
            self._send_whatsapp_announcement()
        except:
            print("Not able to send whatsapp message")
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

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
            task, verbose = self.compute_misfit_and_gradient(task, verbose)
        elif task == "compute_misfit":
            task, verbose = self.compute_misfit(task, verbose)
        elif task == "compute_gradient":
            task, verbose = self.compute_gradient(task, verbose)
        elif task == "select_control_batch":
            task, verbose = self.select_control_batch(task, verbose)
        elif task == "finalize_iteration":
            task, verbose = self.finalize_iteration(task, verbose)
        else:
            raise InversionsonError(f"Don't know task: {task}")

        return task, verbose

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
        self.task = task

        while True:
            task, verbose = self.assign_task_to_function(task, verbose)


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
    # info_toml = input("Give me a path to your information_toml \n\n")
    # Tired of writing it in, I'll do this quick mix for now
    # print("Give me a path to your information_toml \n\n")
    # time.sleep(1)
    # print("Just kidding, I know it")
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
