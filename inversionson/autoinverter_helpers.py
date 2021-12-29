from typing import Dict, List
import emoji
from colorama import init
from colorama import Fore
import time
import os
import inspect
import warnings
import glob
import shutil
import toml
from tqdm import tqdm
from inversionson import InversionsonError, InversionsonWarning
from salvus.flow.api import get_site
from inversionson.optimizers.adam_optimizer import BOOL_ADAM

CUT_SOURCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
    ),
    "inversionson",
    "remote_scripts",
    "cut_and_clip.py",
)
SUM_GRADIENTS_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
    ),
    "inversionson",
    "remote_scripts",
    "gradient_summing.py",
)

init()


class RemoteJobListener(object):
    """
    Class designed to monitor the status of remote jobs.

    It can handle various types of jobs:
    Forward,
    Adjoint,
    Smoothing,
    Model/Gradient Interpolations.
    """

    def __init__(self, comm, job_type, events=None):
        self.comm = comm
        self.job_type = job_type
        self.events_already_retrieved = []
        self.events_retrieved_now = []
        self.to_repost = []
        if events is None:
            if (
                job_type == "smoothing"
                and self.comm.project.inversion_mode == "mono-batch"
            ):
                self.events = [None]
            else:
                self.events = self.comm.project.events_in_iteration
        else:
            self.events = events

    def monitor_jobs(self):
        """
        Takes the job type of the object and monitors the status of
        all the events in the object.

        :raises InversionsonError: Error if job type not recognized
        """
        if self.job_type == "forward":
            job_dict = self.comm.project.forward_job
        elif self.job_type == "adjoint":
            job_dict = self.comm.project.adjoint_job
        elif self.job_type == "model_interp":
            job_dict = self.comm.project.model_interp_job
        elif self.job_type == "gradient_interp":
            job_dict = self.comm.project.gradient_interp_job
        else:
            job_dict = self.comm.project.smoothing_job
        if self.job_type in [
            "forward",
            "adjoint",
            "model_interp",
            "gradient_interp",
        ]:
            self.__monitor_jobs(job_dict=job_dict)
        elif self.job_type == "smoothing":
            self.__monitor_job_array(job_dict=job_dict)
        else:
            raise InversionsonError(f"Job type {self.job_type} not recognised")

    def __check_status_of_job(
        self, event: str, reposts: int, verbose: bool = False
    ):
        """
        Query Salvus Flow for the status of the job

        :param event: Name of event
        :type event: str
        :param reposts: Number of reposts of the event for the job
        :type reposts: int
        """
        status = self.comm.salvus_flow.get_job_status(
            event, self.job_type
        ).name
        if status == "pending":
            if verbose:
                print(f"Status = {status}, event: {event}")
        elif status == "running":
            if verbose:
                print(f"Status = {status}, event: {event}")
        elif status in ["unknown", "failed"]:
            print(f"{self.job_type} job for {event}, {status}, will resubmit")
            if reposts >= 3:
                print("No I've actually reposted this too often \n")
                print("There must be something wrong.")
                raise InversionsonError("Too many reposts")
            self.to_repost.append(event)
            reposts += 1
            self.comm.project.change_attribute(
                attribute=f'{self.job_type}_job["{event}"]["reposts"]',
                new_value=reposts,
            )
        elif status == "cancelled":
            print("What to do here?")
        elif status == "finished":
            return status
        else:
            warnings.warn(
                f"Inversionson does not recognise job status:  {status}",
                InversionsonWarning,
            )
        return status

    def __check_status_of_job_array(
        self, event: str, reposts: int, verbose: bool = False
    ):
        """
        Query Salvus Flow for the status of the job array

        :param event: Name of event
        :type event: str
        :param reposts: Number of reposts of the event for the job
        :type reposts: int
        """
        status = self.comm.salvus_flow.get_job_status(event, self.job_type)
        params = []
        running = 0
        finished = 0
        pending = 0
        unknown = 0
        i = 0
        for _i, s in enumerate(status):
            if s.name == "finished":
                params.append(s)
                finished += 1
            else:
                if s.name in ["pending", "running"]:
                    if verbose:
                        print(
                            f"Status = {s.name}, event: {event} "
                            f"for smoothing job {_i}/{len(status)}"
                        )
                    if s.name == "pending":
                        pending += 1
                    elif s.name == "running":
                        running += 1
                    continue
                elif s.name in ("failed", "unknown"):
                    if i == 0:
                        print(f"Job {s.name}, will resubmit event {event}")
                        self.to_repost.append(event)
                        reposts += 1
                        if reposts >= 3:
                            print(
                                "No I've actually reposted this too often \n"
                            )
                            print("There must be something wrong.")
                            raise InversionsonError("Too many reposts")
                        if event is None:
                            self.comm.project.change_attribute(
                                attribute=f'{self.job_type}_job["reposts"]',
                                new_value=reposts,
                            )
                        else:
                            self.comm.project.change_attribute(
                                attribute=f'{self.job_type}_job["{event}"]["reposts"]',
                                new_value=reposts,
                            )
                        i += 1

                elif s.name == "cancelled":
                    print(f"Job cancelled for event {event}")

                else:
                    warnings.warn(
                        f"Inversionson does not recognise job status:  {status}",
                        InversionsonWarning,
                    )
        if verbose:
            if running > 0:
                print(f"{running}/{len(status)} of jobs running: {event}")
            if pending > 0:
                print(f"{pending}/{len(status)} of jobs pending: {event}")
            if finished > 0:
                print(f"{finished}/{len(status)} of jobs finished: {event}")
        if len(params) == len(status):
            return "finished"

    def __monitor_jobs(
        self, job_dict: Dict, events: List[str] = None, verbose=False
    ):
        """
        Takes the job type of the object and monitors the status of
        all the events in the object.

        :param job_dict: Information on jobs
        :type job_dict: Dict
        :param events: List of events, None results in object events,
            defaults to None
        :type events: List[str], optional
        :param verbose: Print information, defaults to False
        :type verbose: bool, optional
        """
        if events is None:
            events = self.events
        events_left = list(set(events) - set(self.events_already_retrieved))
        finished = len(self.events) - len(events_left)
        running = 0
        pending = 0
        print(f"Checking Jobs for {self.job_type}:")
        for event in tqdm(events_left):
            if job_dict[event]["retrieved"]:
                self.events_already_retrieved.append(event)
                finished += 1
                continue
            else:
                reposts = job_dict[event]["reposts"]
                if not job_dict[event]["submitted"]:
                    status = "unsubmitted"
                    continue
                status = self.__check_status_of_job(
                    event, reposts, verbose=verbose
                )
            if status == "finished":
                self.events_retrieved_now.append(event)
                finished += 1
            elif status == "pending":
                pending += 1
            elif status == "running":
                running += 1

        if finished > 0:
            print(f"{finished}/{len(events)}jobs finished")
        if running > 0:
            print(f"{running}/{len(events)} jobs running")
        if pending > 0:
            print(f"{pending}/{len(events)} jobs pending")

        self.comm.project.update_iteration_toml()

    def __monitor_job_array(self, job_dict, events=None):
        """
        Takes the job type of the object and monitors the status of
        all the events in the object.

        :param job_dict: Information on jobs
        :type job_dict: Dict
        :param events: List of events, None results in object events,
            defaults to None
        :type events: List[str], optional
        """
        finished = 0

        if events is None:
            events = self.events
        if self.comm.project.inversion_mode == "mono-batch":
            if job_dict["retrieved"]:
                self.events_already_retrieved = events
                finished += 1
            else:
                reposts = job_dict["reposts"]
                status = self.__check_status_of_job_array(None, reposts)
                if status == "finished":
                    self.events_retrieved_now = events
                    finished += 1
        else:
            events_left = list(
                set(events) - set(self.events_already_retrieved)
            )
            finished = len(self.events) - len(events_left)
            print("Monitoring Smoothing jobs")
            for event in tqdm(events_left):
                if job_dict[event]["retrieved"]:
                    finished += 1
                    self.events_already_retrieved.append(event)
                    continue
                else:
                    reposts = job_dict[event]["reposts"]
                    status = self.__check_status_of_job_array(
                        event, reposts, verbose=False
                    )
                if status == "finished":
                    self.events_retrieved_now.append(event)
                    finished += 1
            self.comm.project.update_iteration_toml()
            print("\n\n ============= Report ================= \n\n")
            print(f"{finished}/{len(events)} jobs fully finished \n")


class ForwardHelper(object):
    """
    Class which assist with everything related to the forward job

    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def dispatch_forward_simulations(self, verbose=False):
        """
        Dispatch all forward simulations to the remote machine.
        If interpolations are needed, this takes care of that too.

        :param verbose: Print information, defaults to False
        :type verbose: bool, optional
        """
        iteration = self.comm.project.current_iteration
        if (
            self.comm.project.meshes == "multi-mesh"
            and self.comm.project.interpolation_mode == "remote"
        ):
            if "validation_" in iteration:
                self.__dispatch_validation_forwards_remote_interps(verbose)
            else:
                self.__dispatch_forwards_remote_interpolations(verbose)
        else:
            if "validation_" in iteration:
                self.__dispatch_validation_forwards_normal(verbose)
            else:
                self.__dispatch_forwards_normal(verbose)

    def retrieve_forward_simulations(
        self,
        events=None,
        adjoint=False,
        windows=True,
        window_set=None,
        verbose=False,
        validation=False,
    ):
        """
        Get the data from the forward simulations and perform whatever
        operations on them which are requested.
        """
        if events is None:
            events = self.events
        self.__retrieve_forward_simulations(
            events=events,
            adjoint=adjoint,
            windows=windows,
            window_set=window_set,
            verbose=verbose,
            validation=validation,
        )

    def report_total_validation_misfit(self):
        """
        Write the computed validation misfit for the iteration into the
        right place
        """
        iteration = self.comm.project.current_iteration
        self.comm.storyteller.report_validation_misfit(
            iteration=iteration,
            event=None,
            total_sum=True,
        )

    def assert_all_simulations_dispatched(self) -> bool:
        """
        Check whether all simulations have been dispatched

        :return: The answer to your question
        :rtype: bool
        """
        all = True
        for event in self.events:
            submitted, _ = self.__submitted_retrieved(event)
            if not submitted:
                all = False
                break
        return all

    def assert_all_simulations_retrieved(self):
        """
        Check whether all simulations have been retrieved

        :return: The answer to your question
        :rtype: bool
        """
        all = True
        for event in self.events:
            _, retrieved = self.__submitted_retrieved(event)
            if not retrieved:
                all = False
                break
        return all

    def __interpolate_model(self, event: str, mode: str, validation=False):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        :param mode: either "remote" or "local"
        :type mode: str
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
        if mode == "local":
            interp_folder = os.path.join(
                self.comm.project.inversion_root,
                "INTERPOLATION",
                event,
                "model",
            )
            if not os.path.exists(interp_folder):
                os.makedirs(interp_folder)

        if mode == "remote":
            if self.comm.project.model_interp_job[event]["submitted"]:
                print(
                    f"Interpolation for event {event} has already been "
                    "submitted. Will not do interpolation."
                )
                return
            hpc_cluster = get_site(self.comm.project.interpolation_site)
            username = hpc_cluster.config["ssh_settings"]["username"]
            interp_folder = os.path.join(
                "/scratch/snx3000",
                username,
                "INTERPOLATION_WEIGHTS",
                "MODELS",
                event,
            )
            if not hpc_cluster.remote_exists(interp_folder):
                hpc_cluster.remote_mkdir(interp_folder)

        self.comm.multi_mesh.interpolate_to_simulation_mesh(
            event,
            interp_folder=interp_folder,
        )
        if mode == "local":
            self.comm.project.change_attribute(
                attribute=f'forward_job["{event}"]["interpolated"]',
                new_value=True,
            )
        self.comm.project.update_iteration_toml()

    def __submitted_retrieved(self, event: str, sim_type="forward"):
        """
        Get a tuple of boolean values whether job as been submitted
        and retrieved

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation
        :type sim_type: str
        """
        if sim_type == "forward":
            job_info = self.comm.project.forward_job[event]
        elif sim_type == "adjoint":
            job_info = self.comm.project.adjoint_job[event]
        elif sim_type == "model_interp":
            job_info = self.comm.project.model_interp_job[event]
        return job_info["submitted"], job_info["retrieved"]

    def __run_forward_simulation(self, event: str, verbose=False):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        # Check status of simulation
        submitted, retrieved = self.__submitted_retrieved(event)
        iteration = self.comm.project.current_iteration
        if submitted:
            return
        if verbose:
            print(Fore.YELLOW + "\n ============================ \n")
            print(
                emoji.emojize(
                    ":rocket: | Run forward simulation", use_aliases=True
                )
            )
            print(f"Event: {event}")
        receivers = self.comm.salvus_flow.get_receivers(event)
        source = self.comm.salvus_flow.get_source_object(event)
        w = self.comm.salvus_flow.construct_simulation(
            event, source, receivers
        )

        if (
            self.comm.project.remote_mesh is not None
            and self.comm.project.meshes == "mono-mesh"
        ):
            w.set_mesh(self.comm.project.remote_mesh)
        # if self.comm.project.meshes == "mono-mesh":
        #     w.set_mesh("REMOTE:" +
        #         str(self.comm.lasif.find_remote_mesh(
        #             event=None,
        #             interpolate_to=False,
        #             iteration=iteration,
        #             validation="validation" in iteration,
        #         ))
        #     )

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks,
        )

        print(f"Submitted forward job for event: {event}")

    def __compute_station_weights(self, event: str, verbose=False):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        if verbose:
            print(Fore.RED + "\n =========================== \n")
            print(
                emoji.emojize(
                    ":trident: | Calculate station weights", use_aliases=True
                )
            )
        self.comm.lasif.calculate_station_weights(event)

    def __retrieve_seismograms(self, event: str, verbose=False):
        self.comm.salvus_flow.retrieve_outputs(
            event_name=event, sim_type="forward"
        )
        if verbose:
            print(f"Copied seismograms for event {event} to lasif folder")

    def __process_data(self, event: str):
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

    def __select_windows(self, event: str):
        """
        Select the windows for the event and the iteration

        :param event: Name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        if self.comm.project.inversion_mode == "mini-batch":
            window_set_name = iteration + "_" + event
        else:
            window_set_name = event
            if "validation_" not in iteration:
                self.comm.lasif.select_windows(
                    window_set_name=window_set_name, event=event
                )
                return

        if "validation_" in iteration:
            window_set_name = iteration
            if self.comm.project.forward_job[event]["windows_selected"]:
                print(f"Windows already selected for event {event}")
                return
            self.comm.lasif.select_windows(
                window_set_name=window_set_name,
                event=event,
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
            iteration != "it0000_model" and not BOOL_ADAM
            and event in self.comm.project.old_control_group
        ):

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
                window_set_name=window_set_name, event=event
            )

    def __need_misfit_quantification(self, iteration, event, window_set):
        """
        Check whether validation misfit needs to be computed or not

        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event
        :type event: str
        :param window_set: Name of window set
        :type window_set: str
        """
        validation_dict = self.comm.storyteller.validation_dict

        quantify_misfit = True
        if iteration in validation_dict.keys():
            if event in validation_dict[iteration]["events"].keys():
                if (
                    window_set
                    in validation_dict[iteration]["events"][event].keys()
                ):
                    if (
                        validation_dict[iteration]["events"][event][window_set]
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

    def __validation_misfit_quantification(self, event: str, window_set: str):

        iteration = self.comm.project.current_iteration

        if self.__need_misfit_quantification(
            iteration=iteration, event=event, window_set=window_set
        ):
            self.comm.lasif.misfit_quantification(
                event, validation=True, window_set=window_set
            )
            self.comm.storyteller.report_validation_misfit(
                iteration=iteration,
                event=event,
                total_sum=False,
            )

            self.comm.storyteller.report_validation_misfit(
                iteration=self.comm.project.current_iteration,
                event=event,
                total_sum=True,
            )

    def __misfit_quantification(
        self,
        event: str,
        window_set=None,
        validation=False,
    ):
        """
        Compute Misfits and Adjoint sources

        :param event: Name of event
        :type event: str
        """
        if validation:
            self.__validation_misfit_quantification(
                event=event, window_set=self.comm.project.current_iteration
            )
            return
        misfit = self.comm.lasif.misfit_quantification(
            event, validation=validation, window_set=window_set
        )

        self.comm.project.change_attribute(
            attribute=f'misfits["{event}"]', new_value=misfit
        )
        self.comm.project.update_iteration_toml()

    def __dispatch_adjoint_simulation(self, event: str, verbose=False):
        """
        Dispatch an adjoint simulation after finishing the forward
        processing

        :param event: Name of event
        :type event: str
        """
        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        iteration = self.comm.project.current_iteration
        if submitted:
            return

        if verbose:
            print(Fore.YELLOW + "\n ============================ \n")
            print(
                emoji.emojize(
                    ":rocket: | Run adjoint simulation", use_aliases=True
                )
            )
            print(f"Event: {event}")
        adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
        w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
            event, adj_src
        )

        if (
            self.comm.project.remote_mesh is not None
            and self.comm.project.meshes == "mono-mesh"
        ):
            w_adjoint.set_mesh(self.comm.project.remote_mesh)
        # if self.comm.project.meshes == "mono-mesh":
        #     w_adjoint.set_mesh("REMOTE:" +
        #         str(self.comm.lasif.find_remote_mesh(
        #             event=None,
        #             gradient=False,
        #             inverpolate_to=False,
        #             check_if_exists=True,
        #             iteration=iteration,
        #             validatio="validation" in iteration,
        #         ))
        #     )

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

    def __work_with_retrieved_seismograms(
        self,
        event: str,
        windows: bool,
        window_set: str,
        validation=False,
        verbose=False,
    ):
        """
        Process data, select windows, compute adjoint sources

        :param event: Name of event
        :type event: str
        :param windows: Should windows be selected?
        :type windows: bool
        """
        if verbose:
            print(Fore.GREEN + "\n ===================== \n")
            print(
                emoji.emojize(
                    ":floppy_disk: | Process data if needed",
                    use_aliases=True,
                )
            )

        self.__process_data(event)
        if windows:
            if verbose:
                print(Fore.WHITE + "\n ===================== \n")
                print(
                    emoji.emojize(":foggy: | Select windows", use_aliases=True)
                )
            self.__select_windows(event)

        if verbose:
            print(Fore.MAGENTA + "\n ==================== \n")
            print(emoji.emojize(":zap: | Quantify Misfit", use_aliases=True))

        self.__misfit_quantification(
            event, window_set=window_set, validation=validation
        )

    def __dispatch_forwards_remote_interpolations(self, verbose):
        """
        Dispatch remote interpolation jobs,
        Monitor them, as soon as one finishes, dispatch forward job
        Compute station weights
        """
        if verbose:
            print(Fore.CYAN + "\n ============================= \n")
            print(
                emoji.emojize(
                    ":globe_with_meridians: :point_right: "
                    ":globe_with_meridians: | Interpolation Stage",
                    use_aliases=True,
                )
            )
        print("Will dispatch all interpolation jobs \n")
        for _i, event in enumerate(self.events):
            if verbose:
                print(f"Event {_i+1}/{len(self.events)}:  {event}")
            self.__interpolate_model(event=event, mode="remote")
        print("All interpolations have been dispatched")

        int_job_listener = RemoteJobListener(
            comm=self.comm, job_type="model_interp", events=self.events
        )
        j = 0
        while len(int_job_listener.events_already_retrieved) != len(
            self.events
        ):
            int_job_listener.monitor_jobs()
            for event in int_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose)
                self.__compute_station_weights(event, verbose)
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
            for event in int_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__interpolate_model(event, mode="remote")
            print(
                f"We dispatched {len(int_job_listener.events_retrieved_now)} "
                "simulations"
            )
            if len(int_job_listener.events_already_retrieved) + len(
                int_job_listener.events_retrieved_now
            ) == len(self.events):
                j = 0
            int_job_listener.to_repost = []
            int_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 30 seconds before trying again")
                time.sleep(30)
            else:
                j += 1
        # In case of failure:
        if not self.assert_all_simulations_dispatched():
            self.__dispatch_remaining_forwards(verbose=verbose)
        # Here I need to check if all forwards have been dispatched.
        # It can for example fail if the code crashes in the middle.

    def __dispatch_remaining_forwards(self, verbose):
        # Check whether all forwards have been dispatched
        events_left = []
        for event in self.events:
            submitted, _ = self.__submitted_retrieved(
                event, sim_type="forward"
            )
            if not submitted:
                m_submitted, m_retrieved = self.__submitted_retrieved(
                    event, "model_interp"
                )
                if m_retrieved:
                    self.__run_forward_simulation(event, verbose)
                    self.__compute_station_weights(event, verbose)
                elif not m_submitted:
                    events_left.append()
                    self.__interpolate_model(event, mode="remote")
                    self.comm.project.change_attribute(
                        attribute=f'model_interp_job["{event}"]["submitted"]',
                        new_value=False,
                    )
                    self.comm.project.update_iteration_toml()
        if len(events_left) == 0:
            return
        int_job_listener = RemoteJobListener(
            comm=self.comm, job_type="model_interp", events=events_left
        )
        while len(int_job_listener.events_already_retrieved) != len(
            events_left
        ):
            int_job_listener.monitor_jobs()
            for event in int_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose)
                self.__compute_station_weights(event, verbose)
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
            for event in int_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__interpolate_model(event, mode="remote")
            print(
                f"We dispatched {len(int_job_listener.events_retrieved_now)} "
                "simulations"
            )

            int_job_listener.to_repost = []
            int_job_listener.events_retrieved_now = []
            print("Waiting 30 seconds before trying again")
            time.sleep(30)

    def __dispatch_forwards_normal(self, verbose):
        """
        for event:
            (Interpolate)
            Dispatch forward
            Compute station weights
        """
        for event in self.events:
            print(Fore.GREEN + "\n ============================= \n")
            print(f"Event: {event}")
            if self.comm.project.meshes == "multi_mesh":
                if verbose:
                    print(Fore.CYAN + "\n ============================= \n")
                    print(
                        emoji.emojize(
                            ":globe_with_meridians: :point_right: "
                            ":globe_with_meridians: | Interpolation Stage",
                            use_aliases=True,
                        )
                    )
                self.__interpolate_model(event=event, mode="local")
            self.__run_forward_simulation(event, verbose)
            self.__compute_station_weights(event, verbose)
        print("All forward simulations have been dispatched")

    def __dispatch_validation_forwards_remote_interps(self, verbose):
        if verbose:
            print(Fore.CYAN + "\n ============================= \n")
            print(
                emoji.emojize(
                    ":globe_with_meridians: :point_right: "
                    ":globe_with_meridians: | Interpolation Stage",
                    use_aliases=True,
                )
            )
        print("Will dispatch all interpolation jobs \n")
        for _i, event in enumerate(self.events):
            if verbose:
                print(f"Event {_i+1}/{len(self.events)}:  {event}")
            self.__interpolate_model(
                event=event, mode="remote", validation=True
            )
        print("All interpolations have been dispatched")

        vint_job_listener = RemoteJobListener(
            comm=self.comm, job_type="model_interp", events=self.events
        )
        j = 0
        while len(vint_job_listener.events_already_retrieved) != len(
            self.events
        ):
            vint_job_listener.monitor_jobs()
            for event in vint_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose)
                self.__compute_station_weights(event, verbose)
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
            for event in vint_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'model_interp_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__interpolate_model(
                    event=event, mode="remote", validation=True
                )
            print(
                f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                "simulations"
            )
            if len(vint_job_listener.events_already_retrieved) + len(
                vint_job_listener.events_retrieved_now
            ) == len(self.events):
                j = 0
            vint_job_listener.to_repost = []
            vint_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 30 seconds before trying again")
                time.sleep(30)
            else:
                j += 1

    def __dispatch_validation_forwards_normal(self, verbose):
        for event in self.comm.project.validation_dataset:
            if self.comm.project.meshes == "multi-mesh":
                if verbose:
                    print(Fore.CYAN + "\n ============================= \n")
                    print(
                        emoji.emojize(
                            ":globe_with_meridians: :point_right: "
                            ":globe_with_meridians: | Interpolation Stage",
                            use_aliases=True,
                        )
                    )
                    print(f"{event} interpolation")

                self.__interpolate_model(
                    event, validation=True, verbose=verbose
                )
            self.__run_forward_simulation(event, verbose)
            self.__compute_station_weights(event, verbose)

    def __retrieve_forward_simulations(
        self,
        events,
        adjoint,
        windows,
        window_set,
        verbose,
        validation,
    ):
        for_job_listener = RemoteJobListener(
            comm=self.comm, job_type="forward", events=events
        )
        j = 0
        while len(for_job_listener.events_already_retrieved) != len(events):
            for_job_listener.monitor_jobs()
            for event in for_job_listener.events_retrieved_now:
                self.__retrieve_seismograms(event=event, verbose=verbose)

                self.__work_with_retrieved_seismograms(
                    event,
                    windows,
                    window_set,
                    validation,
                    verbose,
                )
                self.comm.project.change_attribute(
                    attribute=f'forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
                if adjoint:
                    self.__dispatch_adjoint_simulation(event, verbose)
            for event in for_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'forward_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__run_forward_simulation(event=event)
            print(
                f"Retrieved {len(for_job_listener.events_retrieved_now)} "
                "seismograms"
            )
            if len(for_job_listener.events_retrieved_now) + len(
                for_job_listener.events_already_retrieved
            ) == len(events):
                j = 0
            for_job_listener.to_repost = []
            for_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 300 seconds before trying again")
                time.sleep(30)
            else:
                j += 1


class AdjointHelper(object):
    """
    A class assisting with everything related to the adjoint simulations

    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def dispatch_adjoint_simulations(self, verbose=False):
        """
        Dispatching all adjoint simulations
        """
        for event in self.events:
            self.__dispatch_adjoint_simulation(event, verbose=verbose)

    def process_gradients(self, events=None, interpolate=False, verbose=False):
        """
        Wait for adjoint simulations. As soon as one is finished,
        we do the appropriate processing of the gradient.
        In the multi-mesh case, that involves an interpolation
        to the inversion grid.
        """
        if events is None:
            events = self.events
        self.__process_gradients(
            events=events, interpolate=interpolate, verbose=verbose
        )

    def assert_all_simulations_dispatched(self):
        all = True
        for event in self.events:
            submitted, _ = self.__submitted_retrieved(event)
            if not submitted:
                all = False
                break
        return all

    def assert_all_simulations_retrieved(self):
        all = True
        for event in self.events:
            _, retrieved = self.__submitted_retrieved(event)
            if not retrieved:
                all = False
                break
        return all

    def __submitted_retrieved(self, event: str, sim_type="adjoint"):
        """
        Get a tuple of boolean values whether job as been submitted
        and retrieved

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation
        :type sim_type: str
        """
        if sim_type == "adjoint":
            job_info = self.comm.project.adjoint_job[event]
        elif sim_type == "gradient_interp":
            job_info = self.comm.project.gradient_interp_job[event]
        elif sim_type == "smoothing":
            if self.comm.project.inversion_mode == "mini-batch":
                job_info = self.comm.project.smoothing_job[event]
            else:
                job_info = self.comm.project.smoothing_job
        return job_info["submitted"], job_info["retrieved"]

    def __process_gradients(
        self, events: list, interpolate: bool, verbose: bool
    ):

        adj_job_listener = RemoteJobListener(
            comm=self.comm, job_type="adjoint", events=events
        )
        if interpolate:
            interp_job_listener = RemoteJobListener(
                comm=self.comm, job_type="gradient_interp", events=events
            )
            mode = self.comm.project.interpolation_mode
        j = 0
        while len(adj_job_listener.events_already_retrieved) != len(events):
            adj_job_listener.monitor_jobs()
            for event in adj_job_listener.events_retrieved_now:
                self.__cut_and_clip_gradient(event=event, verbose=verbose)
                self.comm.project.change_attribute(
                    attribute=f'adjoint_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
                if interpolate:
                    if mode == "remote":
                        self.__dispatch_raw_gradient_interpolation(
                            event, verbose=verbose
                        )
                    else:
                        # Here we do interpolate as false as the interpolate
                        # refers to remote interpolation in this case.
                        # It is related to where the gradient can be found.
                        self.__dispatch_smoothing(
                            event, interpolate=False, verbose=verbose
                        )
                else:
                    if not BOOL_ADAM:
                        self.__dispatch_smoothing(
                            event, interpolate, verbose=verbose
                        )

            for event in adj_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'adjoint_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__dispatch_adjoint_simulation(
                    event=event, verbose=verbose
                )
            print(
                f"Sent {len(adj_job_listener.events_retrieved_now)} "
                "gradients to regularisation"
            )
            if interpolate:
                interp_job_listener.monitor_jobs()
                for event in interp_job_listener.events_retrieved_now:
                    self.comm.project.change_attribute(
                        attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
                    self.comm.project.update_iteration_toml()
                    self.__dispatch_smoothing(
                        event, interpolate, verbose=verbose
                    )
                for event in interp_job_listener.to_repost:
                    self.comm.project.change_attribute(
                        attribute=f'gradient_interp_job["{event}"]["submitted"]',
                        new_value=False,
                    )
                    self.comm.project.update_iteration_toml()
                    self.__dispatch_raw_gradient_interpolation(event)
                interp_job_listener.events_retrieved_now = []
                interp_job_listener.to_repost = []
            # Making sure we don't wait if everything is retrieved already
            if len(adj_job_listener.events_already_retrieved) + len(
                adj_job_listener.events_retrieved_now
            ) == len(events):
                j = 0
            adj_job_listener.to_repost = []
            adj_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 300 seconds before trying again")
                time.sleep(30)
            else:
                j += 1

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose=False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid prior to smoothing.
        """
        submitted, retrieved = self.__submitted_retrieved(
            event, "gradient_interp"
        )
        if submitted:
            if verbose:
                print(
                    f"Interpolation for gradient {event} "
                    "has already been submitted"
                )
            return
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        username = hpc_cluster.config["ssh_settings"]["username"]
        interp_folder = os.path.join(
            "/scratch/snx3000",
            username,
            "INTERPOLATION_WEIGHTS",
            "GRADIENTS",
            event,
        )
        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)
        # Here I need to make sure that the correct layers are interpolated
        # I can just do this by specifying the layers, rather than saying
        # nocore. That's less nice though of course. Could be specified
        # in the config file. Then it should work fine.
        self.comm.multi_mesh.interpolate_gradient_to_model(
            event, smooth=False, interp_folder=interp_folder
        )

    def __dispatch_adjoint_simulation(self, event: str, verbose=False):
        """
        Dispatch an adjoint simulation

        :param event: Name of event
        :type event: str
        """
        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        iteration = self.comm.project.current_iteration
        if submitted:
            return
        if verbose:
            print(f"Event: {event}")
        adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
        w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
            event, adj_src
        )

        if (
            self.comm.project.remote_mesh is not None
            and self.comm.project.meshes == "mono-mesh"
        ):
            w_adjoint.set_mesh(self.comm.project.remote_mesh)

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
            wall_time=self.comm.project.wall_time,
            ranks=self.comm.project.ranks,
        )

    def __dispatch_smoothing(
        self, event: str, interpolate: bool, verbose: bool = False
    ):
        """
        Dispatch a smoothing job for event

        :param event: Name of event
        :type event: str
        :param interpolate: Are we using the multi_mesh approach?
        :type interpolate: bool
        :param verbose: Print information, defaults to False
        :type verbose: bool, optional
        """
        submitted, _ = self.__submitted_retrieved(event, "smoothing")
        if submitted:
            if verbose:
                print(f"Already submitted event {event} for smoothing")
            return

        if interpolate:
            # make sure interpolation has been retrieved
            _, retrieved = self.__submitted_retrieved(event, "gradient_interp")
            if not retrieved:
                if verbose:
                    print(f"Event {event} has not been interpolated")
                return
        if self.comm.project.inversion_mode == "mono-batch":
            self.comm.salvus_flow.retrieve_outputs(
                event_name=event, sim_type="adjoint"
            )
            print(f"Gradient for event {event} has been retrieved.")
        else:
            self.comm.smoother.run_remote_smoother(event)

    def __cut_and_clip_gradient(self, event, verbose=False):
        """
        Cut sources and receivers from gradient before smoothing.
        We also clip the gradient to some percentile
        This can all be configured in information toml.

        :param event: name of the event
        """
        job = self.comm.salvus_flow.get_job(event, "adjoint")
        output_files = job.get_output_files()
        gradient_path = output_files[0][
            ("adjoint", "gradient", "output_filename")
        ]
        # Connect to daint
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_inversionson_dir = os.path.join(
            self.comm.project.remote_diff_model_dir, "..", "smoothing_info"
        )

        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        # copy processing script to hpc
        remote_script = os.path.join(
            remote_inversionson_dir, "cut_and_clip.py"
        )
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)

        if self.comm.project.cut_receiver_radius > 0.0:
            raise InversionsonError(
                "Remote receiver cutting not implemented yet."
            )

        info = {}
        info["filename"] = str(gradient_path)
        info["cutout_radius_in_km"] = self.comm.project.cut_source_radius
        info["source_location"] = self.comm.lasif.get_source(event_name=event)

        info["clipping_percentile"] = self.comm.project.clip_gradient
        info["parameters"] = self.comm.project.inversion_params

        toml_filename = f"{event}_gradient_process.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # put toml on daint and remove local toml
        remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
        hpc_cluster.remote_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Call script
        print(
            hpc_cluster.run_ssh_command(
                f"python {remote_script} {remote_toml}"
            )
        )


class SmoothingHelper(object):
    """
    A class related to everything regarding the smoothing simulations
    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def __remote_summing(self, events, verbose=False):
        """
        Sum gradients on remote for mono-batch case in preparation for.
        smoothing.

        Stores the summed gradient in the local lasif project.

        :param events: List of events to be summed.
        """
        gradient_paths = []
        for event in events:
            job = self.comm.salvus_flow.get_job(event, "adjoint")
            output_files = job.get_output_files()
            gradient_path = output_files[0][
                ("adjoint", "gradient", "output_filename")
            ]
            gradient_paths.append(str(gradient_path))

        # Connect to daint
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_inversionson_dir = os.path.join(
            self.comm.project.remote_diff_model_dir, "..", "summing_dir"
        )
        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        remote_output_path = os.path.join(remote_inversionson_dir,
                                          "summed_gradient.h5")

        # copy summing script to hpc
        remote_script = os.path.join(
            remote_inversionson_dir, "gradient_summing.py"
        )
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(SUM_GRADIENTS_SCRIPT_PATH, remote_script)

        info = {}
        info["filenames"] = gradient_paths
        info["parameters"] = self.comm.project.inversion_params
        info["output_gradient"] = remote_output_path

        toml_filename = f"gradient_sum.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # put toml on daint and remove local toml
        remote_toml = os.path.join(remote_inversionson_dir, toml_filename)
        hpc_cluster.remote_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Call script
        print(
            hpc_cluster.run_ssh_command(
                f"python {remote_script} {remote_toml}"
            )
        )

        # copy summed gradient over to lasif project
        gradients = self.comm.lasif.lasif_comm.project.paths["gradients"]
        iteration = self.comm.project.current_iteration
        gradient = os.path.join(
            gradients,
            f"ITERATION_{iteration}",
            "summed_gradient.h5",)
        hpc_cluster.remote_get(remote_output_path, gradient)


    def dispatch_smoothing_simulations(self, verbose=False):
        """
        Dispatch smoothing simulations. If interpolations needed, they
        are done first.

        :param verbose: Print information, defaults to False
        :type verbose: bool, optional
        """
        if self.comm.project.inversion_mode == "mini-batch" and not BOOL_ADAM:
            if (
                self.comm.project.interpolation_mode == "remote"
                and self.comm.project.meshes == "multi-mesh"
            ):
                interpolate = True
                self.__put_standard_gradient_to_cluster()
            else:
                interpolate = False
            for event in self.events:
                self.__dispatch_smoothing_simulation(
                    event, interpolate=interpolate, verbose=verbose
                )
        else:
            self.__dispatch_smoothing_simulation(event=None, verbose=verbose)

    def monitor_interpolations_send_out_smoothjobs(self, verbose=False):
        """
        Monitor the status of the interpolations, as soon as one is done,
        the smoothing simulation is dispatched
        """

        events = self.events
        int_job_listener = RemoteJobListener(
            comm=self.comm,
            job_type="gradient_interp",
            events=events,
        )
        j = 0
        while len(int_job_listener.events_already_retrieved) != len(events):
            int_job_listener.monitor_jobs()
            for event in int_job_listener.events_retrieved_now:
                self.comm.project.change_attribute(
                    attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
                self.__dispatch_smoothing_simulation(
                    event=event,
                    verbose=verbose,
                    interpolate=True,
                )

            for event in int_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'gradient_interp_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__dispatch_raw_gradient_interpolation(
                    event=event, verbose=verbose
                )
            print(
                f"Dispatched {len(int_job_listener.events_retrieved_now)} "
                "Smoothing jobs"
            )
            if len(int_job_listener.events_retrieved_now) + len(
                int_job_listener.events_already_retrieved
            ) == len(events):
                j = 0
            int_job_listener.to_repost = []
            int_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 30 seconds before trying again")
                time.sleep(30)
            else:
                j += 1

    def sum_gradients(self):
        from inversionson.utils import sum_gradients

        if self.events is None:
            events = self.comm.project.events_in_iteration
        else:
            events = self.events

        if self.comm.project.inversion_mode == "mono-batch" or BOOL_ADAM:
            self.__remote_summing(events)
            return

        grad_mesh = self.comm.lasif.find_gradient(
            iteration=self.comm.project.current_iteration,
            event=None,
            summed=True,
            smooth=False,
            just_give_path=True,
        )
        if os.path.exists(grad_mesh):
            print("Gradient has already been summed. Moving on")
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

    def __submitted_retrieved(self, event: str, sim_type="smoothing"):

        if sim_type == "smoothing":
            if self.comm.project.inversion_mode == "mini-batch" and not BOOL_ADAM:
                job_info = self.comm.project.smoothing_job[event]
            else:
                job_info = self.comm.project.smoothing_job

        elif sim_type == "gradient_interp":
            job_info = self.comm.project.gradient_interp_job[event]

        return job_info["submitted"], job_info["retrieved"]

    def __dispatch_smoothing_simulation(
        self, event: str, interpolate: bool = False, verbose: bool = False
    ):
        submitted, retrieved = self.__submitted_retrieved(event)
        # See if smoothing job already submitted
        # TODO this needs to be fixed for remote summing case

        if submitted:
            sub_ret = "submitted"
            if retrieved:
                sub_ret = "retrieved"
            if verbose:
                print(f"Event {event} has been {sub_ret}. Moving on.")
            return
        if event is None: # mono-batch case, no events
            config = self.comm.smoother.generate_smoothing_config()
            self.comm.smoother.run_smoother(
                config,
                event=None,
                iteration=self.comm.project.current_iteration,
            )
            return
        if not interpolate:
            if verbose:
                print(f"Submitting smoothing for {event}")
            self.comm.smoother.run_remote_smoother(event)

        if interpolate:
            submitted, retrieved = self.__submitted_retrieved(
                event, sim_type="gradient_interp"
            )
            if not submitted:
                hpc_cluster = get_site(self.comm.project.interpolation_site)
                username = hpc_cluster.config["ssh_settings"]["username"]
                interp_folder = os.path.join(
                    "/scratch/snx3000",
                    username,
                    "INTERPOLATION_WEIGHTS",
                    "GRADIENTS",
                    event,
                )
                if not hpc_cluster.remote_exists(interp_folder):
                    hpc_cluster.remote_mkdir(interp_folder)
                self.comm.multi_mesh.interpolate_gradient_to_model(
                    event,
                    smooth=False,
                    interp_folder=interp_folder,
                )
            else:
                if retrieved:
                    print(
                        f"I'm running the remote smoother now for event {event}"
                    )
                    self.comm.smoother.run_remote_smoother(event)
                else:
                    if verbose:
                        print(
                            f"Event {event} is being interpolated,"
                            " can't smooth yet"
                        )

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose=False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid prior to smoothing.
        """
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        username = hpc_cluster.config["ssh_settings"]["username"]
        interp_folder = os.path.join(
            "/scratch/snx3000",
            username,
            "INTERPOLATION_WEIGHTS",
            "GRADIENTS",
            event,
        )
        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)
        self.comm.multi_mesh.interpolate_gradient_to_model(
            event, smooth=False, interp_folder=interp_folder
        )

    def retrieve_smooth_gradients(self, events=None, verbose=False):
        if events is None:
            events = self.events
        if self.comm.project.inversion_mode == "mono-batch":
            events = [events]
        smooth_job_listener = RemoteJobListener(self.comm, "smoothing")
        j = 0
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
        while len(smooth_job_listener.events_already_retrieved) != len(events):
            smooth_job_listener.monitor_jobs()
            for event in smooth_job_listener.events_retrieved_now:
                self.comm.smoother.retrieve_smooth_gradient(event_name=event)
                if self.comm.project.inversion_mode == "mono-batch":
                    attribute = 'smoothing_job["retrieved"]'
                else:
                    attribute = f'smoothing_job["{event}"]["retrieved"]'
                self.comm.project.change_attribute(
                    attribute=attribute,
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
            for event in smooth_job_listener.to_repost:
                if self.comm.project.inversion_mode == "mono-batch":
                    attribute = 'smoothing_job["submitted"]'
                else:
                    attribute = f'smoothing_job["{event}"]["submitted"]'

                self.comm.project.change_attribute(
                    attribute=attribute,
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                print(f"Dispatching smoothing simulation via repost: {event}")
                self.__dispatch_smoothing_simulation(
                    event=event, interpolate=interpolate, verbose=verbose
                )
            print(
                f"Retrieved {len(smooth_job_listener.events_retrieved_now)} "
                "Smooth gradients"
            )
            if len(smooth_job_listener.events_already_retrieved) + len(
                smooth_job_listener.events_retrieved_now
            ) == len(events):
                j = 0
            smooth_job_listener.to_repost = []
            smooth_job_listener.events_retrieved_now = []
            if j != 0:
                print("Waiting 180 seconds before trying again")
                time.sleep(30)
            else:
                j += 1

    def assert_all_simulations_dispatched(self):
        all = True
        if self.comm.project.inversion_mode == "mono-batch":
            submitted, _ = self.__submitted_retrieved(None)
            if submitted:
                return True
            else:
                return False
        for event in self.events:
            submitted, _ = self.__submitted_retrieved(event)
            if not submitted:
                all = False
                break
        return all

    def assert_all_simulations_retrieved(self):
        all = True
        if self.comm.project.inversion_mode == "mono-batch":
            _, retrieved = self.__submitted_retrieved(None)
            if retrieved:
                return True
            else:
                return False
        for event in self.events:
            _, retrieved = self.__submitted_retrieved(event)
            if not retrieved:
                all = False
                break
        return all

    def __put_standard_gradient_to_cluster(self):
        self.comm.lasif.move_gradient_to_cluster()
