import emoji
from colorama import init
from colorama import Fore, Style
import time
import os
import warnings
import glob
import shutil
from inversionson import InversionsonError, InversionsonWarning

init()


class RemoteJobListener(object):
    def __init__(self, comm, job_type, events=None):
        self.comm = comm
        self.job_type = job_type
        self.events_already_retrieved = []
        self.events_retrieved_now = []
        self.to_repost = []
        if events is None:
            events = self.comm.project.events_in_iteration

    def monitor_jobs(self):
        if self.job_type in [
            "forward",
            "adjoint",
            "model_interp",
            "gradient_interp",
        ]:
            self.__monitor_jobs()
        elif self.job_type == "smoothing":
            self.__monitor_job_array()
        else:
            raise InversionsonError(f"Job type {self.job_type} not recognised")

    def __check_status_of_job(self, event: str, reposts: int):
        """
        Query Salvus Flow for the status of the job

        :param event: Name of event
        :type event: str
        """
        status = self.comm.salvus_flow.get_job_status(
            event, self.job_type
        ).name
        if status == "pending":
            print(f"Status = {status}, event: {event}")
        elif status == "running":
            print(f"Status = {status}, event: {event}")
        elif status in ["unknown", "failed"]:
            print(f"Job {status}, will resubmit")
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
        else:
            warnings.warn(
                f"Inversionson does not recognise job status: " f"{status}",
                InversionsonWarning,
            )
        return status

    def __monitor_jobs(self, job_dict, events=None):
        if events is None:
            events = self.events
        events_left = list(set(events) - set(self.events_already_retrieved))
        for event in events_left:
            if job_dict[event]["retrieved"]:
                self.events_already_retrieved.append(event)
                continue
            else:
                reposts = job_dict[event]["reposts"]
                status = self.__check_status_of_job(event, reposts)
            if status == "finished":
                self.events_retrieved_now.append(event)
        self.comm.project.update_iteration_toml()

    def __monitor_job_array(self, job_dict):
        pass


class ForwardHelper(object):
    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def dispatch_forward_simulations(self, verbose=False):
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
        Get the data from the forward simulations
        """
        if events is None:
            events = self.events
        self.__retrieve_forward_simulations(
            self,
            events=events,
            adjoint=adjoint,
            windows=windows,
            window_set=window_set,
            verbose=verbose,
            validation=validation,
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
        else:
            interp_folder = None
        self.comm.multi_mesh.interpolate_to_simulation_mesh(
            event,
            interp_folder=interp_folder,
            validation=validation,
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
        return job_info["submitted"], job_info["retrieved"]

    def __run_forward_simulation(self, event: str, verbose=False):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        # Check status of simulation
        submitted, retrieved = self.__submitted_retrieved(event)
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
            iteration != "it0000_model"
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

        validation_iterations = (
            self.comm.lasif.get_validation_iteration_numbers()
        )
        iteration = self.comm.project.current_iteration

        if self.__need_misfit_quantification(
            iteration=iteration, event=event, window_set=window_set
        ):
            self.comm.lasif.misfit_quantification(
                event, validation=True, window_set=window_set
            )
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
                        event,
                        validation=True,
                        window_set=last_iteration,
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

    def __dispatch_adjoint_simulation(self, event: str):
        """
        Dispatch an adjoint simulation after finishing the forward
        processing

        :param event: Name of event
        :type event: str
        """
        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return
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

        while len(int_job_listener.events_already_retrieved) != len(
            self.events
        ):
            int_job_listener.monitor_jobs()
            for event in int_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose)
                self.__compute_station_weights(event, verbose)
            for event in int_job_listener.to_repost:
                self.__run_forward_simulation(event, verbose)
            print(
                f"We dispatched {len(int_job_listener.events_retrieved_now)} "
                "simulations"
            )
            print("Waiting 15 seconds before trying again")
            int_job_listener.to_repost = []
            int_job_listener.events_retrieved_now = []
            time.sleep(15)

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

        while len(vint_job_listener.events_already_retrieved) != len(
            self.events
        ):
            vint_job_listener.monitor_jobs()
            for event in vint_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose)
                self.__compute_station_weights(event, verbose)
            for event in vint_job_listener.to_repost:
                self.__run_forward_simulation(event, verbose)
            print(
                f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                "simulations"
            )
            print("Waiting 15 seconds before trying again")
            vint_job_listener.to_repost = []
            vint_job_listener.events_retrieved_now = []
            time.sleep(15)

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
        while len(for_job_listener.events_already_retrieved) != len(events):
            for_job_listener.monitor_jobs()
            for event in for_job_listener.events_retrieved_now:
                self.__retrieve_seismograms(event=event, verbose=verbose)
                self.comm.project.change_attribute(
                    attribute=f'forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
                self.__work_with_retrieved_seismograms(
                    event,
                    windows,
                    window_set,
                    validation,
                    verbose,
                )
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
            print("Waiting 15 seconds before trying again")
            for_job_listener.to_repost = []
            for_job_listener.events_retrieved_now = []
            time.sleep(15)


# Might be a good idea to do Interpolations and then Smoothing
# Will be way less transfer of data
# I will have to do some h5py magic on daint inside the second
# interpolation job though.
# That should be doable though.
# Question is... what do I need?
# I have a mesh which I interpolate to. This mesh can be ready
# The output gradient may need a few things, not sure.
# Maybe I have a problem with stuff being added to the core via smoothing
# But that doesn't have to be a problem.
class AdjointHelper(object):
    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def dispatch_adjoint_simulations(self):
        """
        Dispatching all adjoint simulations
        """
        for event in self.events:
            self.__dispatch_adjoint_simulation(event)

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
                    self.__dispatch_raw_gradient_interpolation(
                        event, verbose=verbose
                    )
                else:
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
            print("Waiting 15 seconds before trying again")
            adj_job_listener.to_repost = []
            adj_job_listener.events_retrieved_now = []
            time.sleep(15)

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose=False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid prior to smoothing.
        """
        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return

        # Find the adjoint job
        # Find path to gradient
        # Create interpolation job based on this gradient
        # And one kept in the Project folder.
        # Submit the job

    def __dispatch_adjoint_simulation(self, event: str, verbose=False):
        """
        Dispatch an adjoint simulation

        :param event: Name of event
        :type event: str
        """
        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return
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
        self.comm.project.change_attribute(
            attribute=f'adjoint_job["{event}"]["submitted"]', new_value=True
        )
        self.comm.project.update_iteration_toml()

        # Can't really recall how we do with retrieving gradients now.
        # I'll have to work on that a bit.

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
        self.comm.smoothing.run_remote_smoother(event)