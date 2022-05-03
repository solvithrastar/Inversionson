import os
import inspect
import toml
import salvus.flow.api as sapi
from salvus.flow.api import get_site

from inversionson.helpers.remote_job_listener import RemoteJobListener
from inversionson.utils import sleep_or_process

CUT_SOURCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "remote_scripts",
    "cut_and_clip.py",
)

PROCESS_OUTPUT_SCRIPT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
    "remote_scripts",
    "window_and_calc_adj_src.py",
)


class IterationListener(object):
    """
    Class that can handle an entire iteration until it's done.
    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def print(
        self,
        message: str,
        color="yellow",
        line_above=False,
        line_below=False,
        emoji_alias=None,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def __prepare_forward(self, event: str):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        """

        submitted, _ = self.__submitted_retrieved(event, sim_type="prepare_forward")
        if submitted:
            return

        hpc_cluster = get_site(self.comm.project.interpolation_site)
        interp_folder = os.path.join(
            self.comm.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            "MODELS",
            event,
        )
        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)

        self.comm.multi_mesh.prepare_forward(event=event)
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
        elif sim_type == "prepare_forward":
            job_info = self.comm.project.prepare_forward_job[event]
        elif sim_type == "hpc_processing":
            job_info = self.comm.project.hpc_processing_job[event]
        elif sim_type == "gradient_interp":
            job_info = self.comm.project.gradient_interp_job[event]
        return job_info["submitted"], job_info["retrieved"]

    def __run_forward_simulation(self, event: str, verbose=False):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        # Check status of simulation
        submitted, retrieved = self.__submitted_retrieved(event,
                                                          sim_type="forward")
        if submitted:
            return

        # In the case of a prepare_forward job, assume dict is created remotely
        if self.comm.project.prepare_forward:
            simulation_created_remotely = True
        else:
            simulation_created_remotely = False

        if verbose:
            self.print(
                "Run forward simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        if simulation_created_remotely:
            w = self.comm.salvus_flow.construct_simulation_from_dict(event)
        else:
            receivers = self.comm.salvus_flow.get_receivers(event)
            source = self.comm.salvus_flow.get_source_object(event)
            w = self.comm.salvus_flow.construct_simulation(event, source, receivers)

        if self.comm.project.meshes == "multi-mesh":
            already_interpolated = True
        else:
            already_interpolated = False

        # Get the average model when validation event
        if (
            self.comm.project.is_validation_event(event)
            and self.comm.project.use_model_averaging
            and "00000" not in self.comm.project.current_iteration
        ):
            validation = True
        else:
            validation = False
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        remote_mesh = self.comm.lasif.find_remote_mesh(
            event=event,
            gradient=False,
            interpolate_to=False,
            hpc_cluster=hpc_cluster,
            validation=validation,
            already_interpolated=already_interpolated,
        )
        w.set_mesh("REMOTE:" + str(remote_mesh))
        # make the mesh use

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.comm.project.site_name,
            ranks=self.comm.project.ranks,
        )

        self.print(f"Submitted forward job for event: {event}")

    def __compute_station_weights(self, event: str, verbose=False):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        # Skip this in the event of remote weight set calculations
        # as part of the HPC processing job.
        if self.comm.project.hpc_processing:
            return

        if self.comm.project.is_validation_event(event):
            return

        if verbose:
            self.print(
                "Calculate station weights",
                color="red",
                line_above=True,
                emoji_alias=":trident:",
            )
        self.comm.lasif.calculate_station_weights(event)

    def __retrieve_seismograms(self, event: str, verbose=False):
        self.comm.salvus_flow.retrieve_outputs(event_name=event, sim_type="forward")
        if verbose:
            self.print(f"Copied seismograms for event {event} to lasif folder")

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

    def _launch_hpc_processing_job(self, event):
        """
        Here, we launch a job to select windows and get adjoint sources
        for an event.

        """
        submitted, _ = self.__submitted_retrieved(event, "hpc_processing")
        if submitted:
            return

        if not self.comm.project.remote_data_processing:
            self.__process_data(event)

        iteration = self.comm.project.current_iteration
        forward_job = sapi.get_job(
            site_name=self.comm.project.site_name,
            job_name=self.comm.salvus_flow.get_job_name(
                event=event, sim_type="forward"
            ),
        )

        # Get forward paths
        remote_syn_path = str(forward_job.output_path / "receivers.h5")
        forward_meta_json_filename = str(forward_job.output_path / "meta.json")

        # Get local proc filename
        lasif_root = self.comm.project.lasif_root
        proc_filename = (
            f"preprocessed_{int(self.comm.project.min_period)}s_"
            f"to_{int(self.comm.project.max_period)}s.h5"
        )
        local_proc_file = os.path.join(
            lasif_root, "PROCESSED_DATA", "EARTHQUAKES", event, proc_filename
        )

        remote_proc_file_name = f"{event}_{proc_filename}"
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_processed_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "PROCESSED_DATA"
        )
        remote_adj_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "ADJOINT_SOURCES"
        )
        remote_receiver_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "RECEIVERS"
        )
        for dir_name in [remote_processed_dir, remote_adj_dir, remote_receiver_dir]:
            if not hpc_cluster.remote_exists(dir_name):
                hpc_cluster.remote_mkdir(remote_processed_dir)

        remote_proc_path = os.path.join(remote_processed_dir, remote_proc_file_name)
        tmp_remote_path = remote_proc_path + "_tmp"
        if not hpc_cluster.remote_exists(remote_proc_path):
            hpc_cluster.remote_put(local_proc_file, tmp_remote_path)
            hpc_cluster.run_ssh_command(f"mv {tmp_remote_path} {remote_proc_path}")

        remote_adj_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "ADJOINT_SOURCES"
        )

        if "VPV" in self.comm.project.inversion_params:
            parameterization = "tti"
        elif "VP" in self.comm.project.inversion_params:
            parameterization = "rho-vp-vs"

        info = dict(
            processed_filename=remote_proc_path,
            synthetic_filename=remote_syn_path,
            forward_meta_json_filename=forward_meta_json_filename,
            parameterization=parameterization,
            event_name=event,
            delta=self.comm.project.simulation_dict["time_step"],
            npts=self.comm.project.simulation_dict["number_of_time_steps"],
            iteration_name=iteration,
            minimum_period=self.comm.project.min_period,
            maximum_period=self.comm.project.max_period,
            start_time_in_s=self.comm.project.simulation_dict["start_time"],
            receiver_json_path=os.path.join(
                remote_receiver_dir, f"{event}_receivers.json"
            ),
            ad_src_type=self.comm.project.ad_src_type,
        )

        toml_filename = f"{iteration}_{event}_adj_info.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # Put info toml on daint and remove local toml
        remote_toml = os.path.join(remote_adj_dir, toml_filename)
        hpc_cluster.remote_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Copy processing script to hpc
        remote_script = os.path.join(remote_adj_dir, "window_and_calc_adj_src.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(PROCESS_OUTPUT_SCRIPT_PATH, remote_script)

        # Now submit the job
        description = f"HPC processing of {event} for iteration {iteration}"

        # use interp wall time for now
        wall_time = self.comm.project.hpc_processing_wall_time
        from salvus.flow.sites import job, remote_io_site

        commands = [
            remote_io_site.site_utils.RemoteCommand(
                command="mkdir output", execute_with_mpi=False
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"python {remote_script} {remote_toml}", execute_with_mpi=False
            ),
        ]
        # Allow to set conda environment first
        if self.comm.project.remote_conda_env:
            conda_command = [
                remote_io_site.site_utils.RemoteCommand(
                    command=f"conda activate {self.comm.project.remote_conda_env}",
                    execute_with_mpi=False,
                )
            ]
            commands = conda_command + commands

        job = job.Job(
            site=sapi.get_site(self.comm.project.interpolation_site),
            commands=commands,
            job_type="hpc_processing",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )

        self.comm.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["name"]',
            new_value=job.job_name,
        )
        job.launch()
        self.comm.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["submitted"]',
            new_value=True,
        )
        self.print(f"HPC Processing job for event {event} submitted")
        self.comm.project.update_iteration_toml()

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

        self.comm.lasif.select_windows(window_set_name=window_set_name, event=event)

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
                if window_set in validation_dict[iteration]["events"][event].keys():
                    if validation_dict[iteration]["events"][event][window_set] != 0.0:
                        quantify_misfit = False

        if not quantify_misfit:
            message = (
                f"Will not quantify misfit for event {event}, "
                f"iteration {iteration} "
                f"window set {window_set}. If you want it computed, "
                f"change value in validation toml to 0.0"
            )
            self.print(message)

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
    ):
        """
        Compute Misfits and Adjoint sources

        :param event: Name of event
        :type event: str
        """
        if self.comm.project.is_validation_event(event):
            self.__validation_misfit_quantification(
                event=event, window_set=self.comm.project.current_iteration
            )
            return
        misfit = self.comm.lasif.misfit_quantification(event, window_set=window_set)

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
        :param hpc_processing: Use reomate adjoint file
        :type hpc_processing: bool
        """
        if self.comm.project.is_validation_event(event):
            return

        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return

        if verbose:
            self.print(
                "Run adjoint simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        if self.comm.project.meshes == "multi-mesh" or self.comm.project.hpc_processing:
            simulation_created_remotely = True
        else:
            simulation_created_remotely = False
        if simulation_created_remotely:
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation_from_dict(
                event
            )
        else:
            adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
                event, adj_src
            )

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
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
            self.print(
                "Process data if needed",
                line_above=True,
                emoji_alias=":floppy_disk:",
                color="green",
            )

        self.__process_data(event)

        # Skip window selection in case of validation data
        if windows and not self.comm.project.is_validation_event(event):
            if verbose:
                self.print(
                    "Select windows",
                    color="white",
                    line_above=True,
                    emoji_alias=":foggy:",
                )
            self.__select_windows(event)

        if verbose:
            self.print(
                "Quantify Misfit", color="magenta", line_above=True, emoji_alias=":zap:"
            )

        self.__misfit_quantification(event, window_set=window_set)

    def dispatch_forward_simulations(self, verbose):
        """
        Dispatches the forward events
        """

        if verbose and self.comm.project.prepare_forward:
            self.print(
                "Prepare forward Stage",
                line_above=True,
                emoji_alias=[
                    ":globe_with_meridians:",
                    ":point_right:",
                    ":globe_with_meridians:",
                ],
            )

        events = self.events

        if self.comm.project.prepare_forward:
            self.print("Will dispatch all prepare_forward jobs")
            for _i, event in enumerate(events):
                if verbose:
                    self.print(f"Event {_i+1}/{len(self.events)}:  {event}")
                self.__prepare_forward(event=event)
            self.print("All prepare_forward jobs have been dispatched")

            self.__listen_to_prepare_forward(events=events, verbose=verbose)

        #
        for _i, event in enumerate(events):
            self.__run_forward_simulation(event, verbose=verbose)
            self.__compute_station_weights(event, verbose)

    def __listen_to_prepare_forward(self, events, verbose):
        """
        Listens to prepare forward jobs and waits for them to be done.
        Also submits simulations.
        """
        anything_retrieved = False
        vint_job_listener = RemoteJobListener(
            comm=self.comm, job_type="prepare_forward", events=events
        )
        vint_job_listener.monitor_jobs()
        for event in vint_job_listener.events_retrieved_now:
            self.__run_forward_simulation(event, verbose=verbose)
            self.__compute_station_weights(event, verbose)
            self.comm.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["retrieved"]',
                new_value=True,
            )
            self.comm.project.update_iteration_toml()
            vint_job_listener.events_already_retrieved.append(event)
        for event in vint_job_listener.to_repost:
            self.comm.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.comm.project.update_iteration_toml()
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration=self.comm.project.current_iteration,
                sim_type="prepare_forward",
                event_name=event,
            )
            self.__prepare_forward(event=event)
        if len(vint_job_listener.events_retrieved_now) > 0:
            self.print(
                f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                "simulations"
            )
        # In the case where the processed data already exists, it will be
        # retrieved without ever entering the events_retrieved_now loop.
        # In that case we should still submit these jobs.
        for event in vint_job_listener.events_already_retrieved:
            self.__run_forward_simulation(event, verbose=verbose)
            self.__compute_station_weights(event, verbose)

        if len(vint_job_listener.events_retrieved_now) > 0:
            self.print(
                f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                "simulations"
            )

        if vint_job_listener.events_retrieved_now:
            anything_retrieved = True

        self.comm.project.update_iteration_toml()
        return anything_retrieved, vint_job_listener.events_already_retrieved

    def __listen_to_forward(self, events,
                            adjoint=True,
                            windows=True,
                            window_set=None,
                            verbose=False
                            ):

        anything_retrieved = False

        for_job_listener = RemoteJobListener(
            comm=self.comm, job_type="forward", events=events
        )
        for_job_listener.monitor_jobs()

        # submit remote jobs for the ones that did not get
        # submitted yet, although forwards are done.
        for event in for_job_listener.events_already_retrieved:
            if (
                self.comm.project.hpc_processing
                and not self.comm.project.is_validation_event(event)
            ):
                self._launch_hpc_processing_job(event)

        for event in for_job_listener.events_retrieved_now:
            # Still retrieve synthetics for validation data. NO QA
            if (
                not self.comm.project.hpc_processing
                or self.comm.project.is_validation_event(event)
            ):
                self.__retrieve_seismograms(event=event, verbose=verbose)

            # Here I need to replace this with remote hpc job,
            # then this actually needs be finished before any adjoint
            # jobs are launched
            if (
                self.comm.project.hpc_processing
                and not self.comm.project.is_validation_event(event)
            ):
                self._launch_hpc_processing_job(event)
            else:
                self.__work_with_retrieved_seismograms(
                    event,
                    windows,
                    window_set,
                    verbose,
                )
            self.comm.project.change_attribute(
                attribute=f'forward_job["{event}"]["retrieved"]',
                new_value=True,
            )
            self.comm.project.update_iteration_toml()
            if adjoint and not self.comm.project.hpc_processing:
                self.__dispatch_adjoint_simulation(event, verbose)
            for_job_listener.events_already_retrieved.append(event)
        for event in for_job_listener.to_repost:
            self.comm.project.change_attribute(
                attribute=f'forward_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.comm.project.update_iteration_toml()
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration=self.comm.project.current_iteration,
                sim_type="forward",
                event_name=event,
            )
            self.__run_forward_simulation(event=event)
        if len(for_job_listener.events_retrieved_now) > 0:
            self.print(
                f"Retrieved {len(for_job_listener.events_retrieved_now)} "
                "seismograms"
            )

        if len(for_job_listener.events_retrieved_now):
            anything_retrieved = True

        self.comm.project.update_iteration_toml()
        return anything_retrieved, for_job_listener.events_already_retrieved

    def __listen_to_hpc_processing(self, events, adjoint=True,
                                   verbose=False):
        """
        Here we listen hpc_processing. It is important that only
        events enter here that actually need to be listened to.
        So no validation events for example.
        """
        anything_retrieved = False

        hpc_proc_job_listener = RemoteJobListener(
            comm=self.comm,
            job_type="hpc_processing",
            events=events,
        )
        hpc_proc_job_listener.monitor_jobs()

        for event in hpc_proc_job_listener.events_retrieved_now:
            self.comm.project.change_attribute(
                attribute=f'hpc_processing_job["{event}"]["retrieved"]',
                new_value=True,
            )
            if adjoint and self.comm.project.hpc_processing:
                self.__dispatch_adjoint_simulation(event, verbose)
            hpc_proc_job_listener.events_already_retrieved.append(event)
        for event in hpc_proc_job_listener.to_repost:
            self.comm.project.change_attribute(
                attribute=f'hpc_processing_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.comm.project.update_iteration_toml()
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration=self.comm.project.current_iteration,
                sim_type="hpc_processing",
                event_name=event,
            )
            self._launch_hpc_processing_job(event)

        if hpc_proc_job_listener.events_retrieved_now:
            anything_retrieved = True

        self.comm.project.update_iteration_toml()
        return anything_retrieved, hpc_proc_job_listener.events_already_retrieved

    def __listen_to_adjoint(self, events: list, verbose: bool= False):
        """
        Here we listen to the adjoint jobs.
        Again it is important that only candidate events enter here.
        So no validation events.
        """
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
        else:
            interpolate = False
        anything_retrieved = False

        adj_job_listener = RemoteJobListener(
            comm=self.comm, job_type="adjoint", events=events
        )

        adj_job_listener.monitor_jobs()
        for event in adj_job_listener.events_retrieved_now:
            if not (
                self.comm.project.meshes == "multi-mesh"
                and self.comm.project.interpolation_mode == "remote"
            ):
                self.__cut_and_clip_gradient(event=event, verbose=verbose)
            self.comm.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["retrieved"]',
                new_value=True,
            )
            self.comm.project.update_iteration_toml()
            if interpolate:
                if self.comm.project.interpolation_mode == "remote":
                    self.__dispatch_raw_gradient_interpolation(
                        event, verbose=verbose
                    )
            adj_job_listener.events_already_retrieved.append(event)

        for event in adj_job_listener.to_repost:
            self.comm.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.comm.project.update_iteration_toml()
            self.comm.salvus_flow.delete_stored_wavefields(
                iteration=self.comm.project.current_iteration,
                sim_type="adjoint",
                event_name=event,
            )
            self.__dispatch_adjoint_simulation(event=event, verbose=verbose)

        if adj_job_listener.events_retrieved_now:
            anything_retrieved = True

        self.comm.project.update_iteration_toml()
        return anything_retrieved, adj_job_listener.events_already_retrieved

    def __listen_to_gradient_interp(self, events, verbose=False):
        """
        Monitor the status of the interpolations.

        It is important that only candidate events enter here
        """
        anything_retrieved = False

        if not self.comm.project.meshes == "multi-mesh":
            raise Exception("This should not enter here")

        int_job_listener = RemoteJobListener(
            comm=self.comm,
            job_type="gradient_interp",
            events=events,
        )

        int_job_listener.monitor_jobs()
        for event in int_job_listener.events_retrieved_now:
            self.comm.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                new_value=True,
            )
            self.comm.project.update_iteration_toml()
            int_job_listener.events_already_retrieved.append(event)

        for event in int_job_listener.to_repost:
            self.comm.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.comm.project.update_iteration_toml()
            self.__dispatch_raw_gradient_interpolation(event=event)

        if int_job_listener.events_retrieved_now:
            anything_retrieved = True

        self.comm.project.update_iteration_toml()
        return anything_retrieved, int_job_listener.events_already_retrieved

    def listen(self, verbose=False):
        """
        Listen to all steps in the iteration.
        It will listen to prepare forward (if needed) and forward
        for all jobs. And it will listen to hpc proc (if needed) and
        adjoint, and gradient interp (if needed) jobs.
        """

        # Initialize variables
        all_pf_retrieved_events = []
        all_f_retrieved_events = []
        all_hpc_proc_retrieved_events = []
        all_adj_retrieved_events = []
        all_gi_retrieved_events = []

        len_all_events = len(self.events)
        len_non_validation_events = len(list(set(self.events) - set(self.comm.project.validation_dataset)))

        while True:
            anything_retrieved_f = False
            anything_retrieved_hpc_proc = False
            anything_adj_retrieved = False
            anything_gi_retrieved = False
            anything_retrieved = False

            if self.comm.project.remote_data_processing or \
                    self.comm.project.meshes == "multi-mesh":
                for event in self.events:
                    self.__prepare_forward(event)
                if len(all_pf_retrieved_events) != len_all_events:
                    anything_retrieved_pf, all_pf_retrieved_events = \
                        self.__listen_to_prepare_forward(events=self.events, verbose=verbose)
                    if anything_retrieved_pf:
                        anything_retrieved = True
            # Then we listen to forward for the already retrieved events in
            # prepare_forward.
            # Here it actually
            # important that we dispatch everything first if there is no prepare
            # forward and that we then listen to all events.
            if self.comm.project.remote_data_processing or \
                    self.comm.project.meshes == "multi-mesh":
                if len(all_pf_retrieved_events) > 0 and len(all_f_retrieved_events) != len_all_events:
                    anything_retrieved_f, all_f_retrieved_events = \
                        self.__listen_to_forward(all_pf_retrieved_events, verbose=verbose)
            else:
                for _i, event in enumerate(self.events):
                    self.__run_forward_simulation(event, verbose=verbose)
                    self.__compute_station_weights(event, verbose)
                if len(all_f_retrieved_events) != len_all_events:
                    anything_retrieved_f, all_f_retrieved_events = \
                        self.__listen_to_forward(self.events, verbose=verbose)
            if anything_retrieved_f:
                anything_retrieved = True

            ####################################################################
            # The rest is only for the non validation events.
            ####################################################################
            all_f_retrieved_events = list(set(all_f_retrieved_events) - set(
                self.comm.project.validation_dataset))
            # Now we start listening to the hpc_proc jobs. Only for the ones that
            # finished and only if applicable.

            if self.comm.project.hpc_processing:
                if len(all_f_retrieved_events) > 0 and len(all_hpc_proc_retrieved_events) != len_non_validation_events:
                    anything_retrieved_hpc_proc, all_hpc_proc_retrieved_events = self.__listen_to_hpc_processing(all_f_retrieved_events)
                if anything_retrieved_hpc_proc:
                    anything_retrieved = True
            # Now we start listening to the adjoint jobs. If hpc proc,
            # we only listen to the ones that finished there already.
            # Otherwise we listen to the ones that finished forward
            if self.comm.project.hpc_processing:
                if len(all_hpc_proc_retrieved_events) > 0 and len(all_adj_retrieved_events) != len_non_validation_events:
                    anything_adj_retrieved, all_adj_retrieved_events = self.__listen_to_adjoint(all_hpc_proc_retrieved_events)
            else:
                if len(all_f_retrieved_events) > 0 and len(all_adj_retrieved_events) != len_non_validation_events:
                    anything_adj_retrieved, all_adj_retrieved_events = self.__listen_to_adjoint(all_f_retrieved_events)
            if anything_adj_retrieved:
                anything_retrieved = True
            # Now we listen to the gradient interp jobs in the multi_mesh case
            # otherwise we are done here.

            if self.comm.project.meshes == "multi-mesh":
                if len(all_adj_retrieved_events) > 0 and len(all_gi_retrieved_events) != len_non_validation_events:
                    anything_gi_retrieved, all_gi_retrieved_events = self.__listen_to_gradient_interp(all_adj_retrieved_events)
                    if len(all_gi_retrieved_events) == len_non_validation_events:
                        break
                if anything_gi_retrieved:
                    anything_retrieved = True

            else:
                if len(all_adj_retrieved_events) == len_non_validation_events:
                    break

            if not anything_retrieved:
                sleep_or_process(self.comm)

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose=False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid.
        """
        submitted, retrieved = self.__submitted_retrieved(event, "gradient_interp")
        if submitted:
            if verbose:
                self.print(
                    f"Interpolation for gradient {event} " "has already been submitted"
                )
            return
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        if hpc_cluster.config["site_type"] == "local":
            interp_folder = os.path.join(
                self.comm.project.remote_inversionson_dir,
                "INTERPOLATION_WEIGHTS",
                "GRADIENTS",
                event,
            )
        else:
            interp_folder = os.path.join(
                self.comm.project.remote_inversionson_dir,
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

    def __cut_and_clip_gradient(self, event, verbose=False):
        """
        Cut sources and receivers from gradient before summing or potential
        smoothing.
        We also clip the gradient to some percentile
        This can all be configured in information toml.

        :param event: name of the event
        """
        job = self.comm.salvus_flow.get_job(event, "adjoint")
        output_files = job.get_output_files()
        gradient_path = output_files[0][("adjoint", "gradient", "output_filename")]
        # Connect to daint
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_inversionson_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "GRADIENT_PROCESSING"
        )

        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        # copy processing script to hpc
        remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)

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
        print(hpc_cluster.run_ssh_command(f"python {remote_script} {remote_toml}"))


class ForwardHelper(object):
    """
    Class which assist with everything related to the forward job
    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def print(
        self,
        message: str,
        color="yellow",
        line_above=False,
        line_below=False,
        emoji_alias=None,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def retrieve_forward_simulations(
        self,
        events=None,
        adjoint=False,
        windows=True,
        window_set=None,
        verbose=False,
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

    def __prepare_forward(self, event: str):
        """
        Interpolate model to a simulation mesh

        :param event: Name of event
        :type event: str
        """

        hpc_cluster = get_site(self.comm.project.interpolation_site)
        interp_folder = os.path.join(
            self.comm.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            "MODELS",
            event,
        )
        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)

        self.comm.multi_mesh.prepare_forward(event=event)

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
        elif sim_type == "prepare_forward":
            job_info = self.comm.project.prepare_forward_job[event]
        elif sim_type == "hpc_processing":
            job_info = self.comm.project.hpc_processing_job[event]
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

        # In the case of a prepare_forward job, assume dict is created remotely
        if self.comm.project.prepare_forward:
            simulation_created_remotely = True
        else:
            simulation_created_remotely = False

        if verbose:
            self.print(
                "Run forward simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        if simulation_created_remotely:
            w = self.comm.salvus_flow.construct_simulation_from_dict(event)
        else:
            receivers = self.comm.salvus_flow.get_receivers(event)
            source = self.comm.salvus_flow.get_source_object(event)
            w = self.comm.salvus_flow.construct_simulation(event, source, receivers)

        if self.comm.project.meshes == "multi-mesh":
            already_interpolated = True
        else:
            already_interpolated = False

        # Get the average model when validation event
        if (
            self.comm.project.is_validation_event(event)
            and self.comm.project.use_model_averaging
            and "00000" not in self.comm.project.current_iteration
        ):
            validation = True
        else:
            validation = False
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        remote_mesh = self.comm.lasif.find_remote_mesh(
            event=event,
            gradient=False,
            interpolate_to=False,
            hpc_cluster=hpc_cluster,
            validation=validation,
            already_interpolated=already_interpolated,
        )
        w.set_mesh("REMOTE:" + str(remote_mesh))
        # make the mesh use

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.comm.project.site_name,
            ranks=self.comm.project.ranks,
        )

        self.print(f"Submitted forward job for event: {event}")

    def __compute_station_weights(self, event: str, verbose=False):
        """
        Calculate station weights to reduce the effect of data coverage

        :param event: Name of event
        :type event: str
        """
        # Skip this in the event of remote weight set calculations
        # as part of the HPC processing job.
        if self.comm.project.hpc_processing:
            return

        if self.comm.project.is_validation_event(event):
            return

        if verbose:
            self.print(
                "Calculate station weights",
                color="red",
                line_above=True,
                emoji_alias=":trident:",
            )
        self.comm.lasif.calculate_station_weights(event)

    def __retrieve_seismograms(self, event: str, verbose=False):
        self.comm.salvus_flow.retrieve_outputs(event_name=event, sim_type="forward")
        if verbose:
            self.print(f"Copied seismograms for event {event} to lasif folder")

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

    def _launch_hpc_processing_job(self, event):
        """
        Here, we launch a job to select windows and get adjoint sources
        for an event.

        """
        submitted, _ = self.__submitted_retrieved(event, "hpc_processing")
        if submitted:
            return

        if not self.comm.project.remote_data_processing:
            self.__process_data(event)

        iteration = self.comm.project.current_iteration
        forward_job = sapi.get_job(
            site_name=self.comm.project.site_name,
            job_name=self.comm.salvus_flow.get_job_name(
                event=event, sim_type="forward"
            ),
        )

        # Get forward paths
        remote_syn_path = str(forward_job.output_path / "receivers.h5")
        forward_meta_json_filename = str(forward_job.output_path / "meta.json")

        # Get local proc filename
        lasif_root = self.comm.project.lasif_root
        proc_filename = (
            f"preprocessed_{int(self.comm.project.min_period)}s_"
            f"to_{int(self.comm.project.max_period)}s.h5"
        )
        local_proc_file = os.path.join(
            lasif_root, "PROCESSED_DATA", "EARTHQUAKES", event, proc_filename
        )

        remote_proc_file_name = f"{event}_{proc_filename}"
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_processed_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "PROCESSED_DATA"
        )
        remote_adj_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "ADJOINT_SOURCES"
        )
        remote_receiver_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "RECEIVERS"
        )
        for dir_name in [remote_processed_dir, remote_adj_dir, remote_receiver_dir]:
            if not hpc_cluster.remote_exists(dir_name):
                hpc_cluster.remote_mkdir(remote_processed_dir)

        remote_proc_path = os.path.join(remote_processed_dir, remote_proc_file_name)
        tmp_remote_path = remote_proc_path + "_tmp"
        if not hpc_cluster.remote_exists(remote_proc_path):
            hpc_cluster.remote_put(local_proc_file, tmp_remote_path)
            hpc_cluster.run_ssh_command(f"mv {tmp_remote_path} {remote_proc_path}")

        remote_adj_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "ADJOINT_SOURCES"
        )

        if "VPV" in self.comm.project.inversion_params:
            parameterization = "tti"
        elif "VP" in self.comm.project.inversion_params:
            parameterization = "rho-vp-vs"

        info = dict(
            processed_filename=remote_proc_path,
            synthetic_filename=remote_syn_path,
            forward_meta_json_filename=forward_meta_json_filename,
            parameterization=parameterization,
            event_name=event,
            delta=self.comm.project.simulation_dict["time_step"],
            npts=self.comm.project.simulation_dict["number_of_time_steps"],
            iteration_name=iteration,
            minimum_period=self.comm.project.min_period,
            maximum_period=self.comm.project.max_period,
            start_time_in_s=self.comm.project.simulation_dict["start_time"],
            receiver_json_path=os.path.join(
                remote_receiver_dir, f"{event}_receivers.json"
            ),
            ad_src_type=self.comm.project.ad_src_type,
        )

        toml_filename = f"{iteration}_{event}_adj_info.toml"
        with open(toml_filename, "w") as fh:
            toml.dump(info, fh)

        # Put info toml on daint and remove local toml
        remote_toml = os.path.join(remote_adj_dir, toml_filename)
        hpc_cluster.remote_put(toml_filename, remote_toml)
        os.remove(toml_filename)

        # Copy processing script to hpc
        remote_script = os.path.join(remote_adj_dir, "window_and_calc_adj_src.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(PROCESS_OUTPUT_SCRIPT_PATH, remote_script)

        # Now submit the job
        description = f"HPC processing of {event} for iteration {iteration}"

        # use interp wall time for now
        wall_time = self.comm.project.hpc_processing_wall_time
        from salvus.flow.sites import job, remote_io_site

        commands = [
            remote_io_site.site_utils.RemoteCommand(
                command="mkdir output", execute_with_mpi=False
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"python {remote_script} {remote_toml}", execute_with_mpi=False
            ),
        ]
        # Allow to set conda environment first
        if self.comm.project.remote_conda_env:
            conda_command = [
                remote_io_site.site_utils.RemoteCommand(
                    command=f"conda activate {self.comm.project.remote_conda_env}",
                    execute_with_mpi=False,
                )
            ]
            commands = conda_command + commands

        job = job.Job(
            site=sapi.get_site(self.comm.project.interpolation_site),
            commands=commands,
            job_type="hpc_processing",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )

        self.comm.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["name"]',
            new_value=job.job_name,
        )
        job.launch()
        self.comm.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["submitted"]',
            new_value=True,
        )
        self.print(f"HPC Processing job for event {event} submitted")
        self.comm.project.update_iteration_toml()

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

        self.comm.lasif.select_windows(window_set_name=window_set_name, event=event)

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
                if window_set in validation_dict[iteration]["events"][event].keys():
                    if validation_dict[iteration]["events"][event][window_set] != 0.0:
                        quantify_misfit = False

        if not quantify_misfit:
            message = (
                f"Will not quantify misfit for event {event}, "
                f"iteration {iteration} "
                f"window set {window_set}. If you want it computed, "
                f"change value in validation toml to 0.0"
            )
            self.print(message)

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
    ):
        """
        Compute Misfits and Adjoint sources

        :param event: Name of event
        :type event: str
        """
        if self.comm.project.is_validation_event(event):
            self.__validation_misfit_quantification(
                event=event, window_set=self.comm.project.current_iteration
            )
            return
        misfit = self.comm.lasif.misfit_quantification(event, window_set=window_set)

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
        :param hpc_processing: Use reomate adjoint file
        :type hpc_processing: bool
        """
        if self.comm.project.is_validation_event(event):
            return

        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return

        if verbose:
            self.print(
                "Run adjoint simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        if self.comm.project.meshes == "multi-mesh" or self.comm.project.hpc_processing:
            simulation_created_remotely = True
        else:
            simulation_created_remotely = False
        if simulation_created_remotely:
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation_from_dict(
                event
            )
        else:
            adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
                event, adj_src
            )

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
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
            self.print(
                "Process data if needed",
                line_above=True,
                emoji_alias=":floppy_disk:",
                color="green",
            )

        self.__process_data(event)

        # Skip window selection in case of validation data
        if windows and not self.comm.project.is_validation_event(event):
            if verbose:
                self.print(
                    "Select windows",
                    color="white",
                    line_above=True,
                    emoji_alias=":foggy:",
                )
            self.__select_windows(event)

        if verbose:
            self.print(
                "Quantify Misfit", color="magenta", line_above=True, emoji_alias=":zap:"
            )

        self.__misfit_quantification(event, window_set=window_set)

    def dispatch_forward_simulations(self, verbose):
        """
        Dispatches the forward events
        """

        if verbose and self.comm.project.prepare_forward:
            self.print(
                "Prepare forward Stage",
                line_above=True,
                emoji_alias=[
                    ":globe_with_meridians:",
                    ":point_right:",
                    ":globe_with_meridians:",
                ],
            )

        events = self.events

        if self.comm.project.prepare_forward:
            self.print("Will dispatch all prepare_forward jobs")
            for _i, event in enumerate(events):
                if verbose:
                    self.print(f"Event {_i+1}/{len(self.events)}:  {event}")
                self.__prepare_forward(event=event)
            self.print("All prepare_forward jobs have been dispatched")

            self.__listen_to_prepare_forward(events=events, verbose=verbose)

        #
        for _i, event in enumerate(events):
            self.__run_forward_simulation(event, verbose=verbose)
            self.__compute_station_weights(event, verbose)

    def __listen_to_prepare_forward(self, events, verbose):
        """
        Listens to prepare forward jobs and waits for them to be done.
        Also submits simulations.
        """
        vint_job_listener = RemoteJobListener(
            comm=self.comm, job_type="prepare_forward", events=events
        )
        while True:
            vint_job_listener.monitor_jobs()
            for event in vint_job_listener.events_retrieved_now:
                self.__run_forward_simulation(event, verbose=verbose)
                self.__compute_station_weights(event, verbose)
                self.comm.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
            for event in vint_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'prepare_forward_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.comm.salvus_flow.delete_stored_wavefields(
                    iteration=self.comm.project.current_iteration,
                    sim_type="prepare_forward",
                    event_name=event,
                )
                self.__prepare_forward(event=event)
            if len(vint_job_listener.events_retrieved_now) > 0:
                self.print(
                    f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                    "simulations"
                )
            if len(vint_job_listener.events_already_retrieved) + len(
                vint_job_listener.events_retrieved_now
            ) == len(self.events):
                break

            if not vint_job_listener.events_retrieved_now:
                sleep_or_process(self.comm)
            vint_job_listener.to_repost = []
            vint_job_listener.events_retrieved_now = []

    def __retrieve_forward_simulations(
        self,
        events,
        adjoint,
        windows,
        window_set,
        verbose,
    ):
        for_job_listener = RemoteJobListener(
            comm=self.comm, job_type="forward", events=events
        )
        hpc_proc_job_listener = RemoteJobListener(
            comm=self.comm,
            job_type="hpc_processing",
            events=self.comm.project.non_val_events_in_iteration,
        )
        while True:
            for_job_listener.monitor_jobs()
            # submit remote jobs for the ones that did not get
            # submitted yet, although forwards are done.
            for event in for_job_listener.events_already_retrieved:
                if (
                    self.comm.project.hpc_processing
                    and not self.comm.project.is_validation_event(event)
                ):
                    self._launch_hpc_processing_job(event)
            for event in for_job_listener.events_retrieved_now:
                # Still retrieve synthetics for validation data. NO QA
                if (
                    not self.comm.project.hpc_processing
                    or self.comm.project.is_validation_event(event)
                ):
                    self.__retrieve_seismograms(event=event, verbose=verbose)

                # Here I need to replace this with remote hpc job,
                # then this actually needs be finished before any adjoint
                # jobs are launched
                if (
                    self.comm.project.hpc_processing
                    and not self.comm.project.is_validation_event(event)
                ):
                    self._launch_hpc_processing_job(event)
                else:
                    self.__work_with_retrieved_seismograms(
                        event,
                        windows,
                        window_set,
                        verbose,
                    )
                self.comm.project.change_attribute(
                    attribute=f'forward_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()
                if adjoint and not self.comm.project.hpc_processing:
                    self.__dispatch_adjoint_simulation(event, verbose)
            for event in for_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'forward_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.comm.salvus_flow.delete_stored_wavefields(
                    iteration=self.comm.project.current_iteration,
                    sim_type="forward",
                    event_name=event,
                )
                self.__run_forward_simulation(event=event)
            if len(for_job_listener.events_retrieved_now) > 0:
                self.print(
                    f"Retrieved {len(for_job_listener.events_retrieved_now)} "
                    "seismograms"
                )
            if (
                len(for_job_listener.events_retrieved_now)
                + len(for_job_listener.events_already_retrieved)
                == len(events)
                and not self.comm.project.hpc_processing
            ):
                break

            if (
                self.comm.project.hpc_processing
                and adjoint
                and len(for_job_listener.events_already_retrieved) > 0
            ):
                hpc_proc_job_listener.monitor_jobs()
                for event in hpc_proc_job_listener.events_retrieved_now:
                    self.comm.project.change_attribute(
                        attribute=f'hpc_processing_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
                    if adjoint and self.comm.project.hpc_processing:
                        self.__dispatch_adjoint_simulation(event, verbose)

                for event in hpc_proc_job_listener.to_repost:
                    self.comm.project.change_attribute(
                        attribute=f'hpc_processing_job["{event}"]["submitted"]',
                        new_value=False,
                    )
                    self.comm.project.update_iteration_toml()
                    self.comm.salvus_flow.delete_stored_wavefields(
                        iteration=self.comm.project.current_iteration,
                        sim_type="hpc_processing",
                        event_name=event,
                    )
                    self._launch_hpc_processing_job(event)
                if len(hpc_proc_job_listener.events_retrieved_now) + len(
                    hpc_proc_job_listener.events_already_retrieved
                ) == len(self.comm.project.non_val_events_in_iteration):
                    break

                hpc_proc_job_listener.to_repost = []
                hpc_proc_job_listener.events_retrieved_now = []

            if (
                not for_job_listener.events_retrieved_now
                and not hpc_proc_job_listener.events_retrieved_now
            ):
                sleep_or_process(self.comm)

            for_job_listener.to_repost = []
            for_job_listener.events_retrieved_now = []

            hpc_proc_job_listener.to_repost = []
            hpc_proc_job_listener.events_retrieved_now = []


class AdjointHelper(object):
    """
    A class assisting with everything related to the adjoint simulations

    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def print(
        self,
        message: str,
        color="cyan",
        line_above=False,
        line_below=False,
        emoji_alias=None,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

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
            events=events,
            interpolate=interpolate,
            verbose=verbose,
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
        return job_info["submitted"], job_info["retrieved"]

    def __process_gradients(self, events: list, interpolate: bool, verbose: bool):

        adj_job_listener = RemoteJobListener(
            comm=self.comm, job_type="adjoint", events=events
        )

        interp_job_listener = RemoteJobListener(
            comm=self.comm, job_type="gradient_interp", events=events
        )

        while True:
            adj_job_listener.monitor_jobs()
            for event in adj_job_listener.events_retrieved_now:
                if not (
                    self.comm.project.meshes == "multi-mesh"
                    and self.comm.project.interpolation_mode == "remote"
                ):
                    self.__cut_and_clip_gradient(event=event, verbose=verbose)
                if interpolate:
                    if self.comm.project.interpolation_mode == "remote":
                        self.__dispatch_raw_gradient_interpolation(
                            event, verbose=verbose
                        )
                self.comm.project.change_attribute(
                    attribute=f'adjoint_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()

            for event in adj_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'adjoint_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.comm.salvus_flow.delete_stored_wavefields(
                    iteration=self.comm.project.current_iteration,
                    sim_type="adjoint",
                    event_name=event,
                )
                self.__dispatch_adjoint_simulation(event=event, verbose=verbose)

            if interpolate and len(adj_job_listener.events_already_retrieved) > 0:
                interp_job_listener.monitor_jobs()
                for event in interp_job_listener.events_retrieved_now:
                    self.comm.project.change_attribute(
                        attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
                    self.comm.project.update_iteration_toml()

                for event in interp_job_listener.to_repost:
                    self.comm.project.change_attribute(
                        attribute=f'gradient_interp_job["{event}"]["submitted"]',
                        new_value=False,
                    )
                    self.comm.project.update_iteration_toml()
                    self.comm.salvus_flow.delete_stored_wavefields(
                        iteration=self.comm.project.current_iteration,
                        sim_type="gradient_interp",
                        event_name=event,
                    )
                    self.__dispatch_raw_gradient_interpolation(event)
                interp_job_listener.events_retrieved_now = []
                interp_job_listener.to_repost = []
            # Making sure we don't wait if everything is retrieved already
            if len(adj_job_listener.events_already_retrieved) + len(
                adj_job_listener.events_retrieved_now
            ) == len(events):
                break

            if (
                not adj_job_listener.events_retrieved_now
                and not interp_job_listener.events_retrieved_now
            ):
                sleep_or_process(self.comm)

            adj_job_listener.to_repost = []
            adj_job_listener.events_retrieved_now = []

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose=False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid.
        """
        submitted, retrieved = self.__submitted_retrieved(event, "gradient_interp")
        if submitted:
            if verbose:
                self.print(
                    f"Interpolation for gradient {event} " "has already been submitted"
                )
            return
        hpc_cluster = get_site(self.comm.project.interpolation_site)
        if hpc_cluster.config["site_type"] == "local":
            interp_folder = os.path.join(
                self.comm.project.remote_inversionson_dir,
                "INTERPOLATION_WEIGHTS",
                "GRADIENTS",
                event,
            )
        else:
            interp_folder = os.path.join(
                self.comm.project.remote_inversionson_dir,
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
        Dispatch an adjoint simulation after finishing the forward
        processing

        :param event: Name of event
        :type event: str
        :param hpc_processing: Use reomate adjoint file
        :type hpc_processing: bool
        """
        if self.comm.project.is_validation_event(event):
            return

        submitted, retrieved = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return

        if verbose:
            self.print(
                "Run adjoint simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        if self.comm.project.meshes == "multi-mesh" or self.comm.project.hpc_processing:
            simulation_created_remotely = True
        else:
            simulation_created_remotely = False
        if simulation_created_remotely:
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation_from_dict(
                event
            )
        else:
            adj_src = self.comm.salvus_flow.get_adjoint_source_object(event)
            w_adjoint = self.comm.salvus_flow.construct_adjoint_simulation(
                event, adj_src
            )

        self.comm.salvus_flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.comm.project.site_name,
            ranks=self.comm.project.ranks,
        )
        self.comm.project.change_attribute(
            attribute=f'adjoint_job["{event}"]["submitted"]', new_value=True
        )
        self.comm.project.update_iteration_toml()

    def __cut_and_clip_gradient(self, event, verbose=False):
        """
        Cut sources and receivers from gradient before summing or potential
        smoothing.
        We also clip the gradient to some percentile
        This can all be configured in information toml.

        :param event: name of the event
        """
        job = self.comm.salvus_flow.get_job(event, "adjoint")
        output_files = job.get_output_files()
        gradient_path = output_files[0][("adjoint", "gradient", "output_filename")]
        # Connect to daint
        hpc_cluster = get_site(self.comm.project.site_name)

        remote_inversionson_dir = os.path.join(
            self.comm.project.remote_inversionson_dir, "GRADIENT_PROCESSING"
        )

        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        # copy processing script to hpc
        remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(CUT_SOURCE_SCRIPT_PATH, remote_script)

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
        print(hpc_cluster.run_ssh_command(f"python {remote_script} {remote_toml}"))
