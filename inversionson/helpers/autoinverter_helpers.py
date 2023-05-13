from __future__ import annotations
from ast import List
import os
import time
from typing import Optional
import toml
import json
from pathlib import Path
import salvus.flow.api as sapi

from inversionson.helpers.remote_job_listener import RemoteJobListener
from inversionson.utils import (
    get_misfits_filename,
    get_window_filename,
)
from inversionson import InversionsonError
from inversionson.project import Project

__CUT_SOURCE_SCRIPT_PATH = Path(__file__).parent / "remote_scripts" / "cut_and_clip.py"
__PROCESS_OUTPUT_SCRIPT_PATH = (
    Path(__file__).parent / "remote_scripts" / "window_and_calc_adj_src.py"
)


class IterationListener(object):
    """
    Class that can handle an entire iteration until it's done.
    """

    def __init__(
        self,
        project: Project,
        events: List[str],
        misfit_only: bool = False,
        submit_adjoint: bool = True,
        control_group_events: Optional[List[str]] = None,
        prev_control_group_events: Optional[List[str]] = None,
        prev_iteration: Optional[str] = None,
    ):
        """
        Extension to include special cases for control group related stuff
        """
        if control_group_events is None:
            control_group_events = []
        if prev_control_group_events is None:
            prev_control_group_events = []
        self.project = project
        self.events = events
        self.control_group_events = control_group_events
        self.prev_control_group_events = prev_control_group_events
        self.misfit_only = misfit_only
        self.submit_adjoint = submit_adjoint
        self.prev_iteration = prev_iteration

    def print(
        self,
        message: str,
        color="yellow",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias=None,
    ):
        self.project.storyteller.printer.print(
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

        interp_folder = self.project.remote_paths.interp_weights_dir / "MODELS" / event
        if not self.project.flow.hpc_cluster.remote_exists(interp_folder):
            self.project.flow.hpc_cluster.remote_mkdir(interp_folder)

        self.project.multi_mesh.prepare_forward(event=event)

    def __submitted_retrieved(self, event: str, sim_type: str = "forward"):
        """
        Get a tuple of boolean values whether job as been submitted
        and retrieved

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation
        :type sim_type: str
        """
        if sim_type == "forward":
            job_info = self.project.forward_job[event]
        elif sim_type == "adjoint":
            job_info = self.project.adjoint_job[event]
        elif sim_type == "prepare_forward":
            job_info = self.project.prepare_forward_job[event]
        elif sim_type == "hpc_processing":
            job_info = self.project.hpc_processing_job[event]
        elif sim_type == "gradient_interp":
            job_info = self.project.gradient_interp_job[event]
        return job_info["submitted"], job_info["retrieved"]

    def __run_forward_simulation(self, event: str, verbose: bool = False):
        """
        Submit forward simulation to daint and possibly monitor aswell

        :param event: Name of event
        :type event: str
        """
        # Check status of simulation
        submitted, _ = self.__submitted_retrieved(event, sim_type="forward")
        if submitted:
            return

        if verbose:
            self.print(
                "Run forward simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        w = self.project.flow.construct_simulation_from_dict(event)
        already_interpolated = self.project.config.meshing.multi_mesh
        # Get the average model when validation event
        validation = bool(
            (
                self.project.is_validation_event(event)
                and self.project.config.monitoring.use_model_averaging
                and "00000" not in self.project.current_iteration
            )
        )
        hpc_cluster = self.project.flow.hpc_cluster

        remote_mesh = self.project.lasif.find_remote_mesh(
            event=event,
            gradient=False,
            interpolate_to=False,
            hpc_cluster=hpc_cluster,
            validation=validation,
            already_interpolated=already_interpolated,
        )

        w.set_mesh(f"REMOTE:{str(remote_mesh)}")

        self.project.flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.project.config.hpc.sitename,
            ranks=self.project.config.hpc.n_wave_ranks,
        )

        self.print(f"Submitted forward job for event: {event}")

    def __retrieve_seismograms(self, event: str, verbose: bool = False):
        self.project.flow.retrieve_outputs(event_name=event, sim_type="forward")
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
        self.project.lasif.process_data(event)

    def _launch_hpc_processing_job(self, event: str):
        """
        Here, we launch a job to select windows and get adjoint sources
        for an event.

        """
        submitted, _ = self.__submitted_retrieved(event, "hpc_processing")
        if submitted:
            return

        iteration = self.project.current_iteration
        forward_job = sapi.get_job(
            site_name=self.project.config.hpc.sitename,
            job_name=self.project.flow.get_job_name(event=event, sim_type="forward"),
        )

        # Get forward paths
        remote_syn_path = str(forward_job.output_path / "receivers.h5")
        forward_meta_json_filename = str(forward_job.output_path / "meta.json")

        # Get local proc filename
        lasif_root = self.project.config.lasif_root
        proc_filename = (
            f"preprocessed_{int(self.project.min_period)}s_"
            f"to_{int(self.project.max_period)}s.h5"
        )
        local_proc_file = os.path.join(
            lasif_root, "PROCESSED_DATA", "EARTHQUAKES", event, proc_filename
        )

        remote_proc_file_name = f"{event}_{proc_filename}"
        hpc_cluster = self.project.flow.hpc_cluster

        remote_proc_path = os.path.join(
            self.project.remote_paths.proc_data_dir, remote_proc_file_name
        )
        tmp_remote_path = f"{remote_proc_path}_tmp"
        if not hpc_cluster.remote_exists(remote_proc_path):
            hpc_cluster.remote_put(local_proc_file, tmp_remote_path)
            hpc_cluster.run_ssh_command(f"mv {tmp_remote_path} {remote_proc_path}")

        if "VPV" in self.project.config.inversion.inversion_parameters:
            parameterization = "tti"
        elif "VP" in self.project.config.inversion.inversion_parameters:
            parameterization = "rho-vp-vs"

        remote_window_dir = self.project.remote_paths.window_dir
        if event in self.prev_control_group_events:
            windowing_needed = False
            window_path = remote_window_dir / get_window_filename(
                event, self.prev_iteration
            )
            new_window_path = remote_window_dir / get_window_filename(event, iteration)
            # copy the windows over to ensure it works in the future.
            hpc_cluster.run_ssh_command(f"cp {window_path} {new_window_path}")
        else:
            windowing_needed = True
            window_path = (remote_window_dir / get_window_filename(event, iteration),)

        misfits_path = remote_window_dir / get_misfits_filename(event, iteration)
        info = dict(
            processed_filename=remote_proc_path,
            synthetic_filename=remote_syn_path,
            forward_meta_json_filename=forward_meta_json_filename,
            parameterization=parameterization,
            windowing_needed=windowing_needed,
            window_path=window_path,
            event_name=event,
            delta=self.project.simulation_settings.time_step,
            npts=self.project.simulation_settings.number_of_time_steps,
            iteration_name=iteration,
            misfit_json_filename=misfits_path,
            minimum_period=self.project.simulation_settings.min_period,
            maximum_period=self.project.simulation_settings.max_period,
            start_time_in_s=self.project.simulation_settings.start_time,
            receiver_json_path=os.path.join(
                self.project.remote_paths.receiver_dir, f"{event}_receivers.json"
            ),
            ad_src_type=self.project.ad_src_type,
        )

        toml_filename = f"{iteration}_{event}_adj_info.toml"
        remote_toml = self._upload_toml_to_remote(
            toml_filename, info, self.project.remote_paths.adj_src_dir, hpc_cluster
        )
        # Copy processing script to hpc
        remote_script = os.path.join(
            self.project.remote_paths.adj_src_dir, "window_and_calc_adj_src.py"
        )
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(__PROCESS_OUTPUT_SCRIPT_PATH, remote_script)

        # Now submit the job
        description = f"HPC processing of {event} for iteration {iteration}"

        wall_time = self.project.config.hpc.proc_wall_time
        from salvus.flow.sites import job, remote_io_site

        commands = [
            remote_io_site.site_utils.RemoteCommand(
                command="mkdir output", execute_with_mpi=False
            ),
            remote_io_site.site_utils.RemoteCommand(
                command=f"python {remote_script} {remote_toml}", execute_with_mpi=False
            ),
        ]

        if self.project.config.hpc.conda_env_name:
            conda_command = [
                remote_io_site.site_utils.RemoteCommand(
                    command=f"conda activate {self.project.config.hpc.conda_env_name}",
                    execute_with_mpi=False,
                )
            ]
            commands = conda_command + commands
            if self.project.config.hpc.conda_location:
                source_command = [
                    remote_io_site.site_utils.RemoteCommand(
                        command=f"source {self.project.config.hpc.conda_location}",
                        execute_with_mpi=False,
                    )
                ]
                commands = source_command + commands

        job = job.Job(
            site=sapi.get_site(self.project.config.hpc.sitename),
            commands=commands,
            job_type="hpc_processing",
            job_description=description,
            job_info={},
            wall_time_in_seconds=wall_time,
            no_db=False,
        )

        self.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["name"]',
            new_value=job.job_name,
        )
        job.launch()
        self.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["submitted"]',
            new_value=True,
        )
        self.print(f"HPC Processing job for event {event} submitted")

    def __select_windows(self, event: str):
        """
        Select the windows for the event and the iteration

        :param event: Name of event
        :type event: str
        """
        iteration = self.project.current_iteration
        window_set_name = f"{iteration}_{event}"
        self.project.lasif.select_windows(window_set_name=window_set_name, event=event)

    def __need_misfit_quantification(
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
        validation_dict = self.project.storyteller.validation_dict

        quantify_misfit = (
            iteration not in validation_dict.keys()
            or event not in validation_dict[iteration]["events"].keys()
            or validation_dict[iteration]["events"][event] == 0.0
        )
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

        iteration = self.project.current_iteration

        if self.__need_misfit_quantification(
            iteration=iteration, event=event, window_set=window_set
        ):
            self.project.lasif.misfit_quantification(
                event, validation=True, window_set=window_set
            )
            self.project.storyteller.report_validation_misfit(
                iteration=iteration,
                event=event,
                total_sum=False,
            )

            self.project.storyteller.report_validation_misfit(
                iteration=self.project.current_iteration,
                event=event,
                total_sum=True,
            )

    def __misfit_quantification(
        self,
        event: str,
        window_set: Optional[str] = None,
    ):
        """
        Compute Misfits and Adjoint sources

        :param event: Name of event
        :type event: str
        """
        if self.project.is_validation_event(event):
            self.__validation_misfit_quantification(
                event=event, window_set=self.project.current_iteration
            )
            return
        misfit = self.project.lasif.misfit_quantification(event, window_set=window_set)

        self.project.change_attribute(attribute=f'misfits["{event}"]', new_value=misfit)

    def __dispatch_adjoint_simulation(self, event: str, verbose: bool = False):
        """
        Dispatch an adjoint simulation after finishing the forward
        processing

        :param event: Name of event
        :type event: str
        :param hpc_processing: Use reomate adjoint file
        :type hpc_processing: bool
        """
        if self.project.is_validation_event(event):
            return

        submitted, _ = self.__submitted_retrieved(event, "adjoint")
        if submitted:
            return

        if verbose:
            self.print(
                "Run adjoint simulation", line_above=True, emoji_alias=":rocket:"
            )
            self.print(f"Event: {event}")

        w_adjoint = self.project.flow.construct_adjoint_simulation_from_dict(event)
        self.project.flow.submit_job(
            event=event,
            simulation=w_adjoint,
            sim_type="adjoint",
            site=self.project.config.hpc.sitename,
            ranks=self.project.config.hpc.n_wave_ranks,
        )
        self.project.change_attribute(
            attribute=f'adjoint_job["{event}"]["submitted"]', new_value=True
        )

    def __work_with_retrieved_seismograms(
        self,
        event: str,
        windows: bool,
        window_set: str,
        verbose: bool = False,
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
        if windows and not self.project.is_validation_event(event):
            if verbose:
                self.print(
                    "Select windows",
                    color="cyan",
                    line_above=True,
                    emoji_alias=":foggy:",
                )
            self.__select_windows(event)

        if verbose:
            self.print(
                "Quantify Misfit", color="magenta", line_above=True, emoji_alias=":zap:"
            )

        self.__misfit_quantification(event, window_set=window_set)

    def dispatch_forward_simulations(self, verbose: bool):
        """
        Dispatches the forward events
        """

        if verbose:
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

        self.print("Will dispatch all prepare_forward jobs")
        for _i, event in enumerate(events):
            if verbose:
                self.print(f"Event {_i+1}/{len(self.events)}:  {event}")
            self.__prepare_forward(event=event)
        self.print("All prepare_forward jobs have been dispatched")
        self.__listen_to_prepare_forward(events=events, verbose=verbose)

        for event in events:
            self.__run_forward_simulation(event, verbose=verbose)

    def __listen_to_prepare_forward(self, events: List[str], verbose: bool):
        """
        Listens to prepare forward jobs and waits for them to be done.
        Also submits simulations.
        """
        vint_job_listener = RemoteJobListener(
            comm=self.project, job_type="prepare_forward", events=events
        )
        vint_job_listener.monitor_jobs()
        for event in vint_job_listener.events_retrieved_now:
            self.__run_forward_simulation(event, verbose=verbose)
            self.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["retrieved"]',
                new_value=True,
            )
            vint_job_listener.events_already_retrieved.append(event)

        for event in vint_job_listener.to_repost:
            self.project.change_attribute(
                attribute=f'prepare_forward_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.project.flow.delete_stored_wavefields(
                iteration=self.project.current_iteration,
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

        anything_retrieved = bool(vint_job_listener.events_retrieved_now)
        return anything_retrieved, vint_job_listener.events_already_retrieved

    def __listen_to_forward(
        self,
        events: List[str],
        windows: bool = True,
        window_set: Optional[str] = None,
        verbose: bool = False,
    ):

        for_job_listener = RemoteJobListener(
            comm=self.project, job_type="forward", events=events
        )
        for_job_listener.monitor_jobs()

        # submit remote jobs for the ones that did not get
        # submitted yet, although forwards are done.
        for event in for_job_listener.events_already_retrieved:
            if not self.project.is_validation_event(event):
                self._launch_hpc_processing_job(event)

        for event in for_job_listener.events_retrieved_now:
            # Still retrieve synthetics for validation data. NO QA
            if self.project.is_validation_event(event):
                self.__retrieve_seismograms(event=event, verbose=verbose)

            # Here I need to replace this with remote hpc job,
            # then this actually needs be finished before any adjoint
            # jobs are launched
            if not self.project.is_validation_event(event):
                self._launch_hpc_processing_job(event)
            else:
                self.__work_with_retrieved_seismograms(
                    event,
                    windows,
                    window_set,
                    verbose,
                )
            self.project.change_attribute(
                attribute=f'forward_job["{event}"]["retrieved"]',
                new_value=True,
            )
            for_job_listener.events_already_retrieved.append(event)
        for event in for_job_listener.to_repost:
            self.project.change_attribute(
                attribute=f'forward_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.project.flow.delete_stored_wavefields(
                iteration=self.project.current_iteration,
                sim_type="forward",
                event_name=event,
            )
            self.__run_forward_simulation(event=event)
        if len(for_job_listener.events_retrieved_now) > 0:
            self.print(
                f"Retrieved {len(for_job_listener.events_retrieved_now)} " "seismograms"
            )

        anything_retrieved = bool(len(for_job_listener.events_retrieved_now))
        return anything_retrieved, for_job_listener.events_already_retrieved

    def __listen_to_hpc_processing(
        self, events: List[str], adjoint: bool = True, verbose: bool = False
    ):
        """
        Here we listen hpc_processing. It is important that only
        events enter here that actually need to be listened to.
        So no validation events for example.
        """
        iteration = self.project.current_iteration
        hpc_proc_job_listener = RemoteJobListener(
            comm=self.project,
            job_type="hpc_processing",
            events=events,
        )
        hpc_proc_job_listener.monitor_jobs()
        hpc_cluster = self.project.flow.hpc_cluster

        for event in hpc_proc_job_listener.events_retrieved_now:
            self.project.change_attribute(
                attribute=f'hpc_processing_job["{event}"]["retrieved"]',
                new_value=True,
            )
            # TODO, we need to retrieve the misfit here
            remote_misfits = (
                self.project.remote_paths.misfit_dir
                / get_misfits_filename(event, iteration),
            )

            tmp_filename = "tmp_misfits.json"
            hpc_cluster.remote_get(remote_misfits, "tmp_misfits.json")
            with open(tmp_filename, "r") as fh:
                misfit_dict = json.load(fh)

            self.project.change_attribute(
                attribute=f'misfits["{event}"]',
                new_value=misfit_dict[event]["total_misfit"],
            )

            if adjoint:
                self.__dispatch_adjoint_simulation(event, verbose)
            hpc_proc_job_listener.events_already_retrieved.append(event)

        for event in hpc_proc_job_listener.to_repost:
            self.project.change_attribute(
                attribute=f'hpc_processing_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.project.flow.delete_stored_wavefields(
                iteration=self.project.current_iteration,
                sim_type="hpc_processing",
                event_name=event,
            )
            self._launch_hpc_processing_job(event)

        anything_retrieved = bool(hpc_proc_job_listener.events_retrieved_now)
        return anything_retrieved, hpc_proc_job_listener.events_already_retrieved

    def __listen_to_adjoint(self, events: List[str], verbose: bool = False):
        """
        Here we listen to the adjoint jobs.
        Again it is important that only candidate events enter here.
        So no validation events.
        """
        multi_mesh = self.project.config.meshing.multi_mesh
        adj_job_listener = RemoteJobListener(
            comm=self.project, job_type="adjoint", events=events
        )

        adj_job_listener.monitor_jobs()
        for event in adj_job_listener.events_retrieved_now:
            if not multi_mesh or self.project.interpolation_mode != "remote":
                self.__cut_and_clip_gradient(event=event, verbose=verbose)
            self.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["retrieved"]',
                new_value=True,
            )
            if multi_mesh and self.project.interpolation_mode == "remote":
                self.__dispatch_raw_gradient_interpolation(event, verbose=verbose)
            adj_job_listener.events_already_retrieved.append(event)

        for event in adj_job_listener.to_repost:
            self.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.project.flow.delete_stored_wavefields(
                iteration=self.project.current_iteration,
                sim_type="adjoint",
                event_name=event,
            )
            self.__dispatch_adjoint_simulation(event=event, verbose=verbose)

        for event in adj_job_listener.not_submitted:
            self.__dispatch_adjoint_simulation(event=event, verbose=verbose)
            self.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["submitted"]',
                new_value=True,
            )

        anything_retrieved = bool(adj_job_listener.events_retrieved_now)
        return anything_retrieved, adj_job_listener.events_already_retrieved

    def __listen_to_gradient_interp(self, events: List[str], verbose: bool = False):
        """
        Monitor the status of the interpolations.

        It is important that only candidate events enter here
        """
        assert not self.project.config.meshing.multi_mesh

        int_job_listener = RemoteJobListener(
            comm=self.project,
            job_type="gradient_interp",
            events=events,
        )

        int_job_listener.monitor_jobs()
        for event in int_job_listener.events_retrieved_now:
            self.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                new_value=True,
            )
            int_job_listener.events_already_retrieved.append(event)

        for event in int_job_listener.to_repost:
            self.project.change_attribute(
                attribute=f'gradient_interp_job["{event}"]["submitted"]',
                new_value=False,
            )
            self.__dispatch_raw_gradient_interpolation(event=event)

        for event in int_job_listener.not_submitted:
            self.__dispatch_raw_gradient_interpolation(event=event)

        anything_retrieved = bool(int_job_listener.events_retrieved_now)
        return anything_retrieved, int_job_listener.events_already_retrieved

    def listen(self, verbose: bool = False):
        """
        Listen to all steps in the iteration.
        It will listen to prepare forward (if needed) and forward
        for all jobs. And it will listen to hpc proc (if needed) and
        adjoint, and gradient interp (if needed) jobs.
        """
        # TODO when nothing gets submitted it gets stuck here. needs a fix
        # Initialize variables
        all_pf_retrieved_events = []
        all_f_retrieved_events = []
        all_hpc_proc_retrieved_events = []
        all_adj_retrieved_events = []
        all_gi_retrieved_events = []

        len_all_events = len(self.events)
        non_validation_events = list(
            set(self.events) - set(self.project.validation_dataset)
        )
        len_non_validation_events = len(non_validation_events)

        while True:
            anything_retrieved_pf = False
            anything_retrieved_f = False
            anything_retrieved_hpc_proc = False
            anything_adj_retrieved = False
            anything_gi_retrieved = False
            anything_retrieved = False
            anything_checked = False

            if len(all_pf_retrieved_events) != len_all_events:
                for event in self.events:
                    self.__prepare_forward(event)
                (
                    anything_retrieved_pf,
                    all_pf_retrieved_events,
                ) = self.__listen_to_prepare_forward(
                    events=self.events, verbose=verbose
                )
                anything_checked = True
            if anything_retrieved_pf:
                anything_retrieved = True
            # Then we listen to forward for the already retrieved events in
            # prepare_forward.
            # Here it actually
            # important that we dispatch everything first if there is no prepare
            # forward and that we then listen to all events.
            if (
                len(all_pf_retrieved_events) > 0
                and len(all_f_retrieved_events) != len_all_events
            ):
                (
                    anything_retrieved_f,
                    all_f_retrieved_events,
                ) = self.__listen_to_forward(all_pf_retrieved_events, verbose=verbose)
                anything_checked = True
            if anything_retrieved_f:
                anything_retrieved = True

            ####################################################################
            # The rest is only for the non validation events.
            ####################################################################
            all_non_val_f_retrieved_events = list(
                set(all_f_retrieved_events) - set(self.project.validation_dataset)
            )
            # Now we start listening to the hpc_proc jobs. Only for the ones that
            # finished and only if applicable.

            if (
                all_non_val_f_retrieved_events
                and len(all_hpc_proc_retrieved_events) != len_non_validation_events
            ):
                (
                    anything_retrieved_hpc_proc,
                    all_hpc_proc_retrieved_events,
                ) = self.__listen_to_hpc_processing(
                    all_non_val_f_retrieved_events, adjoint=self.submit_adjoint
                )
                anything_checked = True
            if anything_retrieved_hpc_proc:
                anything_retrieved = True

            # Now we start listening to the adjoint jobs. If hpc proc,
            # we only listen to the ones that finished there already.
            # Otherwise we listen to the ones that finished forward
            if not self.misfit_only:
                if (
                    len(all_hpc_proc_retrieved_events) > 0
                    and len(all_adj_retrieved_events) != len_non_validation_events
                ):
                    (
                        anything_adj_retrieved,
                        all_adj_retrieved_events,
                    ) = self.__listen_to_adjoint(all_hpc_proc_retrieved_events)
                    anything_checked = True
                if anything_adj_retrieved:
                    anything_retrieved = True
                # Now we listen to the gradient interp jobs in the multi_mesh case
                # otherwise we are done here.

                if self.project.meshes == "multi-mesh":
                    if (
                        len(all_adj_retrieved_events) > 0
                        and len(all_gi_retrieved_events) != len_non_validation_events
                    ):
                        (
                            anything_gi_retrieved,
                            all_gi_retrieved_events,
                        ) = self.__listen_to_gradient_interp(all_adj_retrieved_events)
                        anything_checked = True
                        if len(all_gi_retrieved_events) == len_non_validation_events:
                            break
                    if anything_gi_retrieved:
                        anything_retrieved = True

                elif len(all_adj_retrieved_events) == len_non_validation_events:
                    break

            if not anything_retrieved:
                time.sleep(self.project.config.hpc.sleep_time_in_seconds)
            if not anything_checked:
                break

        # Finally update the estimated timestep
        for event in non_validation_events[:1]:
            self.project.find_simulation_time_step(event)

    def __dispatch_raw_gradient_interpolation(self, event: str, verbose: bool = False):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid.
        """
        submitted, _ = self.__submitted_retrieved(event, "gradient_interp")
        if submitted:
            if verbose:
                self.print(
                    f"Interpolation for gradient {event} " "has already been submitted"
                )
            return
        hpc_cluster = self.project.flow.hpc_cluster
        interp_folder = os.path.join(
            self.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            "GRADIENTS",
            event,
        )
        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)

        self.project.multi_mesh.interpolate_gradient_to_model(
            event, smooth=False, interp_folder=interp_folder
        )

    def __cut_and_clip_gradient(self, event: str, verbose: bool = False):
        """
        Cut sources and receivers from gradient before summing or potential
        smoothing.
        We also clip the gradient to some percentile
        This can all be configured in information toml.

        :param event: name of the event
        """
        if (
            self.project.config.inversion.source_cut_radius_in_km == 0.0
            and self.project.config.inversion.clipping_percentile == 1.0
        ):
            return

        job = self.project.flow.get_job(event, "adjoint")
        output_files = job.get_output_files()
        gradient_path = output_files[0][("adjoint", "gradient", "output_filename")]
        # Connect to daint
        hpc_cluster = self.project.flow.hpc_cluster

        remote_inversionson_dir = os.path.join(
            self.project.config.hpc.inversionson_folder, "GRADIENT_PROCESSING"
        )
        if not hpc_cluster.remote_exists(remote_inversionson_dir):
            hpc_cluster.remote_mkdir(remote_inversionson_dir)

        # copy processing script to hpc
        remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(__CUT_SOURCE_SCRIPT_PATH, remote_script)

        info = {
            "filename": str(gradient_path),
            "cutout_radius_in_km": self.project.config.inversion.source_cut_radius_in_km,
            "source_location": self.project.lasif.get_source(event_name=event),
        }
        info["clipping_percentile"] = self.project.config.inversion.clipping_percentile
        info["parameters"] = self.project.config.inversion.inversion_parameters

        toml_filename = f"{event}_gradient_process.toml"
        remote_toml = self._upload_toml_to_remote(
            toml_filename, info, remote_inversionson_dir, hpc_cluster
        )
        # Call script
        _, stdout, stderr = hpc_cluster.run_ssh_command(
            f"python {remote_script} {remote_toml}"
        )
        if "Remote source cut completed successfully" in stdout[0]:
            self.print(
                f"Source cut and clip completed for {event}.", emoji_alias=":scissors:"
            )
        else:
            print("Something went wrong in cutting and clipping on the remote.")
            raise InversionsonError(stdout, stderr)

    def _upload_toml_to_remote(
        self, toml_filename, info_dict, remote_folder, hpc_cluster
    ) -> str:
        """Write a dictionary ato toml and copy to the remote.
        Returns the path on the remote
        """
        with open(toml_filename, "w") as fh:
            toml.dump(info_dict, fh)
        result = os.path.join(remote_folder, toml_filename)
        hpc_cluster.remote_put(toml_filename, result)
        os.remove(toml_filename)
        return result
