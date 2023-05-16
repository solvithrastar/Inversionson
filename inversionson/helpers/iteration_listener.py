from __future__ import annotations
import os
import time
from typing import Dict, Optional, List, Tuple, Union
import toml

import json
from pathlib import Path
import salvus.flow.api as sapi  # type: ignore

from inversionson.helpers.remote_job_listener import RemoteJobListener
from salvus.flow.sites import job, remote_io_site  # type: ignore
from inversionson.utils import (
    get_misfits_filename,
    get_window_filename,
)
from inversionson import InversionsonError
from inversionson.project import Project

_CUT_SOURCE_SCRIPT_PATH = (
    Path(__file__).parent.parent / "remote_scripts" / "cut_and_clip.py"
)
_PROCESS_OUTPUT_SCRIPT_PATH = (
    Path(__file__).parent.parent / "remote_scripts" / "window_and_calc_adj_src.py"
)

if __name__ == "__main__":
    print(_PROCESS_OUTPUT_SCRIPT_PATH)


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
        if prev_control_group_events is not None:
            assert prev_iteration is not None
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
        color: str = "yellow",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def _prepare_forward(self, event: str) -> None:
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

    def __submitted_retrieved(
        self, event: str, sim_type: str = "forward"
    ) -> Tuple[bool, bool]:
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

    def _run_forward(self, event: str, verbose: bool = False) -> None:
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

        w = self.project.flow.forward_simulation_from_dict(event)

        # Get the average model when validation event

        if self.project.config.meshing.multi_mesh:
            remote_model = self.project.remote_paths.get_event_specific_model(event)
        else:
            remote_model = self.project.remote_paths.get_master_model_path()

        # remote_mesh = self.project.remote_paths.get_remote_master_model_path
        w.set_mesh(f"REMOTE:{str(remote_model)}")

        self.project.flow.submit_job(
            event=event,
            simulation=w,
            sim_type="forward",
            site=self.project.config.hpc.sitename,
            ranks=self.project.config.hpc.n_wave_ranks,
        )

        self.print(f"Submitted forward job for event: {event}")

    def __retrieve_seismograms(self, event: str, verbose: bool = False) -> None:
        self.project.flow.retrieve_seismograms(event_name=event)
        if verbose:
            self.print(f"Copied seismograms for event {event} to lasif folder")

    def __process_data(self, event: str) -> None:
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

    def _launch_hpc_processing_job(self, event: str) -> None:
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
            f"preprocessed_{int(self.project.lasif_settings.min_period)}s_"
            f"to_{int(self.project.lasif_settings.max_period)}s.h5"
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
            assert self.prev_iteration is not None
            windowing_needed = False
            window_path = remote_window_dir / get_window_filename(
                event, self.prev_iteration
            )
            new_window_path = remote_window_dir / get_window_filename(event, iteration)
            # copy the windows over to ensure it works in the future.
            hpc_cluster.run_ssh_command(f"cp {window_path} {new_window_path}")
        else:
            windowing_needed = True
            window_path = remote_window_dir / get_window_filename(event, iteration)

        misfits_path = self.project.remote_paths.misfit_dir / get_misfits_filename(
            event, iteration
        )
        info = dict(
            processed_filename=remote_proc_path,
            synthetic_filename=remote_syn_path,
            forward_meta_json_filename=forward_meta_json_filename,
            parameterization=parameterization,
            windowing_needed=windowing_needed,
            window_path=str(window_path),
            event_name=event,
            delta=self.project.lasif_settings.time_step,
            npts=self.project.lasif_settings.number_of_time_steps,
            iteration_name=iteration,
            misfit_json_filename=str(misfits_path),
            minimum_period=self.project.lasif_settings.min_period,
            maximum_period=self.project.lasif_settings.max_period,
            start_time_in_s=self.project.lasif_settings.start_time,
            receiver_json_path=str(
                self.project.remote_paths.receiver_dir / f"{event}_receivers.json"
            ),
            ad_src_type=self.project.ad_src_type,
        )

        toml_filename = f"{iteration}_{event}_adj_info.toml"
        remote_toml = self.project.remote_paths.adj_src_dir / toml_filename
        self._write_and_upload_toml(toml_filename, info, remote_toml)
        # Copy processing script to hpc
        remote_script = os.path.join(
            self.project.remote_paths.adj_src_dir, "window_and_calc_adj_src.py"
        )
        if not hpc_cluster.remote_exists(remote_script):
            print(remote_script)
            print(_PROCESS_OUTPUT_SCRIPT_PATH)
            hpc_cluster.remote_put(_PROCESS_OUTPUT_SCRIPT_PATH, remote_script)

        # Now submit the job
        description = f"HPC processing of {event} for iteration {iteration}"

        wall_time = self.project.config.hpc.proc_wall_time

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

        j = job.Job(
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
            new_value=j.job_name,
        )
        j.launch()
        self.project.change_attribute(
            attribute=f'hpc_processing_job["{event}"]["submitted"]',
            new_value=True,
        )
        self.print(f"HPC Processing job for event {event} submitted")

    def _misfit_quantification(
        self,
        event: str,
    ) -> None:
        """
        Compute Misfits and Adjoint sources

        :param event: Name of event
        :type event: str
        """
        if self.project.is_validation_event(event):
            self.project.lasif.calculate_validation_misfit(event)
            self.project.storyteller.report_validation_misfit(
                iteration=self.project.current_iteration,
                event=event,
            )

    def __dispatch_adjoint_simulation(self, event: str, verbose: bool = False) -> None:
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
    ) -> None:
        """
        Process data, select windows, compute adjoint sources

        :param event: Name of event
        :type event: str
        :param windows: Should windows be selected?
        :type windows: bool
        """
        self.__process_data(event)
        self._misfit_quantification(event)

    def __listen_to_prepare_forward(
        self, events: List[str], verbose: bool
    ) -> Tuple[bool, List[str]]:
        """
        Listens to prepare forward jobs and waits for them to be done.
        Also submits simulations.
        """
        vint_job_listener = RemoteJobListener(
            project=self.project, job_type="prepare_forward", events=events
        )
        vint_job_listener.monitor_jobs()
        for event in vint_job_listener.events_retrieved_now:
            self._run_forward(event, verbose=verbose)
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
            self._prepare_forward(event=event)
        if len(vint_job_listener.events_retrieved_now) > 0:
            self.print(
                f"We dispatched {len(vint_job_listener.events_retrieved_now)} "
                "simulations"
            )
        # In the case where the processed data already exists, it will be
        # retrieved without ever entering the events_retrieved_now loop.
        # In that case we should still submit these jobs.
        for event in vint_job_listener.events_already_retrieved:
            self._run_forward(event, verbose=verbose)

        anything_retrieved = bool(vint_job_listener.events_retrieved_now)
        return anything_retrieved, vint_job_listener.events_already_retrieved

    def _listen_to_forward(
        self,
        events: List[str],
        windows: bool = True,
        verbose: bool = False,
    ) -> Tuple[bool, List[str]]:

        for_job_listener = RemoteJobListener(
            project=self.project, job_type="forward", events=events
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
            self._run_forward(event=event)
        if len(for_job_listener.events_retrieved_now) > 0:
            self.print(
                f"Retrieved {len(for_job_listener.events_retrieved_now)} " "seismograms"
            )

        anything_retrieved = bool(len(for_job_listener.events_retrieved_now))
        return anything_retrieved, for_job_listener.events_already_retrieved

    def __listen_to_hpc_processing(
        self, events: List[str], adjoint: bool = True, verbose: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Here we listen hpc_processing. It is important that only
        events enter here that actually need to be listened to.
        So no validation events for example.
        """
        iteration = self.project.current_iteration
        hpc_proc_job_listener = RemoteJobListener(
            project=self.project,
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
                / get_misfits_filename(event, iteration)
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

    def _listen_to_adjoint(
        self, events: List[str], verbose: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Here we listen to the adjoint jobs.
        Again it is important that only candidate events enter here.
        So no validation events.
        """
        adj_job_listener = RemoteJobListener(
            project=self.project, job_type="adjoint", events=events
        )

        adj_job_listener.monitor_jobs()
        for event in adj_job_listener.events_retrieved_now:
            self._cut_and_clip_gradient(event=event)
            self.project.change_attribute(
                attribute=f'adjoint_job["{event}"]["retrieved"]',
                new_value=True,
            )
            if self.project.config.meshing.multi_mesh:
                self._dispatch_raw_gradient_interpolation(event, verbose=verbose)
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

    def listen_to_gradient_interp(self, events: List[str]) -> Tuple[bool, List[str]]:
        """
        Monitor the status of the interpolations.

        It is important that only candidate events enter here
        """
        assert not self.project.config.meshing.multi_mesh

        int_job_listener = RemoteJobListener(
            project=self.project,
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
            self._dispatch_raw_gradient_interpolation(event=event)

        for event in int_job_listener.not_submitted:
            self._dispatch_raw_gradient_interpolation(event=event)

        anything_retrieved = bool(int_job_listener.events_retrieved_now)
        return anything_retrieved, int_job_listener.events_already_retrieved

    def listen(self, verbose: bool = False) -> None:
        """
        Listen to all steps in the iteration.
        It will listen to prepare forward (if needed) and forward
        for all jobs. And it will listen to hpc proc (if needed) and
        adjoint, and gradient interp (if needed) jobs.
        """
        # TODO when nothing gets submitted it gets stuck here. needs a fix
        # Initialize variables
        all_pf_retrieved_events: List[str] = []
        all_f_retrieved_events: List[str] = []
        all_hpc_proc_retrieved_events: List[str] = []
        all_adj_retrieved_events: List[str] = []
        all_gi_retrieved_events: List[str] = []

        num_events = len(self.events)
        non_validation_events = list(
            set(self.events) - set(self.project.config.monitoring.validation_dataset)
        )
        num_non_validation_events = len(non_validation_events)

        # Stage data
        self.project.mesh.move_model_to_cluster
        self.project.lasif.upload_stf(iteration=self.project.current_iteration)

        while True:
            any_retrieved_pf = False
            any_retrieved_f = False
            any_retrieved_hpc_proc = False
            any_adj_retrieved = False
            any_gi_retrieved = False
            any_retrieved = False
            any_checked = False

            if len(all_pf_retrieved_events) != num_events:
                for event in self.events:
                    self._prepare_forward(event)
                (
                    any_retrieved_pf,
                    all_pf_retrieved_events,
                ) = self.__listen_to_prepare_forward(
                    events=self.events, verbose=verbose
                )
                any_checked = True
            if any_retrieved_pf:
                any_retrieved = True
            # Then we listen to forward for the already retrieved events in
            # prepare_forward.
            # Here it actually
            # important that we dispatch everything first if there is no prepare
            # forward and that we then listen to all events.
            if (
                len(all_pf_retrieved_events) > 0
                and len(all_f_retrieved_events) != num_events
            ):
                (
                    any_retrieved_f,
                    all_f_retrieved_events,
                ) = self._listen_to_forward(all_pf_retrieved_events, verbose=verbose)
                any_checked = True
            if any_retrieved_f:
                any_retrieved = True

            ####################################################################
            # The rest is only for the non validation events.
            ####################################################################
            all_non_val_f_retrieved_events = list(
                set(all_f_retrieved_events)
                - set(self.project.config.monitoring.validation_dataset)
            )
            # Now we start listening to the hpc_proc jobs. Only for the ones that
            # finished and only if applicable.

            if (
                all_non_val_f_retrieved_events
                and len(all_hpc_proc_retrieved_events) != num_non_validation_events
            ):
                (
                    any_retrieved_hpc_proc,
                    all_hpc_proc_retrieved_events,
                ) = self.__listen_to_hpc_processing(
                    all_non_val_f_retrieved_events, adjoint=self.submit_adjoint
                )
                any_checked = True
            if any_retrieved_hpc_proc:
                any_retrieved = True

            # Now we start listening to the adjoint jobs. If hpc proc,
            # we only listen to the ones that finished there already.
            # Otherwise we listen to the ones that finished forward
            if not self.misfit_only:
                if (
                    len(all_hpc_proc_retrieved_events) > 0
                    and len(all_adj_retrieved_events) != num_non_validation_events
                ):
                    (
                        any_adj_retrieved,
                        all_adj_retrieved_events,
                    ) = self._listen_to_adjoint(all_hpc_proc_retrieved_events)
                    any_checked = True
                if any_adj_retrieved:
                    any_retrieved = True
                # Now we listen to the gradient interp jobs in the multi_mesh case
                # otherwise we are done here.

                if self.project.config.meshing.multi_mesh:
                    if (
                        len(all_adj_retrieved_events) > 0
                        and len(all_gi_retrieved_events) != num_non_validation_events
                    ):
                        (
                            any_gi_retrieved,
                            all_gi_retrieved_events,
                        ) = self.listen_to_gradient_interp(all_adj_retrieved_events)
                        any_checked = True
                        if len(all_gi_retrieved_events) == num_non_validation_events:
                            break
                    if any_gi_retrieved:
                        any_retrieved = True

                elif len(all_adj_retrieved_events) == num_non_validation_events:
                    break

            if not any_retrieved:
                print(
                    f"Waiting for {self.project.config.hpc.sleep_time_in_seconds} seconds."
                )
                time.sleep(self.project.config.hpc.sleep_time_in_seconds)
            if not any_checked:
                break

        # Finally update the estimated timestep
        for event in non_validation_events[:1]:
            self.project.find_simulation_time_step(event)

    def _dispatch_raw_gradient_interpolation(
        self, event: str, verbose: bool = False
    ) -> None:
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
        interp_folder = (
            self.project.remote_paths.interp_weights_dir / "GRADIENTS" / event
        )

        if not hpc_cluster.remote_exists(interp_folder):
            hpc_cluster.remote_mkdir(interp_folder)

        self.project.multi_mesh.interpolate_gradient_to_model(event)

    def _cut_and_clip_gradient(self, event: str) -> None:
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
        hpc_cluster = self.project.flow.hpc_cluster
        remote_inversionson_dir = self.project.remote_paths.gradient_proc_dir

        # copy processing script to hpc
        remote_script = os.path.join(remote_inversionson_dir, "cut_and_clip.py")
        if not hpc_cluster.remote_exists(remote_script):
            hpc_cluster.remote_put(_CUT_SOURCE_SCRIPT_PATH, remote_script)

        info = {
            "filename": str(gradient_path),
            "cutout_radius_in_km": self.project.config.inversion.source_cut_radius_in_km,
            "source_location": self.project.lasif.get_source(event_name=event),
        }
        info["clipping_percentile"] = self.project.config.inversion.clipping_percentile
        info["parameters"] = self.project.config.inversion.inversion_parameters

        toml_filename = f"{event}_gradient_process.toml"
        remote_toml = remote_inversionson_dir / toml_filename
        self._write_and_upload_toml(toml_filename, info, remote_toml)
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

    def _write_and_upload_toml(
        self, toml_filename: str, info_dict: Dict, remote_toml_path: Union[Path, str]
    ) -> None:
        """Write a dictionary ato toml and copy to the remote.
        Returns the path on the remote
        """
        with open(toml_filename, "w") as fh:
            toml.dump(info_dict, fh)
        self.project.flow.safe_put(toml_filename, remote_toml_path)
        os.remove(toml_filename)
