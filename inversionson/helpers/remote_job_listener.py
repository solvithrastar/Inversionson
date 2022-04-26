import warnings

from typing import Dict, List

import emoji
from tqdm import tqdm

from inversionson import InversionsonError, InversionsonWarning


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
        self.not_submitted = []
        if events is None:
            if job_type == "smoothing" and (
                self.comm.project.inversion_mode == "mono-batch"
            ):
                self.events = [None]
            else:
                self.events = self.comm.project.events_in_iteration
        else:
            self.events = events

    def print(
        self,
        message: str,
        color="white",
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

    def monitor_jobs(self, smooth_individual=False):
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
        elif self.job_type == "hpc_processing":
            job_dict = self.comm.project.hpc_processing_job
        else:
            job_dict = self.comm.project.smoothing_job
        if self.job_type in [
            "forward",
            "adjoint",
            "model_interp",
            "gradient_interp",
            "hpc_processing",
        ]:
            self.__monitor_jobs(job_dict=job_dict)
        elif self.job_type == "smoothing":
            self.__monitor_job_array(
                job_dict=job_dict, smooth_individual=smooth_individual
            )
        else:
            raise InversionsonError(f"Job type {self.job_type} not recognised")

    def __check_status_of_job(self, event: str, reposts: int, verbose: bool = False):
        """
        Query Salvus Flow for the status of the job

        :param event: Name of event
        :type event: str
        :param reposts: Number of reposts of the event for the job
        :type reposts: int
        """
        status = self.comm.salvus_flow.get_job_status(event, self.job_type).name
        if status == "pending":
            if verbose:
                self.print(f"Status = {status}, event: {event}")
        elif status == "running":
            if verbose:
                self.print(f"Status = {status}, event: {event}")
        elif status in ["unknown", "failed"]:
            self.print(f"{self.job_type} job for {event}, {status}, will resubmit")
            if reposts >= self.comm.project.max_reposts:
                self.print(
                    "No I've actually reposted this too often \n"
                    "There must be something wrong."
                )
                raise InversionsonError("Too many reposts")
            self.to_repost.append(event)
            reposts += 1
            self.comm.project.change_attribute(
                attribute=f'{self.job_type}_job["{event}"]["reposts"]',
                new_value=reposts,
            )
        elif status == "cancelled":
            self.print("What to do here?")
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
        i = 0
        for _i, s in enumerate(status):
            if s.name == "finished":
                params.append(s)
                finished += 1
            else:
                if s.name in ["pending", "running"]:
                    if verbose:
                        self.print(
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
                        self.print(f"Job {s.name}, will resubmit event {event}")
                        self.to_repost.append(event)
                        reposts += 1
                        if reposts >= self.comm.project.max_reposts:
                            print("No I've actually reposted this too often \n")
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
                    self.print(f"Job cancelled for event {event}")

                else:
                    warnings.warn(
                        f"Inversionson does not recognise job status:  {status}",
                        InversionsonWarning,
                    )
        if verbose:
            if running > 0:
                self.print(f"{running}/{len(status)} of jobs running: {event}")
            if pending > 0:
                self.print(f"{pending}/{len(status)} of jobs pending: {event}")
            if finished > 0:
                self.print(f"{finished}/{len(status)} of jobs finished: {event}")
        if len(params) == len(status):
            return "finished"

    def __monitor_jobs(self, job_dict: Dict, events: List[str] = None, verbose=False):
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
        self.print(
            f"Checking Jobs for {self.job_type}:", line_above=True, emoji_alias=":ear:"
        )
        for event in tqdm(
            events_left, desc=emoji.emojize(":ear: | ", use_aliases=True)
        ):
            if job_dict[event]["retrieved"]:
                self.events_already_retrieved.append(event)
                finished += 1
                continue
            else:
                reposts = job_dict[event]["reposts"]
                if not job_dict[event]["submitted"]:
                    status = "unsubmitted"
                    self.not_submitted.append(event)
                    continue
                status = self.__check_status_of_job(event, reposts, verbose=verbose)
            if status == "finished":
                self.events_retrieved_now.append(event)
                finished += 1
                if self.job_type == "gradient_interp":
                    self.comm.project.change_attribute(
                        attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                        new_value=True,
                    )
            elif status == "pending":
                pending += 1
            elif status == "running":
                running += 1

        if finished > 0:
            self.print(f"{finished}/{len(events)} jobs finished", emoji_alias=None)
        if running > 0:
            self.print(f"{running}/{len(events)} jobs running", emoji_alias=None)
        if pending > 0:
            self.print(f"{pending}/{len(events)} jobs pending", emoji_alias=None)

        self.comm.project.update_iteration_toml()

    def __monitor_job_array(self, job_dict, events=None, smooth_individual=False):
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
        if not smooth_individual:
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
            events_left = list(set(events) - set(self.events_already_retrieved))
            finished = len(self.events) - len(events_left)
            self.print(
                "Monitoring Smoothing jobs", line_above=True, emoji_alias=":ear:"
            )
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
            self.print("\n\n ============= Report ================= \n\n")
            self.print(f"{finished}/{len(events)} jobs fully finished \n")