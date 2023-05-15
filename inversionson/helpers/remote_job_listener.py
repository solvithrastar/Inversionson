import warnings

from typing import Dict, List, Optional

import emoji  # type: ignore
from tqdm import tqdm

from inversionson import InversionsonError, InversionsonWarning
from inversionson.project import Project


class RemoteJobListener(object):
    """
    Class designed to monitor the status of remote jobs.
    """

    def __init__(self, project: Project, job_type: str, events: List[str]):
        assert job_type in {
            "forward",
            "adjoint",
            "prepare_forward",
            "gradient_interp",
            "hpc_processing",
        }
        self.project = project

        self._job_type = job_type
        self.events_already_retrieved: List[str] = []
        self.events_retrieved_now: List[str] = []
        self.to_repost: List[str] = []
        self.not_submitted: List[str] = []
        self.events = events or self.project.events_in_iteration

    def print(
        self,
        message: str,
        color="cyan",
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias=None,
    ):
        if self._job_type == "prepare_forward":
            color = "lightred"
        if self._job_type == "forward":
            color = "yellow"
        if self._job_type == "hpc_processing":
            color = "green"
        if self._job_type == "adjoint":
            color = "blue"
        if self._job_type == "gradient_interp":
            color = "magenta"

        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def monitor_jobs(self) -> None:
        """
        Takes the job type of the object and monitors the status of
        all the events in the object.

        :raises InversionsonError: Error if job type not recognized
        """
        if self._job_type == "forward":
            job_dict = self.project.forward_job
        elif self._job_type == "adjoint":
            job_dict = self.project.adjoint_job
        elif self._job_type == "prepare_forward":
            job_dict = self.project.prepare_forward_job
        elif self._job_type == "gradient_interp":
            job_dict = self.project.gradient_interp_job
        elif self._job_type == "hpc_processing":
            job_dict = self.project.hpc_processing_job

        self.__monitor_jobs(job_dict=job_dict)

    def __check_status_of_job(
        self, event: str, reposts: int, verbose: bool = False
    ) -> str:
        """
        Query Salvus Flow for the status of the job

        :param event: Name of event
        :type event: str
        :param reposts: Number of reposts of the event for the job
        :type reposts: int
        """
        status = self.project.flow.get_job_status(event, self._job_type).name
        if status == "pending":
            if verbose:
                self.print(f"Status = {status}, event: {event}")
        elif status == "running":
            if verbose:
                self.print(f"Status = {status}, event: {event}")
        elif status in ["unknown", "failed"]:
            self.print(f"{self._job_type} job for {event}, {status}, will resubmit")
            if reposts >= self.project.config.hpc.max_reposts:
                self.print(
                    "No I've actually reposted this too often \n"
                    "There must be something wrong."
                )
                raise InversionsonError("Too many reposts")
            self.to_repost.append(event)
            reposts += 1
            self.project.change_attribute(
                attribute=f'{self._job_type}_job["{event}"]["reposts"]',
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

    def __monitor_jobs(
        self, job_dict: Dict, events: Optional[List[str]] = None, verbose: bool = False
    ) -> None:
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
            f"Checking Jobs for {self._job_type}:", line_above=True, emoji_alias=":ear:"
        )
        for event in tqdm(
            events_left, desc=emoji.emojize(":ear: | ", use_aliases=True), leave=False
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
                if self._job_type == "gradient_interp":
                    self.project.change_attribute(
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
