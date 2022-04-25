import os

from inversionson.helpers.autoinverter_helpers import RemoteJobListener
from inversionson.utils import sleep_or_process


class InterpolationListener(object):
    """
    A class used to monitor (submitted
    """

    def __init__(self, comm, events):
        self.comm = comm
        self.events = events

    def monitor_interpolations(self):
        """
        Monitor the status of the interpolations, as soon as one is done,
        the smoothing simulation is dispatched
        """
        if not self.comm.project.meshes == "multi-mesh":
            return

        events = self.events
        int_job_listener = RemoteJobListener(
            comm=self.comm,
            job_type="gradient_interp",
            events=events,
        )
        while True:
            int_job_listener.monitor_jobs()
            for event in int_job_listener.events_retrieved_now:
                self.comm.project.change_attribute(
                    attribute=f'gradient_interp_job["{event}"]["retrieved"]',
                    new_value=True,
                )
                self.comm.project.update_iteration_toml()

            for event in int_job_listener.to_repost:
                self.comm.project.change_attribute(
                    attribute=f'gradient_interp_job["{event}"]["submitted"]',
                    new_value=False,
                )
                self.comm.project.update_iteration_toml()
                self.__dispatch_raw_gradient_interpolation(event=event)

            if len(int_job_listener.events_retrieved_now) + len(
                    int_job_listener.events_already_retrieved
            ) == len(events):
                break

            if not int_job_listener.events_retrieved_now:
                sleep_or_process(self.comm)

            int_job_listener.to_repost = []
            int_job_listener.events_retrieved_now = []

    def __dispatch_raw_gradient_interpolation(self, event: str):
        """
        Take the gradient out of the adjoint simulations and
        interpolate them to the inversion grid prior to smoothing.
        """
        interp_folder = os.path.join(
            self.comm.project.remote_inversionson_dir,
            "INTERPOLATION_WEIGHTS",
            "GRADIENTS",
            event,
        )
        self.comm.multi_mesh.interpolate_gradient_to_model(
            event, smooth=False, interp_folder=interp_folder
        )
