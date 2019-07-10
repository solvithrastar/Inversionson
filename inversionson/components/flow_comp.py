from .component import Component
import salvus_flow.api as sapi
from inversionson import InversionsonError


class SalvusFlowComponent(Component):
    """
    A class which handles all dealings with salvus flow.
    """

    def __init__(self, communicator, component_name):
        super(SalvusFlowComponent, self).__init__(communicator, component_name)

    def _get_job_name(self, event: str,
                      sim_type: str, new=True, iteration="current") -> str:
        """
        We need to relate iteration and event to job name. Here you can find
        it. Currently not used. Removed it from the workflow

        :param event: Name of event
        :type event: str
        :param sim_type: Options are "forward" and "adjoint"
        :type sim_type: str
        :param new: If we need a new job name. Otherwise we look for an
        existing one
        :type new: bool
        :param iteration: Name of iteration: defaults to "current"
        :type iteration: str
        :return: Job name
        :rtype: str
        """
        inversion_id = self.comm.project.inversion_id
        old_iter = True
        if iteration == "current":
            iteration = self.comm.project.current_iteration
            old_iter = False

        if sim_type not in ["forward", "adjoint"]:
            raise ValueError(
                f"Simulation type {sim_type} not supported. Only supported "
                f"ones are forward and adjoint")

        if new:
            import random
            unique_id = ''.join(random.choice('0123456789ABCDEF')
                                for i in range(8))
            job = iteration + "_" + inversion_id + "_" + sim_type + "_" + unique_id
            if sim_type == "forward":
                self.comm.project.forward_job[event]["name"] = job
            else:
                self.comm.project.adjoint_job[event]["name"] = job
        self.comm.project.update_iteration_toml()
        # Here we just want to return a previously defined job name
        else:
            if old_iter:
                iteration_info = self.comm.project.get_old_iteration_info(
                    iteration)
                job = iteration_info["events"][event]["jobs"][sim_type]["name"]
            else:
                if sim_type == "forward":
                    job = self.comm.project.forward_job[event]["name"]
                else:
                    job = self.comm.project.adjoint_job[event]["name"]

        return job

    def get_source_object(self, event_name: str):
        """
        Create the source object that the simulation wants

        :param event_name: Name of event
        :type event_name: str
        """

        from salvus_flow.simple_config import source
        from salvus_flow.simple_config import stf

        src_info = self.comm.lasif.get_source(event_name)

        src = source.seismology.MomentTensorPoint3D(
            latitude=src_info["latitude"],
            longitude=src_info["longitude"],
            depth_in_m=src_info["depth_in_m"],
            mrr=src_info["m_rr"],
            mtt=src_info["m_tt"],
            mpp=src_info["m_pp"],
            mtp=src_info["m_tp"],
            mrp=src_info["m_rp"],
            mrt=src_info["m_rt"],
            source_time_function=stf.Custom(filename=src_info["stf_file"],
                                            dataset_name=src_info["source"])
        )

        return src

    def get_adjoint_source_object(self, event_name: str) -> object:
        """
        Generate the adjoint source object for the respective event

        :param event_name: Name of event
        :type event_name: str
        :return: Adjoint source object for salvus
        :rtype: object
        """
        from salvus_flow.simple_config import source
        from salvus_flow.simple_config import stf
        iteration = self.comm.project.current_iteration
        receivers = self.get_receivers(event)
        adjoint_filename = self.comm.lasif.get_adjoint_source_file(
            event=event_name,
            iteration=iteration
        )

        adj_src = [source.seismology.VectorPoint3D(
            latitude=rec["latitude"],
            longitude=rec["longitude"],
            source_time_function=stf.Custom(
                filename=adjoint_filename,
                dataset_name="auxiliary_data/AdjointSources/" +
                              rec["network-code"] + "_" + rec["station-code"]
        ) for rec in receivers]

        return adj_src

    def get_receivers(self, event: str):
        """
        Locate receivers and get them in a format that salvus flow
        can use.

        :param event: Name of event to get the receivers for
        :type event: str
        """
        from salvus_flow.simple_config import receiver

        recs=self.comm.lasif.get_receivers(event)

        receivers=[receiver.seismology.SideSetPoint3D(
            latitude=rec["latitude"],
            longitude=rec["longitude"],
            network_code=rec["network-code"],
            station_code=rec["station-code"],
            fields=["displacement"]
        ) for rec in recs]

        return receivers

    def construct_simulation(self, event: str, sources: object,
        receivers: object):
        """
        Generate the simulation object which salvus flow loves

        :param event: Name of event
        :type event: str
        :param sources: Information regarding source
        :type sources: source object
        :param receivers: Information regarding receivers
        :type receivers: list of receiver objects
        """

        from salvus_flow.simple_config import simulation

        mesh=self.comm.lasif.get_simulation_mesh(event)

        w=simulation.Waveform(
            mesh=mesh, sources=sources, receivers=receivers)

        w.physics.wave_equation.end_time_in_seconds=self.comm.project.end_time
        w.physics.wave_equation.time_step_in_seconds=self.comm.project.time_step
        w.physics.wave_equation.start_time_in_seconds=self.comm.project.start_time

        # For gradient computation

        w.output.volume_data.format="hdf5"
        w.output.volume_data.filename="output.h5"
        w.output.volume_data.fields=["adjoint-checkpoint"]
        w.output.volume_data.sampling_interval_in_time_steps="auto-for-checkpointing"

        w.validate()

        return w

    def construct_adjoint_simulation(self, event: str, adj_src: object) -> object:
        """
        Create the adjoint simulation object that salvus flow needs

        :param event: Name of event
        :type event: str
        :param adj_src: List of adjoint source objects
        :type adj_src: object
        :return: Simulation object
        :rtype: object
        """
        from salvus_flow.simple_config import simulation

        mesh=self.comm.lasif.get_simulation_mesh(event)

        w=simulation.Waveform(mesh=mesh)
        w.adjoint.forward_meta_json_filename="blabla"
        w.adjoint.gradient.parameterization="rho-vp-vs"  # temporary
        w.adjoint.gradient.output_filename="gradient.h5"
        w.adjoint.point_source=adj_src

        w.validate()

    def submit_job(self, event: str, simulation: object,
                   sim_type: str, site="daint", wall_time=3600, ranks=1024):
        """
        Submit a job with some information. Salvus flow returns an object
        which can be used to interact with job.

        :param event: Name of event
        :type event: str
        :param simulation: Simulation object constructed beforehand
        :type simulation: object
        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        :param site: Name of site in salvus flow config file, defaults
        to "daint"
        :type site: str, optional
        :param wall_time: In what time the site kills your job [seconds],
        defaults to 3600
        :type wall_time: int, optional
        :param ranks: How many cores to run on. (A multiple of 12 on daint),
        defaults to 1024
        :type ranks: int, optional
        """

        job=sapi.run_async(
            site_name=site,
            input_file=simulation,
            ranks=ranks,
            wall_time_in_seconds=wall_time,
        )
        if sim_type == "forward":
            self.comm.project.forward_job[event]["name"]=job.name

        elif sim_type == "adjoint":
            self.comm.project.adjoint_job[event]["name"]=job.name
        self.comm.project.update_iteration_toml()

    def get_job_status(self, event: str, sim_type: str, iteration="current") -> str:
        """
        Check the status of a salvus opt job

        :param event: Name of event
        :type event: str
        :param iteration: Name of iteration. "current" if current iteration
        :type iteration: str
        :return: status of job
        :rtype: str
        """
        if iteration == "current":
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["submitted"]:
                    job_name=self.comm.project.forward_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Forward job for event: {event} has not been "
                        "submitted")
            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["submitted"]:
                    job_name=self.comm.project.adjoint_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Adjoint job for event: {event} has not been "
                        "submitted")
        else:
            it_dict=self.comm.project.get_old_iteration_info(iteration)
            job_name=it_dict["events"][event]["jobs"][sim_type]["name"]

        job=sapi.get_job(
            job_name=job_name,
            site_name=self.comm.project.site_name)

        return job.update_status()
