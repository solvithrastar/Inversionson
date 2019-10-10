from .component import Component
import salvus_flow.api as sapi
from inversionson import InversionsonError
import os
import numpy as np


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
        self.comm.project.update_iteration_toml()
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
        if isinstance(src_info, list):
            src_info = src_info[0]

        src = source.seismology.MomentTensorPoint3D(
            latitude=src_info["latitude"],
            longitude=src_info["longitude"],
            depth_in_m=src_info["depth_in_m"],
            mrr=src_info["mrr"],
            mtt=src_info["mtt"],
            mpp=src_info["mpp"],
            mtp=src_info["mtp"],
            mrp=src_info["mrp"],
            mrt=src_info["mrt"],
            source_time_function=stf.Custom(filename=src_info["stf_file"],
                                            dataset_name=src_info["dataset"])
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
        import h5py
        from salvus_flow.simple_config import source
        from salvus_flow.simple_config import stf
        iteration = self.comm.project.current_iteration
        receivers = self.comm.lasif.get_receivers(event_name)
        adjoint_filename = self.comm.lasif.get_adjoint_source_file(
            event=event_name,
            iteration=iteration
            )
        # A workaround needed for a current salvus bug:
        stf_forward = os.path.join(
                self.comm.project.lasif_root,
                "SALVUS_INPUT_FILES",
                f"ITERATION_{iteration}",
                "stf.h5")
        f = h5py.File(stf_forward)
        stf_source = f['source'][()]
        p = h5py.File(adjoint_filename)
        if 'source' in p.keys():
            del p['source']
        adjoint_recs = list(p.keys())
        p.create_dataset(name='source', data=stf_source)
        p["source"].attrs["sampling_rate_in_hertz"] = 1 / self.comm.project.time_step
        p["source"].attrs["spatial-type"] = np.string_("moment_tensor")
        p["source"].attrs["start_time_in_seconds"] = -self.comm.project.time_step
        f.close()
        #rec = receivers[0]
        #Need to make sure I only take receivers with an adjoint source
        adjoint_sources = []
        for rec in receivers:
            if rec["network-code"] + "_" + rec["station-code"] in adjoint_recs:
                adjoint_sources.append(rec)

        p.close()
        adj_src = [source.seismology.VectorPoint3DZNE(
            latitude=rec["latitude"],
            longitude=rec["longitude"],
            fz=1.0,
            fn=1.0,
            fe=1.0,
            source_time_function=stf.Custom(
                filename=adjoint_filename,
                dataset_name="/" + rec["network-code"] + "_" + rec["station-code"])) for rec in adjoint_sources]

        return adj_src

    def get_receivers(self, event: str):
        """
        Locate receivers and get them in a format that salvus flow
        can use.

        :param event: Name of event to get the receivers for
        :type event: str
        """
        from salvus_flow.simple_config import receiver

        recs = self.comm.lasif.get_receivers(event)
        # TODO: Find out how the smoothiesem side sets work.
        receivers = [receiver.seismology.Point3D(
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

        mesh = self.comm.lasif.get_simulation_mesh(event)

        w = simulation.Waveform(
            mesh=mesh, sources=sources, receivers=receivers)

        w.physics.wave_equation.end_time_in_seconds = self.comm.project.end_time
        w.physics.wave_equation.time_step_in_seconds = self.comm.project.time_step
        w.physics.wave_equation.start_time_in_seconds = self.comm.project.start_time

        # For gradient computation

        w.output.volume_data.format = "hdf5"
        w.output.volume_data.filename = "output.h5"
        w.output.volume_data.fields = ["adjoint-checkpoint"]
        w.output.volume_data.sampling_interval_in_time_steps = "auto-for-checkpointing"

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

        mesh = self.comm.lasif.get_simulation_mesh(event)
        forward_job_name = self.comm.project.forward_job[event]["name"]
        forward_job_path = sapi.get_job(
                site_name=self.comm.project.site_name,
                job_name=forward_job_name).output_path
        meta = os.path.join(forward_job_path, "meta.json")
        print(forward_job_path)

        gradient = os.path.join(
                self.comm.lasif.lasif_root,
                "GRADIENTS",
                f"ITERATION_{self.comm.project.current_iteration}",
                event,
                "gradient.h5")

        w = simulation.Waveform(mesh=mesh)
        w.adjoint.forward_meta_json_filename = meta
        w.adjoint.gradient.parameterization = "rho-vp-vs"
        w.adjoint.gradient.output_filename = gradient
        w.adjoint.point_source = adj_src

        w.validate()

        return w

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
        iteration = self.comm.project.current_iteration
        output_folder = os.path.join(
        self.comm.lasif.lasif_root,
                "SYNTHETICS",
                "EARTHQUAKES",
                f"ITERATION_{iteration}",
                event)
        job = sapi.run_async(
            site_name=site,
            input_file=simulation,
            ranks=ranks
            #wall_time_in_seconds=wall_time
            #output_folder=output_folder
        )
        #sapi.run(
        #        site_name=site,
        #        input_file=simulation,
        #        output_folder=output_folder,
        #        ranks=8,
        #        overwrite=True)
        if sim_type == "forward":
            self.comm.project.change_attribute(f"forward_job[\"{event}\"][\"name\"]", job.job_name)
            self.comm.project.change_attribute(f"forward_job[\"{event}\"][\"submitted\"]", True)

        elif sim_type == "adjoint":
            self.comm.project.change_attribute(f"adjoint_job[\"{event}\"][\"name\"]", job.job_name)
            self.comm.project.change_attribute(f"adjoint_job[\"{event}\"][\"submitted\"]", True)
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
                    job_name = self.comm.project.forward_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Forward job for event: {event} has not been "
                        "submitted")
            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["submitted"]:
                    job_name = self.comm.project.adjoint_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Adjoint job for event: {event} has not been "
                        "submitted")
        else:
            it_dict = self.comm.project.get_old_iteration_info(iteration)
            job_name = it_dict["events"][event]["jobs"][sim_type]["name"]

        job = sapi.get_job(
            job_name=job_name,
            site_name=self.comm.project.site_name)

        return job.update_status()

    def get_job_file_paths(self, event: str, sim_type: str) -> dict:
        """
        Get the output folder for an event

        :param event: Name of event
        :type event: str
        :param sim_type: Forward or adjoint simulation
        :type sim_type: str
        """
        if sim_type == "forward":
            job_name = self.comm.project.forward_job[event]["name"]
        elif sim_type == "adjoint":
            job_name = self.comm.project.adjoint_job[event]["name"]
        else:
            raise InversionsonError(f"Don't recognise sim_type {sim_type}")

        job = sapi.get_job(
                job_name=job_name,
                site_name=self.comm.project.site_name)

        return job.get_output_files()

    def submit_smoothing_job(self, event: str, smooth, simulations):
        """
        Submit the salvus diffusion equation smoothing job

        :param event: name of event
        :type event: str
        :param simulation: Simulation object required by salvus flow
        :type simulation: object
        """
        output_folder = os.path.join(
            self.comm.lasif.lasif_root,
            "GRADIENTS",
            f"ITERATION_{self.comm.project.current_iteration}",
            event,
            "smoother_output"
        )
        from salvus_mesh.unstructured_mesh import UnstructuredMesh

        for par in simulations.keys():
            sapi.run(
                #site_name="swp_smooth",
                site_name=self.comm.project.site_name,
                input_file=simulations[par],
                output_folder=output_folder,
                overwrite=True,
                ranks=8,
                get_all=True)
            
            smoothed = UnstructuredMesh.from_h5(os.path.join(output_folder, "smooth_gradient.h5"))
            smooth.attach_field(par, smoothed.elemental_fields[par])
        output_folder = os.path.join(
            self.comm.lasif.lasif_root,
            "GRADIENTS",
            f"ITERATION_{self.comm.project.current_iteration}",
            event
        )
        smooth.write_h5(os.path.join(output_folder, "smooth_gradient.h5"))
