import salvus.flow.api as sapi
import os
import time
import numpy as np
import json
import pathlib

from lasif.components.component import Component
from salvus.flow.sites import job as s_job
from inversionson import InversionsonError
from inversionson.optimizers.adam_optimizer import BOOL_ADAM


class SalvusFlowComponent(Component):
    """
    A class which handles all dealings with salvus flow.
    """

    def __init__(self, communicator, component_name):
        super(SalvusFlowComponent, self).__init__(communicator, component_name)

    def _get_job_name(
        self, event: str, sim_type: str, new=True, iteration="current"
    ) -> str:
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
        sim_types = [
            "forward",
            "adjoint",
            "smoothing",
            "model_interp",
            "gradient_interp",
        ]
        if iteration == "current":
            iteration = self.comm.project.current_iteration
        if iteration == self.comm.project.current_iteration:
            old_iter = False
        if sim_type not in sim_types:
            raise ValueError(
                f"Simulation type {sim_type} not supported. Only supported "
                f"ones are {sim_types}"
            )

        if new:
            import random

            unique_id = "".join(
                random.choice("0123456789ABCDEF") for i in range(8)
            )
            job = (
                iteration
                + "_"
                + inversion_id
                + "_"
                + sim_type
                + "_"
                + unique_id
            )
            if sim_type == "forward":
                self.comm.project.forward_job[event]["name"] = job
            elif sim_type == "adjoint":
                self.comm.project.adjoint_job[event]["name"] = job
            else:
                raise InversionsonError("This isn't even used anyway")
        # Here we just want to return a previously defined job name
        else:
            if old_iter:
                iteration_info = self.comm.project.get_old_iteration_info(
                    iteration
                )
                event_index = self.comm.project.get_key_number_for_event(
                    event, iteration
                )
                if (
                    (self.comm.project.inversion_mode == "mono-batch" or BOOL_ADAM)
                    and sim_type == "smoothing"
                ):
                    job = iteration_info[sim_type]["name"]
                else:
                    job = iteration_info["events"][event_index]["job_info"][
                        sim_type
                    ]["name"]
            else:
                if sim_type == "forward":
                    job = self.comm.project.forward_job[event]["name"]
                elif sim_type == "adjoint":
                    job = self.comm.project.adjoint_job[event]["name"]
                elif sim_type == "model_interp":
                    job = self.comm.project.model_interp_job[event]["name"]
                elif sim_type == "gradient_interp":
                    job = self.comm.project.gradient_interp_job[event]["name"]
                else:
                    if self.comm.project.inversion_mode == "mono-batch" or BOOL_ADAM:
                        job = self.comm.project.smoothing_job["name"]
                    else:
                        job = self.comm.project.smoothing_job[event]["name"]
        self.comm.project.update_iteration_toml()
        return job

    def get_job_name(self, event: str, sim_type: str, iteration="current"):
        return self._get_job_name(
            event=event, sim_type=sim_type, new=False, iteration=iteration
        )

    def get_job(
        self, event: str, sim_type: str, iteration="current"
    ) -> object:
        """
        Get Salvus.Flow Job Object, or JobArray Object

        :param event: Name of event
        :type event: str
        :param sim_type: type of simulation
        :type sim_type: str
        :param iteration: name of iteration, defaults to "current"
        :type iteration: str, optional
        """
        if (
            iteration == "current"
            or iteration == self.comm.project.current_iteration
        ):
            if "interp" in sim_type:
                return self.__get_custom_job(event=event, sim_type=sim_type)
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["submitted"]:
                    job_name = self.comm.project.forward_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Forward job for event: {event} has not been "
                        "submitted"
                    )
            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["submitted"]:
                    job_name = self.comm.project.adjoint_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Adjoint job for event: {event} has not been "
                        "submitted"
                    )
            elif sim_type == "smoothing":
                if self.comm.project.inversion_mode == "mono-batch" or BOOL_ADAM:
                    smoothing_job = self.comm.project.smoothing_job
                else:
                    smoothing_job = self.comm.project.smoothing_job[event]

                if smoothing_job["submitted"]:
                    job_name = smoothing_job["name"]
                else:
                    raise InversionsonError(
                        f"Smoothing job for event: {event} has not been "
                        "submitted"
                    )
        else:
            it_dict = self.comm.project.get_old_iteration_info(iteration)
            if (
                sim_type == "smoothing"
                and self.comm.project.inversion_mode == "mono-batch" or BOOL_ADAM
            ):
                job_name = it_dict["smoothing"]["name"]
            else:
                event_index = self.comm.project.get_key_number_for_event(
                    event=event, iteration=iteration
                )
                job_name = it_dict["events"][event_index]["job_info"][
                    sim_type
                ]["name"]
        if sim_type == "smoothing":
            site_name = self.comm.project.smoothing_site_name
            job = sapi.get_job_array(
                job_array_name=job_name, site_name=site_name
            )
        else:
            site_name = self.comm.project.site_name
            job = sapi.get_job(job_name=job_name, site_name=site_name)

        return job

    def __get_custom_job(self, event: str, sim_type: str):
        """
        A get_job function which handles job types which are not of type
        salvus.flow.sites.salvus_job.SalvusJob

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation
        :type sim_type: str
        """
        gradient = False
        hpc_cluster = sapi.get_site(self.comm.project.interpolation_site)
        username = hpc_cluster.config["ssh_settings"]["username"]
        interp_folder = os.path.join(
            "/scratch/snx3000",
            username,
            "INTERPOLATION_WEIGHTS",
        )
        if sim_type == "model_interp":
            if self.comm.project.model_interp_job[event]["submitted"]:
                job_name = self.comm.project.model_interp_job[event]["name"]
            else:
                raise InversionsonError(
                    f"Model interpolation job for event: {event} "
                    "has not been submitted"
                )
        elif sim_type == "gradient_interp":
            gradient = True
            if self.comm.project.gradient_interp_job[event]["submitted"]:
                job_name = self.comm.project.gradient_interp_job[event]["name"]
            else:
                raise InversionsonError(
                    f"Gradient interpolation job for event: {event} "
                    "has not been submitted"
                )
        interp_folder = os.path.join(
            interp_folder, "GRADIENTS" if gradient else "MODELS", event
        )
        site_name = self.comm.project.interpolation_site
        db_job = sapi._get_config()["db"].get_jobs(
            limit=1,
            site_name=site_name,
            job_name=job_name,
        )[0]

        job = s_job.Job(
            site=sapi.get_site(site_name=db_job.site.site_name),
            commands=self.comm.multi_mesh.get_interp_commands(
                event, gradient, interp_folder=interp_folder
            ),
            job_type=db_job.job_type,
            job_info=db_job.info,
            jobname=db_job.job_name,
            job_description=db_job.description,
            wall_time_in_seconds=db_job.wall_time_in_seconds,
            working_dir=pathlib.Path(db_job.working_directory),
            tmpdir_root=pathlib.Path(db_job.temp_directory_root)
            if db_job.temp_directory_root
            else None,
            rundir_root=pathlib.Path(db_job.run_directory_root)
            if db_job.run_directory_root
            else None,
            job_groups=[i.group_name for i in db_job.groups],
            initialize_on_site=False,
        )
        return job

    def retrieve_outputs(self, event_name: str, sim_type: str):
        """
        Currently we need to use command line salvus opt to
        retrieve the seismograms. There must be some better way
        though.

        :param event_name: Name of event
        :type event_name: str
        :param sim_type: Type of simulation, forward, adjoint
        :type sim_type: str
        """

        job_name = self._get_job_name(
            event=event_name, sim_type=sim_type, new=False
        )
        salvus_job = sapi.get_job(
            site_name=self.comm.project.site_name, job_name=job_name
        )
        if sim_type == "forward":
            destination = self.comm.lasif.find_seismograms(
                event=event_name, iteration=self.comm.project.current_iteration
            )

        elif sim_type == "adjoint":
            destination = self.comm.lasif.find_gradient(
                iteration=self.comm.project.current_iteration,
                event=event_name,
                smooth=False,
                inversion_grid=False,
                just_give_path=True,
            )

        else:
            raise InversionsonError(
                f"Simulation type {sim_type} not supported in this function"
            )
        salvus_job.copy_output(
            destination=os.path.dirname(destination),
            allow_existing_destination_folder=True,
        )

    def get_source_object(self, event_name: str):
        """
        Create the source object that the simulation wants

        :param event_name: Name of event
        :type event_name: str
        """

        from salvus.flow.simple_config import source
        from salvus.flow.simple_config import stf

        iteration = self.comm.project.current_iteration
        src_info = self.comm.lasif.get_source(event_name)
        stf_file = self.comm.lasif.find_stf(iteration)
        side_set = "r1"
        if isinstance(src_info, list):
            src_info = src_info[0]
        if self.comm.project.meshes == "multi-mesh":
            src = source.seismology.SideSetMomentTensorPoint3D(
                latitude=src_info["latitude"],
                longitude=src_info["longitude"],
                depth_in_m=src_info["depth_in_m"],
                mrr=src_info["mrr"],
                mtt=src_info["mtt"],
                mpp=src_info["mpp"],
                mtp=src_info["mtp"],
                mrp=src_info["mrp"],
                mrt=src_info["mrt"],
                source_time_function=stf.Custom(
                    filename=stf_file, dataset_name="/source"
                ),
                side_set_name=side_set,
            )
            # print(f"Source info: {src_info}")
        else:
            if self.comm.project.ocean_loading["use"]:
                side_set = "r1_ol"
            src = source.seismology.SideSetMomentTensorPoint3D(
                latitude=src_info["latitude"],
                longitude=src_info["longitude"],
                depth_in_m=src_info["depth_in_m"],
                mrr=src_info["mrr"],
                mtt=src_info["mtt"],
                mpp=src_info["mpp"],
                mtp=src_info["mtp"],
                mrp=src_info["mrp"],
                mrt=src_info["mrt"],
                side_set_name=side_set,
                source_time_function=stf.Custom(
                    filename=stf_file, dataset_name="/source"
                ),
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
        from salvus.flow.simple_config import source, stf

        iteration = self.comm.project.current_iteration
        receivers = self.comm.lasif.get_receivers(event_name)
        adjoint_filename = self.comm.lasif.get_adjoint_source_file(
            event=event_name, iteration=iteration
        )
        # A workaround needed for a current salvus bug:
        # stf_forward = os.path.join(
        #         self.comm.project.lasif_root,
        #         "SALVUS_INPUT_FILES",
        #         f"ITERATION_{iteration}",
        #         "custom_stf.h5")
        # job_name = self._get_job_name(
        #     event=event_name,
        #     sim_type="forward",
        #     new=False)
        # stf_forward_path = f"/scratch/snx3000/tsolvi/salvus_flow/run/{job_name}/input/custom_stf.h5"
        # copy it:
        # os.system(f"scp daint:{stf_forward_path} {stf_forward}")
        # f = h5py.File(stf_forward)
        # stf_source = f['stf'][()]
        p = h5py.File(adjoint_filename, "r")
        # if 'stf' in p.keys():
        # del p['stf']
        adjoint_recs = list(p.keys())
        # p.create_dataset(name='stf', data=stf_source)
        # p["stf"].attrs["sampling_rate_in_hertz"] = 1 / self.comm.project.time_step
        # p["source"].attrs["spatial-type"] = np.string_("moment_tensor")
        # p["stf"].attrs["start_time_in_seconds"] = -self.comm.project.time_step
        # f.close()
        # rec = receivers[0]
        # Need to make sure I only take receivers with an adjoint source
        adjoint_sources = []
        for rec in receivers:
            if rec["network-code"] + "_" + rec["station-code"] in adjoint_recs:
                adjoint_sources.append(rec)

        p.close()
        # if self.comm.project.meshes == "multi-mesh":
        #     adj_src = [
        #         source.seismology.VectorPoint3DZNE(
        #             latitude=rec["latitude"],
        #             longitude=rec["longitude"],
        #             fz=1.0,
        #             fn=1.0,
        #             fe=1.0,
        #             source_time_function=stf.Custom(
        #                 filename=adjoint_filename,
        #                 dataset_name="/"
        #                 + rec["network-code"]
        #                 + "_"
        #                 + rec["station-code"],
        #             ),
        #         )
        #         for rec in adjoint_sources
        #     ]
        #     return adj_src
        # Get path to meta.json to obtain receiver position, use again for adjoint
        meta_json_filename = os.path.join(
            self.comm.project.lasif_root,
            "SYNTHETICS",
            "EARTHQUAKES",
            f"ITERATION_{iteration}",
            event_name,
            "meta.json",
        )

        # Build meta info dict

        with open(meta_json_filename) as json_file:
            data = json.load(json_file)
        meta_recs = data["forward_run_input"]["output"]["point_data"][
            "receiver"
        ]
        meta_info_dict = {}
        for rec in meta_recs:
            if rec["network_code"] + "_" + rec["station_code"] in adjoint_recs:
                rec_name = rec["network_code"] + "_" + rec["station_code"]
                meta_info_dict[rec_name] = {}
                # this is the rotation from XYZ to ZNE,
                # we still need to transpose to get ZNE -> XYZ
                meta_info_dict[rec_name]["rotation_on_input"] = {
                    "matrix": np.array(
                        rec["rotation_on_output"]["matrix"]
                    ).T.tolist()
                }
                meta_info_dict[rec_name]["location"] = rec["location"]

        adj_src = [
            source.cartesian.VectorPoint3D(
                x=meta_info_dict[
                    rec["network-code"] + "_" + rec["station-code"]
                ]["location"][0],
                y=meta_info_dict[
                    rec["network-code"] + "_" + rec["station-code"]
                ]["location"][1],
                z=meta_info_dict[
                    rec["network-code"] + "_" + rec["station-code"]
                ]["location"][2],
                fx=1.0,
                fy=1.0,
                fz=1.0,
                source_time_function=stf.Custom(
                    filename=adjoint_filename,
                    dataset_name="/"
                    + rec["network-code"]
                    + "_"
                    + rec["station-code"],
                ),
                rotation_on_input=meta_info_dict[
                    rec["network-code"] + "_" + rec["station-code"]
                ]["rotation_on_input"],
            )
            for rec in adjoint_sources
        ]

        return adj_src

    def get_receivers(self, event: str):
        """
        Locate receivers and get them in a format that salvus flow
        can use.

        :param event: Name of event to get the receivers for
        :type event: str
        """
        from salvus.flow.simple_config import receiver

        side_set = "r1"

        recs = self.comm.lasif.get_receivers(event)
        if self.comm.project.meshes == "multi-mesh":
            receivers = [
                receiver.seismology.SideSetPoint3D(
                    latitude=rec["latitude"],
                    longitude=rec["longitude"],
                    network_code=rec["network-code"],
                    station_code=rec["station-code"],
                    depth_in_m=0.0,
                    fields=["displacement"],
                    side_set_name=side_set,
                )
                for rec in recs
            ]
        else:
            if self.comm.project.ocean_loading["use"]:
                side_set = "r1_ol"
            receivers = [
                receiver.seismology.SideSetPoint3D(
                    latitude=rec["latitude"],
                    longitude=rec["longitude"],
                    network_code=rec["network-code"],
                    station_code=rec["station-code"],
                    depth_in_m=0.0,
                    fields=["displacement"],
                    side_set_name=side_set,
                )
                for rec in recs
            ]
        return receivers

    def construct_simulation(
        self, event: str, sources: object, receivers: object
    ):
        """
        Generate the simulation object which salvus flow loves

        :param event: Name of event
        :type event: str
        :param sources: Information regarding source
        :type sources: source object
        :param receivers: Information regarding receivers
        :type receivers: list of receiver objects
        """
        import salvus.flow.simple_config as sc

        if self.comm.project.meshes == "multi-mesh":
            mesh = self.comm.lasif.get_simulation_mesh(event)
            if self.comm.project.interpolation_mode == "remote":
                use_mesh = f"REMOTE:{mesh}"
            mesh = self.comm.lasif.find_event_mesh(event=event)
        else:
            mesh = self.comm.lasif.get_simulation_mesh(event)
        w = sc.simulation.Waveform(
            mesh=mesh, sources=sources, receivers=receivers
        )

        w.physics.wave_equation.end_time_in_seconds = (
            self.comm.project.end_time
        )
        w.physics.wave_equation.time_step_in_seconds = (
            self.comm.project.time_step
        )
        w.physics.wave_equation.start_time_in_seconds = (
            self.comm.project.start_time
        )
        w.physics.wave_equation.attenuation = self.comm.project.attenuation
        bound = False
        boundaries = []

        if self.comm.project.absorbing_boundaries:
            bound = True
            if (
                "inner_boundary"
                in self.comm.lasif.lasif_comm.project.domain.get_side_set_names()
            ):
                side_sets = ["inner_boundary"]
            else:
                side_sets = [
                    "r0",
                    "t0",
                    "t1",
                    "p0",
                    "p1",
                ]
            absorbing = sc.boundary.Absorbing(
                width_in_meters=self.comm.project.abs_bound_length * 1000.0,
                side_sets=side_sets,
                taper_amplitude=1.0
                / self.comm.lasif.lasif_comm.project.simulation_settings[
                    "minimum_period_in_s"
                ],
            )
            boundaries.append(absorbing)
        if self.comm.project.ocean_loading["use"]:
            bound = True
            if self.comm.project.meshes == "multi-mesh":
                side_set = "r1"
            else:
                side_set = "r1_ol"
            ocean_loading = sc.boundary.OceanLoading(side_sets=[side_set])
            boundaries.append(ocean_loading)
        if bound:
            w.physics.wave_equation.boundaries = boundaries

        # For gradient computation

        w.output.volume_data.format = "hdf5"
        w.output.volume_data.filename = "output.h5"
        w.output.volume_data.fields = ["adjoint-checkpoint"]
        w.output.volume_data.sampling_interval_in_time_steps = (
            "auto-for-checkpointing"
        )
        if self.comm.project.meshes == "multi-mesh":
            w.set_mesh(use_mesh)

        w.validate()

        return w

    def construct_adjoint_simulation(
        self, event: str, adj_src: object
    ) -> object:
        """
        Create the adjoint simulation object that salvus flow needs

        :param event: Name of event
        :type event: str
        :param adj_src: List of adjoint source objects
        :type adj_src: object
        :return: Simulation object
        :rtype: object
        """
        print("Constructing Adjoint Simulation now")
        from salvus.flow.simple_config import simulation

        remote_interp = False
        if (
            self.comm.project.interpolation_mode == "remote"
            and self.comm.project.meshes == "multi-mesh"
        ):
            remote_interp = True

        mesh = self.comm.lasif.find_event_mesh(event)
        if remote_interp:
            interp_job = self.get_job(event=event, sim_type="model_interp")
        forward_job_name = self.comm.project.forward_job[event]["name"]
        forward_job = self.get_job(event=event, sim_type="forward")
        meta = forward_job.output_path / "meta.json"
        if remote_interp:
            remote_mesh = interp_job.path / "output" / "mesh.h5"
        else:
            remote_mesh = forward_job.input_path / "mesh.h5"

        # gradient = os.path.join(
        #     self.comm.lasif.lasif_root,
        #     "GRADIENTS",
        #     f"ITERATION_{self.comm.project.current_iteration}",
        #     event,
        #     "gradient.h5",
        # )
        gradient = "gradient.h5"
        # if not os.path.exists(os.path.dirname(gradient)):
        #     os.makedirs(os.path.dirname(gradient))

        w = simulation.Waveform(mesh=mesh)
        w.adjoint.forward_meta_json_filename = f"REMOTE:{meta}"
        if "VPV" in self.comm.project.inversion_params:
            parameterization = "tti"
        elif "VP" in self.comm.project.inversion_params:
            parameterization = "rho-vp-vs"
        w.adjoint.gradient.parameterization = parameterization
        w.adjoint.gradient.output_filename = gradient
        w.adjoint.gradient.format = "hdf5-full"
        w.adjoint.point_source = adj_src

        # Now set a remote mesh
        w.set_mesh("REMOTE:" + str(remote_mesh))
        w.validate()

        return w

    def submit_job(
        self,
        event: str,
        simulation: object,
        sim_type: str,
        site="daint",
        wall_time=3600,
        ranks=1024,
    ):
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
        # iteration = self.comm.project.current_iteration
        # output_folder = os.path.join(
        # self.comm.lasif.lasif_root,
        #         "SYNTHETICS",
        #         "EARTHQUAKES",
        #         f"ITERATION_{iteration}",
        #         event)

        # Adjoint simulation takes longer and seems to be less predictable
        # we thus give it a longer wall time.
        start = time.time()
        if sim_type == "adjoint":
            wall_time = self.comm.project.wall_time * 1.5
        else:
            wall_time = self.comm.project.wall_time
        job = sapi.run_async(
            site_name=site,
            input_file=simulation,
            ranks=ranks,
            wall_time_in_seconds=wall_time,
        )

        if (
            self.comm.project.remote_mesh is None
            and self.comm.project.meshes == "mono-mesh"
        ):
            self.comm.project.change_attribute(
                "remote_mesh",
                "REMOTE:"
                + str(
                    job.input_path
                    / pathlib.Path(
                        self.comm.lasif.get_simulation_mesh(event_name=event)
                    ).name
                ),
            )

        if sim_type == "forward":
            self.comm.project.change_attribute(
                f'forward_job["{event}"]["name"]', job.job_name
            )
            self.comm.project.change_attribute(
                f'forward_job["{event}"]["submitted"]', True
            )

        elif sim_type == "adjoint":
            self.comm.project.change_attribute(
                f'adjoint_job["{event}"]["name"]', job.job_name
            )
            self.comm.project.change_attribute(
                f'adjoint_job["{event}"]["submitted"]', True
            )
        self.comm.project.update_iteration_toml()
        end = time.time()
        print(f"Submitting took {end - start} seconds")

    def get_job_status(
        self, event: str, sim_type: str, iteration="current"
    ) -> str:
        """
        Check the status of a salvus opt job

        :param event: Name of event
        :type event: str
        :param sim_type: Type of simulation: forward, adjoint or smoothing
        :type sim_type: str
        :param iteration: Name of iteration. "current" if current iteration
        :type iteration: str
        :return: status of job
        :rtype: str
        """

        job = self.get_job(
            event=event,
            sim_type=sim_type,
            iteration=iteration,
        )
        return job.update_status(force_update=True)

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
            raise InversionsonError(f"Don't recognize sim_type {sim_type}")

        job = sapi.get_job(
            job_name=job_name, site_name=self.comm.project.site_name
        )

        return job.get_output_files()

    def delete_stored_wavefields(self, iteration: str, sim_type: str):
        """
        Delete all stored jobs for a certain simulation type of an iteration

        :param iteration: Name of iteration
        :type iteration: str
        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        """
        iter_info = self.comm.project.get_old_iteration_info(iteration)

        events_in_iteration = self.comm.lasif.list_events(iteration=iteration)

        for _i, event in enumerate(events_in_iteration):
            try:
                job = self.get_job(event=event, sim_type=sim_type)
                job.delete()
            except:
                print(f"Could not delete job {sim_type} for event {event}")

    def submit_smoothing_job(self, event: str, simulation, par):
        """
        Submit the salvus diffusion equation smoothing job

        :param event: name of event
        :type event: str
        :param simulation: Simulation object required by salvus flow
        :type simulation: object
        :param par: Parameter to smooth
        :type par: str
        """
        # output_folder = os.path.join(
        #     self.comm.lasif.lasif_root,
        #     "GRADIENTS",
        #     f"ITERATION_{self.comm.project.current_iteration}",
        #     event,
        #     "smoother_output"
        # )
        # from salvus_mesh.unstructured_mesh import UnstructuredMesh
        # if self.comm.project.site_name == "swp":
        #     for par in simulations.keys():
        #         sapi.run(
        #             #site_name="swp_smooth",
        #             site_name=self.comm.project.site_name,
        #             input_file=simulations[par],
        #             output_folder=output_folder,
        #             overwrite=True,
        #             ranks=8,
        #             get_all=True)
        #
        #         smoothed = UnstructuredMesh.from_h5(os.path.join(output_folder, "smooth_gradient.h5"))
        #         smooth.attach_field(par, smoothed.elemental_fields[par])
        #     output_folder = os.path.join(
        #         self.comm.lasif.lasif_root,
        #         "GRADIENTS",
        #         f"ITERATION_{self.comm.project.current_iteration}",
        #         event
        #     )
        #     smooth.write_h5(os.path.join(output_folder, "smooth_gradient.h5"))
        job = sapi.run_async(
            site_name=self.comm.project.smoothing_site_name,
            input_file=simulation,
            ranks=self.comm.project.smoothing_ranks,
            wall_time_in_seconds=self.comm.project.smoothing_wall_time,
        )
        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["{par}"]["name"]', job.job_name
        )
        self.comm.project.change_attribute(
            f'smoothing_job["{event}"]["{par}"]["submitted"]', True
        )
