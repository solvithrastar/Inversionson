import salvus.flow.api as sapi
import os
import time
import numpy as np
import json
import pathlib
import toml
from typing import Union, List

from lasif.components.component import Component
from salvus.flow import schema_validator
import typing
from salvus.flow.simple_config.simulation import Waveform
from salvus.flow.sites import job as s_job
from salvus.flow.simple_config import simulation, source, stf, receiver
from salvus.flow.api import get_site
from inversionson import InversionsonError
from salvus.mesh.unstructured_mesh import UnstructuredMesh


class SalvusFlowComponent(Component):
    """
    A class which handles all dealings with salvus flow.
    """

    def __init__(self, communicator, component_name):
        super(SalvusFlowComponent, self).__init__(communicator, component_name)

    def print(
        self,
        message: str,
        color: str = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Union[str, List[str]] = None,
    ):
        self.comm.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

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
        old_iter = True
        sim_types = [
            "forward",
            "adjoint",
            "smoothing",
            "prepare_forward",
            "gradient_interp",
            "hpc_processing",
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

            unique_id = "".join(random.choice("0123456789ABCDEF") for i in range(8))
            job = iteration + "_" + sim_type + "_" + unique_id
            if sim_type == "forward":
                self.comm.project.forward_job[event]["name"] = job
            elif sim_type == "adjoint":
                self.comm.project.adjoint_job[event]["name"] = job
            else:
                raise InversionsonError("This isn't even used anyway")
        # Here we just want to return a previously defined job name
        else:
            if old_iter:
                iteration_info = self.comm.project.get_old_iteration_info(iteration)
                event_index = self.comm.project.get_key_number_for_event(
                    event, iteration
                )
                if (
                    self.comm.project.inversion_mode == "mono-batch"
                    or self.comm.project.optimizer == "adam"
                ) and sim_type == "smoothing":
                    job = iteration_info[sim_type]["name"]
                else:
                    job = iteration_info["events"][event_index]["job_info"][sim_type][
                        "name"
                    ]
            else:
                if sim_type == "forward":
                    job = self.comm.project.forward_job[event]["name"]
                elif sim_type == "adjoint":
                    job = self.comm.project.adjoint_job[event]["name"]
                elif sim_type == "prepare_forward":
                    job = self.comm.project.prepare_forward_job[event]["name"]
                elif sim_type == "gradient_interp":
                    job = self.comm.project.gradient_interp_job[event]["name"]
                elif sim_type == "hpc_processing":
                    job = self.comm.project.hpc_processing_job[event]["name"]
                else:
                    job = self.comm.project.smoothing_job["name"]

        self.comm.project.update_iteration_toml()
        return job

    def get_job_name(self, event: str, sim_type: str, iteration="current"):
        return self._get_job_name(
            event=event, sim_type=sim_type, new=False, iteration=iteration
        )

    def get_job(self, event: str, sim_type: str, iteration="current") -> object:
        """
        Get Salvus.Flow Job Object, or JobArray Object

        :param event: Name of event
        :type event: str
        :param sim_type: type of simulation
        :type sim_type: str
        :param iteration: name of iteration, defaults to "current"
        :type iteration: str, optional
        """
        if iteration == "current" or iteration == self.comm.project.current_iteration:
            if sim_type in ["gradient_interp", "prepare_forward", "hpc_processing"]:
                return self.__get_custom_job(event=event, sim_type=sim_type)
            if sim_type == "forward":
                if self.comm.project.forward_job[event]["submitted"]:
                    job_name = self.comm.project.forward_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Forward job for event: {event} has not been " "submitted"
                    )
            elif sim_type == "adjoint":
                if self.comm.project.adjoint_job[event]["submitted"]:
                    job_name = self.comm.project.adjoint_job[event]["name"]
                else:
                    raise InversionsonError(
                        f"Adjoint job for event: {event} has not been " "submitted"
                    )
            elif sim_type == "smoothing":
                if (
                    self.comm.project.inversion_mode == "mono-batch"
                    or self.comm.project.optimizer == "adam"
                ):
                    smoothing_job = self.comm.project.smoothing_job
                else:
                    smoothing_job = self.comm.project.smoothing_job[event]

                if smoothing_job["submitted"]:
                    job_name = smoothing_job["name"]
                else:
                    raise InversionsonError(
                        f"Smoothing job for event: {event} has not been " "submitted"
                    )
        else:
            it_dict = self.comm.project.get_old_iteration_info(iteration)
            if (
                sim_type == "smoothing"
                and self.comm.project.inversion_mode == "mono-batch"
                or self.comm.project.optimizer == "adam"
            ):
                job_name = it_dict["smoothing"]["name"]
            else:
                event_index = self.comm.project.get_key_number_for_event(
                    event=event, iteration=iteration
                )
                job_name = it_dict["events"][event_index]["job_info"][sim_type]["name"]
        if sim_type == "smoothing":
            site_name = self.comm.project.smoothing_site_name
            job = sapi.get_job_array(job_array_name=job_name, site_name=site_name)
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

        if sim_type == "prepare_forward":
            if self.comm.project.prepare_forward_job[event]["submitted"]:
                job_name = self.comm.project.prepare_forward_job[event]["name"]
            else:
                raise InversionsonError(
                    f"Model interpolation job for event: {event} "
                    "has not been submitted"
                )
        if sim_type == "hpc_processing":
            if self.comm.project.hpc_processing_job[event]["submitted"]:
                job_name = self.comm.project.hpc_processing_job[event]["name"]
            else:
                raise InversionsonError(
                    f"HPC processing job for event: {event} " "has not been submitted"
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
        site_name = self.comm.project.interpolation_site
        db_job = sapi._get_config()["db"].get_jobs(
            limit=1,
            site_name=site_name,
            job_name=job_name,
        )[0]

        job = s_job.Job(
            site=sapi.get_site(site_name=db_job.site.site_name),
            commands=self.comm.multi_mesh.get_interp_commands(event, gradient),
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

        job_name = self._get_job_name(event=event_name, sim_type=sim_type, new=False)
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

        iteration = self.comm.project.current_iteration
        receivers = self.comm.lasif.get_receivers(event_name)
        if not self.comm.project.hpc_processing:
            adjoint_filename = self.comm.lasif.get_adjoint_source_file(
                event=event_name, iteration=iteration
            )

        if not self.comm.project.hpc_processing:
            p = h5py.File(adjoint_filename, "r")
            adjoint_recs = list(p.keys())
            p.close()
        else:
            forward_job = self.get_job(event_name, sim_type="forward")

            # remote synthetics
            remote_meta_path = forward_job.output_path / "meta.json"
            hpc_cluster = get_site(self.comm.project.site_name)
            meta_json_filename = "meta.json"
            if os.path.exists(meta_json_filename):
                os.remove(meta_json_filename)
            hpc_cluster.remote_get(remote_meta_path, meta_json_filename)

            proc_job = self.get_job(event_name, sim_type="hpc_processing")
            remote_misfit_dict_toml = str(
                proc_job.stdout_path.parent / "output" / "misfit_dict.toml"
            )
            adjoint_filename = "REMOTE:" + str(
                proc_job.stdout_path.parent / "output" / "stf.h5"
            )
            local_misfit_dict = "misfit_dict.toml"
            if os.path.exists(local_misfit_dict):
                os.remove(local_misfit_dict)
            hpc_cluster.remote_get(remote_misfit_dict_toml, local_misfit_dict)
            misfits = toml.load(local_misfit_dict)
            adjoint_recs = list(misfits[event_name].keys())
            if os.path.exists(local_misfit_dict):
                os.remove(local_misfit_dict)

        # Need to make sure I only take receivers with an adjoint source
        adjoint_sources = []
        for rec in receivers:
            if (
                rec["network-code"] + "_" + rec["station-code"] in adjoint_recs
                or rec["network-code"] + "." + rec["station-code"] in adjoint_recs
            ):
                adjoint_sources.append(rec)

        # print(adjoint_sources)

        # Get path to meta.json to obtain receiver position, use again for adjoint
        if not self.comm.project.hpc_processing:
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
        meta_recs = data["forward_run_input"]["output"]["point_data"]["receiver"]
        meta_info_dict = {}
        for rec in meta_recs:
            if (
                rec["network_code"] + "_" + rec["station_code"] in adjoint_recs
                or rec["network_code"] + "." + rec["station_code"] in adjoint_recs
            ):
                rec_name = rec["network_code"] + "_" + rec["station_code"]
                meta_info_dict[rec_name] = {}
                # this is the rotation from XYZ to ZNE,
                # we still need to transpose to get ZNE -> XYZ
                meta_info_dict[rec_name]["rotation_on_input"] = {
                    "matrix": np.array(rec["rotation_on_output"]["matrix"]).T.tolist()
                }
                meta_info_dict[rec_name]["location"] = rec["location"]

        adj_src = [
            source.cartesian.VectorPoint3D(
                x=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                    "location"
                ][0],
                y=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                    "location"
                ][1],
                z=meta_info_dict[rec["network-code"] + "_" + rec["station-code"]][
                    "location"
                ][2],
                fx=1.0,
                fy=1.0,
                fz=1.0,
                source_time_function=stf.Custom(
                    filename=adjoint_filename,
                    dataset_name="/" + rec["network-code"] + "_" + rec["station-code"],
                ),
                rotation_on_input=meta_info_dict[
                    rec["network-code"] + "_" + rec["station-code"]
                ]["rotation_on_input"],
            )
            for rec in adjoint_sources
        ]
        if os.path.exists(meta_json_filename) and self.comm.project.hpc_processing:
            os.remove(meta_json_filename)

        return adj_src

    def get_receivers(self, event: str):
        """
        Locate receivers and get them in a format that salvus flow
        can use.

        :param event: Name of event to get the receivers for
        :type event: str
        """

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

    def construct_simulation(self, event: str, sources: object, receivers: object):
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

        mesh = self.comm.lasif.get_master_model()
        w = sc.simulation.Waveform(mesh=mesh, sources=sources, receivers=receivers)

        w.physics.wave_equation.end_time_in_seconds = self.comm.project.end_time
        # w.physics.wave_equation.time_step_in_seconds = self.comm.project.time_step
        w.physics.wave_equation.start_time_in_seconds = self.comm.project.start_time
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

        # Compute wavefield subsampling factor.
        if self.comm.project.simulation_time_step:
            # Compute wavefield subsampling factor.
            samples_per_min_period = (
                    self.comm.project.min_period / self.comm.project.simulation_time_step
            )
            min_samples_per_min_period = 30.0
            reduction_factor = int(
                samples_per_min_period / min_samples_per_min_period)
            reduction_factor_syn = int(
                samples_per_min_period / 40.0)
            if reduction_factor_syn >= 2:
                w.output.point_data.sampling_interval_in_time_steps = reduction_factor_syn
            if reduction_factor >= 2:
                checkpointing_flag = f"auto-for-checkpointing_{reduction_factor}"
            else:
                checkpointing_flag = "auto-for-checkpointing"
        else:
            checkpointing_flag = "auto-for-checkpointing_10"
        # For gradient computation
        w.output.volume_data.format = "hdf5"
        w.output.volume_data.filename = "output.h5"
        w.output.volume_data.fields = ["adjoint-checkpoint"]
        w.output.volume_data.sampling_interval_in_time_steps = checkpointing_flag

        w.validate()

        return w

    def construct_simulation_from_dict(self, event: str):
        """
        Download a dictionary with the simulation object and use it to create a local simulation object
        without having any of the relevant data locally.
        Only used to submit a job to the remote without having to store anything locally.

        :param event: Name of event
        :type event: str
        """

        # Always write events to the same folder
        destination = (
            self.comm.lasif.lasif_comm.project.paths["salvus_files"]
            / f"SIMULATION_DICTS"
            / event
            / "simulation_dict.toml"
        )
        if not os.path.exists(destination.parent):
            os.makedirs(destination.parent)

        if not os.path.exists(destination):
            hpc_cluster = sapi.get_site(self.comm.project.site_name)
            interp_job = self.get_job(event, sim_type="prepare_forward")
            remote_dict = (
                interp_job.stdout_path.parent / "output" / "simulation_dict.toml"
            )
            hpc_cluster.remote_get(remotepath=remote_dict, localpath=destination)

        sim_dict = toml.load(destination)

        local_dummy_mesh_path = self.comm.lasif.get_master_model()
        local_dummy_mesh = self.comm.lasif.get_master_mesh()
        for key in ["mesh", "model", "geometry"]:
            sim_dict["domain"][key]["filename"] = local_dummy_mesh_path

        w = self.simulation_from_dict(sim_dict, local_dummy_mesh)

        return w

    @classmethod
    def simulation_from_dict(cls, dictionary: typing.Dict,
                             mesh_object: UnstructuredMesh) -> "Waveform":
        """
        Custom version of the from Waveform.from_dict method
        to allow passing of mesh objects and prevent reading mesh files
        again and again.

        Must be initialized with a locally existing mesh. but can
        be modified to use a remote mesh after creation

        Args:
            dictionary: Dictionary
            mesh_object: salvus mesh object
        """

        w = Waveform()

        # make sure the dictionary is compatible
        schema_validator.validate(
            value=dictionary, schema=w._schema, pretty_error=True
        )

        # ensure the same mesh file is given for mesh, model and geometry
        filenames = [
            dictionary["domain"]["mesh"]["filename"],
            dictionary["domain"]["model"]["filename"],
            dictionary["domain"]["geometry"]["filename"],
        ]

        if len(set(filenames)) != 1:
            msg = (
                "This method currently only supports unique file names "
                "for mesh, model and geometry."
            )
            raise ValueError(msg)

        mesh = dictionary["domain"]["mesh"]["filename"]
        if mesh == "__SALVUS_FLOW_SPECIAL_TEMP__":
            msg = "The dictionary does not contain a path to a mesh file."
            raise ValueError(msg)

        w.set_mesh(mesh_object)
        w.apply(dictionary)
        w.validate()

        return w


    def construct_adjoint_simulation_from_dict(self, event: str):
        """
        Download a dictionary with the simulation object and use it to create a local simulation object
        without having any of the relevant data locally.
        Only used to submit a job to the remote without having to store anything locally.

        :param event: Name of event
        :type event: str
        """

        hpc_cluster = sapi.get_site(self.comm.project.site_name)
        hpc_proc_job = self.get_job(event, sim_type="hpc_processing")

        # Always write events to the same folder
        destination = (
            self.comm.lasif.lasif_comm.project.paths["salvus_files"]
            / f"SIMULATION_DICTS"
            / event
            / "adjoint_simulation_dict.toml"
        )
        if not os.path.exists(destination.parent):
            os.makedirs(destination.parent)

        if os.path.exists(destination):
            os.remove(destination)
        remote_dict = (
            hpc_proc_job.stdout_path.parent / "output" / "adjoint_simulation_dict.toml"
        )
        hpc_cluster.remote_get(remotepath=remote_dict, localpath=destination)

        adjoint_sim_dict = toml.load(destination)
        remote_mesh = adjoint_sim_dict["domain"]["mesh"]["filename"]
        local_dummy_mesh_path = self.comm.lasif.get_master_model()
        local_dummy_mesh = self.comm.lasif.get_master_mesh()
        for key in ["mesh", "model", "geometry"]:
            adjoint_sim_dict["domain"][key]["filename"] = local_dummy_mesh_path

        w = self.simulation_from_dict(adjoint_sim_dict, local_dummy_mesh)
        w.set_mesh("REMOTE:" + str(remote_mesh))
        return w

    def construct_adjoint_simulation(self, event: str, adj_src: object) -> object:
        """
        Create the adjoint simulation object that salvus flow needs.
        This only gets used in the non HPC processing case.

        :param event: Name of event
        :type event: str
        :param adj_src: List of adjoint source objects
        :type adj_src: object
        :return: Simulation object
        :rtype: object
        """
        self.print("Constructing Adjoint Simulation now", emoji_alias=":wrench:")

        remote_interp = False
        if (
            self.comm.project.interpolation_mode == "remote"
            and self.comm.project.meshes == "multi-mesh"
        ):
            remote_interp = True

        # mesh = self.comm.lasif.find_event_mesh(event)
        mesh = self.comm.lasif.lasif_comm.project.lasif_config["domain_settings"][
            "domain_file"
        ]

        if remote_interp:
            interp_job = self.get_job(event=event, sim_type="prepare_forward")

        forward_job = self.get_job(event=event, sim_type="forward")
        meta = forward_job.output_path / "meta.json"
        if remote_interp:
            remote_mesh = interp_job.path / "output" / "mesh.h5"
        else:
            local_meta = os.path.join(
                self.comm.project.lasif_root,
                "SYNTHETICS",
                "EARTHQUAKES",
                f"ITERATION_{self.comm.project.current_iteration}",
                event,
                "meta.json",
            )
            with open(local_meta, "r") as fh:
                meta_info = json.load(fh)
            remote_mesh = meta_info["forward_run_input"]["domain"]["mesh"][
                "filename"]

        gradient = "gradient.h5"



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
        :param ranks: How many cores to run on. (A multiple of 12 on daint),
        defaults to 1024
        :type ranks: int, optional
        """
        # Adjoint simulation takes longer and seems to be less predictable
        # we thus give it a longer wall time.
        if sim_type == "adjoint":
            wall_time = self.comm.project.wall_time * 1.5
        else:
            wall_time = self.comm.project.wall_time
        start_submit = time.time()
        job = sapi.run_async(
            site_name=site,
            input_file=simulation,
            ranks=ranks,
            wall_time_in_seconds=wall_time,
        )
        end_submit = time.time()
        self.print(
            f"Submitting took {end_submit - start_submit:.3f} seconds",
            emoji_alias=":hourglass:",
            color="magenta",
        )
        hpc_cluster = sapi.get_site(self.comm.project.site_name)

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
        if hpc_cluster.config["site_type"] == "local":
            self.print(f"Running {sim_type} simulation...")
            job.wait(poll_interval_in_seconds=self.comm.project.sleep_time_in_s)

    def get_job_status(self, event: str, sim_type: str, iteration="current") -> str:
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

        job = sapi.get_job(job_name=job_name, site_name=self.comm.project.site_name)

        return job.get_output_files()

    def delete_stored_wavefields(
        self, iteration: str, sim_type: str, event_name: str = None
    ):
        """
        Delete all stored jobs for a certain simulation type of an iteration

        :param iteration: Name of iteration
        :type iteration: str
        :param sim_type: Type of simulation, forward or adjoint
        :type sim_type: str
        :param event_name: Name of an event if only for a single event, this can
            for example be used after a job fails and needs to be reposted.
            Defaults to None
        :type event_name: str, optional
        """
        if event_name is not None:
            try:
                job = self.get_job(event=event_name, sim_type=sim_type)
                job.delete()
            except:
                self.print(
                    f"Could not delete job {sim_type} for event {event_name}",
                    emoji_alias=":hankey:",
                )
            return
        events_in_iteration = self.comm.lasif.list_events(iteration=iteration)
        non_val_tasks = ["gradient_interp", "hpc_processing"]
        for _i, event in enumerate(events_in_iteration):
            if (
                self.comm.project.is_validation_event(event)
                and sim_type in non_val_tasks
            ):
                continue
            try:
                job = self.get_job(event=event, sim_type=sim_type)
                job.delete()
            except:
                self.print(
                    f"Could not delete job {sim_type} for event {event}",
                    emoji_alias=":hankey:",
                )

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
