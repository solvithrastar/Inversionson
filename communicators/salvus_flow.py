class salvus_flow_comm(object):
    """
    A class which handles all dealings with salvus flow.
    """
    import salvus_flow.api as sapi

    def __init__(self, info_dict: dict, simulation_dict: dict):
        self.lasif_root = info_dict["lasif_project"]
        self.end_time = simulation_dict["end_time_in_seconds"]
        self.time_step = simulation_dict["time_step_in_seconds"]
        self.start_time = simulation_dict["start_time_in_seconds"]

    def get_source_object(self, iteration: str, event_name: str):
        """
        Create the source object that the simulation wants

        :param iteration: Name of iteration
        :type iteration: str
        :param event_name: Name of event
        :type event_name: str
        """
        import lasif.api as lapi
        from salvus_flow.simple_config import source
        from salvus_flow.simple_config import stf

        src_info = lapi.get_source(self.lasif_root, event_name, iteration)

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

    def get_receivers(self, event: str):
        """
        Locate receivers and get them in a format that salvus flow
        can use.

        :param event: Name of event to get the receivers for
        :type event: str
        """
        import lasif.api as lapi
        from salvus_flow.simple_config import receiver

        recs = lapi.get_receivers(self.lasif_root, event)

        receivers = [receiver.seismology.SideSetPoint3D(
            latitude=rec["latitude"],
            longitude=rec["longitude"],
            network_code=rec["network-code"],
            station_code=rec["station-code"],
            fields=["displacement"]
        ) for rec in recs]

        return receivers

    def construct_simulation(self, iteration: str, event: str, sources,
                             receivers: dict):
        """
        Generate the simulation object which salvus flow loves

        :param iteration: Name of iteration
        :type iteration: str
        :param event: Name of event
        :type event: str
        :param sources: Information regarding source
        :type sources: source object
        :param receivers: Information regarding receivers
        :type receivers: dict
        """
        import lasif.api as lapi
        from salvus_flow.simple_config import simulation

        mesh = lapi.get_simulation_mesh(self.lasif_root, event, iteration)

        w = simulation.Waveform(
            mesh=mesh, sources=sources, receivers=receivers)

        w.physics.wave_equation.end_time_in_seconds = self.end_time
        w.physics.wave_equation.time_step_in_seconds = self.time_step
        w.physics.wave_equation.start_time_in_seconds = self.start_time

        # For gradient computation

        w.output.volume_data.format = "hdf5"
        w.output.volume_data.filename = "output.h5"
        w.output.volume_data.fields = ["adjoint-checkpoint"]
        w.output.volume_data.sampling_interval_in_time_steps = "auto-for-checkpointing"

        w.validate()
