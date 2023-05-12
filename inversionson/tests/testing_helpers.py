import os

from inversionson.create_dummy_info_file import create_info
from inversionson import project

import lasif.api


class DummyProject:
    def __init__(self, dir):
        self.root_folder = os.path.join(dir, "dummy_project")
        os.mkdir(self.root_folder)
        self._dummy_salvus_opt()
        info = create_info(root=os.path.join(self.root_folder))
        lasif.api.init_project(os.path.join(self.root_folder, "LASIF_PROJECT"))
        self.comm = project.ProjectComponent(info).get_communicator()
        self._dummy_events_to_lasif()

    def dummy_file(self, path):
        with open(path, "a"):
            os.utime(path, None)

    def _dummy_salvus_opt(self):
        salvus_opt = os.path.join(self.root_folder, "SALVUS_OPT")
        os.mkdir(salvus_opt)
        os.mkdir(os.path.join(salvus_opt, "PHYSICAL_MODELS"))
        os.mkdir(os.path.join(salvus_opt, "INVERSION_MODELS"))
        os.mkdir(os.path.join(salvus_opt, "BACKUP"))
        self.dummy_file(os.path.join(salvus_opt, "inversion.toml"))
        self.dummy_file(os.path.join(salvus_opt, "PHYSICAL_MODELS", "it0000_model.h5"))

    def _dummy_events_to_lasif(self):
        events = [
            "http://ds.iris.edu/spud/momenttensor/988455",
            "http://ds.iris.edu/spud/momenttensor/735711",
        ]

        for event in events:
            lasif.api.add_spud_event(lasif_root=self.comm.project.lasif_root, url=event)
        # This is only here as a reminder
        event_names = [
            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14",
            "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20",
        ]
        event_mesh = os.path.join(
            self.comm.project.lasif_root,
            "MODELS",
            "EVENT_MESHES",
            event_names[0],
            "mesh.h5",
        )
        os.makedirs(os.path.dirname(event_mesh))
        self.dummy_file(event_mesh)
        self.comm.lasif.set_up_iteration(name="it0000_model")
        sim_mesh = os.path.join(
            self.comm.project.lasif_root,
            "MODELS",
            "ITERATION_it0000_model",
            event_names[0],
            "mesh.h5",
        )
        if not os.path.exists(os.path.dirname(sim_mesh)):
            os.makedirs(os.path.dirname(sim_mesh))

        self.dummy_file(sim_mesh)
