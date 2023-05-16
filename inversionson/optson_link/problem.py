"""We need a few things. 

Firstly, we need a way to go from salvus meshes to optson vec and back

Then we need to be able to compute f and g

# Within f

# We need to create an iteration if needed
# and otherwise load the iteration attributes

# then we need to select samples


"""
from optson.problem import AbstractProblem, CallCounter
from optson.vector import OptsonVec
from inversionson.file_templates.inversion_info_template import InversionsonConfig
from inversionson.helpers.iteration_listener import IterationListener
from .helpers import vector_to_mesh
from inversionson.project import Project


class Problem(AbstractProblem):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def _get_iteration(self, iteration: str):
        if (
            self.project.lasif.has_iteration(iteration)
            and self.project.paths.get_iteration_toml(iteration).exists()
        ):
            self.project.set_iteration_attributes(iteration=iteration)
            return

        # We need a bunch of events, mocking this noew with a single event
        events = self.project.event_db.get_event_names([0])

        self.project.lasif.set_up_iteration(iteration, events)
        self.project.create_iteration_toml(iteration, events)
        self.project.set_iteration_attributes(iteration)

        # We still need to upload the model.

        hpc_cluster = self.project.flow.hpc_cluster
        remote_model_file = self.project.remote_paths.get_master_model_path(iteration)
        if not hpc_cluster.remote_exists(remote_model_file):
            self.project.flow.safe_put(
                self.project.paths.get_model_path(iteration), remote_model_file
            )

    @CallCounter
    def f(self, x: OptsonVec) -> float:
        # Step 1:: Write the mesh file
        model_file = self.project.paths.get_model_path(x.descriptor)
        if not model_file.exists():
            m = vector_to_mesh(
                x,
                self.project.lasif.master_mesh,
                self.project.config.inversion.inversion_parameters,
            )
            m.write_h5(model_file)

        self._get_iteration(iteration=x.descriptor)

        it_listen = IterationListener(
            project=self.project,
            events=self.project.events_in_iteration,
            control_group_events=self.project.events_in_iteration,
            prev_control_group_events=None,
            misfit_only=True,
            prev_iteration=None,
            submit_adjoint=False,
        )
        it_listen.listen()

        # We now get to the end

    @CallCounter
    def g(self, x: OptsonVec) -> OptsonVec:
        self.f(x)
