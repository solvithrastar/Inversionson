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
from inversionson.helpers.gradient_summer import GradientSummer
from inversionson.helpers.iteration_listener import IterationListener
from inversionson.utils import write_xdmf
from .helpers import mesh_to_vector, vector_to_mesh
from salvus.mesh.unstructured_mesh import UnstructuredMesh as UM
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

    def _prepare_iteration(self, x: OptsonVec) -> None:
        model_file = self.project.paths.get_model_path(x.descriptor)
        if not model_file.exists():
            m = vector_to_mesh(
                x,
                self.project.lasif.master_mesh,
                self.project.config.inversion.inversion_parameters,
            )
            m.write_h5(model_file)
        self._get_iteration(iteration=x.descriptor)

    @CallCounter
    def f(self, x: OptsonVec) -> float:
        # Step 1:: Write the mesh file
        self._prepare_iteration(x)

        IterationListener(
            project=self.project,
            events=self.project.events_in_iteration,
            control_group_events=self.project.events_in_iteration,
            prev_control_group_events=None,
            misfit_only=True,
            prev_iteration=None,
            submit_adjoint=False,
        ).listen()

        events = set(self.project.events_in_iteration) - set(
            self.project.config.monitoring.validation_dataset
        )
        total_misfit = 0.0
        for event in events:
            total_misfit += self.project.misfits[event]
        total_misfit /= len(events)
        return total_misfit

    @CallCounter
    def g(self, x: OptsonVec) -> OptsonVec:
        self.f(x)

        IterationListener(
            project=self.project,
            events=self.project.events_in_iteration,
            control_group_events=self.project.events_in_iteration,
            prev_control_group_events=None,
            misfit_only=False,
            prev_iteration=None,
            submit_adjoint=True,
        ).listen()

        grad_summer = GradientSummer(project=self.project)
        events = set(self.project.events_in_iteration) - set(
            self.project.config.monitoring.validation_dataset
        )
        raw_grad_f = self.project.paths.get_raw_gradient_path(x.descriptor)
        grad_summer.sum_gradients(
            events=events,
            output_location=raw_grad_f,
            batch_average=True,
            sum_vpv_vph=False,
            store_norms=True,
        )
        write_xdmf(raw_grad_f)

        return mesh_to_vector(
            UM.from_h5(raw_grad_f), self.project.config.inversion.inversion_parameters
        )
