"""_summary_
TODO: add smoothing option
TODO: Support control groups/adam/event selection etc.
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

    def _get_iteration(self, iteration: str) -> None:
        if self.project.paths.get_iteration_toml(iteration).exists():
            self.project.set_iteration_attributes(iteration=iteration)
            return

        # We need a bunch of events, mocking this for now with a single event
        events = self.project.event_db.get_event_names([0])
        self.project.lasif.set_up_iteration(iteration, events)
        self.project.create_iteration_toml(iteration, events)

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
            prev_control_group_events=None,
            misfit_only=True,
            prev_iteration=None,
            submit_adjoint=False,
        ).listen()

        total_misfit = 0.0
        for event in self.project.non_val_events_in_iteration:
            total_misfit += self.project.misfits[event]
        total_misfit /= len(self.project.non_val_events_in_iteration)
        return total_misfit

    @CallCounter
    def g(self, x: OptsonVec) -> OptsonVec:
        self.f(x)  # Ensure forward is completed.
        raw_grad_f = self.project.paths.get_raw_gradient_path(x.descriptor)

        if not raw_grad_f.exists():
            IterationListener(
                project=self.project,
                events=self.project.events_in_iteration,
                prev_control_group_events=None,
                misfit_only=False,
                prev_iteration=None,
                submit_adjoint=True,
            ).listen()

            GradientSummer(project=self.project).sum_gradients(
                events=self.project.non_val_events_in_iteration,
                output_location=raw_grad_f,
                batch_average=True,
                sum_vpv_vph=False,
                store_norms=True,
            )

        return mesh_to_vector(
            raw_grad_f, self.project.config.inversion.inversion_parameters
        )
