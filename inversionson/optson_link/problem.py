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
from .helpers import vector_to_mesh


from inversionson.project import Project


class Problem(AbstractProblem):
    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def _get_iteration(self, iteration: str):
        if self.project.lasif.has_iteration(iteration):
            self.project.set_iteration_attributes(iteration=iteration)
            return

        # We need a bunch of events, mocking this noew with a single event
        events = self.project.event_db.get_event_names([0])

        self.project.lasif.set_up_iteration(iteration, events)
        self.project.create_iteration_toml(iteration)
        self.project.set_iteration_attributes(iteration)

        # We still need to upload the model.

        hpc_cluster = self.project.flow.hpc_cluster
        remote_model_file = self.project.remote_paths.get_master_model_path(iteration)
        if not hpc_cluster.remote_exists(remote_model_file):
            self.project.flow.safe_put(self.project.paths.get_model_path(iteration), remote_model_file)


    @CallCounter
    def f(self, x: OptsonVec) -> float:
        # Step 1:: Write the mesh file
        model_file = self.paths.get_model_path(x.descriptor)
        if not model_file.exists():
            m = vector_to_mesh(x, self.project.lasif.master_mesh)
            m.write(model_file)
        
        self._get_iteration(iteration=x.descriptor)

    @CallCounter
    def g(self, x: OptsonVec) -> OptsonVec:
        pass]
