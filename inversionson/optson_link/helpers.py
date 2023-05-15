from typing import List
from salvus.mesh.unstructured_mesh import UnstructuredMesh as UM
from optson.problem import OptsonVec
import numpy as np


def mesh_to_vector(m: UM, params_to_invert: List[str]) -> OptsonVec:
    par_list = [m.element_nodal_fields[param].flatten() for param in params_to_invert]
    return OptsonVec(np.concatenate(par_list))


def vector_to_mesh(x: OptsonVec, target_mesh: UM, params_to_invert=List[str]) -> UM:
    par_vals = np.array_split(x, len(params_to_invert))
    m = target_mesh.copy()

    for idx, param in enumerate(params_to_invert):
        m.element_nodal_fields[param][:] = par_vals[idx].reshape(
            m.element_nodal_fields[param].shape
        )
    return m
