from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, TYPE_CHECKING

from .component import Component

if TYPE_CHECKING:
    from inversionson.project import Project


class Mesh(Component):
    """
    Communications with Salvus Mesh.
    This will have to be done in a temporary way to begin with
    as it is not possible to make smoothiesem meshes through
    a nice config as things stand.

    :param infodict: Information related to inversion project
    :type infodict: Dictionary
    """

    def __init__(self, project: Project):
        super().__init__(project)

    def print(
        self,
        message: str,
        color: Optional[str] = None,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = ":globe_with_meridians:",
    ) -> None:
        self.project.storyteller.printer.print(
            message=message,
            color=color,
            line_above=line_above,
            line_below=line_below,
            emoji_alias=emoji_alias,
        )

    def fill_inversion_params_with_zeroes(self, mesh: Union[Path, str]) -> None:
        """
        This is done because we don't interpolate every layer and then
        we want to make sure there is nothing sneaking into the gradients

        :param mesh: Path to mesh
        :type mesh: str
        """
        self.print("Filling inversion parameters with zeros before interpolation")
        m = self.project.lasif.master_mesh.copy()
        parameters = self.project.config.inversion.inversion_parameters
        zero_element_nodal = np.zeros_like(m.element_nodal_fields[parameters[0]])

        for param in parameters:
            m.attach_field(param, zero_element_nodal)
        m.write_h5(mesh)

    def move_model_to_cluster(self, iteration: Optional[str] = None):
        iteration = iteration or self.project.current_iteration
        local_model = self.project.paths.get_model_path(iteration)
        remote_model = self.project.remote_paths.get_master_model_path(iteration)

        hpc_cluster = self.project.flow.hpc_cluster
        if not hpc_cluster.remote_exists(remote_model):
            self.project.flow.safe_put(local_model, remote_model)
