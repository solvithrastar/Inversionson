from .component import Component
import os
import shutil
import multi_mesh.api as mapi
import lasif.api as lapi


class MultiMeshComponent(Component):
    """
    Communication with Lasif
    """

    def __init__(self, communicator, component_name):
        super(MultiMeshComponent, self).__init__(communicator, component_name)
        self.physical_models = self.comm.salvus_opt.models

    def interpolate_to_simulation_mesh(self, event: str):
        """
        Interpolate current master model to a simulation mesh.

        :param event: Name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.model_interpolation_mode
        simulation_mesh = lapi.get_simulation_mesh(self.comm.lasif.lasif_comm,
                                                   event, iteration)
        if mode == "gll2gll":
            model = os.path.join(self.physical_models, iteration + ".h5")
            # There are many more knobs to tune but for now lets stick to
            # defaults.
            mapi.gll_2_gll(
                from_gll=model,
                to_gll=simulation_mesh,
                nelem_to_search=50,
                from_model_path="MODEL/data",
                to_model_path="MODEL/data",
                from_coordinates_path="MODEL/coordinates",
                to_coordinates_path="MODEL/coordinates",
                parameters=["VP", "VS", "RHO"]
            )
        elif mode == "ex2gll":
            model = os.path.join(self.physical_models, iteration + ".e")
            # This function can be further specified for different inputs.
            # For now, let's leave it at the default values.
            # This part is not really maintained for now
            mapi.exodus2gll(
                mesh=model,
                gll_model=simulation_mesh
            )
        else:
            raise ValueError(f"Mode: {mode} not supported")

    def interpolate_gradient_to_model(self, event: str, first=False):
        """
        Interpolate gradient parameters from simulation mesh to master
        dicretisation
        
        :param event: Name of event
        :type event: str
        :param first: First gradient to be interpolated. If it should be
        summed on top of an existing gradient, put False, defaults to False
        :type first: bool, optional
        """
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.gradient_interpolation_mode
        gradient = self.comm.lasif.find_gradient(iteration, event)
        if first:
            master_model = self.comm.lasif.get_master_model()
            summed_gradient = self.comm.salvus_opt.get_model_path(
                iteration, gradient=True)
            shutil.copy(master_model, summed_gradient)
        else:
            summed_gradient = self.comm.salvus_opt.get_model_path(
                iteration, gradient=True)
        if mode == "gll2gll":
            mapi.gll_2_gll(
                from_gll=gradient,
                to_gll=summed_gradient,
                nelem_to_search=5,
                parameters=["VS", "VP", "RHO"],
                gradient=True
                )
        elif mode == "gll2exo":
            # Not being maintained currently
            mapi.gll_2_exodus(
                gll_model=gradient,
                exodus_model=summed_gradient,
                nelem_to_search=5,
                parameters=["VS", "VP", "RHO"],
                gradient=True,
                first=first
            )
        else:
            raise ValueError(f"Mode: {mode} not implemented")


