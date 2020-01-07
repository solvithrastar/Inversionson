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

    def interpolate_to_simulation_mesh(self, event: str, interp_folder=None):
        """
        Interpolate current master model to a simulation mesh.

        :param event: Name of event
        :type event: str
        """
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.model_interpolation_mode
        simulation_mesh = lapi.get_simulation_mesh(
            self.comm.lasif.lasif_comm, event, iteration
        )
        if mode == "gll_2_gll":
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
                parameters=self.comm.project.modelling_params,
                stored_array=interp_folder
            )
        elif mode == "exodus_2_gll":
            model = os.path.join(self.physical_models, iteration + ".e")
            # This function can be further specified for different inputs.
            # For now, let's leave it at the default values.
            # This part is not really maintained for now
            mapi.exodus2gll(mesh=model, gll_model=simulation_mesh)
        else:
            raise ValueError(f"Mode: {mode} not supported")

    def interpolate_gradient_to_model(self, event: str, smooth=True,
                                      interp_folder=None):
        """
        Interpolate gradient parameters from simulation mesh to master
        dicretisation. In minibatch approach gradients are not summed,
        they are all interpolated to the same discretisation and salvus opt
        deals with them individually.
        
        :param event: Name of event
        :type event: str
        :param smooth: Whether the smoothed gradient should be used
        :type smooth: bool, optional
        :param interp_folder: Pass a path if you want the matrix of the
        interpolation to be saved and then it can be used later on. Also
        pass this if the directory exists and you want to use the matrices
        """
        iteration = self.comm.project.current_iteration
        mode = self.comm.project.gradient_interpolation_mode
        gradient = self.comm.lasif.find_gradient(iteration, event, smooth=smooth)

        master_model = self.comm.lasif.get_master_model()
        # summed_gradient = self.comm.salvus_opt.get_model_path(
        #     iteration, gradient=True)
        seperator = "/"
        master_disc_gradient = (
            seperator.join(gradient.split(seperator)[:-1]) + "/smooth_grad_master.h5"
        )
        shutil.copy(master_model, master_disc_gradient)

        if mode == "gll_2_gll":
            mapi.gll_2_gll(
                from_gll=gradient,
                to_gll=master_disc_gradient,
                nelem_to_search=300,
                parameters=self.comm.project.inversion_params,
                gradient=True,
                stored_array=interp_folder
            )
            self.comm.salvus_mesher.write_xdmf(master_disc_gradient)
        elif mode == "gll2exo":  # This will probably be removed soon
            mapi.gll_2_exodus(
                gll_model=gradient,
                exodus_model=summed_gradient,
                nelem_to_search=5,
                parameters=self.comm.project.inversion_params,
                gradient=True,
                first=True,
            )
            self.comm.salvus_mesher
        else:
            raise ValueError(f"Mode: {mode} not implemented")
