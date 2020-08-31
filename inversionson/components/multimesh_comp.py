from salvus.flow.simple_config import simulation
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
            if "validation" in iteration:
                iteration = iteration.replace("validation_", "")

                if (
                    self.comm.project.when_to_validate > 1
                    and iteration != "it0000_model"
                ):
                    it_number = (
                        self.comm.salvus_opt.get_number_of_newest_iteration()
                    )
                    old_it = it_number - self.comm.project.when_to_validate + 1
                    model = (
                        self.comm.salvus_mesher.average_meshes
                        / f"it_{old_it}_to_{it_number}"
                        / "mesh.h5"
                    )
                else:
                    model = os.path.join(
                        self.physical_models, iteration + ".h5"
                    )

            # There are many more knobs to tune but for now lets stick to
            # defaults.
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="layer",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="fluid",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=self.comm.project.domain_file,
                to_mesh=model,
                field_name="moho_idx",
                global_string=True,
                overwrite=False,
            )
            mapi.gll_2_gll_layered(
                from_gll=model,
                to_gll=simulation_mesh,
                layers="nocore",
                nelem_to_search=20,
                parameters=self.comm.project.modelling_params,
                stored_array=interp_folder,
            )
        elif mode == "exodus_2_gll":
            model = os.path.join(self.physical_models, iteration + ".e")
            # This function can be further specified for different inputs.
            # For now, let's leave it at the default values.
            # This part is not really maintained for now
            mapi.exodus2gll(mesh=model, gll_model=simulation_mesh)
        else:
            raise ValueError(f"Mode: {mode} not supported")

    def interpolate_gradient_to_model(
        self, event: str, smooth=True, interp_folder=None
    ):
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
        gradient = self.comm.lasif.find_gradient(
            iteration, event, smooth=smooth
        )
        simulation_mesh = self.comm.lasif.get_simulation_mesh(event_name=event)

        master_model = self.comm.lasif.get_master_model()
        # summed_gradient = self.comm.salvus_opt.get_model_path(
        #     iteration, gradient=True)
        # seperator = "/"
        # master_disc_gradient = (
        #     seperator.join(gradient.split(seperator)[:-1])
        #     + "/smooth_grad_master.h5"
        # )
        master_disc_gradient = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event,
            smooth=True,
            inversion_grid=True,
            just_give_path=True,
        )
        shutil.copy(master_model, master_disc_gradient)

        if mode == "gll_2_gll":
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=simulation_mesh,
                to_mesh=gradient,
                field_name="layer",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=simulation_mesh,
                to_mesh=gradient,
                field_name="fluid",
                elemental=True,
                overwrite=False,
            )
            self.comm.salvus_mesher.add_field_from_one_mesh_to_another(
                from_mesh=master_model,
                to_mesh=gradient,
                field_name="moho_idx",
                global_string=True,
                overwrite=False,
            )
            mapi.gll_2_gll_layered(
                from_gll=gradient,
                to_gll=master_disc_gradient,
                nelem_to_search=20,
                layers="nocore",
                parameters=self.comm.project.inversion_params,
                stored_array=interp_folder,
            )
            self.comm.salvus_mesher.write_xdmf(master_disc_gradient)
        else:
            raise ValueError(f"Mode: {mode} not implemented")
