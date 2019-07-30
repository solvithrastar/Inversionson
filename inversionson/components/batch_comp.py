"""
A class which takes care of everything related to selecting
batches and control groups for coming iterations.
"""

from .component import Component
import numpy as np
import h5py


class BatchComponent(Component):
    """
    Computations that need to be done in order to do a mini batch inversion.
    """

    def __init__(self, communicator, component_name):
        super(BatchComponent, self).__init__(communicator, component_name)

    def _dropout(self, events: list) -> list:
        """
        Takes in a list of events. Picks random events based on a probability
        and drops them out of the list. This is a form of regularization
        used to get rid of events which have a high influence on the total
        gradient and stay in the control group for too long. This event
        might be a flawed event and thus we don't want any event to stay
        for too long in the control group.

        :param events: The events picked for control group
        :type events: list
        :return: A list of events which have been dropped.
        :rtype: list
        """

    def _assert_parameter_in_mesh(self, mesh: str):
        """
        It is not trivial to find parameters in an hdf5 mesh. This function
        takes care of that and returns an index where the parameter is kept.
        The name of the parameter has to be exactly the same as in the mesh.
        I rely on the arrange parameters method in project to arrange
        parameters well

        :param mesh: Path to mesh
        :type mesh: str
        :param parameter: Name of parameter
        :type parameter: str
        :return: Index where parameter is kept
        :rtype: int
        """
        # TODO: Check on multimesh to see how it's done there.
        #       remember to raise a warning when it doesn't fit with
        #       what is expected.
        with h5py.File(mesh, 'r') as mesh:
            params = mesh["MODEL/data"].attrs.get(
                "DIMENSION_LABELS")[1].decode()
            params = params[2:-2].replace(" ",
                                          "").replace("grad", "").split("|")
        # Not sure if I should replace the "grad" in this case
        msg = "Parameters in gradient are not the same as inversion perameters"
        assert params == self.comm.project.inversion_params, msg

    def _angle_between(self, gradient_1, gradient_2) -> float:
        """
        Compute the angle between two gradients
        
        :param gradient_1: numpy array with gradient info
        :type gradient_1: numpy array
        :param gradient_2: numpy array with gradient info
        :type gradient_2: numpy array
        :return: angle between the two
        :rtype: float
        """
        norm_1 = np.linalg.norm(gradient_1)
        norm_2 = np.linalg.norm(gradient_2)
        angle = np.arccos(
            np.dot(gradient_1, gradient_2) /
            (norm_1 * norm_2) / np.pi * 180.0
        )
        return angle

    def _compute_angular_change(self, full_gradient, full_norm,
                                individual_gradient) -> float:
        """
        Compute the angular change fo the full gradient, when the individual
        gradient is removed from it.

        :param full_gradient: Numpy array, containing the full summed gradient
        :type full_gradient: np.array, np.float64
        :param full_norm: The norm of the full_gradient
        :type full_norm: float
        :param individual_gradient: Numpy array with the gradient to be removed
        :type individual_gradient: np.array, np.float64
        :return: The angular difference resulting from removing gradient
        :rtype: float
        """
        test_grad = np.copy(full_gradient) - individual_gradient
        test_grad_norm = np.linalg.norm(test_grad)
        angle = np.arccos(
            np.dot(test_grad, full_gradient) /
            (test_grad_norm * full_norm) / np.pi * 180.0
        )
        return angle

    def select_optimal_control_group(self) -> list:
        """
        Takes the computed gradients, figures out which are the most
        influential for the inversion and selects them as a control group.
        Currently designed for a dynamic batch size. Maybe implemented for
        a constant batch size later

        :return: List of events to be used in the control group
        :rtype: list
        """
        # Compute full gradient
        # sequentially remove one gradient and compute angle
        # remove the event with minimal angle
        # look for event to remove until a certain total angle
        # is reached or minimum control group is reached.
        # We just use the bulk norm of the gradients it seems
        events = self.comm.project.events_used
        ctrl_group = events
        max_ctrl = self.comm.project.max_ctrl_group_size
        min_ctrl = self.comm.project.min_ctrl_group_size
        iteration = self.comm.project.current_iteration
        gradient_paths = []
        for _i, event in enumerate(events):
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event
            )
            gradient_paths.append(gradient)
            with h5py.File(gradient, "r") as f:
                grad = f["MODEL/data"]
                if _i == 0:
                    full_grad = grad
                else:
                    full_grad += grad
        # TODO: Monitor event quality
        # TODO: Can I not just store the angle for all and remove like that?
        #       We want to approximate the total gradient so that might
        #       even be better right?
        full_grad_norm = np.linalg.norm(full_grad)
        ctrl_group_grad = np.copy(full_grad)
        ctrl_group_norm = np.linalg.norm(ctrl_group_grad)

        angular_changes = {}
        for event in events:
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event
            )
            with h5py.File(gradient, "r") as f:
                individual_gradient = f["MODEL/data"]
            angle = self._compute_angular_change(
                full_gradient=full_grad,
                full_norm=full_grad_norm,
                individual_gradient=individual_gradient
            )
            angular_changes[event] = angle
        all_angular_changes = angular_changes 
        batch_grad = np.copy(full_grad)
        while len(ctrl_group) >= min_ctrl or len(ctrl_group) > max_ctrl:
            redundant_gradient = min(angular_changes, key=angular_changes.get)

            with h5py.File(redundant_gradient, "r") as f:
                batch_grad -= f["MODEL/data"]
            angle = self._angle_between(full_grad, batch_grad)

            if angle >= self.comm.project.maximum_grad_divergence_angle:
                #TODO: In this case I actually want the previous one.
                break
            
        #TODO: Use all_angular_changes to score event quality

        while len(ctrl_group) >= min_ctrl or len(ctrl_group) > max_ctrl:
            min_angle = 180.0
            for event in events:
                gradient = self.comm.lasif.find_gradient(
                    iteration=iteration,
                    event=event
                )
                with h5py.File(gradient, "r") as f:
                    individual_gradient = f["MODEL/data"]
                angle = self._compute_angular_change(
                    full_gradient=full_grad,
                    full_norm=full_grad_norm,
                    individual_gradient=individual_gradient
                )
                if angle < min_angle:
                    min_angle = angle
                    redundant_gradient = gradient


