"""
A class which takes care of everything related to selecting
batches and control groups for coming iterations.
"""

from .component import Component
import numpy as np
import h5py
import random
import math
from colorama import Fore, Back, Style
import time
import toml
from tqdm import tqdm
from typing import Union


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
        dropout = []
        # Find the current control group and if an event was not in it,
        # we don't want to drop it out. Every event selected for a control
        # group thus gets a chance to stay in at least once.
        # make sure it's not more likely that early events are dropped
        random.shuffle(events)
        for event in events:
            if event not in self.comm.project.old_control_group:
                continue
            if random.random() < self.comm.project.dropout_probability:
                dropout.append(event)
        return dropout

    def _assert_parameter_in_mesh(self, mesh: str):  # Not used
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
        with h5py.File(mesh, "r") as mesh:
            params = mesh["MODEL/data"].attrs.get("DIMENSION_LABELS")[1]
            params = (
                params[1:-1].replace(" ", "").replace("grad", "").split("|")
            )
        # Not sure if I should replace the "grad" in this case
        msg = "Parameters in gradient are not the same as inversion perameters"
        assert params == self.comm.project.inversion_params, msg

    def _get_unique_points(self, points: np.ndarray):
        """
        Take an array of coordinates and find the unique coordinates. Returns
        the unique coordinates and an array of indices that can be used to
        reconstruct the previous array.

        :param points: Coordinates, or a file
        :type points: numpy.ndarray
        """
        all_points = points.reshape(
            (points.shape[0] * points.shape[1], points.shape[2])
        )
        return np.unique(all_points, return_index=True, axis=0)

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
        value = np.dot(gradient_1, gradient_2) / (norm_1 * norm_2)
        eps = 1.0e-6
        if value > 1.0 and (value - 1.0) < eps:
            value = 1.0
        # print(f"Value passed to arccos in angle between: {value}")
        angle = np.arccos(value) / np.pi * 180.0
        return angle

    def _compute_angular_change(
        self,
        full_gradient,
        full_norm,
        individual_gradient,
        ctrl_grp_grad: np.ndarray = None,
    ) -> float:
        """
        Compute the angular change fo the full gradient, when the individual
        gradient is removed from it.

        :param full_gradient: Numpy array, containing the full summed gradient
        :type full_gradient: np.array, np.float64
        :param full_norm: The norm of the full_gradient
        :type full_norm: float
        :param cntrl_gradient: Numpy array that is the sum of all control
        gradients
        :type cntrl_gradient: np.array
        :param individual_gradient: Numpy array with the gradient to be removed
        :type individual_gradient: np.array, np.float64
        :param ctrl_grp_grad: Numpy array, containing the current control group
            gradient. Used if you want to remove individual grad from
            control group gradient. If you just want to check the individual
            gradient with the full gradient, just pass None
        :type ctrl_grp_grad: np.array, np.float64
        :return: The angular difference resulting from removing gradient
        :rtype: float
        """
        if ctrl_grp_grad is not None:
            test_grad = np.copy(ctrl_grp_grad) - individual_gradient
        else:
            test_grad = np.copy(full_gradient) - individual_gradient
        test_grad_norm = np.linalg.norm(test_grad)
        value = np.dot(test_grad, full_gradient) / (test_grad_norm * full_norm)
        eps = 1.0e-6
        if value > 1.0 and (value - 1.0) < eps:
            value = 1.0
        # print(f"Value passed to arccos: {value}")
        angle = np.arccos(value) / np.pi * 180.0
        return angle

    def _sum_relevant_values(
        self, grad, param_ind: list, unique_indices: np.ndarray
    ) -> np.ndarray:
        """
        Take the gradient, find inverted parameters and sum them together.
        Reduces a 3D array to a 2D array.

        :param grad: Numpy array with gradient values
        :type grad: numpy.ndarray
        :param parameters: A list of indices where relevant gradient is kept
        :type parameters: list
        :param unique_indices: indices of the unique points in mesh
        :type unique_indices: numpy.ndarray
        :rtype: numpy.ndarray
        """
        # shapegrad = grad.element_nodal_fields[parameters[0]].shape
        # shapegrad = grad.shape
        # summed_grad = np.zeros(
        #     shape=(shapegrad[0] * shapegrad[1], shapegrad[2])
        # )
        # summed_grad = np.zeros_like(grad.element_nodal_fields[parameters[0]])
        for _i, ind in enumerate(param_ind):
            if _i == 0:
                summed_grad = grad[:, ind, :].reshape(
                    grad.shape[0] * grad.shape[2]
                )[unique_indices]
            else:
                summed_grad = np.concatenate(
                    (
                        summed_grad,
                        grad[:, ind, :].reshape(grad.shape[0] * grad.shape[2])[
                            unique_indices
                        ],
                    ),
                    axis=0,
                )
            # summed_grad[
            #     _i * shapegrad[0] : (_i + 1) * shapegrad[0], :
            # ] = grad.element_nodal_fields[param]
        return summed_grad

    def _get_vector_of_values(
        self, gradient, parameters: list, unique_indices: np.ndarray
    ) -> np.ndarray:
        """
        Take a full gradient, find it's unique values and relevant parameters,
        manipulate all of these into a vector of summed parameter values.

        :param gradient: Array of gradient values
        :type gradient: numpy.ndarray
        :param parameters: A list of dimension labels in gradient
        :type parameters: list
        :param unique_indices: indices of the unique points in mesh
        :type unique_indices: numpy.ndarray
        :return: 1D vector with gradient values
        :rtype: numpy.ndarray
        """
        with h5py.File(gradient, mode="r") as f:
            grad = f["MODEL/data"][()]
            dim_labels = (
                f["MODEL/data"]
                .attrs.get("DIMENSION_LABELS")[1][1:-1]
                .replace(" ", "")
                .split("|")
            )
            grad = grad[()]
            indices = []
            for param in parameters:
                indices.append(dim_labels.index(param))
            # relevant_grad = grad[:, indices, :]
        return self._sum_relevant_values(
            grad=grad, param_ind=indices, unique_indices=unique_indices
        )

    def _remove_individual_grad_from_full_grad(
        self,
        full_grad: np.ndarray,
        event: str,
        unique_indices=np.ndarray,
    ) -> np.ndarray:
        """
        Remove one gradient from the full gradient

        :param full_grad: A sum of all gradients
        :type full_grad: np.ndarray
        :param event: An event name of the individual gradient to be removed
        :type event: str
        """
        inversion_grid = False
        if self.comm.project.meshes == "multi-mesh":
            inversion_grid = True
        iteration = self.comm.project.current_iteration
        parameters = self.comm.project.inversion_params
        gradient = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=event,
            smooth=True,
            inversion_grid=inversion_grid,
        )
        # individual_gradient = um.from_h5(gradient)

        individual_gradient = self._get_vector_of_values(
            gradient=gradient,
            parameters=parameters,
            unique_indices=unique_indices,
        )
        return full_grad - individual_gradient

    def _find_most_useless_event(
        self,
        full_gradient: np.ndarray,
        events: list,
        unique_indices: np.ndarray,
        cntrl_gradient: np.ndarray,
    ) -> Union[str, np.ndarray]:
        """
        For a given gradient, which of the events which compose the full_grad
        is has the smallest influence on its summed direction?

        :param full_gradient: Summed gradient for all events in events
        :type full_gradient: np.ndarray
        :param events: A list of event names
        :type events: list
        :param cntrl_gradient: Summed gradient for all events in current
        control group
        :type cntrl_gradient: np.ndarray
        :return: Name of the event and the reduced gradient
        :rtype: Union[str, np.ndarray]
        """

        inversion_grid = False
        if self.comm.project.meshes == "multi-mesh":
            inversion_grid = True
        iteration = self.comm.project.current_iteration
        parameters = self.comm.project.inversion_params
        full_gradient_norm = np.linalg.norm(full_gradient)
        event_angles = {}
        for event in events:
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=inversion_grid,
            )
            # individual_gradient = um.from_h5(gradient)

            individual_gradient = self._get_vector_of_values(
                gradient=gradient,
                parameters=parameters,
                unique_indices=unique_indices,
            )
            assert not np.any(
                np.isnan(individual_gradient)
            ), f"Nan values in individual_gradient for {event}"

            angle = self._compute_angular_change(
                full_gradient=full_gradient,
                full_norm=full_gradient_norm,
                individual_gradient=individual_gradient,
                ctrl_grp_grad=cntrl_gradient,
            )
            event_angles[event] = angle
            print(f"Angle computed for event: {event}: {angle}")
        redundant_gradients = sorted(
            event_angles.items(), key=lambda x: x[1], reverse=False
        )
        # redundant_gradient = min(event_angles, key=event_angles.get)
        # print(f"Most redundant: {redundant_gradient}")

        # reduced_gradient = self._remove_individual_grad_from_full_grad(
        #     cntrl_gradient, redundant_gradient, unique_indices=unique_indices,
        # )

        return redundant_gradients  # , reduced_gradient

    def get_random_event(
        self, n: int, existing: list, avail_events: list
    ) -> list:
        """
        Get an n number of events based on the probabilities defined
        in the event_quality toml file

        :param n: Number of events to randomly choose
        :type n: int
        :param existing: Events blocked from selection
        :type existing: list
        :param avail_events: Events available in lasif (important if some)
            were removed at any point but remain in events_quality
        :type avail_events: list
        :return: List of events randomly picked
        :rtype: list
        """
        events_quality = toml.load(self.comm.storyteller.events_quality_toml)
        not_usable = list(set(events_quality.keys()) - set(avail_events))
        for k in existing:
            del events_quality[k]
        for k in self.comm.project.validation_dataset:
            if k in events_quality.keys():
                del events_quality[k]
        for k in self.comm.project.test_dataset:
            if k in events_quality.keys():
                del events_quality[k]
        if len(not_usable) > 0:
            for k in not_usable:
                if k in events_quality.keys():
                    del events_quality[k]
        list_of_events = list(events_quality.keys())
        list_of_probabilities = list(events_quality.values())
        list_of_probabilities /= np.sum(list_of_probabilities)

        chosen_events = list(
            np.random.choice(
                list_of_events, n, replace=False, p=list_of_probabilities
            )
        )
        return chosen_events

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

        events = self.comm.project.events_in_iteration
        n_events_in_project = len(self.comm.lasif.list_events())
        n_events_in_project -= len(self.comm.project.validation_dataset)
        n_events_in_project -= len(self.comm.project.test_dataset)
        max_ctrl_group = math.floor(n_events_in_project / 2.0)
        print(f"Control batch events: {events}")
        ctrl_group = events.copy()
        min_ctrl = self.comm.project.min_ctrl_group_size
        iteration = self.comm.project.current_iteration
        gradient_paths = []
        inversion_grid = False
        if self.comm.project.meshes == "multi-mesh":
            inversion_grid = True
        parameters = self.comm.project.inversion_params
        print("Summing Gradients: \n")
        for _i, event in tqdm(enumerate(events), total=len(events)):
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=inversion_grid,
            )
            if _i == 0:
                with h5py.File(gradient, mode="r") as f:
                    _, unique_indices = self._get_unique_points(
                        points=f["MODEL/coordinates"][()]
                    )
            gradient_paths.append(gradient)
            # grad = um.from_h5(gradient)
            if _i == 0:
                full_grad = self._get_vector_of_values(
                    gradient=gradient,
                    parameters=parameters,
                    unique_indices=unique_indices,
                )
            else:
                full_grad += self._get_vector_of_values(
                    gradient=gradient,
                    parameters=parameters,
                    unique_indices=unique_indices,
                )

        full_grad_norm = np.linalg.norm(full_grad)
        print(f"Full grad norm: {full_grad_norm}")
        assert not np.any(np.isnan(full_grad)), "Nan values in full gradient"
        event_quality = {}
        batch_grad = np.copy(full_grad)
        removal_order = 0
        # Dropout now.
        if "it0000" not in iteration:
            dropped_events = self._dropout(ctrl_group.copy())

            for event in dropped_events:
                event_quality[event] = 0.75
                batch_grad = self._remove_individual_grad_from_full_grad(
                    batch_grad,
                    event,
                    unique_indices=unique_indices,
                )
                ctrl_group.remove(event)
                print(f"Event: {event} randomly dropped from ctrl group")

        while len(ctrl_group) > min_ctrl:
            removal_order += 1
            event_names = self._find_most_useless_event(
                full_gradient=full_grad,
                cntrl_gradient=batch_grad,
                events=ctrl_group,
                unique_indices=unique_indices,
            )
            i = 0
            while i < 5 and len(ctrl_group) > min_ctrl:
                event_name = event_names[i][0]
                test_batch_grad = self._remove_individual_grad_from_full_grad(
                    batch_grad,
                    event_name,
                    unique_indices=unique_indices,
                )
                angle = self._angle_between(
                    full_grad,
                    test_batch_grad,
                )
                print(f"Angle between test_batch and full gradient: {angle}")
                if (
                    angle >= self.comm.project.maximum_grad_divergence_angle
                    and len(ctrl_group) <= max_ctrl_group
                ):
                    break
                else:
                    batch_grad = np.copy(test_batch_grad)
                    # del angular_changes[redundant_gradient]
                    event_quality[event_name] = removal_order / len(events)
                    # event_quality[event_name] = 1 / len(ctrl_group)
                    ctrl_group.remove(event_name)
                    print(f"{event_name} does not continue to next iteration")
                    print(f"Current size of control group: {len(ctrl_group)}")
                    i += 1
            if (
                angle >= self.comm.project.maximum_grad_divergence_angle
                and len(ctrl_group) <= max_ctrl_group
            ):
                break

        for key, val in event_quality.items():
            self.comm.project.event_quality[key] = val
        print(f"Control batch events: {ctrl_group}")
        print("\n \n ============================= \n \n")
        print(f"Number of Control group events: {len(ctrl_group)}")

        return ctrl_group

    def increase_control_group_size(self):
        """
        If the control group is not deemed good enough, we need to add more
        events into the control group.
        We thus find the highest quality events from the iteration and add
        to the batch.
        """
        events = self.comm.project.events_in_iteration
        ctrl_group = self.comm.project.new_control_group
        events_quality = self.comm.project.event_quality

        non_ctrl_events = list(set(events) - set(ctrl_group))
        best = 0.0
        for event in non_ctrl_events:
            if events_quality[event] > best:
                add_to_ctrl_group = event
        events_quality[add_to_ctrl_group] = 0.0
        ctrl_group.append(add_to_ctrl_group)
        print(f"\n Event: {add_to_ctrl_group} added to control group \n")
        self.comm.project.change_attribute(
            attribute="new_control_group", new_value=ctrl_group
        )
        self.comm.project.update_control_group_toml(new=True)
        self.comm.project.update_iteration_toml()

    def print_dp(self):
        """
        Print DP's face. Important to give proper credit.
        """

        string = """
                                `. .:oydhy/..`        `.`                                           
                          `-+osdMNNMMMMNMMMmNNh+//osh+osdmdo-                                       
                       `.-yNMMMMMMMMNMNmNhNNMMMNNNddso+-/ysNhs:.+//                                 
                     .`.-sMMMMMMMMMMMMNMMMMMMMMMMMMms+syyyyNNdNmh:h`                                
                   `-`--omMMMMMMMMMMMMMMMMMMMMMMMMMMNNMMMMMMMmMNNNd:-`./syo+/-`                     
                  .ssyyhmMMMMMMMMMMMMMMMMMMNMMMMMMMMMMMMMMMMMMMMMMMmdNNmNMMMNNms:                   
                .odsymmMMMMMMMMMMMMMMMMMMMMNMMMMMMMMMMMMMMMMMMMMMMMMMMMdddMMMMmdo/                  
              `.y+hdmNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMdMMMMNMdy.                 
              //ysmdymMNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMmMMMMNMhm- `               
              sdhdmmmNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNMMMMMNm/.-               
             :hdmNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmho``             
           `/ymdNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNMNMNh/`            
           -symNNMMMMMMMMMMMMMMMMMMMMMNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNh+-`           
           .+hNMMMMMMMMMMMMMNNmmmdddddddddmmmmmmddddmmmmmmmmmmmmmNNMMMMMMMMMMMMMMMMMNNdso+-`        
         ...syMMMMMMMMMNmdhyyssooooooosssssssssoooooooossssooooosydmmmmmMMMMMMMMMMMMMNNh+/:.        
         s:+ddMMMMMMNmhysoo+++++////////::::---::/:::::///+////+oyhyhyddddmMMMMMMMMMMMMNmdho`       
        -osdNMMMNddhso++++////:::-----------.--::::::::::::::://oysyyymhssydmNMMMMMMMMMMMMMMy-      
       .+hmNMMNy+++///////////:::------.....-------::-::--:::///+ossyshhsssosyhdmNMMMMMMMNmmdh-     
       /NNNNMd+::-::///::::::::::::------..-----..---:---:::::///+++ossssso++++osymMMMMMMMMMNNh.    
      /mNmMMd/:-::::///::::/::::--------.--..-.-.--.--------::://///++++++///////+sdMMMMMMMMMMN:    
     :mMNMMd/:::::://:::::::::-------..................--------::::///////////////+sdMMMMNMMMNNy`   
    -dNmMMm+::::////::::::----.........````.....``..........---:::::::::///////////+yNNMMMMMMMNd/   
    .smNNmy/:://////::::::----..........``.................---------:::::://////////+dNMMMMMMMMNo   
    :hmNddo/://////////::-------..```....````.........-------------:::::::://///////+ymNMMMMMMMMm.  
  `ydmmmdyo////////////:::::-----........```````.....-------::::::::/://:://///////+oymmNNNMMMMMMs  
  :hhmdhyys+///////////::::::-----------............-----:::::::::////////////////++ohmNmmNMMMMNMN- 
  hhdhdddhs+///////////::::::::------------........-------::::::://///////////////++shmNmNMMMMMmMM+ 
 -dmNNNmmdy+/////+////::------------------..........-----..-----::::::///////////++oshdNNNNMMMMNmNh 
 yNmNNNNmds+////////::--...............---...--....--.-..``......-----::://///++/++osydmNNmMNNNNNds 
:hmmmMMNdy+/////+//:::-.....``````````.......-........``````````......--://///++++++oyhmNNNMNNNNNN- 
-hmdNMNds+////++///:::----....``````````````````.```````````````..------://///++++++osyhNMNMMNNMNm- 
 sNmMMms////++++ooo++++++++////::...``````````````````````..-::::://////+++o+++++++++ooydMMMMNMNNd` 
 :NNMMs////+++oydddddddmNNNNNNNmdh+/:/:-.``````````.-:-//oydmmmmmmmdddhyshhhys++++o+++++smMMMMNNMm. 
  dNMNo////+oydmNNNNNMMMMMMMMMMMMNmhyoo/:-.`````..-:/+oyddNMMMMMMMMMMMMNNNmmhdhso+++++/++hMMMMNNMN- 
 `+MMN+////+ydmmNMMMMMNNmmNNNNNNNNNmdys+/:--.....-:/+syhddNNNNNNNNNNNMMMMMMMNmddhs+o+///+yMMMMMNNs` 
 `+NMN+::/+ohmmNNmmmmhso+/++oossyyhhhhso++:--...-:/+osyhyyyyssoo++++osyyhdmNmmmmdho+++//+yMMMMMNN.  
  -hMM+::/+syhhhhyso+/:::--://+++oossyyys+/-.``.-/+syyyyso++///::-...-:/+osyhdddddso+///+hMMNMMMy   
 `-hMMo:://+oooooo++//++oossyyyssoo+oosyyo/:-...-+syysso+oossyyyyysso++++++ossyyyhyo+///+hMMMMMd.   
 -smMM+::://////+oyhdmmNMMNNNNNdhhhyo++syyo+//::+oyys//+shhdhyhMNmmNMMNNmdhysooosoo++///+dMMMMNo.   
 /+omN+:::///+oshmNNmo:yMMNNNMN/-oyhy+-/yyso+//+osyso--oyyss:`oMNmmNMMdhmNNmdyso++++////+mMMMMmy+-  
 :o/dN+::://+ossyyhyys+/hNNNNmhossyso++syys+++++osyysoo//osss+odNNNNNhoyhhddhyyso+++////omMMMMhyyo  
 .::sN+:-::/++++oo++++ooosooooosssooosyyys++++++osyyyss++ooooo++oooss+oooooosooo+++/////omMMMNyos/  
 .-:+d/--://++++++++//////+++ooo++++syyysooo++++osyyyyso+//++oo+++//::///++++++++++////+smMMNd+++`  
  .::d/--:///////+///+++///////::-:+syyssoo+++ooosssyyyo+:::://++ooooooooo++/+++++++///+oNMMdo///   
  .:+d/--:///////////////////:::://osysoooo++++osssssyys+/:-:::://++++///+//////+++/////oNMd+/::.   
  .-+h/--:::/::::::::://///:::::/+osssoooo++///oosssyyyyo+:----:::////::::::::///++////+oNNs+/:-    
  `-+ds:-:::::::::::::::::::::://++ossoooo+////+oossyyyso+//-..---:::::-------:::/+///++yNmoo/:`    
   :sdm:-::::------------:-----::/osysooooo+//:/+osyyyyys+/:-------......------::/+///+omMho+//     
    -:m/::::----.............---/osssoosss+/::-:/+oysssyys/:----.....``.......-:////++oydmsso/-     
     .h+:///:--:-...........--:/oso++++oo+/.````.:oosooosyo+/:-......`........-////+++shms-.::      
     .hyo/::--:-..-......--:/+++os+////+//-...``./ooso++syo+//:-......-.------:/+/+++ohdm/``-`      
     .yms+/:::///-----..-://+++osyssyhyso+/////:/ooyhdhyyys+//::-.----:::::////+o++ooydNm/-`        
      smyo+///o+/::--.-:/oo++oosyhdddhhysyysyysssyhdmmmdhyys+/::---:-::///+++++++ooosdmNh:-.        
      /mss++oyso/:::-:/ossoosssyydddmddhhhdddmddmddmmmmddhyyss+////::-:/++ooooo+ooysyymNy/-`        
      .Nhsosshmy+/:::/oyyhhyyysyhddddddhdmmmmmNmdddddmmdhhyyssyssyso/////+oossss+syyhmNN+.`         
      `hdhsoshhy+://++hhddhyyyyhhhhddhhhhmNNmmNmdhhyhdhhyhyhhyhhhyhhyoo+/+osyyssoshyydNm`           
       sNmdhoysoo+/+oyddhyysssoooosooo+o+osysssssysssyyyyyyyyyshhhhhhyso//osyys+oshymmNo            
       .NNdhyysssoo+oyyysssssssysssssssyysso+ossyyyssyyyyyyyyhhhhyyhhhhs++shhhssydhymmm-            
        yNmhyhdhy++/ooosooshhdmmmmmmdddhhhhyyyhhhhhhhhhhhhddmdhyyhyyyss+/+ydmdhyyddydNs             
        :mNddhhyyso++ooosssooosyyhhhhhhhhhhhhhhhhhyyyyyyyyyyso++/ossoo///ohhyyhdsydmmm.             
         smmdhdshhhs/+ooyso++oo++osyyysssoosssssoossyyyyyso++++++ooys+//oyddhyhhhdmmNo              
         `dmNmdddhddoosssooo+++++//++osssooo++ooooossoo+/////+++oooyyyyyhhhydhdddmNNm`              
          :NNNNmmmmmhhhhhyso+////////://+++o+++++////////+///++++ooshdmdNNmhmmNNMMMm:               
           oMMNNMMNNmmdmmdhso+////:/++++osssyhyshyyyso++/:://+ooosyhdmNNNNNMNMNMMMm:                
           `+NMMMMMNNmmNNmyyyo++///////+osyhhmddmhhyoo+//::/++ssyyhmmNNNmNMMMMMMMM/                 
             -mMMMNNMMNNNmyysssoo++/++oossyhhhdddhhysoo++///+ssyyhdmmNmNNNMMMMMMd+                  
              .yNMMMMNMMNNmdydyssssssyyssossoosyyyysysssoosssyhdhhdNNMNNNMNNMMMm`                   
                -mMMMNNMMNmmmdyshhysysyssyssysyysssyyssssyhhdddmmmNNNMMNMMNMMNh/                    
                 /smMMMMMMNNmdhdhhhhysyssssssyyyyssyssyyyhdmmmNmNmNMMMMMMMMMms+-                    
                 :++dNMMMMMNNmmddmhyyyssssssyyyyysoyssyyhyddmmmNNNNMMMMMMMMmyo+-                    
                 /+/+hmNMMMMNMNmdddyysosyyssyhdhhosysssydddmmNNMNMMMMMMMNNmho+/.                    
                 :+o+oymNNMMMMMNmddhhhyyyyhyhmNmmyyyhysshmmmNNNMMMMMMMNNNmds+++.                    
                `:::o++yhdNMMMMMMNNNmmmddhddNMMNmmhhyhhhdmmNMMMMMMMNNmmmhys++++-                    
              `-.::::++/oydNNNMNNMMMNNNNNNNNMMMMNNNmdmmNMMMMMMMNMNNNmmdsss//+++/o+:`                
        ``:ohmh/----..://:/sdNMNddmmmNMMMMMMMMMMMMMMMNMMNNMMNmdmmmdhso//////++o+/smMdo-"""

        print(Fore.BLACK + "\n =================== \n")
        print(Back.WHITE)
        print(Style.DIM)

        print(string)
        print(Style.RESET_ALL)
        time.sleep(1)
        print(Fore.YELLOW)
        print(Back.BLACK)
        print("Now Dirk-Philip will select a control group for you!")
        print("van Herwaarden et al. 2020!")
