"""
A class which takes care of everything related to selecting
batches and control groups for coming iterations.
"""

from .component import Component
import numpy as np
import h5py
import random
from colorama import Fore, Back, Style
import sys
import time


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
        for event in events:
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
            (norm_1 * norm_2)) / np.pi * 180.0
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
            (test_grad_norm * full_norm)
        ) / np.pi * 180.0
        return angle

    def _sum_relevant_values(self, grad, parameters: list):
        """
        Take the gradient, find inverted parameters and sum them together.
        Reduces a 3D array to a 2D array.
        
        :param grad: Numpy array with gradient values
        :type grad: numpy.ndarray
        :param parameters: A list of dimension labels in gradient
        :return: list
        :rtype: numpy.ndarray
        """
        inversion_params = self.comm.project.inversion_params
        indices = []
        for param in inversion_params:
            indices.append(parameters.index(param))
        summed_grad = np.zeros(shape=(grad.shape[0]))
        
        for i in indices:
            summed_grad += grad[:, i]
        return summed_grad

    def _get_vector_of_values(self, gradient,
                              unique,
                              parameters: list):
        """
        Take a full gradient, find it's unique values and relevant parameters,
        manipulate all of these into a vector of summed parameter values.
        
        :param gradient: Array of gradient values
        :type gradient: numpy.ndarray
        :param unique: Array of unique indices spatially
        :type unique: numpy.ndarray
        :param parameters: A list of dimension labels in gradient
        :return: list
        :return: 1D vector with gradient values
        :rtype: numpy.ndarray
        """
        gradient = np.swapaxes(a=gradient, axis1=1, axis2=2)
        gradient = np.reshape(a=gradient,
                              newshape=(gradient.shape[0]*gradient.shape[1],
                                        gradient.shape[2]))
        gradient = gradient[unique]
        gradient = self._sum_relevant_values(
            grad=gradient,
            parameters=parameters)
        return gradient

    def get_random_event(self, n: int, existing: list) -> list:
        """
        Get an n number of events based on the probabilities defined
        in the event_quality toml file

        :param n: Number of events to randomly choose
        :type n: int
        :param existing: Events blocked from selection
        :type existing: list
        :return: List of events randomly picked
        :rtype: list
        """
        events_quality = self.comm.storyteller.event_quality
        for k in existing:
            del events_quality[k]
        list_of_events = list(events_quality.keys())
        list_of_probabilities = list(events_quality.values())
        list_of_probabilities /= np.sum(list_of_probabilities)

        chosen_events = list(np.random.choice(
            list_of_events, n, replace=False, p=list_of_probabilities
        ))
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
        print(f"Control batch events: {events}")
        ctrl_group = events.copy()
        max_ctrl = self.comm.project.max_ctrl_group_size
        min_ctrl = self.comm.project.min_ctrl_group_size
        iteration = self.comm.project.current_iteration
        gradient_paths = []
        for _i, event in enumerate(events):
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=True
            )
            gradient_paths.append(gradient)
            with h5py.File(gradient, "r") as f:
                grad = f["MODEL/data"]
                if _i == 0:
                    parameters = grad.attrs.get("DIMENSION_LABELS")[1].decode()
                    parameters = parameters[2:-2].replace(" ", "").replace("grad", "").split("|")
                    coordinates = f["MODEL/coordinates"][()]
                    init_shape = coordinates.shape
                    coordinates = np.reshape(a=coordinates,
                        newshape=(init_shape[0] * init_shape[1], init_shape[2]))
                    _, unique_indices = np.unique(
                        ar=coordinates,
                        return_index=True,
                        axis=0)
                if _i == 0:
                    full_grad = np.zeros_like(grad)
                # else:
                full_grad += grad[()]
        # Select only the relevant parameters and sum them together.
        # We also need to make sure we don't use points more often than ones.
        full_grad = self._get_vector_of_values(
            gradient=full_grad,
            unique=unique_indices,
            parameters=parameters)

        full_grad_norm = np.linalg.norm(full_grad)

        angular_changes = {}
        event_quality = {}
        for event in events:
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=event,
                smooth=True,
                inversion_grid=True
            )
            with h5py.File(gradient, "r") as f:
                individual_gradient = f["MODEL/data"][()]
                individual_gradient = self._get_vector_of_values(
                    gradient=individual_gradient,
                    unique=unique_indices,
                    parameters=parameters
                )

            angle = self._compute_angular_change(
                full_gradient=full_grad,
                full_norm=full_grad_norm,
                individual_gradient=individual_gradient
            )
            angular_changes[event] = angle
            event_quality[event] = 0.0
        batch_grad = np.copy(full_grad)
        test_batch_grad = np.copy(batch_grad)

        while len(ctrl_group) > min_ctrl or len(ctrl_group) > max_ctrl:
            redundant_gradient = min(angular_changes, key=angular_changes.get)
            gradient = self.comm.lasif.find_gradient(
                iteration=iteration,
                event=redundant_gradient,
                smooth=True,
                inversion_grid=True
            )
            with h5py.File(gradient, "r") as f:
                removal_grad = self._get_vector_of_values(
                    gradient=f["MODEL/data"][()],
                    unique=unique_indices,
                    parameters=parameters)
                test_batch_grad -= removal_grad
            angle = self._angle_between(full_grad, batch_grad)
            #TODO: Figure out problem with small angle.
            print(f"Angle: {angle}")
            if angle >= self.comm.project.maximum_grad_divergence_angle:
                break
            else:
                batch_grad = np.copy(test_batch_grad)
                del angular_changes[redundant_gradient]
                event_quality[redundant_gradient] = 1/len(ctrl_group)
                ctrl_group.remove(redundant_gradient)
        
        grads_dropped = self._dropout(ctrl_group)
        tmp_event_qual = event_quality
        best_non_ctrl_group_event = max(tmp_event_qual, key=tmp_event_qual.get)
        for grad in grads_dropped:
            non_ctrl_group_event = max(tmp_event_qual, key=tmp_event_qual.get)
            print(f"Best non: {best_non_ctrl_group_event}")
            print(f"Event Quality: {event_quality}")
            event_quality[grad] = event_quality[best_non_ctrl_group_event]
            ctrl_group.remove(grad)
            ctrl_group.append(non_ctrl_group_event)
            del tmp_event_qual[non_ctrl_group_event]
            print(f"Event: {grad} randomly dropped from control group.\n")
            print(f"Replaced by event: {non_ctrl_group_event} \n")

        for key, val in event_quality.items():
            self.comm.project.event_quality[key] = val
        print(f"Control batch events: {self.comm.project.events_in_iteration}")

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
            attribute="new_control_group",
            new_value=ctrl_group
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
        
        # for line in string:
        #     sys.stdout.write(line)
        #     time.sleep(.1)
        print(string)
        print(Style.RESET_ALL)
        time.sleep(1)
        print(Fore.YELLOW)
        print(Back.BLACK)
        print("Now Dirk-Philip will select a control group for you!")
        #time.sleep(2)
        print("van Herwaarden et al. 2019!")
        #time.sleep(2)
        
        # print(string)
