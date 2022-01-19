import os
import shutil

import emoji
from inversionson import InversionsonError
import toml
from colorama import init
from colorama import Fore, Style
from salvus.flow.api import get_site
from inversionson import autoinverter_helpers as helpers
from inversionson.optimizers.adam_optimizer import AdamOptimizer

init()
from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset


def _find_project_comm(info):
    """
    Get Inversionson communicator.
    """
    from inversionson.components.project import ProjectComponent

    return ProjectComponent(info).get_communicator()


class AutoInverter(object):
    """
    A class which takes care of automating a Full-Waveform Inversion.

    It works for mono-batch or mini-batch, mono-mesh or multi-mesh
    or any combination of the two.
    It uses Salvus, Lasif and Multimesh to perform most of its actions.
    This is a class which wraps the three packages together to perform an
    automatic Full-Waveform Inversion
    """

    def __init__(self, info_dict: dict, manual_mode=False):
        self.info = info_dict
        print(Fore.RED + "Will make communicator now")
        self.comm = _find_project_comm(self.info)
        print(Fore.GREEN + "Now I want to start running the inversion")
        print(Style.RESET_ALL)
        self.task = None
        if not manual_mode:
            self.run_inversion()

    def _send_whatsapp_announcement(self):
        """
        Send a quick announcement via whatsapp when an iteration is done
        You need to have a twilio account, a twilio phone number
        and enviroment variables:
        MY_NUMBER
        TWILIO_NUMBER
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        """
        from twilio.rest import Client

        client = Client()
        from_whatsapp = f"whatsapp:{os.environ['TWILIO_NUMBER']}"
        to_whatsapp = f"whatsapp:{os.environ['MY_NUMBER']}"
        iteration = self.comm.project.current_iteration
        string = f"Your Inversionson code is DONE_WITH_{iteration}"
        client.messages.create(
            body=string, from_=from_whatsapp, to=to_whatsapp
        )

    def prepare_iteration(self, first=False, validation=False):
        """
        Prepare iteration.
        Get iteration name from salvus opt
        Modify name in inversion status
        Pick events
        Create iteration
        Make meshes if needed
        Update information in iteration dictionary.
        """
        if self.comm.project.AdamOpt:
            adam_opt = AdamOptimizer(
                inversion_root=self.comm.project.paths["inversion_root"])
            it_name = adam_opt.get_iteration_name()
        else:
            it_name = self.comm.salvus_opt.get_newest_iteration_name()

        if validation:
            it_name = f"validation_{it_name}"
        if self.comm.project.AdamOpt:
            move_meshes = "00000" in it_name if validation else True
        else:
            move_meshes = "it0000" in it_name if validation else True

        if self.comm.project.AdamOpt:
            first_try = True
        else:
            first_try = self.comm.salvus_opt.first_trial_model_of_iteration()
        self.comm.project.change_attribute("current_iteration", it_name)
        it_toml = os.path.join(
            self.comm.project.paths["iteration_tomls"], it_name + ".toml"
        )
        if self.comm.lasif.has_iteration(it_name):
            if not os.path.exists(it_toml):
                self.comm.project.create_iteration_toml(it_name)
            self.comm.project.get_iteration_attributes(validation=validation)
            # If the iteration toml was just created but
            # not the iteration, we finish making the iteration
            # Should never happen though
            if len(self.comm.project.events_in_iteration) != 0:
                if self.comm.project.meshes == "multi-mesh" and move_meshes:
                    self.comm.multi_mesh.add_fields_for_interpolation_to_mesh()
                    self.comm.lasif.move_mesh(
                        event=None,
                        iteration=it_name,
                        hpc_cluster=None,
                    )
                    for event in self.comm.project.events_in_iteration:
                        if not self.comm.lasif.has_mesh(event):
                            self.comm.salvus_mesher.create_mesh(
                                event=event,
                            )
                            self.comm.salvus_mesher.add_region_of_interest(
                                event=event
                            )
                            self.comm.lasif.move_mesh(event, it_name)
                        else:
                            self.comm.lasif.move_mesh(event, it_name)
                elif self.comm.project.meshes == "mono-mesh" and move_meshes:
                    self.comm.lasif.move_mesh(event=None, iteration=it_name)
                return
        if first_try and not validation:
            if self.comm.project.inversion_mode == "mini-batch":
                print("Getting minibatch")
                if self.comm.project.AdamOpt:
                    all_events = self.comm.lasif.list_events()
                    valid_data = list(
                        set(
                            self.comm.project.validation_dataset
                            + self.comm.project.test_dataset
                        )
                    )
                    all_events = list(set(all_events) - set(valid_data))
                    n_events = self.comm.project.initial_batch_size
                    # choose random events
                    doc_path = os.path.join(
                        self.comm.project.paths["inversion_root"],
                        "DOCUMENTATION")
                    all_norms_path = os.path.join(doc_path,
                                                  "all_norms.toml")
                    if os.path.exists(all_norms_path):
                        norm_dict = toml.load(all_norms_path)
                        unused_events = list(
                            set(all_events).difference(set(norm_dict.keys())))
                        n_unused_events = len(unused_events)
                        if n_unused_events >= n_events:
                            events = get_random_mitchell_subset(
                                self.comm.lasif.lasif_comm, n_events,
                                unused_events)
                        else:
                            existing_events = []
                            if len(unused_events) > 0:
                                existing_events = get_random_mitchell_subset(
                                    self.comm.lasif.lasif_comm,
                                    n_unused_events, unused_events)
                            remaining_events = list(
                                set(all_events) - set(existing_events))
                            new_events = get_random_mitchell_subset(
                                self.comm.lasif.lasif_comm,
                                n_events - n_unused_events, remaining_events,
                                norm_dict, existing_events)
                            events = existing_events + new_events
                    else:
                        events = get_random_mitchell_subset(
                            self.comm.lasif.lasif_comm, n_events,
                            all_events)
                else:
                    events = self.comm.lasif.get_minibatch(first)
            else:
                events = self.comm.lasif.list_events()
        elif validation:
            events = self.comm.project.validation_dataset
        else:
            prev_try = self.comm.salvus_opt.get_previous_iteration_name(
                tr_region=True
            )
            events = self.comm.lasif.list_events(iteration=prev_try)
        self.comm.project.change_attribute("current_iteration", it_name)
        self.comm.lasif.set_up_iteration(it_name, events)
        self.comm.project.create_iteration_toml(it_name)
        self.comm.project.get_iteration_attributes(validation)
        if self.comm.project.meshes == "multi-mesh" and move_meshes:
            if self.comm.project.interpolation_mode == "remote":
                interp_site = get_site(self.comm.project.interpolation_site)
            else:
                interp_site = None
            self.comm.multi_mesh.add_fields_for_interpolation_to_mesh()
            self.comm.lasif.move_mesh(
                event=None, iteration=it_name, hpc_cluster=interp_site
            )
            for event in events:
                if not self.comm.lasif.has_mesh(
                        event, hpc_cluster=interp_site
                ):
                    self.comm.salvus_mesher.create_mesh(event=event)
                    self.comm.lasif.move_mesh(
                        event, it_name, hpc_cluster=interp_site
                    )
                else:
                    self.comm.lasif.move_mesh(
                        event, it_name, hpc_cluster=interp_site
                    )
        elif self.comm.project.meshes == "mono-mesh" and move_meshes:
            self.comm.lasif.move_mesh(event=None, iteration=it_name)

        if not validation and not self.comm.project.AdamOpt and \
                self.comm.project.inversion_mode == "mini-batch":
            self.comm.project.update_control_group_toml(first=first)
        # self.comm.project.create_iteration_toml(it_name)
        # self.comm.project.get_iteration_attributes(validation)

    def time_for_validation(self) -> bool:
        """
        Check whether it is time to run a validation iteration
        """
        run_function = False
        if self.comm.project.AdamOpt:
            iteration_number = AdamOptimizer.get_iteration_number()
            if iteration_number == 0:
                run_function == True
        else:
            iteration_number = (
                self.comm.salvus_opt.get_number_of_newest_iteration()
            )

        # We execute the validation check if iteration is either:
        # a) The initial iteration or
        # b) When #Iteration + 1 mod when_to_validate = 0
        if self.comm.project.current_iteration == "it0000_model":
            run_function = True
        if (iteration_number + 1) % self.comm.project.when_to_validate == 0:
            run_function = True
        return run_function

    def compute_misfit_on_validation_data(self):
        """
        We define a validation dataset and compute misfits on it for an average
        model of the past few iterations.
        We will both compute the misfit on the initial window set and two
        newer sets.

        Probably a good idea to only run this after a model has been accepted
        and on the first one of course.
        """
        if self.comm.project.when_to_validate == 0:
            return
        if not self.time_for_validation():
            print("Not time for a validation")
            return
        if self.comm.project.AdamOpt:
            iteration_number = AdamOptimizer.get_iteration_number()
        else:
            iteration_number = (
                self.comm.salvus_opt.get_number_of_newest_iteration()
            )
        print(Fore.GREEN + "\n ================== \n")
        print(
            emoji.emojize(
                ":white_check_mark: | Computing misfit on validation dataset",
                use_aliases=True,
            )
        )
        print("\n\n")
        # Prepare validation iteration.
        # Simple enough, just create an iteration with the validation events
        print("Preparing iteration for validation set")
        self.prepare_iteration(validation=True)

        # Prepare simulations and submit them
        # I need something to monitor them.
        # There are definitely complications there
        print(Fore.YELLOW + "\n ============================ \n")
        print(
            emoji.emojize(
                ":rocket: | Run forward simulations", use_aliases=True
            )
        )
        if (
                self.comm.project.when_to_validate > 1
                and "it0000_model" not in self.comm.project.current_iteration and "00000" not in self.comm.project.current_iteration
        ):
            # Find iteration range
            to_it = iteration_number
            from_it = iteration_number - self.comm.project.when_to_validate + 1
            self.comm.salvus_mesher.get_average_model(
                iteration_range=(from_it, to_it)
            )
            self.comm.multi_mesh.add_fields_for_interpolation_to_mesh()
            if self.comm.project.interpolation_mode == "remote":
                self.comm.lasif.move_mesh(
                    event=None,
                    iteration=None,
                    validation=True,
                )

        val_forward_helper = helpers.ForwardHelper(
            self.comm, self.comm.project.validation_dataset
        )
        assert "validation_" in self.comm.project.current_iteration
        val_forward_helper.dispatch_forward_simulations(verbose=True)
        assert val_forward_helper.assert_all_simulations_dispatched()
        val_forward_helper.retrieve_forward_simulations(
            adjoint=False,
            verbose=True,
            validation=True,
        )
        assert val_forward_helper.assert_all_simulations_retrieved()
        val_forward_helper.report_total_validation_misfit()

        iteration = self.comm.project.current_iteration
        # leave the validation iteration.
        iteration = iteration[11:]
        self.comm.project.change_attribute(
            attribute="current_iteration",
            new_value=iteration,
        )
        self.comm.project.get_iteration_attributes()

    def compute_misfit_and_gradient(self, task: str, verbose: str):
        """
        A task associated with the initial iteration of FWI

        :param task: Task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print("Will prepare iteration")

        self.prepare_iteration(first=True)

        print(
            emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True)
        )

        print(f"Current Iteration: {self.comm.project.current_iteration}")

        forward_helper = helpers.ForwardHelper(
            self.comm, self.comm.project.events_in_iteration
        )
        forward_helper.dispatch_forward_simulations(verbose=True)
        print("Making sure all forward simulations have been dispatched")
        assert forward_helper.assert_all_simulations_dispatched()
        print("Retrieving forward simulations")
        forward_helper.retrieve_forward_simulations(adjoint=True, verbose=True)

        self.compute_misfit_on_validation_data()

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        assert forward_helper.assert_all_simulations_retrieved()

        adjoint_helper = helpers.AdjointHelper(
            self.comm, self.comm.project.events_in_iteration
        )
        adjoint_helper.dispatch_adjoint_simulations(verbose=True)
        assert adjoint_helper.assert_all_simulations_dispatched()
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
        adjoint_helper.process_gradients(interpolate=interpolate, verbose=True)
        assert adjoint_helper.assert_all_simulations_retrieved()

        if self.comm.project.inversion_mode == "mini-batch":
            smoothing_helper = helpers.SmoothingHelper(
                self.comm, self.comm.project.events_in_iteration
            )
            smoothing_helper.dispatch_smoothing_simulations(verbose=True)
            if self.comm.project.meshes == "multi-mesh":
                if self.comm.project.interpolation_mode == "remote":
                    smoothing_helper.monitor_interpolations(
                        verbose=True
                    )
                    smoothing_helper.dispatch_smoothing_simulations(
                        verbose=True
                    )
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients()
        else:
            smoothing_helper = helpers.SmoothingHelper(self.comm, events=None)
            # what seems to be missing here is remote summing of the gradient
            smoothing_helper.sum_gradients()
            smoothing_helper.dispatch_smoothing_simulations()
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients()
        assert smoothing_helper.assert_all_simulations_retrieved()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )
        self.comm.salvus_opt.write_misfit_and_gradient_to_task_toml()
        self.comm.project.update_iteration_toml()
        self.comm.storyteller.document_task(task)
        # Try to make ray-density plot work
        self.comm.salvus_opt.close_salvus_opt_task()
        # Bypass a Salvus Opt Error
        self.comm.salvus_opt.quickfix_delete_old_gradient_files()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                f":grinning: | Iteration "
                f"{self.comm.project.current_iteration} done",
                use_aliases=True,
            )
        )
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

    def compute_gradient_for_adam(self, task: str, verbose: str):
        """
        A task associated with the task in Adam optimization.

        TODO: implement the following:
        - fix documentation
        - search for bottlenecks
        - Use interpolation weights, and improve remote paths.
        - fix validation misfit stuff

        :param task: Task issued by the Adam Optimizer
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print("Will prepare iteration")

        self.prepare_iteration(first=True)

        print(
            emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True)
        )

        print(f"Current Iteration: {self.comm.project.current_iteration}")

        forward_helper = helpers.ForwardHelper(
            self.comm, self.comm.project.events_in_iteration
        )
        forward_helper.dispatch_forward_simulations(verbose=True)
        print("Making sure all forward simulations have been dispatched")
        assert forward_helper.assert_all_simulations_dispatched()
        print("Retrieving forward simulations")
        forward_helper.retrieve_forward_simulations(adjoint=True, verbose=True)

        self.compute_misfit_on_validation_data()

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        assert forward_helper.assert_all_simulations_retrieved()

        adjoint_helper = helpers.AdjointHelper(
            self.comm, self.comm.project.events_in_iteration
        )
        adjoint_helper.dispatch_adjoint_simulations(verbose=True)
        assert adjoint_helper.assert_all_simulations_dispatched()
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True

        # make sure standard gradient is there to interpolate to.
        self.comm.lasif.move_gradient_to_cluster()
        adjoint_helper.process_gradients(interpolate=interpolate, verbose=True)
        assert adjoint_helper.assert_all_simulations_retrieved()

        # Here we directly sum the gradients on the remote and recover the
        # raw summed gradient and pass it to Adam
        gradients = self.comm.lasif.lasif_comm.project.paths["gradients"]
        iteration = self.comm.project.current_iteration
        gradient = os.path.join(
            gradients,
            f"ITERATION_{iteration}",
            "summed_gradient.h5")

        smoothing_helper = helpers.SmoothingHelper(self.comm, events=self.comm.project.events_in_iteration)
        if not os.path.exists(gradient):
            if self.comm.project.meshes == "multi-mesh":
                smoothing_helper.monitor_interpolations(verbose=True, smooth_all=False)
            smoothing_helper.sum_gradients()

        adam_opt = AdamOptimizer(inversion_root=
                                 self.comm.project.paths["inversion_root"])
        adam_grad = adam_opt.get_gradient_path()
        shutil.copy(gradient, adam_grad)
        adam_opt.set_gradient_task_to_finished()
        if not os.path.exists(adam_opt.get_raw_update_path()):
            adam_opt.compute_raw_update()

        smoothing_helper.dispatch_smoothing_simulations()
        assert smoothing_helper.assert_all_simulations_dispatched()
        smoothing_helper.retrieve_smooth_gradients()
        assert smoothing_helper.assert_all_simulations_retrieved()

        # Now adam_opt still needs the update and apply the update
        # for now we used the mono-batch smoother, so it is written to the summed
        # gradient location. Get it there and pass it to Adam
        smooth_update = self.comm.lasif.find_gradient(
            iteration=iteration,
            event=None,
            smooth=True,
            summed=True,
        )
        adam_smooth = adam_opt.get_smooth_path()
        shutil.copy(smooth_update, adam_smooth)
        adam_opt.set_smoothing_task_to_finished()
        adam_opt.apply_smooth_update()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )

        self.comm.project.update_iteration_toml()
        self.comm.storyteller.document_task(task)
        # Here we mainly call this to
        task, verbose = self.finalize_iteration(task, verbose)

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                f":grinning: | Iteration "
                f"{self.comm.project.current_iteration} done",
                use_aliases=True,
            )
        )
        return task, verbose

    def compute_misfit(self, task: str, verbose: str):
        """
        Compute misfit for a test model

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """

        print(Fore.RED + "\n =================== \n")
        print("Will prepare iteration")
        self.prepare_iteration()

        print(
            emoji.emojize("Iteration prepared | :thumbsup:", use_aliases=True)
        )

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        print(f"More specifically: {verbose}")
        adjoint = True

        if (
                "compute misfit for" in verbose
                and self.comm.project.inversion_mode == "mini-batch"
        ):
            events_to_use = self.comm.project.old_control_group
            adjoint = False
        elif self.comm.project.inversion_mode == "mini-batch":
            # If model is accepted we consider looking into validation data.
            self.compute_misfit_on_validation_data()
            events_to_use = list(
                set(self.comm.project.events_in_iteration)
                - set(self.comm.project.old_control_group)
            )
        else:
            events_to_use = self.comm.project.events_in_iteration
        forward_helper = helpers.ForwardHelper(self.comm, events_to_use)
        forward_helper.dispatch_forward_simulations(verbose=True)
        assert forward_helper.assert_all_simulations_dispatched()
        forward_helper.retrieve_forward_simulations(
            adjoint=adjoint, verbose=True
        )

        print(Fore.BLUE + "\n ========================= \n")
        print(
            emoji.emojize(
                ":hourglass: | Making sure all forward jobs are " "done",
                use_aliases=True,
            )
        )

        assert forward_helper.assert_all_simulations_retrieved()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )
        self.comm.storyteller.document_task(task, verbose)
        if "compute additional" in verbose:
            events_to_use = None
        self.comm.salvus_opt.write_misfit_to_task_toml(events=events_to_use)
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

    def compute_gradient(self, task: str, verbose: str):
        """
        Compute gradient for accepted trial model

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        iteration = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.change_attribute(
            attribute="current_iteration", new_value=iteration
        )
        self.comm.project.get_iteration_attributes()

        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")

        adjoint_helper = helpers.AdjointHelper(
            self.comm, self.comm.project.events_in_iteration
        )
        adjoint_helper.dispatch_adjoint_simulations(verbose=True)
        assert adjoint_helper.assert_all_simulations_dispatched()
        interpolate = False
        if self.comm.project.meshes == "multi-mesh":
            interpolate = True
        adjoint_helper.process_gradients(interpolate=interpolate, verbose=True)
        assert adjoint_helper.assert_all_simulations_retrieved()

        if self.comm.project.inversion_mode == "mini-batch":
            smoothing_helper = helpers.SmoothingHelper(
                self.comm, self.comm.project.events_in_iteration
            )
            smoothing_helper.dispatch_smoothing_simulations()
            if self.comm.project.meshes == "multi-mesh":
                if self.comm.project.interpolation_mode == "remote":
                    smoothing_helper.monitor_interpolations(
                        verbose=True
                    )
                    smoothing_helper.dispatch_smoothing_simulations(
                        verbose=True
                    )
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients()
        else:
            smoothing_helper = helpers.SmoothingHelper(self.comm, None)
            smoothing_helper.sum_gradients()
            smoothing_helper.dispatch_smoothing_simulations()
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients()
        assert smoothing_helper.assert_all_simulations_retrieved()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )
        self.comm.salvus_opt.write_gradient_to_task_toml()
        self.comm.storyteller.document_task(task)
        # bypassing an opt bug
        self.comm.salvus_opt.quickfix_delete_old_gradient_files()
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

    def select_control_batch(self, task, verbose):
        """
        Select events that will carry on to the next iteration

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task} \n")
        print("Selection of Dynamic Mini-Batch Control Group:")
        print(f"More specifically: {verbose}")
        self.comm.project.get_iteration_attributes()
        if "increase control group size" in verbose:
            self.comm.minibatch.increase_control_group_size()
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=self.comm.project.new_control_group
            )
            self.comm.storyteller.document_task(task, verbose)

        else:
            if self.comm.project.current_iteration == "it0000_model":
                self.comm.project.update_control_group_toml(first=True)
            self.comm.minibatch.print_dp()

            control_group = self.comm.minibatch.select_optimal_control_group()
            # print(f"Selected Control group: {control_group}")
            self.comm.salvus_opt.write_control_group_to_task_toml(
                control_group=control_group
            )
            self.comm.project.change_attribute(
                attribute="new_control_group", new_value=control_group
            )
            first = False

            self.comm.project.update_control_group_toml(new=True, first=first)
            self.comm.storyteller.document_task(task)
        self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()
        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

    def finalize_iteration(self, task, verbose):
        """
        Wrap up loose ends and get on to next iteration.

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(Fore.RED + "\n ==============finalize_iter===== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        if self.comm.project.AdamOpt:
            adam_opt = AdamOptimizer(
                inversion_root=self.comm.project.paths["inversion_root"])
            # adam already went for the next iter, so query the previous one.
            iteration = adam_opt.get_iteration_name()
        else:
            iteration = self.comm.salvus_opt.get_newest_iteration_name()
        if not self.comm.project.AdamOpt:  # TODO sort this 2 lines below out
            self.comm.project.get_iteration_attributes()
            self.comm.storyteller.document_task(task)
        if not self.comm.project.AdamOpt:
            self.comm.salvus_opt.close_salvus_opt_task()
        self.comm.project.update_iteration_toml()

        self.comm.salvus_flow.delete_stored_wavefields(iteration, "forward")
        self.comm.salvus_flow.delete_stored_wavefields(iteration, "adjoint")
        self.comm.salvus_flow.delete_stored_wavefields(
            iteration, "model_interp"
        )
        self.comm.salvus_flow.delete_stored_wavefields(
            iteration, "gradient_interp"
        )
        try:
            self._send_whatsapp_announcement()
        except:
            print("Not able to send whatsapp message")

        if self.comm.project.AdamOpt:
            adam_opt.finalize_iteration()
            task = adam_opt.get_inversionson_task()
            verbose = "Compute gradient for Adam optimizer."
            return task, verbose

        self.comm.salvus_opt.run_salvus_opt()
        task_2, verbose_2 = self.comm.salvus_opt.read_salvus_opt_task()
        if task_2 == task and verbose_2 == verbose:
            message = "Salvus Opt did not run properly "
            message += "It gave an error and the task.toml has not been "
            message += "updated."
            raise InversionsonError(message)
        return task_2, verbose_2

    def assign_task_to_function(self, task: str, verbose: str):
        """
        Input a salvus opt task and send to correct function

        :param task: task issued by salvus opt
        :type task: str
        :param verbose: Detailed info regarding task
        :type verbose: str
        """
        print(f"\nNext salvus opt task is: {task}\n")
        print(f"More specifically: {verbose}")

        if task == "compute_misfit_and_gradient":
            task, verbose = self.compute_misfit_and_gradient(task, verbose)
        elif task == "compute_gradient_for_adam":
            task, verbose = self.compute_gradient_for_adam(task, verbose)
        elif task == "compute_misfit":
            task, verbose = self.compute_misfit(task, verbose)
        elif task == "compute_gradient":
            task, verbose = self.compute_gradient(task, verbose)
        elif task == "select_control_batch":
            task, verbose = self.select_control_batch(task, verbose)
        elif task == "finalize_iteration":
            task, verbose = self.finalize_iteration(task, verbose)
        else:
            raise InversionsonError(f"Don't know task: {task}")

        return task, verbose

    def run_inversion(self):
        """
        Workflow:
                Read Salvus opt,
                Perform task,
                Document it
                Close task, repeat.
        """
        # Always do this as a first thing, Might write a different function
        # for checking status
        # self.initialize_inversion()

        if self.comm.project.AdamOpt:

            adam_opt = AdamOptimizer(inversion_root=
                                     self.comm.project.paths["inversion_root"])

            adam_config = toml.load(adam_opt.config_file)
            if adam_config["initial_model"] == "":
                raise Exception("Set adam config file and provide initial"
                                " model.")

            task = adam_opt.get_inversionson_task()
            self.task = task
            verbose = "Computing gradient for Adam optimization."
        else:
            task, verbose = self.comm.salvus_opt.read_salvus_opt_task()
            self.task = task

        while True:
            task, verbose = self.assign_task_to_function(task, verbose)


def read_information_toml(info_toml_path: str):
    """
    Read a toml file with inversion information into a dictionary

    :param info_toml_path: Path to the toml file
    :type info_toml_path: str
    """
    return toml.load(info_toml_path)


if __name__ == "__main__":
    print(
        emoji.emojize(
            "\n :flag_for_Iceland: | Welcome to Inversionson | :flag_for_Iceland: \n",
            use_aliases=True,
        )
    )
    # info_toml = input("Give me a path to your information_toml \n\n")
    # Tired of writing it in, I'll do this quick mix for now
    # print("Give me a path to your information_toml \n\n")
    # time.sleep(1)
    # print("Just kidding, I know it")
    info_toml = "inversion_info.toml"
    if not info_toml.startswith("/"):
        cwd = os.getcwd()
        if info_toml.startswith("./"):
            info_toml = os.path.join(cwd, info_toml[2:])
        elif info_toml.startswith("."):
            info_toml = os.path.join(cwd, info_toml[1:])
        else:
            info_toml = os.path.join(cwd, info_toml)
    info = read_information_toml(info_toml)
    invert = AutoInverter(info)
