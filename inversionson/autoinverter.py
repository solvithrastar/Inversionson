import numpy as np
import time
import os
import shutil
import sys
import warnings
import emoji
from inversionson import InversionsonError, InversionsonWarning
import toml
from colorama import init
from colorama import Fore, Style
from typing import Union, List
from inversionson.remote_helpers.helpers import preprocess_remote_gradient
from salvus.flow.api import get_site
from inversionson import autoinverter_helpers as helpers

init()


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
        it_name = self.comm.salvus_opt.get_newest_iteration_name()
        if validation:
            it_name = f"validation_{it_name}"
        move_meshes = "it0000" in it_name if validation else True
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
                        event=None, iteration=it_name, hpc_cluster=None,
                    )
                    for event in self.comm.project.events_in_iteration:
                        if not self.comm.lasif.has_mesh(event):
                            self.comm.salvus_mesher.create_mesh(event=event,)
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
        elif self.comm.project.meshes == "mono-mesh":
            self.comm.lasif.move_mesh(event=None, iteration=it_name)

        if not validation and self.comm.project.inversion_mode == "mini-batch":
            self.comm.project.update_control_group_toml(first=first)
        # self.comm.project.create_iteration_toml(it_name)
        # self.comm.project.get_iteration_attributes(validation)

    def time_for_validation(self) -> bool:
        """
        Check whether it is time to run a validation iteration
        """
        run_function = False

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
        model of the past few iterations. *Currently we do not average*
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
            and "it0000_model" not in self.comm.project.current_iteration
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
                    event=None, iteration=None, validation=True,
                )

        val_forward_helper = helpers.ForwardHelper(
            self.comm, self.comm.project.validation_dataset
        )
        assert "validation_" in self.comm.project.current_iteration
        val_forward_helper.dispatch_forward_simulations(verbose=True)
        assert val_forward_helper.assert_all_simulations_dispatched()
        val_forward_helper.retrieve_forward_simulations(
            adjoint=False, verbose=True, validation=True,
        )
        assert val_forward_helper.assert_all_simulations_retrieved()
        val_forward_helper.report_total_validation_misfit()

        iteration = self.comm.project.current_iteration
        # leave the validation iteration.
        iteration = iteration[11:]
        self.comm.project.change_attribute(
            attribute="current_iteration", new_value=iteration,
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
                    smoothing_helper.monitor_interpolations_send_out_smoothjobs(
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
            smoothing_helper.dispatch_smoothing_simulations(events=None)
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients(events=None)
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
                    smoothing_helper.monitor_interpolations_send_out_smoothjobs(
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
            smoothing_helper.dispatch_smoothing_simulations(events=None)
            assert smoothing_helper.assert_all_simulations_dispatched()
            smoothing_helper.retrieve_smooth_gradients(events=None)
        assert smoothing_helper.assert_all_simulations_retrieved()

        print(Fore.RED + "\n =================== \n")
        print(
            emoji.emojize(
                ":love_letter: | Finalizing iteration " "documentation",
                use_aliases=True,
            )
        )
        sys.exit("Run salvus opt in shell")
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
        print(Fore.RED + "\n =================== \n")
        print(f"Current Iteration: {self.comm.project.current_iteration}")
        print(f"Current Task: {task}")
        iteration = self.comm.salvus_opt.get_newest_iteration_name()
        self.comm.project.get_iteration_attributes()
        self.comm.storyteller.document_task(task)
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
