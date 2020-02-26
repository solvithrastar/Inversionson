from .component import Component
import os
import shutil
import toml

from inversionson import InversionsonError


class StoryTellerComponent(Component):
    """
    A class in charge of documentation of inversion.

    Monitors a file which tells the actual story of the inversion

    Keeps track of a few things:
    - For each iteration:
    -- Which events are used
    -- What was the control group
    -- What was the misfit
    -- Type of adjoint source.

    - During inversion
    -- How often each event has been used
    -- How influential the event was on the inversion
    -- It allows for the addition of new data by regularly
        querying Lasif project to look for new data and then
        updates list of how often events have been used.

    Preferably this should be done in a way that it should be
    easy to work with data afterwards. Currently using toml files
    but would be nice to have a better option.
    """

    def __init__(self, communicator, component_name):
        super(StoryTellerComponent, self).__init__(communicator, component_name)
        self.root, self.backup = self._create_root_folder()
        self.iteration_tomls = self.comm.project.paths["iteration_tomls"]
        self.story_file = os.path.join(self.root, "inversion.md")
        self.all_events = os.path.join(self.root, "all_events.txt")
        self.events_used_toml = os.path.join(self.root, "events_used.toml")
        self.events_quality_toml = os.path.join(self.root, "events_quality.toml")
        if os.path.exists(self.events_used_toml):
            self.events_used = toml.load(self.events_used_toml)
        else:
            self._create_initial_events_used_toml()
        if os.path.exists(self.events_quality_toml):
            self.events_quality = toml.load(self.events_quality_toml)
        else:
            self._create_initial_events_quality_toml()
        self.markdown = MarkDown(self.story_file)

    def _create_root_folder(self):
        """
        Initiate the folder structure if needed
        """
        root = self.comm.project.paths["documentation"]
        backup = os.path.join(root, "BACKUP")
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(backup):
            os.mkdir(backup)
        return root, backup

    def _backup_files(self):
        """
        Backup all information at the end of each iteration.
        """
        tmpdir = os.path.join(self.backup, "..", "tmp")
        if os.path.exists(self.backup):
            shutil.copytree(self.backup, tmpdir)
            shutil.rmtree(self.backup)
            shutil.copytree(self.root, self.backup)
            shutil.rmtree(tmpdir)
        else:
            shutil.copytree(self.root, self.backup)

    def _backup_story_file(self):
        """
        Protects the valuable story file from being overwritten.
        """
        backup_loc = os.path.join(self.backup, "inversion.md")
        shutil.copy(self.story_file, backup_loc)

    def _create_story_file(self):
        """
        Create a markdown file which will be used to tell the story
        of the inversion automatically.
        """
        if os.path.isfile(self.story_file):
            raise InversionsonError(
                f"File {self.story_file} already exists."
                f" Will stop here so that it does not get"
                f" overwritten."
            )
        header = self.comm.project.inversion_id
        self.markdown.add_header(header_style=1, text=header, new=True)

        text = f"Welcome to the automatic documentation of the inversion"
        text += f" project {self.comm.project.inversion_id}. Here the "
        text += f"inversion is documented iteration by iteration. \n"
        text += f"This is currently just a test but hopefully it will "
        text += f"work out beautifully."

        self.markdown.add_paragraph(text)

    def _write_list_of_all_events(self):
        """
        Write out a list of all events included in lasif project
        """
        all_events = self.comm.lasif.list_events()
        with open(self.all_events, "w+") as fh:
            fh.writelines(f"{event}\n" for event in all_events)

    def _create_initial_events_used_toml(self):
        """
        Initialize the toml files which keeps track of usage of events
        """
        all_events = self.comm.lasif.list_events()
        self.events_used = {}
        for event in all_events:
            self.events_used[event] = 0
        with open(self.events_used_toml, "w+") as fh:
            toml.dump(self.events_used, fh)

    def _create_initial_events_quality_toml(self):
        """
        Initialize the toml files which keeps track of usage of events
        """
        all_events = self.comm.lasif.list_events()
        self.events_quality = {}
        for event in all_events:
            self.events_quality[event] = 0.0
        with open(self.events_quality_toml, "w+") as fh:
            toml.dump(self.events_quality, fh)

    def _update_list_of_events(self):
        """
        In order to be able to add events to inversion we 
        need to update the list of used events.
        """
        all_events = self.comm.lasif.list_events()
        already_in_list = list(self.events_used)
        new = [x for x in all_events if x not in already_in_list]
        if len(new) == 0:
            return
        else:
            for event in new:
                self.events_quality[event] = 0.0
                self.events_used[event] = 0
            with open(self.events_used_toml, "w") as fh:
                toml.dump(self.events_used, fh)
            with open(self.events_quality_toml, "w") as fh:
                toml.dump(self.events_quality, fh)

    def _update_usage_of_events(self):
        """
        To keep track of how often events are used.
        """
        for event in self.comm.project.events_in_iteration:
            if not self.comm.project.updated[event]:
                self.events_used[event] += 1
                self.comm.project.change_attribute(
                    attribute=f'updated["{event}"]', new_value=True
                )
                self.comm.project.update_iteration_toml()
        with open(self.events_used_toml, "w") as fh:
            toml.dump(self.events_used, fh)

    def _update_event_quality(self):
        """
        Keep track of event quality, which is based on removal order of
        batch gradients in control group selection.
        """
        for key, val in self.comm.project.event_quality.items():
            self.events_quality[key] = val
        with open(self.events_quality_toml, "w") as fh:
            toml.dump(self.events_quality, fh)

    def _start_entry_for_iteration(self):
        """
        Start a new section in the story file
        """
        iteration = self.comm.project.current_iteration
        if iteration.startswith("it0000_model"):
            iteration_number = 0
        else:
            iteration_number = int(iteration.split("_")[0][2:].strip("0"))
        self.markdown.add_header(header_style=2, text=f"Iteration: {iteration_number}")
        text = f"Here you can read all about what happened in iteration "
        text += f"{iteration_number}."

        self.markdown.add_paragraph(text=text)

    def _add_image_of_data_coverage(self):
        """
        Include an image of event distribution to story file.
        """
        self.markdown.add_header(header_style=3, text="Data Used")
        im_file = self.comm.lasif.plot_iteration_events()
        self.markdown.add_image(
            image_url=im_file,
            image_title=f"Event distribution for "
            f"{self.comm.project.current_iteration}",
            alt_text="text",
        )
        print("Preparing Ray density image")
        ray_file = self.comm.lasif.plot_iteration_raydensity()
        self.markdown.add_image(
            image_url=ray_file,
            image_title=f"Ray density plot for "
            f"{self.comm.project.current_iteration}",
            alt_text="text",
        )

    def _report_acceptance_of_model(self):
        """
        When model gets accepted and we compute additional misfits,
        we report it to story file.
        """
        iteration = self.comm.project.current_iteration
        if iteration.startswith("it0000_model"):
            iteration_number = 0
        else:
            iteration_number = int(iteration.split("_")[0][2:].strip("0"))
        tr_region = float(iteration.split("_")[-1][:-2])
        text = f"Model for Iteration {iteration_number} accepted for"
        text += f" trust region: {tr_region}."

        self.markdown.add_paragraph(text=text, textstyle="bold")

    def _report_shrinking_of_trust_region(self):
        """
        When model gets accepted and we compute additional misfits,
        we report it to story file.
        """
        iteration = self.comm.project.current_iteration
        if iteration.startswith("it0000_model"):
            iteration_number = 0
        else:
            iteration_number = int(iteration.split("_")[0][2:].strip("0"))
        tr_region = float(iteration.split("_")[-1][:-2])
        text = f"Model for Iteration {iteration_number} was rejected "
        text += f"so now we shrink the trust region to: {tr_region} "
        text += f"and try again."

        self.markdown.add_paragraph(text=text)

    def _get_misfit_reduction(self):
        """
        Compute misfit reduction between previous two iterations
        """
        # We start with misfit of previous iteration
        # TODO: Something wrong going on here and needs fixing
        prev_iter = self.comm.salvus_opt.get_previous_iteration_name()
        prev_it_dict = self.comm.project.get_old_iteration_info(prev_iter)

        prev_total_misfit = 0.0
        prev_cg_misfit = 0.0
        for key in prev_it_dict["events"].keys():
            prev_total_misfit += prev_it_dict["events"][key]["misfit"]
            for key in prev_it_dict["new_control_group"]:
                prev_cg_misfit += prev_it_dict["events"][key]["misfit"]

        current_total_misfit = 0.0
        current_cg_misfit = 0.0
        for key, value in self.comm.project.misfits.items():
            current_total_misfit += value
            if key in self.comm.project.old_control_group:
                current_cg_misfit += value

        tot_red = (prev_total_misfit - current_total_misfit) / prev_total_misfit
        cg_red = (prev_cg_misfit - current_cg_misfit) / prev_cg_misfit

        return tot_red, cg_red

    def _add_table_of_events_and_misfits(self, verbose=None, task=None):
        """
        Include a table of events and corresponding misfits to
        the story file.
        """
        self.markdown.add_header(header_style=3, text="Misfits")
        if not verbose:
            text = "The events used in the iteration along with their misfits"
            text += " are displayed below:"

        if verbose and "additional" not in verbose:
            text = "We have computed misfits for the control group events. "
            text += "The misfits are displayed below. The additional events "
            text += "are displayed with 0.0 misfit values."

        if verbose and "additional" in verbose:
            text = "We have now computed the misfits for all the events of "
            text += "the iteration. These are displayed below."

        self.markdown.add_paragraph(text=text)
        # iteration = self.comm.project.current_iteration
        self.comm.project.get_iteration_attributes()
        self.markdown.add_table(
            data=self.comm.project.misfits, headers=["Events", "Misfits"]
        )
        if task == "compute_misfit_and_gradient":
            total_misfit = 0.0
            for key in self.comm.project.misfits.keys():
                total_misfit += self.comm.project.misfits[key]
            text = f"Total misfit for iteration: {total_misfit} \n"
            self.markdown.add_paragraph(text=text)
            return

        if verbose and "additional" in verbose:
            total_misfit = 0.0
            old_control_group_misfit = 0.0
            for key, value in self.comm.project.misfits.items():
                total_misfit += value
                if key in self.comm.project.old_control_group:
                    old_control_group_misfit += value

            _, cg_red = self._get_misfit_reduction()

            text = f"Total misfit for iteration: {total_misfit} \n"
            text += f"Misfit for the old control group: "
            text += f"{old_control_group_misfit}"
            text += f"\n Misfit reduction between the iterations: {cg_red}"

        if verbose and "additional" not in verbose:
            old_control_group_misfit = 0.0
            for key, value in self.comm.project.misfits.items():
                if key in self.comm.project.old_control_group:
                    old_control_group_misfit += value

            _, cg_red = self._get_misfit_reduction()

            text = f"Misfit for the old control group: "
            text += f"{old_control_group_misfit}"
            if cg_red >= 0.0:
                text += f"\n Misfit increase between the iterations: {cg_red}%"
            else:
                text += f"\n Misfit reduction between the iterations: {cg_red}%"

        self.markdown.add_paragraph(text=text)

    def _report_control_group(self):
        """
        Report what the new control group is and what the current misfit is.
        """
        self.markdown.add_header(header_style=4, text="Selection of New Control Group")
        text = "The events which will continue on to the next iteration are "
        text += "listed below."

        self.markdown.add_paragraph(text=text)
        self.markdown.add_list(items=self.comm.project.new_control_group)

        cg_misfit = 0.0
        for key, value in self.comm.project.misfits.items():
            if key in self.comm.project.new_control_group:
                cg_misfit += value

        text = f"The current misfit for the control group is {cg_misfit}"
        self.markdown.add_paragraph(text=text)

    def _report_increase_in_control_group_size(self):
        """
        The control group needs to be enlarged. This is reported here.
        """
        text = "Control group was not good enough, so we increase it with "
        text += "one extra event."

        self.markdown.add_paragraph(text=text)

    def _report_number_of_used_events(self):
        """
        At the end of each iteration we report how many events have been
        uses in inversion.
        """
        num_events = len([x for x in list(self.events_used.values()) if x != 0])

        text = f"We have now used {num_events} events during the inversion."

        self.markdown.add_paragraph(text=text)

    def _initiate_gradient_computation_task(self):
        """
        Write a quick paragraph reporting that we will now compute gradients
        for the accepted trial model.
        """
        text = "Since model has been accepted, we will now compute "
        text += "gradients for all batch events for the accepted model."

        self.markdown.add_paragraph(text=text)

    def document_task(self, task: str, verbose=None):
        """
        Depending on what kind of task it is, the function makes
        sure that there exists proper documentation of what happened
        in that task

        :param task: Type of task
        :type task: str
        :param verbose: Additional information regarding task, optional.
        :type verbose: str
        """
        if task == "compute_misfit_and_gradient":
            # The compute misfit and gradient task is always associated
            # with the first iteration
            # This is the absolute first iteration
            # We need to create all necessary files
            iteration = self.comm.project.current_iteration
            if iteration.startswith("it0000_model"):
                iteration_number = 0
            else:
                iteration_number = 1
            self._create_story_file()
            self._start_entry_for_iteration()
            if self.comm.project.inversion_mode == "mini-batch" or iteration_number == 0:
                self._write_list_of_all_events()
                self._update_usage_of_events()
                self._add_image_of_data_coverage()
            self._add_table_of_events_and_misfits(task=task)
            # self._report_control_group()
            # self._update_event_quality()
        if task == "compute_misfit":
            first_try = self.comm.salvus_opt.first_trial_model_of_iteration()
            if "additional" in verbose:
                self._report_acceptance_of_model()
                self._update_usage_of_events()
            else:
                if first_try:
                    self._start_entry_for_iteration()
                    if self.comm.project.inversion_mode == "mini-batch":
                        self._add_image_of_data_coverage()
                if not first_try:
                    self._report_shrinking_of_trust_region()
            if self.comm.project.inversion_mode == "mini-batch":
                self._add_table_of_events_and_misfits(verbose)

        if task == "compute_gradient":
            self._initiate_gradient_computation_task()

        if task == "select_control_batch":
            if verbose and "increase" in verbose:
                self._report_increase_in_control_group_size()
            self._report_control_group()
            self._update_event_quality()

        if task == "finalize_iteration":
            if self.comm.project.inversion_mode == "mini-batch":
                self._report_number_of_used_events()
            self._backup_files()


class MarkDown(StoryTellerComponent):
    """
    A little class designed to contain a few helper functions
    to write text in Markdown style
    """

    def __init__(self, file_name):
        self.file = file_name
        self.text_styles = ["normal", "italic", "bold"]
        self.stream = ""

    def _read_file(self):
        with open(self.file, "r") as fh:
            self.stream = fh.read()

    def _append_to_file(self):
        with open(self.file, "a") as fh:
            fh.write(self.stream)

    def _write_to_file(self):
        with open(self.file, "w") as fh:
            fh.write(self.stream)

    def _add_line_break(self):
        self.stream += "\n "

    def add_header(self, header_style: int, text: str, new=False):
        """
        Add a header to a markdown file. The header style
        has to be between 1 and 6

        :param header_style: Style of header, 1-6
        :type header_style: int
        :param text: Content of header
        :type text: str
        :param new: Add it to a new file?, defaults to False
        :type new: bool
        """
        if header_style < 1 or header_style > 6:
            raise ValueError("Header style must be an integer between 1 and 6")
        self.stream = text
        self._transform_special_characters()
        self.stream = "#" * int(header_style) + " " + self.stream
        self._add_line_break()
        self._add_line_break()

        if new:
            self._write_to_file()
        else:
            self._append_to_file()

    def _transform_special_characters(self, string=None):
        """
        Take special markdown characters from string
        and make sure they are interpreted correctly
        """
        output = True
        if not string:
            string = self.stream
            output = False
        string = string.replace("*", "\*")
        string = string.replace("`", "\`")
        string = string.replace("_", "\_")
        string = string.replace("{", "\{")
        string = string.replace("}", "\}")
        string = string.replace("[", "\[")
        string = string.replace("]", "\]")
        string = string.replace("(", "\(")
        string = string.replace(")", "\)")
        string = string.replace("#", "\#")
        string = string.replace("+", "\+")
        string = string.replace("-", "\-")
        string = string.replace("!", "\!")
        string = string.replace("&", "&amp;")
        string = string.replace("<", "&lt;")
        if output:
            return string
        self.stream = string

    def add_paragraph(self, text: str, textstyle="normal"):
        """
        Add a brand new paragraph to the markdown file

        :param text: Content of paragraph
        :type text: str
        :param textstyle: Style of text, defaults to 'normal'
        :type textstyle: str, optional
        """
        if textstyle not in self.text_styles:
            raise ValueError(f"Text style {textstyle} is not available")

        self.stream = text
        self._transform_special_characters()

        if textstyle != self.text_styles[0]:
            text = self.stream
            text = "_" + text + "_"
            if textstyle == self.text_styles[2]:
                text = "_" + text + "_"
            self.stream = text

        self._add_line_break()
        self._add_line_break()
        self._append_to_file()

    def add_image(self, image_url: str, image_title="", alt_text="text"):
        """
        Add an image to a markdown file

        :param image_url: Location of an image, I think this can be a file
        when using a local markdown and not an online one.
        :type image_url: str
        :param image_title: Title when hovering on pic, defaults to ""
        :type image_title: str, optional
        :param alt_text: Text when pic doesn't appear, defaults to "text"
        :type alt_text: str, optional
        """
        self.stream = f"![Alt {alt_text}]"
        self.stream += f'({image_url} "{image_title}")'
        self._add_line_break()
        self._append_to_file()

    def add_table(self, data: dict, headers=["Events", "Misfits"]):
        """
        Add a table to a markdown file. Currently only for 2 column
        based data.

        :param data: Data to display in table
        :type data: dict
        :param headers: Table headers, defaults to ["Events", "Misfits"]
        :type headers: list, optional
        """
        self.stream = ""
        string = f"| {headers[0]} | {headers[1]} |\n"
        fixed_string = self._transform_special_characters(string)
        self.stream += fixed_string
        self.stream += "| --- | ---: | \n"

        for key in data.keys():
            string = f"| {key} | {data[key]} |\n"
            fixed_string = self._transform_special_characters(string)
            self.stream += fixed_string

        self._add_line_break()
        self._add_line_break()
        self._append_to_file()

    def add_list(self, items: list):
        """
        Add an unordered list to a markdown file.

        :param items: Items to be listed
        :type items: list
        """
        self.stream = ""

        for _i, item in enumerate(items):
            if _i != 0:
                self.stream += " "
            self.stream += f"* {self._transform_special_characters(item)} \n"

        self._add_line_break()
        self._add_line_break()
        self._append_to_file()
