from .component import Component
import os
import shutil

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
    -- It allows for the addition of new data by regularly
        querying Lasif project to look for new data and then
        updates list of how often events have been used.

    Preferably this should be done in a way that it should be
    easy to work with data afterwards. Currently using toml files
    but would be nice to have a better option.
    """

    def __init__(self, communicator, component_name):
        super(StoryTellerComponent, self).__init__(
            communicator, component_name)
        self.root, self.backup = self._create_root_folder()
        self.iteration_tomls = self.comm.project.paths["iteration_tomls"]
        self.story_file = os.path.join(self.root, "inversion.md")
        self.all_events = os.path.join(self.root, "all_events.txt")
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
        if os.path.exists(self.story_file):
            raise InversionsonError(f"File {self.story_file} already exists."
                                    f" Will stop here so that it does not get"
                                    f" overwritten.")
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
                self.events_used[event] = 0
    
    def _update_usage_of_events(self):
        """
        To keep track of how often events are used.
        """
        for event in self.comm.project.events_in_iteration:
            self.events_used[event] += 1
    
    def _start_entry_for_iteration(self):
        """
        Start a new section in the story file
        """
        self.markdown.add_header(
            header_style=2,
            text=self.comm.project.current_iteration
        )
        text = f"Here you can read all about what happened in iteration "
        text += f"{self.comm.project.current_iteration}."

        self.markdown.add_paragraph(text=text)
    
    def _add_image_of_data_coverage(self):
        """
        Include an image of event distribution to story file.
        TODO: Include raydensity plot.
        """
        self.markdown.add_header(
            header_style=3,
            text="Data Used"
        )
        im_file = self.comm.lasif.plot_iteration_events()
        self.markdown.add_image(
            image_url=im_file,
            image_title=f"Event distribution for "
                        f"{self.comm.project.current_iteration}",
            alt_text="text"
        )
    
    def _get_misfit_reduction(self):
        """
        Compute misfit reduction between previous two iterations
        """
        # We start with misfit of previous iteration
        prev_iter = self.comm.salvus_opt.get_previous_iteration_name
        prev_it_dict = self.comm.project.get_old_iteration_info(prev_iter)

        prev_total_misfit = 0.0
        prev_cg_misfit = 0.0
        for key in prev_it_dict["events"]:
            prev_total_misfit += prev_it_dict["events"][key]["misfit"]
            for key in prev_it_dict["new_control_group"]:
                prev_cg_misfit += prev_it_dict["events"][key]["misfit"]
        
        current_total_misfit = 0.0
        current_cg_misfit = 0.0
        for key, value in self.comm.project.misfits:
            current_total_misfit += value
            if key in self.comm.project.old_control_group:
                current_cg_misfit += value
        
        tot_red = (prev_total_misfit - current_total_misfit) / prev_total_misfit
        cg_red = (prev_cg_misfit - current_cg_misfit) / prev_cg_misfit

        return tot_red, cg_red

    def _add_table_of_events_and_misfits(self):
        """
        Include a table of events and corresponding misfits to
        the story file.
        """
        self.markdown.add_header(
            header_style=3,
            text="Misfits"
        )
        text = "The events used in the iteration along with their misfits"
        text += " are displayed below:"
        self.markdown.add_paragraph(text=text)
        iteration = self.comm.project.current_iteration
        self.comm.project.get_iteration_attributes(iteration)
        self.markdown.add_table(
            data=self.comm.project.misfits,
            headers=["Events", "Misfits"]
        )
        total_misfit = 0.0
        old_control_group_misfit = 0.0
        for key, value in self.comm.project.misfits:
            total_misfit += value
            if key in self.comm.project.old_control_group:
                old_control_group_misfit += value
        
        _, cg_red = self._get_misfit_reduction()
        
        text = f"Total misfit for iteration: {total_misfit} \n"
        text += f"Misfit for the old control group: {old_control_group_misfit}"
        text += f"\n Misfit reduction between the control groups: {cg_red}"

        self.markdown.add_paragraph(text=text)
    
    def _report_control_group(self):
        """
        Report what the new control group is and what the current misfit is.
        """
        self.markdown.add_header(
            header_style=4,
            text="Selection of New Control Group"
        )
        text = "The events which will continue on to the next iteration are "
        text += "listed below."

        self.markdown.add_paragraph(
            text=text
        )
        self.markdown.add_list(items=self.comm.project.new_control_group)

        cg_misfit = 0.0
        for key, value in self.comm.project.misfits:
            if key in self.comm.project.new_control_group:
                cg_misfit += value
        
        text = f"The current misfit for the control group is {cg_misfit}"
        self.markdown.add_paragraph(text=text)

    def document_task(self, task: str):
        """
        Depending on what kind of task it is, the function makes
        sure that there exists proper documentation of what happened
        in that task

        :param task: Type of task
        :type task: str
        """
        if task == "compute_misfit_and_gradient":
            # The compute misfit and gradient task is always associated
            # with the first iteration
            # This is the absolute first iteration
            # We need to create all necessary files
            self._create_story_file()
            self._write_list_of_all_events()
            self.events_used = {}
            for event in self.comm.lasif.list_events():
                self.events_used[event] = 0
        if task != "compute_gradient" or task != "finalize_iteration":
            self._update_usage_of_events()
            self._start_entry_for_iteration()
            self._add_image_of_data_coverage()
            self._add_table_of_events_and_misfits()

        elif task == "compute_gradient":
            self._report_control_group()


class MarkDown(StoryTellerComponent):
    """
    A little class designed to contain a few helper functions
    to write text in Markdown style
    """

    def __init__(self, file_name):
        self.file = file_name
        self.text_styles = ['normal', 'italic', 'bold']
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
            raise ValueError(
                "Header style must be an integer between 1 and 6")

        self.stream = "#"*int(header_style) + " "
        self.stream += text
        self._add_line_break()
        self._add_line_break()

        if new:
            self._write_to_file()
        else:
            self._append_to_file()

    def _transform_special_characters(self):
        """
        Take special markdown characters from string
        and make sure they are interpreted correctly
        """
        string = self.stream
        string = string.replace('*', '\*')
        string = string.replace('`', '\`')
        string = string.replace('_', '\_')
        string = string.replace('{', '\{')
        string = string.replace('}', '\}')
        string = string.replace('[', '\[')
        string = string.replace(']', '\]')
        string = string.replace('(', '\(')
        string = string.replace(')', '\)')
        string = string.replace('#', '\#')
        string = string.replace('+', '\+')
        string = string.replace('-', '\-')
        string = string.replace('!', '\!')
        string = string.replace('&', '&amp;')
        string = string.replace('<', '&lt;')
        self.stream = string
    
    def add_paragraph(self, text: str, textstyle='normal'):
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
        self.stream += f"({image_url} \"{image_title}\")"
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
        self.stream += f"| {headers[0]} | {headers[1]} |\n"
        self.stream += "| --- | ---: | \n"
        
        for key, value in data:
            self.stream += f"| {key} | {value} |\n"
        
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

        for item in items:
            self.stream += f"* {item} \n"
        
        self._add_line_break()
        self._add_line_break()
        self._append_to_file()
