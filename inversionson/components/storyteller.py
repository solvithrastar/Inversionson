from __future__ import annotations
import os
import toml
import emoji  # type: ignore
from .component import Component
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from inversionson.project import Project
from colorama import init, Fore

init()


class StoryTeller(Component):
    """
    A class in charge of documentation of the inversion.
    TODO: DP: I want to mostly get rid of this (WIP)
    """

    def __init__(self, project: Project):
        super().__init__(project=project)
        self.root = self.project.paths.doc_dir
        self.iteration_tomls = self.project.paths.iteration_tomls
        self.all_events = self.root / "all_events.txt"
        self.events_used_toml = self.root / "events_used.toml"
        self.validation_toml = self.root / "validation.toml"

        self.events_used = (
            toml.load(self.events_used_toml)
            if self.events_used_toml.exists()
            else self._create_initial_events_used_toml()
        )
        self.printer = PrettyPrinter()

    def _create_initial_events_used_toml(self) -> Dict[str, int]:
        """
        Initialize the toml files which keeps track of usage of events
        """
        all_events = self.project.lasif.list_events()
        events_used = {event: 0 for event in all_events}
        with open(self.events_used_toml, "w+") as fh:
            toml.dump(events_used, fh)
        return events_used

    def _update_usage_of_events(self) -> None:
        """
        Count usage of events.

        """
        # It seems like the below is not fully correct if you would interupt this process in the middle.
        for event in self.project.non_val_events_in_iteration:
            if not self.project.updated[event]:
                if event not in self.events_used.keys():
                    self.events_used[event] = 0
                assert isinstance(self.events_used[event], int)
                self.events_used[event] += 1
                self.project.change_attribute(
                    attribute=f'updated["{event}"]', new_value=True
                )
        with open(self.events_used_toml, "w") as fh:
            toml.dump(self.events_used, fh)

    def report_validation_misfit(
        self,
        iteration: str,
        event: str,
    ) -> None:
        """
        We write misfit of validation dataset for a specific window_set

        :param iteration: Name of validation iteration
        :type iteration: str
        :param window_set: Name of window set
        :type window_set: str
        :param event: Name of event reported
        :type event: str
        :param total_sum: When the total sum for the iteration needs to
            be reported, default False
        :type total_sum: bool, Optional
        """
        validation_dict = (
            toml.load(self.validation_toml)
            if os.path.exists(self.validation_toml)
            else {}
        )
        if iteration not in validation_dict.keys():
            validation_dict[iteration] = {"events": {}, "total": 0.0}

        misfits_toml = os.path.join(
            self.project.lasif.lasif_root,
            "ITERATIONS",
            f"ITERATION_{iteration}",
            "misfits.toml",
        )
        misfits_dict = toml.load(misfits_toml)
        event_misfit = misfits_dict[event]["event_misfit"]
        validation_dict[iteration]["events"][event] = float(event_misfit)

        total = 0.0
        for event in validation_dict[iteration]["events"].keys():
            total += float(validation_dict[iteration]["events"][event])
        validation_dict[iteration]["total"] = total

        validation_dict[iteration]["events"] = dict(
            sorted(validation_dict[iteration]["events"].items())
        )
        with open(self.validation_toml, mode="w") as fh:
            toml.dump(validation_dict, fh)

    def document_iteration(self) -> None:
        """
        Update event usage and possibly other things at the  end of the iteration.
        """
        # TODO: Consider adding this back into Problem.
        self._update_usage_of_events()


class PrettyPrinter(object):
    """
    A class which makes printing in Inversionson pretty and consistant.

    Not too dissimilar from the MarkDown class
    """

    def __init__(self) -> None:
        self.stream = ""
        self.color = Fore.WHITE
        self.color_dict = self.create_color_dict()

    def create_color_dict(self) -> Dict:
        return {
            "white": Fore.WHITE,
            "black": Fore.BLACK,
            "blue": Fore.BLUE,
            "green": Fore.GREEN,
            "red": Fore.RED,
            "cyan": Fore.CYAN,
            "magenta": Fore.MAGENTA,
            "yellow": Fore.YELLOW,
            "lightred": Fore.LIGHTRED_EX,
        }

    def set_color(self, color: str) -> None:
        self.color = self.color_dict[color.lower()]

    def add_emoji(self, emoji_alias: str, vertical_line: bool = True) -> None:
        if not emoji_alias.startswith(":"):
            emoji_alias = f":{emoji_alias}"
        if not emoji_alias.endswith(":"):
            emoji_alias += ":"
        self.stream += f"{emoji.emojize(emoji_alias, language='alias')}"
        self.stream += " | " if vertical_line else " "

    def add_horizontal_line(self) -> None:
        self.stream += "\n ============================== \n"

    def add_message(self, message: str) -> None:
        self.stream += message

    def print(
        self,
        message: str,
        line_above: bool = False,
        line_below: bool = False,
        emoji_alias: Optional[Union[str, List[str]]] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        A printing function which works with the stream and finally prints it and
        resets the stream

        :param message: The string to be printed
        :type message: str
        :param line_above: Print a line above?, defaults to False
        :type line_above: bool, optional
        :param line_below: Print a line below?, defaults to False
        :type line_below: bool, optional
        :param emoji_alias: An emoji at the beginning for good measure? It needs to be a string that
            refers to an emoji, defaults to None
        :type emoji_alias: Union[str, List[str]], optional
        :param emoji_alias: Color to print with. Available colors are:
            [white, black, red, cyan, yellow, magenta, green, blue], defaults to None
        :type emoji_alias: str, optional
        """
        if color is not None:
            self.set_color(color)
        self.stream += f"{self.color} "
        if line_above:
            self.add_horizontal_line()
        if emoji_alias is not None:
            if isinstance(emoji_alias, list):
                for _i, emo in enumerate(emoji_alias):
                    vertical_line = _i == len(emoji_alias) - 1
                    self.add_emoji(emo, vertical_line=vertical_line)
            else:
                self.add_emoji(emoji_alias)
        self.add_message(message)
        if line_below:
            self.add_horizontal_line()
        print(self.stream)
        self.stream = ""
