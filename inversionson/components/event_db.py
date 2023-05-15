from __future__ import annotations
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from inversionson.project import Project

from .component import Component
import json


class EventDataBase(Component):
    """
    This component numbers all data. This is important since Optson labels samples with integer
    values.
    """

    def __init__(self, project: Project):
        super().__init__(project=project)
        self.event_numbering_file = self.project.paths.doc_dir / "event_numbering.json"
        self.enumerate_all_data()

    def enumerate_all_data(self):
        all_events = self.project.lasif.list_events()
        if not self.event_numbering_file.exists():
            self.event_dict = {e: i for i, e in enumerate(all_events)}
        else:
            # Update numbering
            with open(self.event_numbering_file, "r") as fh:
                self.event_dict = json.load(fh)
            idx = max(self.event_dict.values())

            # Clean up files that have been thrown out of the lasif project
            for event in self.event_dict:
                if event not in all_events:
                    del self.event_dict[event]

            # Append new data and continue numbering upwards
            for event in all_events:
                if event not in self.event_dict:
                    idx += 1
                    self.event_dict[event] = idx
        # Write the file.
        with open(self.event_numbering_file, "w") as fh:
            json.dump(self.event_dict, fh)

        self.flipped_event_dict = {
            int(idx): name for name, idx in self.event_dict.items()
        }

    def get_event_idx(self, event: str):
        return self.event_dict[event]

    def get_event_indices(self, events: List[str]):
        return [self.event_dict[event] for event in events]

    def get_event_name(self, event_idx: Union[str, int]) -> str:
        """Gives a list of string names for indices of events. Given on a list of integers given either in the form
        of ints of strings of ints.
        Strings of ints are supported to allow easy conversion from toml files."""
        return self.flipped_event_dict[event_idx]

    def get_event_names(self, event_indices: Union[List[str], List[int]]) -> List[str]:
        """Gives a list of string names for indices of events. Given on a list of integers given either in the form
        of ints of strings of ints.
        Strings of ints are supported as well to allow easy conversion from toml files."""
        return [self.flipped_event_dict[int(event_idx)] for event_idx in event_indices]
