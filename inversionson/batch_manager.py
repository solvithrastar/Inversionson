from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from inversionson.project import Project
from optson.batch_manager import AbstractBatchManager
from lasif.tools.query_gcmt_catalog import get_random_mitchell_subset
import json
import toml
import numpy as np


class InversionsonBatchManager(AbstractBatchManager):
    def __init__(
        self, project: Project, batch_size: int, use_overlapping_batches: bool = True
    ):
        super().__init__()
        self.project = project
        self.batch_size = batch_size
        self.all_samples = self.project.event_db.get_all_event_indices(
            non_validation_only=True
        )
        self.use_overlapping_batches = use_overlapping_batches
        self.batch_json = self.project.paths.batch_json
        self.all_mini_batches: Dict[str, List[int]] = {}
        self.all_control_groups: Dict[str, List[int]] = {}
        self._json_to_dicts()

    def update(self, iteration: int) -> Tuple[Optional[List[int]], List[int]]:
        raise NotImplementedError

    def _dict_to_json(self):
        all_info = {
            "batch_size": self.batch_size,
            "mini_batches": self.all_mini_batches,
            "control_groups": self.all_control_groups,
        }

        with open(self.batch_json, "w") as fh:
            json.dump(all_info, fh)

    def _json_to_dicts(self) -> None:
        if not self.batch_json.exists():
            return

        with open(self.batch_json, "r") as fh:
            all_info = json.load(fh)

        self.batch_size = all_info["batch_size"]
        self.all_mini_batches = all_info["mini_batches"]
        self.all_control_groups = all_info["control_groups"]

    def get_batch(self, iteration: int) -> Optional[List[int]]:
        it_str = str(iteration)
        prev_it_str = str(iteration - 1)
        if it_str in self.all_mini_batches:
            return self.all_mini_batches[it_str]

        if prev_it_str in self.all_control_groups:
            prev_control_group = self.all_control_groups[prev_it_str]
        else:
            prev_control_group = []

        eligible_samples = set(self.all_samples) - set(prev_control_group)
        n_events = self.batch_size - len(self.control_group_previous)

        if n_events > 0:
            new_events = self._get_norm_derived_batch(n_events, list(eligible_samples))
        self.all_mini_batches[it_str] = self.project.event_db.get_event_indices(
            new_events + prev_control_group
        )
        self._dict_to_json()
        return self.all_mini_batches[it_str]

    def _get_norm_derived_batch(
        self, n_events: int, eligible_samples: List[int]
    ) -> List[str]:
        all_events = self.project.event_db.get_event_names(eligible_samples)
        if not self.project.paths.all_gradient_norms_toml.exists():
            return get_random_mitchell_subset(
                self.project.lasif.lasif_comm, n_events, all_events
            )
        norm_dict = toml.load(self.project.paths.all_gradient_norms_toml)
        unused_events = list(set(all_events).difference(set(norm_dict.keys())))
        list_of_vals = np.array(list(norm_dict.values()))
        max_norm = np.max(list_of_vals)

        # Assign high norm values to unused events to make them more likely to be chosen
        for event in unused_events:
            norm_dict[event] = max_norm
        return get_random_mitchell_subset(
            self.project.lasif.lasif_comm, n_events, all_events, norm_dict
        )

    def get_control_group(self, iteration: int) -> List[int]:
        if not self.use_overlapping_batches:
            return []
        it_str = str(iteration)
        if it_str in self.all_control_groups:
            return self.all_control_groups[it_str]

        current_batch = self.all_mini_batches[it_str]
        control_group_size = int(np.ceil(0.5 * len(current_batch)))
        events = self._get_norm_derived_batch(control_group_size, current_batch)
        self.all_control_groups[it_str] = self.project.event_db.get_event_indices(
            events
        )
        self._dict_to_json()
        return self.all_control_groups[it_str]

    def extend_control_group(self, iteration: int) -> bool:
        if not self.use_overlapping_batches:
            raise ValueError("This should not occur")

        it_str = str(iteration)
        current_batch = self.all_mini_batches[it_str]
        current_control_group = self.all_control_groups[it_str]

        available_samples = set(current_batch) - set(current_control_group)
        n_available_samples = len(available_samples)
        if n_available_samples == 0:
            return False
        n_new_samples = int(np.ceil(0.5 * n_available_samples))
        events = self._get_norm_derived_batch(
            n_events=n_new_samples, eligible_samples=list(available_samples)
        )
        event_indices = self.project.event_db.get_event_indices(events)
        self.all_control_groups[it_str] += event_indices
        self._dict_to_json()
        return True

    def save(self, file: Union[Path, str]) -> None:
        pass

    def load(self, file: Union[Path, str]) -> None:
        pass

    @property
    def stochastic(self) -> bool:
        return True
