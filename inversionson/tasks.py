"""
A class with a collection of tasks that need to be done as a part of an inversion.
Created to have an easily accessible collection of methods which different optimizers have in common.
"""
from typing import List, Dict
import autoinverter_helpers as helper


class Task(object):
    def __init__(task_info: List[Dict]):
        self.task_info = task_info


"""
I think this will mostly include basic information.
There can be an inform task object method that gives info
And then there can be the actual tasks in there.
Whether it actually needs the info, I'm not sure. We'll see.
It would be good if the status can somehow be saved on the go.
"""
