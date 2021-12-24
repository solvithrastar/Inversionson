"""
A placeholder Class that will make it possible to develop
the switch to the new version of SalvusOpt
"""

from lasif.components.component import Component


class SalvusOptComponent(Component):
    """
    Communications with Salvus Opt
    """

    def __init__(self, communicator, component_name):
        super(SalvusOptComponent, self).__init__(communicator, component_name)
        self.test = None
