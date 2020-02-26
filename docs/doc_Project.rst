Project Component
=================

The Project Component is used to organize the project. It keeps a record
of what is going on in the current iteration, saves some information
into toml files on the fly and reads from these files aswell. The Project
Component ties together all the other components and is the one that makes
it possible for them to communicate with each other.

The Project Component sets up the project and makes sure that everything
is set up correctly before the project is initiated. It reads information
from the input file and communicates the needed information to the other
components.

The methods of the Project Component are documented here below:

.. autoclass:: inversionson.components.project.ProjectComponent
    :members: