Inversionson's documentation!
========================================

What if I don't care?
=====================

Just run this in your inversionson environment::

    cd <Inversionson code base>
    python create_dummy_info_file.py
    mv inversion_info.toml <Inversionson project>
    # Configure your inversion_info.toml to make sense
    python -m inversionson.autoinverter

And then just wait. This doesn't really work though, you need all sorts
of specific configurations to make things run smoothly, but that will
hopefully be explained here one day.

Guide
^^^^^

.. toctree::
   :maxdepth: 2

   Introduction
   Documentation

