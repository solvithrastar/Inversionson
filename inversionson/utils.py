"""
A collection of useful scripts which don't really fall into one of
the components.
Stuff like cutting away sources and receivers and such.
Maybe one day some of these will be moved to a handyman component
or something like that.
"""

import numpy as np
import os
import h5py

def cut_source_region_from_gradient(mesh: str, source_location: dict,
    radius_to_cut: float):
    """
    Sources often show unreasonable sensitivities. This function
    brings the value of the gradient down to zero for that region.
    I recommend doing this before smoothing.
    
    :param mesh: Path to the mesh
    :type mesh: str
    :param source_location: Source latitude, longitude and depth
    :type source_location: dict
    :param radius_to_cut: Radius to cut in km
    :type radius_to_cut: float
    """
    print("Not at all implemented yet.")
    print("Not even sure in which coordinate system the mesh operates")

def cut_receiver_regions_from_gradient(mesh: str, receivers: dict,
    radius_to_cut: float):
    """
    Remove regions around receivers from gradients. Receivers often have an
    imprint on a model and this aims to fight that effect.
    
    :param mesh: Path to a mesh with a gradient
    :type mesh: str
    :param receivers: key: receivers{'lat': , 'lon':}
    :type receivers: dict
    :param radius_to_cut: Radius to cut gradient in km
    :type radius_to_cut: float
    """
    print("Still need to implement this but should be done soon.")

def clip_gradient(mesh: str, percentile: float):
    """
    Clip the gradient to remove abnormally high/low values from it.
    
    :param mesh: Path to mesh containing gradient
    :type mesh: str
    :param percentile: The percentile at which you want to clip the gradient
    :type percentile: float
    """
    print("Not implemented yet")
