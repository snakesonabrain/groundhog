#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


HYDRAULICCONDUCTIVITY_UNCONFINEDAQUIFER = {
    'radius_1': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'radius_2': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'piezometric_height_1': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'piezometric_height_2': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'flowrate': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

HYDRAULICCONDUCTIVITY_UNCONFINEDAQUIFER_ERRORRETURN = {
    'hydraulic_conductivity [m/s]': np.nan,
}

@Validator(HYDRAULICCONDUCTIVITY_UNCONFINEDAQUIFER, HYDRAULICCONDUCTIVITY_UNCONFINEDAQUIFER_ERRORRETURN)
def hydraulicconductivity_unconfinedaquifer(
    radius_1,radius_2,piezometric_height_1,piezometric_height_2,flowrate,
     **kwargs):

    """
    Calculates the hydraulic conductivity from observing two standpipes in the vicinity of a pumping well. The standpipes should be within the radius of influence of the pumping well.

    The following conditions must be satisfied:

        - Unconfined and non-leaking water layer
        - Open base of the pumping well is below the groundwater level
        - Homogeneous, isotropic soil mass of infinite size
        - Darcy's law applies
        - Radial flow
        - Hydraulic gradient equal to slope of groundwater surface
    
    :param radius_1: Radial distance between the axis of the pumping well and the first standpipe (:math:`r_1`) [:math:`m`] - Suggested range: radius_1 >= 0.0
    :param radius_2: Radial distance between the axis of the pumping well and the second standpipe (:math:`r_2`) [:math:`m`] - Suggested range: radius_2 >= 0.0
    :param piezometric_height_1: Piezometric height in the first standpipe (:math:`h_1`) [:math:`m`] - Suggested range: piezometric_height_1 >= 0.0
    :param piezometric_height_2: Piezometric height in the second standpipe (:math:`h_2`) [:math:`m`] - Suggested range: piezometric_height_2 >= 0.0
    :param flowrate: Flowrate extracted from the pumping well (:math:`q_z`) [:math:`m3/s`] - Suggested range: flowrate >= 0.0
    
    .. math::
        i = \\frac{dz}{dr}
        
        A = 2 \\pi r z
        
        q_z = 2 \\pi r z k \\frac{dz}{dr}
        
        q_z \\int_{r_1)^{r_2} \\frac{dr}{r} = 2 k \\pi \\int_{h_1}^{h_2} z dz
        
        k = \\frac{q_z \\ln \\left( r_2 / r_1 \\right)}{\\pi \\left( h_2^2 - h_1^2 \\right) }
    
    :returns: Dictionary with the following keys:
        
        - 'hydraulic_conductivity [m/s]': Hydraulic conductivity (:math:`k`)  [:math:`m/s`]
    
    .. figure:: images/hydraulicconductivity_unconfinedaquifer_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Geometry of the pumping test (Budhu, 2011)
    
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.

    """
    
    _hydraulic_conductivity = flowrate * np.log(radius_2 / radius_1) / \
        (np.pi * (piezometric_height_2 ** 2 - piezometric_height_1 ** 2))

    return {
        'hydraulic_conductivity [m/s]': _hydraulic_conductivity,
    }