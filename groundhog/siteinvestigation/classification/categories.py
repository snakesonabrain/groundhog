#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.general.validation import Validator



RELATIVEDENSITY_CATEGORIES = {
    'relative_density': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
}

RELATIVEDENSITY_CATEGORIES_ERRORRETURN = {
    'Relative density': None,
}


@Validator(RELATIVEDENSITY_CATEGORIES, RELATIVEDENSITY_CATEGORIES_ERRORRETURN)
def relativedensity_categories(
        relative_density,
        **kwargs):
    """
    Categorizes relative densities according to the following definition:

        - 0 - 0.15: Very loose
        - 0.15 - 0.35: Loose
        - 0.35 - 0.65: Medium dense
        - 0.65 - 0.85: Dense
        - 0.85 - 1: Very dense

    :param relative_density: Relative density of cohesionless material (:math:`D_r`) [:math:`-`] - Suggested range: 0.0 <= relative_density <= 1.0

    .. math::
        D_r = \\frac{e - e_{min}}{e_{max} - e_{min}}

    :returns: Dictionary with the following keys:

        - 'Relative density': Relative density class

    Reference - API RP2 GEO

    """

    if 0 <= relative_density < 0.15:
        _relative_density = "Very loose"
    elif 0.15 <= relative_density < 0.35:
        _relative_density = "Loose"
    elif 0.35 <= relative_density < 0.65:
        _relative_density = "Medium dense"
    elif 0.65 <= relative_density < 0.85:
        _relative_density = "Dense"
    elif relative_density >= 0.85:
        _relative_density = "Very dense"

    return {
        'Relative density': _relative_density,
    }


SU_CATEGORIES = {
    'undrained_shear_strength': {'type': 'float', 'min_value': 0.0, 'max_value': 1000.0},
    'standard': {'type': 'string', 'options': ('BS 5930:2015', 'ASTM D-2488'), 'regex': None},
}

SU_CATEGORIES_ERRORRETURN = {
    'strength class': None,
}


@Validator(SU_CATEGORIES, SU_CATEGORIES_ERRORRETURN)
def su_categories(
        undrained_shear_strength,
        standard="BS 5930:2015", ** kwargs):

    """
    Classifies undrained shear strength in a number of categories.
    The classification system can be selected but the default is BS 5930:2015.
    Classification according to ASTM D-2488 is also available.

    According to BS 5930:2015:

    - Extremely low: < 10kPa
    - Very low: 10 - 20kPa
    - Low: 20 - 40kPa
    - Medium: 40 - 75kPa
    - High: 75 - 150kPa
    - Very high: 150 - 300kPa
    - Extremely high: > 300kPa

    According to ASTM D-2488:

    - Very soft: 0 - 12.5kPa
    - Soft: 12.5 - 25kPa
    - Firm: 25 - 50kPa
    - Stiff: 50 - 100kPa
    - Very stiff: 100 - 200kPa
    - Hard: 200 - 400kPa
    - Very hard: > 400kPa


    :param undrained_shear_strength: Undrained shear strength of the cohesive sample (:math:`S_u`) [:math:`kPa`] - Suggested range: 0.0 <= undrained_shear_strength <= 1000.0
    :param standard: Standard used for the classification (optional, default= 'BS 5930:2015') - Options: ('BS 5930:2015', 'ASTM D-2488')

    :returns: Dictionary with the following keys:

        - 'strength class': Strength class for the selected classification system

    Reference - BS 5930:2015, ASTM D-2488

    """

    if standard == 'BS 5930:2015':
        if undrained_shear_strength < 10:
            _strength_class = 'Extremely low'
        elif 10 <= undrained_shear_strength < 20:
            _strength_class = 'Very low'
        elif 20 <= undrained_shear_strength < 40:
            _strength_class = 'Low'
        elif 40 <= undrained_shear_strength < 75:
            _strength_class = 'Medium'
        elif 75 <= undrained_shear_strength < 150:
            _strength_class = 'High'
        elif 150 <= undrained_shear_strength < 300:
            _strength_class = 'Very high'
        elif 300 <= undrained_shear_strength:
            _strength_class = 'Extremely high'
        else:
            raise ValueError("Undrained shear strength outside bounds")
    elif standard == 'ASTM D-2488':
        if undrained_shear_strength < 12.5:
            _strength_class = 'Very soft'
        elif 13 <= undrained_shear_strength < 25:
            _strength_class = 'Soft'
        elif 25 <= undrained_shear_strength < 50:
            _strength_class = 'Firm'
        elif 50 <= undrained_shear_strength < 100:
            _strength_class = 'Stiff'
        elif 100 <= undrained_shear_strength < 200:
            _strength_class = 'Very stiff'
        elif 200 <= undrained_shear_strength < 400:
            _strength_class = 'Hard'
        elif undrained_shear_strength >= 400:
            _strength_class = 'Very hard'
        else:
            raise ValueError("Undrained shear strength outside bounds")
    else:
        raise ValueError("Standard unknown, choose from 'BS 5930:2015', 'ASTM D-2488'")

    return {
        'strength class': _strength_class,
    }

USCS_DICTIONARY = {
    "GW": "Well graded gravels, gravel- sand mixtures, little or no fines",
    "GP": "Poorly graded gravels, gravel sand mixtures, little or no fines",
    "GM": "Silty gravels, poorly graded gravel-sand-silt mixtures",
    "GC": "Clayey gravels, poorly graded gravel-sand-clay mixtures",
    "SW": "Well graded sands, gravelly sands, little or no fines",
    "SP": "Poorly graded sands, gravelly sands, little or no fines",
    "SM": "Silty sands, poorly graded sand- silt mixtures",
    "SC": "Clayey sands, poorly graded sand-clay mixtures",
    "ML": "Inorganic silts and very fine sands, rock flour, silty or clayey fine sands with slight plasticity",
    "CL": "Inorganic clays of low to medium plasticity, gravelly clays, sandy clays, silty clays, lean clays",
    "OL": "Organic clays organic silt-clays of low plasticity",
    "MH": "Inorganic silts, micaceous or diatomaceous fine sandy or silty soils, elastic silts",
    "CH": "Inorganic clays of high plastic- ity, fat clays",
    "OH": "Organic clays of medium-high plasticity"
}

USCS_CATEGORIES_ERRORRETURN = {
    'Soil type': None,
}

USCS_CATEGORIES = {
    'symbol': {'type': 'string', 'options': ("GW", "GP", "GM", "GC", "SW", "SP", "SM", "SC", "ML", "CL", "OL", "MH", "CH", "OH"), 'regex': None}
}


@Validator(USCS_CATEGORIES, USCS_CATEGORIES_ERRORRETURN)
def uscs_categories(
        symbol,
        **kwargs):
    """
    Provides the verbose description for soil type codes according to USCS. The ``USCS_DICTIONARY`` can also be used in workflows.

    :param symbol: Two character symbol for the soil type according to USCS

    :returns: Dictionary with the following keys:

        - 'Soil type': Verbose description of the soil type

    Reference - USCS

    """

    
    _soiltype = USCS_DICTIONARY[symbol]

    return {
        'Soil type': _soiltype,
    }

