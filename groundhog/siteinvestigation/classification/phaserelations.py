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


VOIDRATIO_POROSITY = {
    'porosity': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
}

VOIDRATIO_POROSITY_ERRORRETURN = {
    'voidratio [-]': np.nan,
}

@Validator(VOIDRATIO_POROSITY, VOIDRATIO_POROSITY_ERRORRETURN)
def voidratio_porosity(
        porosity,
        **kwargs):

    """
    Converts a void ratio into a porosity

    :param porosity: Porosity of the sample defined as the ratio of volume of voids to total volume (:math:`n`) [:math:`-`] - Suggested range: 0.0 <= porosity <= 1.0

    .. math::
        e = \\frac{V_{voids}}{V_{solids}}

        n = \\frac{V_{voids}}{V_{total}}

        e = \\frac{n}{1-n}

    :returns: Dictionary with the following keys:

        - 'voidratio [-]': Void ratio defined as the ratio of volume of voids to volume of solids (:math:`e`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _voidratio = porosity / (1- porosity)

    return {
        'voidratio [-]': _voidratio,
    }


POROSITY_CALC = {
    'voidratio': {'type': 'float', 'min_value': 0.0, 'max_value': 5.0},
}

POROSITY_CALC_ERRORRETURN = {
    'porosity [-]': np.nan,
}


@Validator(POROSITY_CALC, POROSITY_CALC_ERRORRETURN)
def porosity_voidratio(
        voidratio,
        **kwargs):
    """
    Calculates the porosity of sample from the void ratio

    :param voidratio: Void ratio defined as the ratio of volume of voids to volume of solids (:math:`e`) [:math:`-`] - Suggested range: 0.0 <= voidratio <= 5.0

    .. math::
        n = \\frac{e}{e+1}

    :returns: Dictionary with the following keys:

        - 'porosity [-]': Porosity defined as the ratio of the volume of voids to the total volume (:math:`n`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _porosity = voidratio / (1 + voidratio)

    return {
        'porosity [-]': _porosity,
    }


SATURATION_WATERCONTENT = {
    'water_content': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'voidratio': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'specific_gravity': {'type': 'float', 'min_value': 1.0, 'max_value': 3.0},
}

SATURATION_WATERCONTENT_ERRORRETURN = {
    'saturation [-]': np.nan,
}


@Validator(SATURATION_WATERCONTENT, SATURATION_WATERCONTENT_ERRORRETURN)
def saturation_watercontent(
        water_content, voidratio,
        specific_gravity=2.65, **kwargs):
    """
    Calculates the saturation of a sample from the water content, the specific gravity and the void ratio

    :param water_content: Water content of the soil defined as the ratio of weight of water to weight of solids (:math:`w`) [:math:`-`] - Suggested range: 0.0 <= water_content <= 4.0
    :param voidratio: Ratio of volume of voids to volume of solids (:math:`e`) [:math:`-`] - Suggested range: 0.0 <= voidratio <= 4.0
    :param specific_gravity: Specific gravity of the soil grains (:math:`G_s`) [:math:`-`] - Suggested range: 1.0 <= specific_gravity <= 3.0 (optional, default= 2.65)

    .. math::
        S = \\frac{V_{water}}{V_{voids}} = \\frac{w \\cdot G_s}{e}

    :returns: Dictionary with the following keys:

        - 'saturation [-]': Saturation of the sample defined as the ratio of volume of water to volume of voids (:math:`S`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _saturation = water_content * specific_gravity / voidratio

    return {
        'saturation [-]': _saturation,
    }


BULKUNITWEIGHT = {
    'saturation': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'voidratio': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'specific_gravity': {'type': 'float', 'min_value': 1.0, 'max_value': 3.0},
    'unitweight_water': {'type': 'float', 'min_value': 9.0, 'max_value': 11.0},
}

BULKUNITWEIGHT_ERRORRETURN = {
    'bulk unit weight [kN/m3]': np.nan,
    'effective unit weight [kN/m3]': np.nan,
}


@Validator(BULKUNITWEIGHT, BULKUNITWEIGHT_ERRORRETURN)
def bulkunitweight(
        saturation, voidratio,
        specific_gravity=2.65, unitweight_water=10.0, **kwargs):
    """
    Calculates the bulk unit weight from specific gravity, void ratio and saturation

    :param saturation: Saturation of the sample, ratio of volume of water to volume of voids (:math:`S`) [:math:`-`] - Suggested range: 0.0 <= saturation <= 1.0
    :param voidratio: Void ratio, ratio of volume of voids to volume of solids (:math:`e`) [:math:`-`] - Suggested range: 0.0 <= voidratio <= 4.0
    :param specific_gravity: Specific gravity of solid particles (:math:`G_s`) [:math:`-`] - Suggested range: 1.0 <= specific_gravity <= 3.0 (optional, default= 2.65)
    :param unitweight_water: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.0 <= unitweight_water <= 11.0 (optional, default= 10.0)

    .. math::
        \\gamma = \\frac{W}{V} = \\left( \\frac{G_s + S \\cdot e}{1+e} \\right) \\cdot \\gamma_w

    :returns: Dictionary with the following keys:

        - 'bulk unit weight [kN/m3]': Bulk unit weight of the material (:math:`\\gamma`)  [:math:`kN/m3`]
        - 'effective unit weight [kN/m3]': Effective unit weight of the material (:math:`\\gamma^{\\prime}`)  [:math:`kN/m3`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _bulk_unit_weight = unitweight_water * (
            (specific_gravity + saturation * voidratio) / (1 + voidratio))
    _effective_unit_weight = _bulk_unit_weight - unitweight_water

    return {
        'bulk unit weight [kN/m3]': _bulk_unit_weight,
        'effective unit weight [kN/m3]': _effective_unit_weight,
    }


DRYUNITWEIGHT_WATERCONTENT = {
    'watercontent': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'bulkunitweight': {'type': 'float', 'min_value': 10.0, 'max_value': 25.0},
}

DRYUNITWEIGHT_WATERCONTENT_ERRORRETURN = {
    'dry unit weight [kN/m3]': np.nan,
}


@Validator(DRYUNITWEIGHT_WATERCONTENT, DRYUNITWEIGHT_WATERCONTENT_ERRORRETURN)
def dryunitweight_watercontent(
        watercontent, bulkunitweight,
        **kwargs):
    """
    Calculates the dry unit weight of the sample from the water content and the bulk unit weight

    :param watercontent: Water content of the sample, ratio of weight of water to weight of solids (:math:`w`) [:math:`-`] - Suggested range: 0.0 <= watercontent <= 4.0
    :param bulkunitweight: Bulk unit weight of the sample (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 10.0 <= bulkunitweight <= 25.0

    .. math::
        \\gamma_d = \\frac{W_s}{V} = \\left( \\frac{G_s}{1+e} \\right) \\cdot \\gamma_w = \\frac{\\gamma}{1 + w}

    :returns: Dictionary with the following keys:

        - 'dry unit weight [kN/m3]': Dry unit weight, ratio of weight of solids to total volume (:math:`\\gamma_d`)  [:math:`kN/m3`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _dry_unit_weight = bulkunitweight / (1 + watercontent)

    return {
        'dry unit weight [kN/m3]': _dry_unit_weight,
    }


BULKUNITWEIGHT_DRYUNITWEIGHT = {
    'dryunitweight': {'type': 'float', 'min_value': 1.0, 'max_value': 15.0},
    'watercontent': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
}

BULKUNITWEIGHT_DRYUNITWEIGHT_ERRORRETURN = {
    'bulk unit weight [kN/m3]': np.nan,
    'effective unit weight [kN/m3]': np.nan,
}


@Validator(BULKUNITWEIGHT_DRYUNITWEIGHT, BULKUNITWEIGHT_DRYUNITWEIGHT_ERRORRETURN)
def bulkunitweight_dryunitweight(
        dryunitweight, watercontent, unitweight_water=10.0,
        **kwargs):
    """
    Calculates the bulk unit weight from the dry unit weight and the water content

    :param dryunitweight: Dry unit weight, ratio of weight of solids to total volume (:math:`\\gamma_d`) [:math:`kN/m3`] - Suggested range: 1.0 <= dryunitweight <= 15.0
    :param watercontent: Water content, ratio of weight of water to weight of solids (:math:`w`) [:math:`-`] - Suggested range: 0.0 <= watercontent <= 4.0
    :param unitweight_water: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.0 <= unitweight_water <= 11.0 (optional, default= 10.0)

    .. math::
        \\gamma = (1+w) \\cdot \\gamma_d

    :returns: Dictionary with the following keys:

        - 'bulk unit weight [kN/m3]': Bulk unit weight (:math:`\\gamma`)  [:math:`kN/m3`]
        - 'effective unit weight [kN/m3]': Effective unit weight (:math:`\\gamma^{\\prime}`)  [:math:`kN/m3`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _bulk_unit_weight = (1 + watercontent) * dryunitweight
    _effective_unit_weight = _bulk_unit_weight - unitweight_water

    return {
        'bulk unit weight [kN/m3]': _bulk_unit_weight,
        'effective unit weight [kN/m3]': _effective_unit_weight,
    }


RELATIVE_DENSITY = {
    'void_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 5.0},
    'e_min': {'type': 'float', 'min_value': 0.0, 'max_value': 5.0},
    'e_max': {'type': 'float', 'min_value': 0.0, 'max_value': 5.0},
}

RELATIVE_DENSITY_ERRORRETURN = {
    'Dr [-]': np.nan,
}


@Validator(RELATIVE_DENSITY, RELATIVE_DENSITY_ERRORRETURN)
def relative_density(void_ratio, e_min, e_max, **kwargs):
    """
    Calculates the relative density for a cohesionless sample from the measured void ratio, comparing it to the void ratio at minimum and maximum density.

    :param void_ratio: Void ratio of the sample (:math:`e`) [:math:`-`] - Suggested range: 0.0 <= void_ratio <= 5.0
    :param e_min: Void ratio at the minimum density (:math:`e_{min}`) [:math:`-`] - Suggested range: 0.0 <= e_min <= 5.0
    :param e_max: Void ratio at the maximum density (:math:`e_{max}`) [:math:`-`] - Suggested range: 0.0 <= e_max <= 5.0

    .. math::
        D_r = \\frac{e - e_{min}}{e_{max} - e_{min}}

    :returns: Dictionary with the following keys:

        - 'Dr [-]': Relative density (:math:`D_r`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _Dr = (void_ratio - e_min) / (e_max - e_min)

    return {
        'Dr [-]': _Dr,
    }

RELATIVEDENSITY_CATEGORIES = {
    'relative_density': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
}

RELATIVEDENSITY_CATEGORIES_ERRORRETURN = {
    'Relative density []': None,
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
    elif relative_density > 0.85:
        _relative_density = "Very dense"

    return {
        'Relative density': _relative_density,
    }


VOIDRATIO_BULKUNITWEIGHT = {
    'bulkunitweight': {'type': 'float', 'min_value': 10.0, 'max_value': 25.0},
    'saturation': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'specific_gravity': {'type': 'float', 'min_value': 2.4, 'max_value': 2.9},
    'unitweight_water': {'type': 'float', 'min_value': 9.0, 'max_value': 11.0},
}

VOIDRATIO_BULKUNITWEIGHT_ERRORRETURN = {
    'e [-]': np.nan,
}


@Validator(VOIDRATIO_BULKUNITWEIGHT, VOIDRATIO_BULKUNITWEIGHT_ERRORRETURN)
def voidratio_bulkunitweight(
        bulkunitweight,
        saturation=1.0, specific_gravity=2.65, unitweight_water=10.0, **kwargs):
    """
    Calculates the void ratio from the bulk unit weight for a soil with varying saturation.

    Since unit weight is generally better known or measured than void ratio, this conversion can be useful to derive the in-situ void ratio in a soil profile.

    The default behaviour of this function assumes saturated soil but the saturation can be changed for dry or partially saturated soil.

    The water content is also returned.

    :param bulkunitweight: The bulk unit weight of the soil (ratio of weight of water and solids to volume) (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 10.0 <= bulkunitweight <= 25.0
    :param saturation: Saturation of the soil as a number between 0 (dry) and fully saturated (1) (:math:`S`) [:math:`-`] - Suggested range: 0.0 <= saturation <= 1.0 (optional, default= 1.0)
    :param specific_gravity: Specific gravity or the ratio of the weight of soil solids to the weight of an equal volume of water (:math:`G_s`) [:math:`-`] - Suggested range: 2.4 <= specific_gravity <= 2.9 (optional, default= 2.65)
    :param unitweight_water: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.0 <= unitweight_water <= 11.0 (optional, default= 10.0)

    .. math::
        \\gamma = \\left( \\frac{G_s + S e}{1 + e} \\right) \\gamma_w

        \\implies e = \\frac{\\gamma_w G_s - \\gamma}{\\gamma - S \\gamma_w}

        w = \\frac{S e}{G_s}

    :returns: Dictionary with the following keys:

        - 'e [-]': Void ratio of the soil (:math:`e`)  [:math:`-`]
        - 'w [-]': Water content of the soil (:math:`w`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _e = (unitweight_water * specific_gravity - bulkunitweight) / (bulkunitweight - saturation * unitweight_water)
    _w = (saturation * _e) / specific_gravity

    return {
        'e [-]': _e,
        'w [-]': _w
    }


UNITWEIGHT_WATERCONTENT_SATURATED = {
    'water_content': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
    'specific_gravity': {'type': 'float', 'min_value': 2.5, 'max_value': 2.8},
    'gamma_w': {'type': 'float', 'min_value': 9.5, 'max_value': 10.5},
}

UNITWEIGHT_WATERCONTENT_SATURATED_ERRORRETURN = {
    'gamma [kN/m3]': np.nan,
}


@Validator(UNITWEIGHT_WATERCONTENT_SATURATED, UNITWEIGHT_WATERCONTENT_SATURATED_ERRORRETURN)
def unitweight_watercontent_saturated(
        water_content,
        specific_gravity=2.65, gamma_w=10.0, **kwargs):
    """
    Calculates the bulk unit weight from water content for a saturated soil. A specific gravity needs to be assumed or derived from pycnometer test results.

    :param water_content: Water content of the sample (:math:`w`) [:math:`-`] - Suggested range: 0.0 <= water_content <= 2.0
    :param specific_gravity: Specific gravity of the soil (:math:`G_s`) [:math:`-`] - Suggested range: 2.5 <= specific_gravity <= 2.8 (optional, default= 2.65)
    :param gamma_w: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.5 <= gamma_w <= 10.5 (optional, default= 10.0)

    .. math::
        S \\cdot e = w \\cdot G_s

        \\gamma = \\left( \\frac{G_s + S \\cdot e}{1 + e} \\right) \\cdot \\gamma_w

        \\gamma = \\left( \\frac{G_s \\cdot (1 + w)}{1 + w \\cdot G_s} \\right) \\cdot \\gamma_w

    :returns: Dictionary with the following keys:

        - 'gamma [kN/m3]': Bulk unit weight of the saturated sample (:math:`\\gamma`)  [:math:`kN/m3`]

    Reference - UGent In-house practice

    """

    _gamma = ((specific_gravity * (1 + water_content)) / (1 + water_content * specific_gravity)) * gamma_w

    return {
        'gamma [kN/m3]': _gamma,
    }