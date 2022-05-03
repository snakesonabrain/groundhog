#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
from mimetypes import init
import numpy as np

# Project imports
from groundhog.general.validation import Validator


PRIMARYCONSOLIDATIONSETTLEMENT_NC = {
    'initial_height': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'initial_voidratio': {'type': 'float', 'min_value': 0.1, 'max_value': 5.0},
    'initial_effective_stress': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'effective_stress_increase': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'compression_index': {'type': 'float', 'min_value': 0.1, 'max_value': 0.8},
    'e_min': {'type': 'float', 'min_value': 0.1, 'max_value': None},
}

PRIMARYCONSOLIDATIONSETTLEMENT_NC_ERRORRETURN = {
    'delta z [m]': np.nan,
    'delta e [-]': np.nan,
    'e final [-]': np.nan
}


@Validator(PRIMARYCONSOLIDATIONSETTLEMENT_NC, PRIMARYCONSOLIDATIONSETTLEMENT_NC_ERRORRETURN)
def primaryconsolidationsettlement_nc(
        initial_height, initial_voidratio, initial_effective_stress, effective_stress_increase, compression_index, e_min=0.3,
        **kwargs):
    """
    Calculates the primary consolidation settlement for normally consolidated fine grained soil.

    :param initial_height: Initial thickness of the layer (:math:`H_0`) [:math:`m`] - Suggested range: initial_height >= 0.0
    :param initial_voidratio: Initial void ratio of the layer (:math:`e_0`) [:math:`-`] - Suggested range: 0.1 <= initial_voidratio <= 5.0
    :param initial_effective_stress: Initial vertical effective stress in the center of the layer (:math:`\\sigma_{v0}^{\\prime}`) [:math:`kPa`] - Suggested range: initial_effective_stress >= 0.0
    :param effective_stress_increase: Increase in vertical effective stress under the given load (:math:`\\Delta sigma_{v}^{\\prime}`) [:math:`kPa`] - Suggested range: effective_stress_increase >= 0.0
    :param compression_index: Compression index derived from oedometer tests (:math:`C_c`) [:math:`-`] - Suggested range: 0.1 <= compression_index <= 0.8 (derived using logarithm with base 10)
    :param e_min: Minimum void ratio below which no further consolidation occurs (:math:`e_{min}`) [:math:`-`] - Default=0.3

    .. math::
        \\Delta z = \\frac{H_0}{1 + e_0} C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}

        \\Delta e = C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}

    :returns: Dictionary with the following keys:

        - 'delta z [m]': Primary consolidation settlement for normally consolidated soil (:math:`\\Delta z`)  [:math:`m`]
        - 'delta e [-]': Decrease in void ratio for the normally consolidated soil (:math:`\\delta e`)  [:math:`-`]
        - 'e final [-]': Final void ratio after consolidation (:math:` e_{final}`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _delta_e = compression_index * \
        np.log10((initial_effective_stress + effective_stress_increase) / initial_effective_stress)

    if (initial_voidratio - _delta_e) > e_min:
        pass
    else:
        _delta_e = initial_voidratio - e_min    
    
    _delta_z = (initial_height / (1 + initial_voidratio)) * _delta_e
    _e_final = initial_voidratio - _delta_e

    return {
        'delta z [m]': _delta_z,
        'delta e [-]': _delta_e,
        'e final [-]': _e_final
    }


PRIMARYCONSOLIDATIONSETTLEMENT_OC = {
    'initial_height': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'initial_voidratio': {'type': 'float', 'min_value': 0.1, 'max_value': 5.0},
    'initial_effective_stress': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'preconsolidation_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'effective_stress_increase': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'compression_index': {'type': 'float', 'min_value': 0.1, 'max_value': 0.8},
    'recompression_index': {'type': 'float', 'min_value': 0.015, 'max_value': 0.35},
    'e_min': {'type': 'float', 'min_value': 0.1, 'max_value': None},
}

PRIMARYCONSOLIDATIONSETTLEMENT_OC_ERRORRETURN = {
    'delta z [m]': np.nan,
    'delta e [-]': np.nan,
    'e final [-]': np.nan
}


@Validator(PRIMARYCONSOLIDATIONSETTLEMENT_OC, PRIMARYCONSOLIDATIONSETTLEMENT_OC_ERRORRETURN)
def primaryconsolidationsettlement_oc(
        initial_height, initial_voidratio, initial_effective_stress, preconsolidation_pressure,
        effective_stress_increase, compression_index, recompression_index, e_min=0.3,
        **kwargs):
    """
    Calculates the primary consolidation settlement for an overconsolidated clay. This material is characterised using a compression index and a recompression index which can be derived from oedometer tests.

    The settlement depends on whether the stress increase loads the layer beyond the preconsolidation pressure. If stresses remain below the preconsolidation pressure, the recompression index applies. If stresses go beyond the preconsolidation pressure, the compression index will apply for the increase beyond the preconsolidation pressure.

    Note that a minimum void ratio is set to prevent calculated void ratios from dropping below the minimum.

    :param initial_height: Initial thickness of the layer (:math:`H_0`) [:math:`m`] - Suggested range: initial_height >= 0.0
    :param initial_voidratio: Initial void ratio of the layer (:math:`e_0`) [:math:`-`] - Suggested range: 0.1 <= initial_voidratio <= 5.0
    :param initial_effective_stress: Initial vertical effective stress in the center of the layer (:math:`\\sigma_{v0)^{\\prime}`) [:math:`kPa`] - Suggested range: initial_effective_stress >= 0.0
    :param preconsolidation_pressure: Preconsolidation pressure, maximum vertical stress to which the layer has been subjected (:math:`p_c^{\\prime}`) [:math:`kPa`] - Suggested range: preconsolidation_pressure >= 0.0
    :param effective_stress_increase: Increase in vertical effective stress under the given load (:math:`\\Delta sigma_{v}^{\\prime}`) [:math:`kPa`] - Suggested range: effective_stress_increase >= 0.0
    :param compression_index: Compression index derived from oedometer tests (:math:`C_c`) [:math:`-`] - Suggested range: 0.1 <= compression_index <= 0.8
    :param recompression_index: Recompression index derived from the unloading step in oedometer tests (:math:`C_r`) [:math:`-`] - Suggested range: 0.015 <= recompression_index <= 0.35
    :param e_min: Minimum void ratio below which no further consolidation occurs (:math:`e_{min}`) [:math:`-`] - Default=0.3

    .. math::
        \\Delta z = \\frac{H_0}{1 + e_0} C_r \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} < p_c^{\\prime}

        \\Delta z = \\frac{H_0}{1 + e_0} \\left( C_r \\log_{10} \\frac{p_c^{\\prime}}{\\sigma_{v0}^{\\prime}} + C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{p_c^{\\prime}} \\right); \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} > p_c^{\\prime}

        \\Delta e = C_r \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} < p_c^{\\prime}

        \\Delta e = C_r \\log_{10} \\frac{p_c^{\\prime}}{\\sigma_{v0}^{\\prime}} + C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{p_c^{\\prime}} ; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} > p_c^{\\prime}

    :returns: Dictionary with the following keys:

        - 'delta z [m]': Primary consolidation settlement for the overconsolidated soil (:math:`\\delta z`)  [:math:`m`]
        - 'delta e [-]': Decrease in void ratio for the overconsolidated soil (:math:`\\delta e`)  [:math:`-`]
        - 'e final [-]': Final void ratio after consolidation (:math:` e_{final}`)  [:math:`-`]

    .. figure:: images/primaryconsolidation_settlement.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Cases for calculating the primary consolidation settlement

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    if (initial_effective_stress + effective_stress_increase) < preconsolidation_pressure:
        _delta_e = recompression_index * np.log10((initial_effective_stress + effective_stress_increase) / initial_effective_stress)
        
    else:
        _delta_e = \
            recompression_index * np.log10(preconsolidation_pressure / initial_effective_stress) + \
            compression_index * np.log10(
                (initial_effective_stress + effective_stress_increase) / preconsolidation_pressure)

    if (initial_voidratio - _delta_e) > e_min:
        pass
    else:
        _delta_e = initial_voidratio - e_min    
    
    _delta_z = (initial_height / (1 + initial_voidratio)) * _delta_e
    _e_final = initial_voidratio - _delta_e

    return {
        'delta z [m]': _delta_z,
        'delta e [-]': _delta_e,
        'e final [-]': _e_final
    }