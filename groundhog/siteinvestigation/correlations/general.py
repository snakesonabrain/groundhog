#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.siteinvestigation.classification.phaserelations import voidratio_bulkunitweight, porosity_voidratio
from groundhog.general.validation import Validator


ACOUSTICIMPEDANCE_BULKUNITWEIGHT_CHEN = {
    'bulkunitweight': {'type': 'float', 'min_value': 12.0, 'max_value': 22.0},
    'specific_gravity': {'type': 'float', 'min_value': 1.0, 'max_value': 3.0},
    'saturation': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'gamma_w': {'type': 'float', 'min_value': 9.5, 'max_value': 10.5},
    'calibration_factor_4': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_0': {'type': 'float', 'min_value': None, 'max_value': None},
}

ACOUSTICIMPEDANCE_BULKUNITWEIGHT_CHEN_ERRORRETURN = {
    'e [-]': np.nan,
    'w [-]': np.nan,
    'n [-]': np.nan,
    'I [(m/s).(g/cm3)]': np.nan,
}

@Validator(ACOUSTICIMPEDANCE_BULKUNITWEIGHT_CHEN, ACOUSTICIMPEDANCE_BULKUNITWEIGHT_CHEN_ERRORRETURN)
def acousticimpedance_bulkunitweight_chen(
        bulkunitweight,
        specific_gravity=2.65,saturation=1.0,gamma_w=10.0,calibration_factor_4=0.0001315,calibration_factor_3=-0.03776,calibration_factor_2=4.201,calibration_factor_1=-245.0,calibration_factor_0=8603.0, **kwargs):

    """
    Several authors have researched the correlation between porosity and acoustic impedance. Chen et al compiled available measurements for sand and clay and supplemented them with deepwater measurements with the multi-sensor core logger.

    Since porosity is not a parameter which is commonly used, the user can enter bulk unit weight instead which is then converted to porosity for a saturated soil.

    The correlation shows a tight relation between acoustic impedance and porosity. However, soils with in-situ excess pore pressure are not included in this dataset.

    :param bulkunitweight: Bulk (total) unit weight (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 12.0 <= bulkunitweight <= 22.0
    :param specific_gravity: Specific gravity of the soil (:math:`G_s`) [:math:`-`] - Suggested range: 1.0 <= specific_gravity <= 3.0 (optional, default= 2.65)
    :param saturation: Saturation of the soil (fully saturated for offshore soils) (:math:`S`) [:math:`-`] - Suggested range: 0.0 <= saturation <= 1.0 (optional, default= 1.0)
    :param gamma_w: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.5 <= gamma_w <= 10.5 (optional, default= 10.0)
    :param calibration_factor_4: Calibration factor on the fourth order term (:math:``) [:math:`-`] (optional, default= 0.0001315)
    :param calibration_factor_3: Calibration factor on the third order term (:math:``) [:math:`-`] (optional, default= -0.03776)
    :param calibration_factor_2: Calibration factor on the second order term (:math:``) [:math:`-`] (optional, default= 4.201)
    :param calibration_factor_1: Calibration factor on the first order term (:math:``) [:math:`-`] (optional, default= -245.0)
    :param calibration_factor_0: Calibration factor on the zero order term (:math:``) [:math:`-`] (optional, default= 8603.0)

    .. math::
        I =1.315 \\cdot 10^{-4} \\cdot n^4 - 3.776 \\cdot 10^{-2} \\cdot n^3 + 4.201 \\cdot n^2 - 2.450 \\cdot 10^2 \\cdot n + 8.603 \\cdot 10^3

        e = \\frac{\\gamma_w G_s - \\gamma}{\\gamma - S \\gamma_w}

        w = \\frac{S e}{G_s}

        n = \\frac{e}{e+1}

    :returns: Dictionary with the following keys:

        - 'e [-]': Void ratio (:math:`e`)  [:math:`-`]
        - 'we [-]': Water content (:math:`w`)  [:math:`-`]
        - 'n [-]': Porosity (:math:`n`)  [:math:`-`]
        - 'I [(m/s).(g/cm3)]': Acoustic impedance (:math:`I`)  [:math:`(m/s).(g/cm3)`]

    .. figure:: images/acousticimpedance_bulkunitweight_chen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Compiled data from Chen et al

    Reference - Chen et al (2021). Machine Learning Based Digital Integration of Geotechnical and Ultra-High Frequency Geophysical Data for Offshore Site Characterizations. Journal of Geotechnical and Geoenvironmental Engineering.

    """
    _result = voidratio_bulkunitweight(
        bulkunitweight=bulkunitweight,
        saturation=saturation,
        specific_gravity=specific_gravity,
        unitweight_water=gamma_w)
    _e = _result['e [-]']
    _w = _result['w [-]']
    _n = porosity_voidratio(voidratio=_e)['porosity [-]']
    _I = calibration_factor_4 * (100 * _n) ** 4 + \
        calibration_factor_3 * (100 * _n) ** 3 + \
        calibration_factor_2 * (100 * _n) ** 2 + \
        calibration_factor_1 * (100 * _n) + \
        calibration_factor_0

    return {
        'e [-]': _e,
        'n [-]': _n,
        'I [(m/s).(g/cm3)]': _I,
    }