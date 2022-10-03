#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.siteinvestigation.classification.phaserelations import voidratio_bulkunitweight, porosity_voidratio
from groundhog.general.validation import Validator


API_UNIT_SHAFT_FRICTION_SAND_RP2GEO = {
    'api_relativedensity':{'type': 'string','options':("Very loose","Loose","Medium dense","Dense","Very dense"),'regex':None},
    'api_soildescription':{'type': 'string','options':("Sand","Sand-silt","Silt","Gravel"),'regex':None},
    'sigma_vo_eff':{'type': 'float','min_value':0.0,'max_value':None},
    'tension_modifier':{'type': 'float','min_value':0.0,'max_value':1.0}
}

API_UNIT_SHAFT_FRICTION_SAND_RP2GEO_ERRORRETURN = {
    'f_s_comp_out [kPa]': np.NaN,
    'f_s_comp_in [kPa]': np.NaN,
    'f_s_tens_out [kPa]': np.NaN,
    'f_s_tens_in [kPa]': np.NaN,
    'f_s_lim [kPa]': np.NaN,
    'beta [-]': np.NaN,
}

@Validator(API_UNIT_SHAFT_FRICTION_SAND_RP2GEO, API_UNIT_SHAFT_FRICTION_SAND_RP2GEO_ERRORRETURN)
def API_unit_shaft_friction_sand_rp2geo(api_relativedensity, api_soildescription, sigma_vo_eff, fs_limit=False,
                                        tension_modifier=1.0, **kwargs):

    """
    Calculates unit skin friction according to the beta method in API RP 2GEO. The main difference is that the beta-parameter is defined directly in API RP 2GEO whereas API RP 2A WSD (2000) works with a soil pile friction angle.

    Use the string ``'API RP2 GEO Sand'`` to define this method in a ``SoilProfile``.

    :param api_relativedensity: Relative density of the sand (:math:`D_r`) [:math:`-`] Options: ("Very loose","Loose","Medium dense","Dense","Very dense"), regex: None
    :param api_soildescription: Description of the soil type (:math:`-`) [:math:`-`] Options: ("Sand","Sand-silt"), regex: None
    :param sigma_vo_eff: In-situ vertical effective stress (:math:`p'_o`) [:math:`kPa`]  - Suggested range: 0.0<=vertical_effective_stress

    .. math::
        f(z) = \\beta \\cdot p'_o(z)

    :returns:   Unit skin friction (:math:`f_s`) [:math:`kPa`], Unit skin friction limit (:math:`f_{s,lim}`) [:math:`kPa`], Coefficient beta (:math:`\\beta`) [:math:`-`]

    :rtype: Python dictionary with keys ['f_s [kPa]','f_s_lim [kPa]','beta [-]']


    Reference - API RP 2GEO, API RP 2GEO Geotechnical and Foundation Design Considerations, 2011

    """

    if api_soildescription == "Sand":
        if api_relativedensity == "Medium dense":
            beta = 0.37
            f_s_lim = 81.0
        elif api_relativedensity == "Dense":
            beta = 0.46
            f_s_lim = 96.0
        elif api_relativedensity == "Very dense":
            beta = 0.56
            f_s_lim = 115.0
        else:
            raise ValueError("Relative density category not found")
    elif api_soildescription == "Sand-silt":
        if api_relativedensity == "Medium dense":
            beta = 0.29
            f_s_lim = 67.0
        elif api_relativedensity == "Dense":
            beta = 0.37
            f_s_lim = 81.0
        elif api_relativedensity == "Very dense":
            beta = 0.46
            f_s_lim = 96.0
        else:
            raise ValueError("Relative density category not found")
    else:
        raise ValueError("Soil description not found")

    if fs_limit:
        f_s_comp = min(beta * sigma_vo_eff, f_s_lim)
        f_s_tens = tension_modifier * min(beta * sigma_vo_eff, f_s_lim)
    else:
        f_s_comp = beta * sigma_vo_eff
        f_s_tens = tension_modifier * beta * sigma_vo_eff

    return {
        'f_s_comp_out [kPa]': f_s_comp,
        'f_s_comp_in [kPa]': f_s_comp,
        'f_s_tens_out [kPa]': f_s_tens,
        'f_s_tens_in [kPa]': f_s_tens,
        'f_s_lim [kPa]': f_s_lim,
        'beta [-]': beta,
    }


API_UNIT_SHAFT_FRICTION_CLAY = {
    'undrained_shear_strength':{'type': 'float','min_value':0.0,'max_value':400.0},
    'sigma_vo_eff':{'type': 'float','min_value':0.0,'max_value':None},
}

API_UNIT_SHAFT_FRICTION_CLAY_ERRORRETURN = {
    'f_s_comp_out [kPa]': np.NaN,
    'f_s_comp_in [kPa]': np.NaN,
    'f_s_tens_out [kPa]': np.NaN,
    'f_s_tens_in [kPa]': np.NaN,
    'psi [-]': np.NaN,
    'alpha [-]': np.NaN,
}

@Validator(API_UNIT_SHAFT_FRICTION_CLAY, API_UNIT_SHAFT_FRICTION_CLAY_ERRORRETURN)
def API_unit_shaft_friction_clay(undrained_shear_strength, sigma_vo_eff, **kwargs):

    """
    Calculates unit skin friction according to the alpha method in API RP 2GEO. Caution should be exercised in its application as there are many more variables which affect pile capacity than the ones accounted for in this equation. Due to the shortage of pile load tests in soils having ratios of undrained shear strenght to vertical effective stress greater than three. The function should be applied with considerable care for these high ratios. Similar judgment should be applied for deep penetrating piles in soils with high undrained shear strength. Low plasticity clays should be treated with particular caution. For very long piles some reduction in capacity may be warranted, particularly where the shaft friction degrades on continued displacement.

    Use the string ``'API RP2 GEO Clay'`` to define this method in a ``SoilProfile``.

    :param undrained_shear_strength: Undrained shear strength (:math:`S_u`) [:math:`kPa`]  - Suggested range: 0.0<=undrained_shear_strength<=400.0
    :param sigma_vo_eff: In-situ vertical effective stress (:math:`p'_o(z)`) [:math:`kPa`]  - Suggested range: 0.0<=vertical_effective_stress

    .. math::
        f(z) = \\alpha \\cdot S_u

        \\alpha = 0.5 \\cdot \\psi^{-0.5} & \\quad \\text{for } \\psi \\leq 1.0

        \\alpha = 0.5 \\cdot \\psi^{-0.25} & \\quad \\text{for } \\psi > 1.0

        \\psi = \\frac{S_u}{p'_o(z)}

    :returns:   Unit shaft friction (:math:`f(z)`) [:math:`kPa`], Ratio of undrained shear strength to vertical effective stress (:math:`\\psi`) [:math:`-`], Alpha factor (:math:`\\alpha`) [:math:`-`]

    :rtype: Python dictionary with keys ['f_s [kPa]','psi [-]','alpha [-]']

    Reference - API RP 2GEO, API RP 2GEO Geotechnical and Foundation Design Considerations, 2011.

    """
    psi = undrained_shear_strength / sigma_vo_eff

    if psi <= 1.0:
        alpha = 0.5 * (psi ** -0.5)
    else:
        alpha = 0.5 * (psi ** -0.25)

    f_s = alpha * undrained_shear_strength

    return {
        'f_s_comp_out [kPa]': f_s,
        'f_s_comp_in [kPa]': f_s,
        'f_s_tens_out [kPa]': f_s,
        'f_s_tens_in [kPa]': f_s,
        'psi [-]': psi,
        'alpha [-]': alpha,
    }

SKINFRICTION_METHODS = {
    'API RP2 GEO Sand': API_unit_shaft_friction_sand_rp2geo,
    'API RP2 GEO Clay': API_unit_shaft_friction_clay
}

SKINFRICTION_PARAMETERS = {
    'API RP2 GEO Sand': ['api_relativedensity', 'api_soildescription', 'sigma_vo_eff'],
    'API RP2 GEO Clay': ['undrained_shear_strength', 'sigma_vo_eff']
}