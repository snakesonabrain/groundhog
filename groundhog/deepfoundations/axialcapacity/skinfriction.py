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
    'f_s_comp_out [kPa]': np.nan,
    'f_s_comp_in [kPa]': np.nan,
    'f_s_tens_out [kPa]': np.nan,
    'f_s_tens_in [kPa]': np.nan,
    'f_s_lim [kPa]': np.nan,
    'beta [-]': np.nan,
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
    'f_s_comp_out [kPa]': np.nan,
    'f_s_comp_in [kPa]': np.nan,
    'f_s_tens_out [kPa]': np.nan,
    'f_s_tens_in [kPa]': np.nan,
    'psi [-]': np.nan,
    'alpha [-]': np.nan,
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

ALMHAMRE_UNITSKINFRICTION_SAND = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'interface_friction_angle': {'type': 'float', 'min_value': 10.0, 'max_value': 50.0},
    'depth': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'embedded_length': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'shape_factor_multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'fsi_sand_multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'fsi_sand_exponent': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_fsres': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_outside': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_inside': {'type': 'float', 'min_value': None, 'max_value': None},
}

ALMHAMRE_UNITSKINFRICTION_SAND_ERRORRETURN = {
    'f_s_comp_out [kPa]': np.nan,
    'f_s_comp_in [kPa]': np.nan,
    'f_s_tens_out [kPa]': np.nan,
    'f_s_tens_in [kPa]': np.nan,
    'f_s_initial [kPa]': np.nan,
    'f_s_res [kPa]': np.nan,
}


@Validator(ALMHAMRE_UNITSKINFRICTION_SAND, ALMHAMRE_UNITSKINFRICTION_SAND_ERRORRETURN)
def unitskinfriction_sand_almhamre(
        qt, sigma_vo_eff, interface_friction_angle, depth, embedded_length,
        shape_factor_multiplier=80.0, atmospheric_pressure=101.325, fsi_sand_multiplier=0.0132, fsi_sand_exponent=0.13,
        multiplier_fsres=0.2, multiplier_outside=0.5, multiplier_inside=0.5, **kwargs):
    """
    Calculates the unit skin friction in sand according to the method by Alm & Hamre. The unit skin friction includes the effect of friction fatigue based on back-analysis from a number of jacket piles from North Sea Oil & Gas platforms.
    The authors recommend applying 50% of the calculated unit skin friction on the outside of the pile and 50% on the inside.

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\\\sigma_vo^{\\\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param interface_friction_angle: Interface friction angle for sand (ICP recommendations can be used in absence of other data) (:math:`\\\\delta`) [:math:`deg`] - Suggested range: 10.0 <= interface_friction_angle <= 50.0
    :param depth: Depth at which unit skin friction is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param embedded_length: Depth of the pile tip below mudline (:math:`z_{tip}`) [:math:`m`] - Suggested range: embedded_length >= 0.0
    :param shape_factor_multiplier: Factor by which to divide for the shape factor k (:math:``) [:math:`-`] (optional, default= 80.0)
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] (optional, default= 101.325)
    :param fsi_sand_multiplier: Multiplier for initial unit skin friction in sand (:math:``) [:math:`-`] (optional, default= 0.0132)
    :param fsi_sand_exponent: Exponent for initial unit skin friction in sand (:math:``) [:math:`-`] (optional, default= 0.13)
    :param multiplier_fsres: Multiplier on initial skin friction to obtain residual skin friction (:math:``) [:math:`-`] (optional, default= 0.2)
    :param multiplier_outside: Multiplier on calculated unit skin friction to obtain outside unit skin friction (default is 50%) (:math:``) [:math:`-`] (optional, default= 0.5)
    :param multiplier_inside: Multiplier on calculated unit skin friction to obtain inside unit skin friction (default is 50%) (:math:``) [:math:`-`] (optional, default= 0.5)

    .. math::
        f_s = f_{s,res} + (f_{s,i} - f_{s,res}) \\cdot e^{k \\cdot (z-z_{tip})}

        k = \\frac{\\sqrt{q_t / \\sigma_{vo}^{\\prime}}}{80}

        f_{s,i,sand} = 0.0132 \\cdot q_t \\cdot \\left( \\frac{\\sigma_{vo}^{\\prime}}{P_a} \\right)^{0.13} \\cdot \\tan \\delta

        f_{s,res,sand} = 0.2 \\cdot f_{s,i,sand}

        f_{s,out} = 0.5 \\cdot f_s

        f_{s,in} = 0.5 \\cdot f_s

    :returns: Dictionary with the following keys:

        - 'f_s_comp_out [kPa]': Unit skin friction on the outside (with multiplier applied) (:math:`f_{s,out}`)  [:math:`kPa`]
        - 'f_s_comp_in [kPa]': Unit skin friction on the inside (with multiplier applied) (:math:`f_{s,in}`)  [:math:`kPa`]
        - 'f_s_tens_out [kPa]': Not applicable [:math:`kPa`]
        - 'f_s_tens_in [kPa]': Not applicable [:math:`kPa`]
        - 'f_s_initial [kPa]': Initial unit skin friction in sand (without multiplier for inside/outside) (:math:`f_{s,i,sand}`)  [:math:`kPa`]
        - 'f_s_res [kPa]': Residual unit skin friction in sand (without multiplier for inside/outside) (:math:`f_{s,res,sand}`)  [:math:`kPa`]

    Reference - Alm, T., Hamre, L., 2001. Soil model for pile driveability predictions based on CPT interpretations. Presented at the International Conference On Soil Mechanics and Foundation Engineering.
    Alm, T., Hamre, L., 1998. Soil model for driveability predictions. Presented at the OTC 8835, Annual Offshore Technology Conference, Houston, Texas, p. 13.

    """

    _f_s_initial = fsi_sand_multiplier * (1000 * qt) * \
                   ((sigma_vo_eff / atmospheric_pressure) ** fsi_sand_exponent) * \
                   np.tan(np.radians(interface_friction_angle))
    _f_s_res = multiplier_fsres * _f_s_initial
    _shape_factor = np.sqrt(1000 * qt / sigma_vo_eff) / shape_factor_multiplier
    _fs = _f_s_res + (_f_s_initial - _f_s_res) * \
          np.exp(_shape_factor * (depth - embedded_length))

    _f_s_comp_out = multiplier_outside * _fs
    _f_s_comp_in = multiplier_inside * _fs
    _f_s_tens_out = 0
    _f_s_tens_in = 0
    _f_s_initial = _f_s_initial
    _f_s_res = _f_s_res

    return {
        'f_s_comp_out [kPa]': _f_s_comp_out,
        'f_s_comp_in [kPa]': _f_s_comp_in,
        'f_s_tens_out [kPa]': _f_s_tens_out,
        'f_s_tens_in [kPa]': _f_s_tens_in,
        'f_s_initial [kPa]': _f_s_initial,
        'f_s_res [kPa]': _f_s_res,
    }


ALMHAMRE_UNITSKINFRICTION_CLAY = {
    'depth': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'embedded_length': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'fs': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'shape_factor_multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_fsres_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_fsres_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_fs_initial': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_outside': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_inside': {'type': 'float', 'min_value': None, 'max_value': None},
}

ALMHAMRE_UNITSKINFRICTION_CLAY_ERRORRETURN = {
    'f_s_comp_out [kPa]': np.nan,
    'f_s_comp_in [kPa]': np.nan,
    'f_s_tens_out [kPa]': np.nan,
    'f_s_tens_in [kPa]': np.nan,
    'f_s_initial [kPa]': np.nan,
    'f_s_res [kPa]': np.nan,
}


@Validator(ALMHAMRE_UNITSKINFRICTION_CLAY, ALMHAMRE_UNITSKINFRICTION_CLAY_ERRORRETURN)
def unitskinfriction_clay_almhamre(
        depth, embedded_length, qt, fs, sigma_vo_eff,
        shape_factor_multiplier=80.0, multiplier_fsres_1=0.004, multiplier_fsres_2=0.0025, multiplier_fs_initial=1.0,
        multiplier_outside=1.0, multiplier_inside=1.0, **kwargs):
    """
    Calculates the unit skin friction in clay according to the method by Alm & Hamre. The unit skin friction includes the effect of friction fatigue based on back-analysis from a number of jacket piles from North Sea Oil & Gas platforms.
    The authors recommend applying 100% of the calculated unit skin friction on the outside of the pile and 100% on the inside.

    :param depth: Depth at which the unit skin friction is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param embedded_length: Pile tip depth below mudline (:math:`z_{tip}`) [:math:`m`] - Suggested range: embedded_length >= 0.0
    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param fs: Sleeve friction from the CPT (:math:`f_{s,CPT}`) [:math:`MPa`] - Suggested range: sleeve_friction >= 0.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\\\sigma_{vo}^{\\\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param shape_factor_multiplier: Factor by which to divide for the shape factor k (:math:``) [:math:`-`] (optional, default= 80.0)
    :param multiplier_fsres_1: First multiplier on residual skin friction (:math:``) [:math:`-`] (optional, default= 0.004)
    :param multiplier_fsres_2: Second multiplier on residual skin friction (:math:``) [:math:`-`] (optional, default= 0.0025)
    :param multiplier_fs_initial: Multiplier on initial unit skin friction (:math:``) [:math:`-`] (optional, default= 1.0)
    :param multiplier_outside: Multiplier on calculated unit skin friction to obtain outside unit skin friction (default is 50%) (:math:``) [:math:`-`] (optional, default= 1.0)
    :param multiplier_inside: Multiplier on calculated unit skin friction to obtain inside unit skin friction (default is 50%) (:math:``) [:math:`-`] (optional, default= 1.0)

    .. math::
        f_s = f_{s,res} + (f_{s,i} - f_{s,res}) \\cdot e^{k \\cdot (z-z_{tip})}

        k = \\frac{\\sqrt{q_t / \\sigma_{vo}^{\\prime}}}{80}

        f_{s,i,clay} = f_{s,CPT}

        f_{s,res,clay} = 0.004 \\cdot q_t \\cdot \\left( 1 - 0.0025 \\cdot \\frac{q_t}{\\sigma_{vo}^{\\prime}} \\right)

        f_{s,out} = f_s

        f_{s,in} = f_s

    :returns: Dictionary with the following keys:

        - 'f_s_comp_out [kPa]': Unit skin friction on the outside (with multiplier applied) (:math:`f_{s,out}`)  [:math:`kPa`]
        - 'f_s_comp_in [kPa]': Unit skin friction on the inside (with multiplier applied) (:math:`f_{s,in}`)  [:math:`kPa`]
        - 'f_s_tens_out [kPa]': Not applicable [:math:`kPa`]
        - 'f_s_tens_in [kPa]': Not applicable [:math:`kPa`]
        - 'f_s_initial [kPa]': Initial unit skin friction in sand (without multiplier for inside/outside) (:math:`f_{s,i,sand}`)  [:math:`kPa`]
        - 'f_s_res [kPa]': Residual unit skin friction in sand (without multiplier for inside/outside) (:math:`f_{s,res,sand}`)  [:math:`kPa`]

    Reference - Alm, T., Hamre, L., 2001. Soil model for pile driveability predictions based on CPT interpretations. Presented at the International Conference On Soil Mechanics and Foundation Engineering.
    Alm, T., Hamre, L., 1998. Soil model for driveability predictions. Presented at the OTC 8835, Annual Offshore Technology Conference, Houston, Texas, p. 13.

    """
    _f_s_initial = multiplier_fs_initial * 1000 * fs
    _f_s_res = multiplier_fsres_1 * (1000 * qt) * (1 - multiplier_fsres_2 * (1000 * qt / sigma_vo_eff))
    _shape_factor = np.sqrt(1000 * qt / sigma_vo_eff) / shape_factor_multiplier
    _fs = _f_s_res + (_f_s_initial - _f_s_res) * \
          np.exp(_shape_factor * (depth - embedded_length))
    _f_s_comp_out = multiplier_outside * _fs
    _f_s_comp_in = multiplier_inside * _fs
    _f_s_tens_out = 0
    _f_s_tens_in = 0

    return {
        'f_s_comp_out [kPa]': _f_s_comp_out,
        'f_s_comp_in [kPa]': _f_s_comp_in,
        'f_s_tens_out [kPa]': _f_s_tens_out,
        'f_s_tens_in [kPa]': _f_s_tens_in,
        'f_s_initial [kPa]': _f_s_initial,
        'f_s_res [kPa]': _f_s_res,
    }


SKINFRICTION_METHODS = {
    'API RP2 GEO Sand': API_unit_shaft_friction_sand_rp2geo,
    'API RP2 GEO Clay': API_unit_shaft_friction_clay,
    'Alm and Hamre Sand': unitskinfriction_sand_almhamre,
    'Alm and Hamre Clay': unitskinfriction_clay_almhamre
}

SKINFRICTION_PARAMETERS = {
    'API RP2 GEO Sand': ['api_relativedensity', 'api_soildescription', 'sigma_vo_eff'],
    'API RP2 GEO Clay': ['undrained_shear_strength', 'sigma_vo_eff'],
    'Alm and Hamre Sand': ['qt', 'sigma_vo_eff', 'interface_friction_angle', 'depth', 'embedded_length'],
    'Alm and Hamre Clay': ['depth', 'embedded_length', 'qt', 'fs', 'sigma_vo_eff']
}