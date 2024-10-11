#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.siteinvestigation.classification.phaserelations import voidratio_bulkunitweight, porosity_voidratio
from groundhog.general.validation import Validator


API_UNIT_END_BEARING_CLAY = {
    'undrained_shear_strength':{'type': 'float','min_value':0.0,'max_value':400.0},
    'N_c':{'type': 'float','min_value':7.0,'max_value':12.0},
}

API_UNIT_END_BEARING_CLAY_ERRORRETURN = {
    'q_b_coring [kPa]': np.nan,
    'q_b_plugged [kPa]': np.nan,
    'plugged': None,
    'internal_friction': False
}

@Validator(API_UNIT_END_BEARING_CLAY, API_UNIT_END_BEARING_CLAY_ERRORRETURN)
def API_unit_end_bearing_clay(undrained_shear_strength,N_c=9.0,**kwargs):

    """
    Calculates unit end bearing in clay according to API RP 2 GEO. For piles considered to be plugged, the bearing pressure may be assumed to act over the entire cross-section of the pile. For unplugged piles, the bearing pressure acts on the pile wall annulus only. That a pile is considered plugged or unplugged shall be based on static calculations. A pile can be driven in an unplugged condition but behave as plugged under static loads.

    Use the string ``'API RP2 GEO Clay'`` to define this method in a ``SoilProfile``.

    :param undrained_shear_strength: Undrained shear strenght at the pile tip (:math:`S_u`) [:math:`kPa`]  - Suggested range: 0.0<=undrained_shear_strength<=400.0
    :param N_c: Bearing capacity factor (:math:`N_c`) [:math:`-`] (optional, default=9.0) - Suggested range: 7.0<=N_c<=12.0

    .. math::
        q = N_c \\cdot S_u

    :returns:   Unit end bearing (:math:`q`) [:math:`kPa`]

    :rtype: Python dictionary with keys ['q_b [kPa]']


    Reference - API RP 2GEO, API RP 2GEO Geotechnical and Foundation Design Considerations, 2011

    """
    q_b = N_c * undrained_shear_strength

    return {
        'q_b_coring [kPa]': q_b,
        'q_b_plugged [kPa]': q_b,
        'plugged': None,
        'internal_friction': False
    }

API_UNIT_END_BEARING_SAND_RP2GEO = {
    'api_relativedensity':{'type': 'string','options':("Very loose","Loose","Medium dense","Dense","Very dense"),'regex':None},
    'api_soildescription':{'type': 'string','options':("Sand","Sand-silt"),'regex':None},
    'sigma_vo_eff':{'type': 'float','min_value':0.0,'max_value':None},
}

API_UNIT_END_BEARING_SAND_RP2GEO_ERRORRETURN = {
    'q_b_coring [kPa]': np.nan,
    'q_b_plugged [kPa]': np.nan,
    'plugged': None,
    'internal_friction': False,
    'q_b_lim [kPa]': np.nan,
    'Nq [-]': np.nan
}

@Validator(API_UNIT_END_BEARING_SAND_RP2GEO, API_UNIT_END_BEARING_SAND_RP2GEO_ERRORRETURN)
def API_unit_end_bearing_sand_rp2geo(api_relativedensity, api_soildescription, sigma_vo_eff, qb_limit=False, **kwargs):

    """
    Calculates unit end bearing in sand according to API RP2 GEO.

    Use the string ``'API RP2 GEO Sand'`` to define this method in a ``SoilProfile``.

    :param api_relativedensity: Relative density of the sand (:math:`D_r`) [:math:`-`] Options: ("Very loose","Loose","Medium dense","Dense","Very dense"), regex: None
    :param api_soildescription : Description of the soil type (:math:`-`) [:math:`-`] Options: ("Sand","Sand-silt"), regex: None
    :param sigma_vo_eff: In-situ vertical effective stress at the pile tip (:math:`p'_{o,tip}`) [:math:`kPa`]  - Suggested range: 0.0<=vertical_effective_stress

    .. math::
        q = N_q \\cdot p'_{o,tip}

    :returns:   Unit end bearing (:math:`q_b`) [:math:`kPa`], Unit end bearing (limited) (:math:`q_{b,limited}`) [:math:`kPa`], Unit end bearing limit (:math:`q_{b,lim}`) [:math:`kPa`]

    :rtype: Python dictionary with keys ['q_b [kPa]','q_b_with_lim [kPa]','q_b_lim [kPa]']

    .. figure:: images/API_unit_end_bearing_sand_1.PNG
        :figwidth: 500
        :width: 400
        :align: center

        API RP 2 GEO values (TODO: Add figure file in docs)

    Reference - API RP 2GEO, API RP 2GEO Geotechnical and Foundation Design Considerations, 2011

    """
    if api_soildescription == "Sand":
        if api_relativedensity == "Medium dense":
            Nq = 20.0
            qb_lim = 5000.0
        elif api_relativedensity == "Dense":
            Nq = 40.0
            qb_lim = 10000.0
        elif api_relativedensity == "Very dense":
            Nq = 50.0
            qb_lim = 12000.0
        else:
            raise ValueError("Relative density category not found")
    elif api_soildescription == "Sand-silt":
        if api_relativedensity == "Medium dense":
            Nq = 12.0
            qb_lim = 3000.0
        elif api_relativedensity == "Dense":
            Nq = 20.0
            qb_lim = 5000.0
        elif api_relativedensity == "Very dense":
            Nq = 40.0
            qb_lim = 10000.0
        else:
            raise ValueError("Relative density category not found")
    else:
        raise ValueError("Soil description not found")

    if qb_limit:
        q_b = min(Nq * sigma_vo_eff, qb_lim)
    else:
        q_b = Nq * sigma_vo_eff

    return {
        'q_b_coring [kPa]': q_b,
        'q_b_plugged [kPa]': q_b,
        'plugged': None,
        'internal_friction': False,
        'q_b_lim [kPa]': qb_lim,
        'Nq [-]': Nq
    }

UNITENDBEARING_SAND_ALMHAMRE = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent': {'type': 'float', 'min_value': None, 'max_value': None},
}

UNITENDBEARING_SAND_ALMHAMRE_ERRORRETURN = {
    'q_b_coring [kPa]': np.nan,
    'q_b_plugged [kPa]': np.nan,
    'plugged []': False,
    'internal_friction []': True,
}

@Validator(UNITENDBEARING_SAND_ALMHAMRE, UNITENDBEARING_SAND_ALMHAMRE_ERRORRETURN)
def unitendbearing_sand_almhamre(
        qt, sigma_vo_eff,
        multiplier=0.15, exponent=0.2, **kwargs):
    """
    Calculates unit end bearing in sand according to Alm & Hamre.

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param multiplier: Multipier for unit end bearing (:math:``) [:math:`-`] (optional, default= 0.15)
    :param exponent: Exponent for unit end bearing (:math:``) [:math:`-`] (optional, default= 0.2)

    .. math::
        q_{b,sand} = 0.15 \\cdot q_t \\cdot \\left( \\frac{q_t}{\\sigma_{vo}^{\\prime}} \\right)^{0.2}

    :returns: Dictionary with the following keys:

        - 'q_b_coring [kPa]': Unit end bearing for pile driving (:math:`q_{b,sand}`)  [:math:`kPa`]
        - 'q_b_plugged [kPa]': Taken equal to the coring unit end bearing but tubular piles are not assumed to be driven in a plugged manner [:math:`kPa`]
        - 'plugged []': Determines whether the pile is plugged (False for driving piles)
        - 'internal_friction []': Determines whether internal friction is taken into account (True by default for Alm & Hamre formula)

    Reference - Alm, T., Hamre, L., 2001. Soil model for pile driveability predictions based on CPT interpretations. Presented at the International Conference On Soil Mechanics and Foundation Engineering.
    Alm, T., Hamre, L., 1998. Soil model for driveability predictions. Presented at the OTC 8835, Annual Offshore Technology Conference, Houston, Texas, p. 13.

    """

    _q_b_coring = multiplier * (1000 * qt) * ((1000 * qt / sigma_vo_eff) ** exponent)
    _q_b_plugged = _q_b_coring
    _plugged = False
    _internal_friction = True

    return {
        'q_b_coring [kPa]': _q_b_coring,
        'q_b_plugged [kPa]': _q_b_plugged,
        'plugged []': _plugged,
        'internal_friction []': _internal_friction,
    }


UNITENDBEARING_CLAY_ALMHAMRE = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
}

UNITENDBEARING_CLAY_ALMHAMRE_ERRORRETURN = {
    'q_b_coring [kPa]': np.nan,
    'q_b_plugged [kPa]': np.nan,
    'plugged []': False,
    'internal_friction []': True,
}


@Validator(UNITENDBEARING_CLAY_ALMHAMRE, UNITENDBEARING_CLAY_ALMHAMRE_ERRORRETURN)
def unitendbearing_clay_almhamre(
        qt,
        multiplier=0.6, **kwargs):
    """
    Calculates unit end bearing in clay according to Alm & Hamre.

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param multiplier: Multipier for unit end bearing (:math:``) [:math:`-`] (optional, default= 0.6)

    .. math::
        q_{b,clay} = 0.6 \\cdot q_t

    :returns: Dictionary with the following keys:

        - 'q_b_coring [kPa]': Unit end bearing for pile driving (:math:`q_{b,sand}`)  [:math:`kPa`]
        - 'q_b_plugged [kPa]': Not applicable [:math:`kPa`]
        - 'plugged []': Determines whether the pile is plugged (False for driving piles)
        - 'internal_friction []': Determines whether internal friction is taken into account (True by default for Alm & Hamre formula)

    Reference - Alm, T., Hamre, L., 2001. Soil model for pile driveability predictions based on CPT interpretations. Presented at the International Conference On Soil Mechanics and Foundation Engineering.
    Alm, T., Hamre, L., 1998. Soil model for driveability predictions. Presented at the OTC 8835, Annual Offshore Technology Conference, Houston, Texas, p. 13.

    """

    _q_b_coring = multiplier * 1000 * qt
    _q_b_plugged = 0
    _plugged = False
    _internal_friction = True

    return {
        'q_b_coring [kPa]': _q_b_coring,
        'q_b_plugged [kPa]': _q_b_plugged,
        'plugged []': _plugged,
        'internal_friction []': _internal_friction,
    }


ENDBEARING_METHODS = {
    'API RP2 GEO Sand': API_unit_end_bearing_sand_rp2geo,
    'API RP2 GEO Clay': API_unit_end_bearing_clay,
    'Alm and Hamre Sand': unitendbearing_sand_almhamre,
    'Alm and Hamre Clay': unitendbearing_clay_almhamre
}

ENDBEARING_PARAMETERS = {
    'API RP2 GEO Sand': ['api_relativedensity', 'api_soildescription', 'sigma_vo_eff'],
    'API RP2 GEO Clay': ['undrained_shear_strength',],
    'Alm and Hamre Sand': ['qt', 'sigma_vo_eff'],
    'Alm and Hamre Clay': ['qt',]
}