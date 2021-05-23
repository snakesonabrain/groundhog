#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


OVERBURDENCORRECTION_SPT_LIAOWHITMAN = {
    'N': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
}

OVERBURDENCORRECTION_SPT_LIAOWHITMAN_ERRORRETURN = {
    'CN [-]': np.nan,
    'N1 [-]': np.nan,
}

@Validator(OVERBURDENCORRECTION_SPT_LIAOWHITMAN, OVERBURDENCORRECTION_SPT_LIAOWHITMAN_ERRORRETURN)
def overburdencorrection_spt_liaowhitman(
        N,sigma_vo_eff,
        atmospheric_pressure=100.0, **kwargs):

    """
    Applies a correction to the SPT N value to account for the effect of effective overburden pressure in granular soils.
    The relation given by Liao and Whitman (1986) is one of the most commonly used.
    Increasing overburden pressure will lead to less penetration at deeper depths for the same soil type.
    By applying the correction, the field value of N is corrected to a standard effective overburden pressure of 100kPa.

    The standard penetration number corrected for field condition (:math:`N_{60}`) can also be used as an input in which case :math:`\\left( N_1 \\right)_{60}` is obtained.

    :param N: Field value of SPT N number (:math:`N`) or corrected value :math:`N_{60}` [:math:`-`] - Suggested range: N >= 0.0
    :param sigma_vo_eff: Effective overburden pressure (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)

    .. math::
        N_1 = C_N \\cdot N

        C_N = \\left[ \\frac{1}{ \\left( \\frac{\\sigma_{vo}^{\\prime}}{P_a} \\right) } \\right]^{0.5}

    :returns: Dictionary with the following keys:

        - 'CN [-]': Correction factor (:math:`C_N`)  [:math:`-`]
        - 'N1 [-]': Value of SPT N number corrected to an effective overburden pressure of 100kPa (:math:`N_1` or :math:`\\left( N_1 \\right)_{60}` in case :math:`N_{60}` is used as input)  [:math:`-`]

    Reference - Liao SSC, Whitman RV (1986) Overburden correction factors for SPT in sand. J Geotech Eng ASCE 112(3):373–377

    """

    _CN = np.sqrt(1 / (sigma_vo_eff / atmospheric_pressure))
    _N1 = _CN * N

    return {
        'CN [-]': _CN,
        'N1 [-]': _N1,
    }


SPT_N60_CORRECTION = {
    'N': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'borehole_diameter': {'type': 'float', 'min_value': 60.0, 'max_value': 200.0},
    'rod_length': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'country': {'type': 'string', 'options': ('Japan', 'United States', 'Argentina', 'China', 'Other'), 'regex': None},
    'hammertype': {'type': 'string', 'options': ('Donut', 'Safety'), 'regex': None},
    'hammerrelease': {'type': 'string', 'options': ('Free fall', 'Rope and pulley'), 'regex': None},
    'samplertype': {'type': 'string', 'options': ('Standard sampler', 'With liner for dense sand and clay', 'With liner for loose sand'), 'regex': None},
    'eta_H': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'eta_B': {'type': 'float', 'min_value': 1.0, 'max_value': 1.2},
    'eta_S': {'type': 'float', 'min_value': 0.8, 'max_value': 1.0},
    'eta_R': {'type': 'float', 'min_value': 0.75, 'max_value': 1.0},
}

SPT_N60_CORRECTION_ERRORRETURN = {
    'N60 [-]': np.nan,
    'eta_H [pct]': np.nan,
    'eta_B [-]': np.nan,
    'eta_S [-]': np.nan,
    'eta_R [-]': np.nan,
}

@Validator(SPT_N60_CORRECTION, SPT_N60_CORRECTION_ERRORRETURN)
def spt_N60_correction(
        N, borehole_diameter, rod_length, country, hammertype, hammerrelease,
        samplertype='Standard sampler', eta_H=np.nan, eta_B=np.nan, eta_S=np.nan, eta_R=np.nan, **kwargs):

    """
    The performance of the SPT in a given soil type depends on the efficiency of energy transmission to the soil. It is common practice to correct the field value of SPT N number to an equivalent number of blows at an energy ratio of 60% (ratio of energy delivered to the sampler divided by the input energy). The hammer efficiency (:math:`\eta_H`), borehole diameter (:math:`\eta_B`), sampler type (:math:`\eta_S`) and rod length (:math:`\eta_R`) are corrected for.

    The recommendations by Seed et al (1985) and Skempton (1986) as presented by Ameratunga et al (2016) are used by default. The user can specify overrides for each correction factor. If overrides are specified, they take precedence.

    +---------------+-------------+-----------------+----------------------+
    |    Country    | Hammer type |  Hammer release | :math:`\eta_H` [pct] |
    +===============+=============+=================+======================+
    |     Japan     |    Donut    |    Free fall    |          78          |
    |               +-------------+-----------------+----------------------+
    |               |    Donut    | Rope and pulley |          67          |
    +---------------+-------------+-----------------+----------------------+
    | United States |    Safety   | Rope and pulley |          60          |
    |               +-------------+-----------------+----------------------+
    |               |    Donut    | Rope and pulley |          45          |
    +---------------+-------------+-----------------+----------------------+
    |   Argentina   |    Donut    | Rope and pulley |          45          |
    +---------------+-------------+-----------------+----------------------+
    |     China     |    Donut    |    Free fall    |          60          |
    |               +-------------+-----------------+----------------------+
    |               |    Donut    | Rope and pulley |          50          |
    +---------------+-------------+-----------------+----------------------+


    +---------------+---------------------+
    | Diameter [mm] | :math:`\eta_B` [-]  |
    +===============+=====================+
    |     60-120    |          1          |
    +---------------+---------------------+
    |      150      |         1.05        |
    +---------------+---------------------+
    |      200      |         1.15        |
    +---------------+---------------------+


    +------------------------------------+---------------------+
    |            Sampler type            | :math:`\eta_S` [-]  |
    +====================================+=====================+
    |          Standard sampler          |         1.0         |
    +------------------------------------+---------------------+
    | With liner for dense sand and clay |         0.8         |
    +------------------------------------+---------------------+
    |      With liner for loose sand     |         0.9         |
    +------------------------------------+---------------------+


    +----------------+---------------------+
    | Rod length [m] | :math:`\eta_R` [-]  |
    +================+=====================+
    |       >10      |         1.0         |
    +----------------+---------------------+
    |      6-10      |         0.95        |
    +----------------+---------------------+
    |       4-6      |         0.85        |
    +----------------+---------------------+
    |       0-4      |         0.75        |
    +----------------+---------------------+


    :param N: Field value of SPT N number (:math:`N`) [:math:`-`] - Suggested range: N >= 0.0
    :param borehole_diameter: Diameter of the borehole (:math:`D`) [:math:`mm`] - Suggested range: 60.0 <= borehole_diameter <= 200.0
    :param rod_length: Length of rods connecting hammer with sampler (:math:`L`) [:math:`m`] - Suggested range: rod_length >= 0.0
    :param country: Country where SPT test is executed - Options: ('Japan', 'United States', 'Argentina', 'China', 'Other'). If 'Other' is chosen, an override for :math:`\\eta_H` should be specified
    :param hammertype: Type of hammer used - Options: ('Donut', 'Safety')
    :param hammerrelease: Release mechanism for the hammer - Options: ('Free fall', 'Rope and pulley')
    :param samplertype: Type of sampler used (optional, default= 'Standard sampler') - Options: ('Standard sampler', 'With liner for dense sand and clay', 'With liner for loose sand')
    :param eta_H: Correction factor for hammer efficiency (:math:`\\eta_H`) [:math:`pct`] - Suggested range: 0.0 <= eta_H <= 100.0 (optional, default= -10.0)
    :param eta_B: Correction factor for borehole diameter (:math:`\\eta_B`) [:math:`-`] - Suggested range: 1.0 <= eta_B <= 1.2 (optional, default= -10.0)
    :param eta_S: Correction factor for sampler type (:math:`\\eta_S`) [:math:`-`] - Suggested range: 0.8 <= eta_S <= 1.0 (optional, default= -10.0)
    :param eta_R: Correction factor for rod length (:math:`\\eta_R`) [:math:`-`] - Suggested range: 0.75 <= eta_R <= 1.0 (optional, default= -10.0)

    .. math::
        N_{60} = \\frac{N \\cdot \\eta_H \\cdot \\eta_B \\cdot \\eta_S \\cdot \\eta_R}{60}

    :returns: Dictionary with the following keys:

        - 'N60 [-]': SPT N number corrected to 60pct efficiency (:math:`N_{60}`)  [:math:`-`]
        - 'eta_H [pct]': Correction factor for hammer efficiency (:math:`\\eta_H`)  [:math:`pct`]
        - 'eta_B [-]': Correction factor for borehole diameter (:math:`\\eta_B`)  [:math:`-`]
        - 'eta_S [-]': Correction factor for sampler type (:math:`\\eta_S`)  [:math:`-`]
        - 'eta_R [-]': Correction factor for rod length (:math:`\\eta_R`)  [:math:`-`]

    Reference - J. Ameratunga et al., Correlations of Soil and Rock Properties in Geotechnical Engineering, Developments in Geotechnical Engineering, DOI 10.1007/978-81-322-2629-1_4

    """
    # Hammer efficiency correction
    if np.math.isnan(eta_H):
        if country == 'Japan':
            if hammertype == 'Donut':
                if hammerrelease == 'Free fall':
                    _eta_H = 78
                elif hammerrelease == 'Rope and pulley':
                    _eta_H = 67
                else:
                    raise ValueError("Hammer release type not recognised for Japan.")
            else:
                raise ValueError("Only hammertype 'Donut' available for Japan. Use overrides for other combinations.")
        elif country == 'United States':
            if hammertype == 'Safety':
                if hammerrelease == 'Rope and pulley':
                    _eta_H = 60
                else:
                    raise ValueError("Hammer release type not recognised for USA.")
            elif hammertype == 'Donut':
                if hammerrelease == 'Rope and pulley':
                    _eta_H = 45
                else:
                    raise ValueError("Hammer release type not recognised for USA.")
            else:
                raise ValueError("Hammer type not recognised for USA.")
        elif country == 'Argentina':
            if hammertype == 'Donut':
                if hammerrelease == 'Rope and pulley':
                    _eta_H = 45
                else:
                    raise ValueError("Hammer release type not recognised for Argentina.")
            else:
                raise ValueError("Hammer type not recognised for Argentina.")
        elif country == 'China':
            if hammertype == 'Donut':
                if hammerrelease == 'Free fall':
                    _eta_H = 60
                elif hammerrelease == 'Rope and pulley':
                    _eta_H = 50
                else:
                    raise ValueError("Hammer release type not recognised for China.")
            else:
                raise ValueError("Hammer type not recognised for China.")
        elif country == 'Other':
            if np.math.isnan(eta_H):
                raise ValueError("For country='Other', an override for eta_H should be specified.")
            else:
                pass
        else:
            raise ValueError("Country not recognised. Select one from the list and use the overrides for other countries.")
    else:
        _eta_H = eta_H

    # Borehole diameter correction
    if np.math.isnan(eta_B):
        if 60 < borehole_diameter < 120:
            _eta_B = 1
        elif borehole_diameter == 150:
            _eta_B = 1.05
        elif borehole_diameter == 200:
            _eta_B = 1.15
        else:
            raise ValueError("Borehole diameter not in list of allowed diameters.")
    else:
        _eta_B = eta_B

    # Sample type correction
    if np.math.isnan(eta_S):
        if samplertype == 'Standard sampler':
            _eta_S = 1.0
        elif samplertype == 'With liner for dense sand and clay':
            _eta_S = 0.8
        elif samplertype == 'With liner for loose sand':
            _eta_S = 0.9
        else:
            raise ValueError("Sampler type not recognised.")
    else:
        _eta_S = eta_S

    # Rod length correction
    if np.math.isnan(eta_R):
        if 0 <= rod_length < 4:
            _eta_R = 0.75
        elif 4 <= rod_length < 6:
            _eta_R = 0.85
        elif 6 <= rod_length < 10:
            _eta_R = 0.95
        elif rod_length >= 10:
            _eta_R = 1
        else:
            raise ValueError("Rod length not allowed.")
    else:
        _eta_R = eta_R

    _N60 = N * _eta_H * _eta_B * _eta_S * _eta_R / 60

    return {
        'N60 [-]': _N60,
        'eta_H [pct]': _eta_H,
        'eta_B [-]': _eta_B,
        'eta_S [-]': _eta_S,
        'eta_R [-]': _eta_R,
    }