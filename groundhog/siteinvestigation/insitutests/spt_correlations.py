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
        N, sigma_vo_eff, granular=True,
        atmospheric_pressure=100.0, **kwargs):

    """
    Applies a correction to the SPT N value to account for the effect of effective overburden pressure in granular soils.
    The relation given by Liao and Whitman (1986) is one of the most commonly used.
    Increasing overburden pressure will lead to less penetration at deeper depths for the same soil type.
    By applying the correction, the field value of N is corrected to a standard effective overburden pressure of 100kPa.

    The standard penetration number corrected for field condition (:math:`N_{60}`) can also be used as an input in which case :math:`\\left( N_1 \\right)_{60}` is obtained.

    :param N: Field value of SPT N number (:math:`N`) or corrected value :math:`N_{60}` [:math:`-`] - Suggested range: N >= 0.0
    :param sigma_vo_eff: Effective overburden pressure (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param granular: Boolean defining whether the soil behaves in a granular or not. If the behaviour is not granular, the correction factor is taken equal to 1.
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)

    .. math::
        N_1 = C_N \\cdot N

        C_N = \\left[ \\frac{1}{ \\left( \\frac{\\sigma_{vo}^{\\prime}}{P_a} \\right) } \\right]^{0.5}

    :returns: Dictionary with the following keys:

        - 'CN [-]': Correction factor (:math:`C_N`)  [:math:`-`]
        - 'N1 [-]': Value of SPT N number corrected to an effective overburden pressure of 100kPa (:math:`N_1` or :math:`\\left( N_1 \\right)_{60}` in case :math:`N_{60}` is used as input)  [:math:`-`]

    Reference - Liao SSC, Whitman RV (1986) Overburden correction factors for SPT in sand. J Geotech Eng ASCE 112(3):373–377

    """

    if granular:
        _CN = np.sqrt(1 / (sigma_vo_eff / atmospheric_pressure))
    else:
        _CN = 1
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
    'eta_H [-]': np.nan,
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
    :param eta_H: Correction factor for hammer efficiency (:math:`\\eta_H`) [:math:`pct`] - Suggested range: 0.0 <= eta_H <= 100.0 (optional, default= np.nan)
    :param eta_B: Correction factor for borehole diameter (:math:`\\eta_B`) [:math:`-`] - Suggested range: 1.0 <= eta_B <= 1.2 (optional, default= np.nan)
    :param eta_S: Correction factor for sampler type (:math:`\\eta_S`) [:math:`-`] - Suggested range: 0.8 <= eta_S <= 1.0 (optional, default= np.nan)
    :param eta_R: Correction factor for rod length (:math:`\\eta_R`) [:math:`-`] - Suggested range: 0.75 <= eta_R <= 1.0 (optional, default= np.nan)

    .. math::
        N_{60} = \\frac{N \\cdot \\eta_H \\cdot \\eta_B \\cdot \\eta_S \\cdot \\eta_R}{60}

    :returns: Dictionary with the following keys:

        - 'N60 [-]': SPT N number corrected to 60pct efficiency (:math:`N_{60}`)  [:math:`-`]
        - 'eta_H [%]': Correction factor for hammer efficiency (:math:`\\eta_H`)  [:math:`pct`]
        - 'eta_H [-]': : Correction factor for hammer efficiency (:math:`\\eta_H`)  [:math:`-`]
        - 'eta_B [-]': Correction factor for borehole diameter (:math:`\\eta_B`)  [:math:`-`]
        - 'eta_S [-]': Correction factor for sampler type (:math:`\\eta_S`)  [:math:`-`]
        - 'eta_R [-]': Correction factor for rod length (:math:`\\eta_R`)  [:math:`-`]

    Reference - J. Ameratunga et al., Correlations of Soil and Rock Properties in Geotechnical Engineering, Developments in Geotechnical Engineering, DOI 10.1007/978-81-322-2629-1_4

    """
    
    # Hammer efficiency correction
    if np.isnan(eta_H):
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
            if np.isnan(eta_H):
                raise ValueError("For country='Other', an override for eta_H should be specified.")
            else:
                pass
        else:
            raise ValueError("Country not recognised. Select one from the list and use the overrides for other countries.")
    else:
        _eta_H = eta_H

    # Borehole diameter correction
    if np.isnan(eta_B):
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
    if np.isnan(eta_S):
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
    if np.isnan(eta_R):
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
        'eta_H [%]': _eta_H,
        'eta_H [-]': 0.01 * _eta_H,
        'eta_B [-]': _eta_B,
        'eta_S [-]': _eta_S,
        'eta_R [-]': _eta_R,
    }


RELATIVEDENSITY_SPT_KULHAWYMAYNE = {
    'N1_60': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'd_50': {'type': 'float', 'min_value': 0.002, 'max_value': 20.0},
    'calibration_factor_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'time_since_deposition': {'type': 'float', 'min_value': 1.0, 'max_value': None},
    'ocr': {'type': 'float', 'min_value': 1.0, 'max_value': 50.0},
    'ca_override': {'type': 'float', 'min_value': 1.0, 'max_value': None},
    'cocr_override': {'type': 'float', 'min_value': 1.0, 'max_value': None},
}

RELATIVEDENSITY_SPT_KULHAWYMAYNE_ERRORRETURN = {
    'Dr [-]': np.nan,
    'Dr [pct]': np.nan,
    'C_A [-]': np.nan,
    'C_OCR [-]': np.nan,
}

@Validator(RELATIVEDENSITY_SPT_KULHAWYMAYNE, RELATIVEDENSITY_SPT_KULHAWYMAYNE_ERRORRETURN)
def relativedensity_spt_kulhawymayne(
        N1_60, d_50,
        calibration_factor_1=60.0,
        calibration_factor_2=25.0,
        time_since_deposition=1.0,
        ocr=1.0, ca_override=np.nan, cocr_override=np.nan, **kwargs):

    """
    Estimates relative density from SPT test. Although initially proposed based on the results of tests on non-aged, normally consolidated sands, the correlation can account for the effect of ageing and overconsolidation through correction factors. The parameters for these correction factors are not always easy to estimate.

    Note that stress and energy corrections need to be applied to the raw SPT data before applying the correlation.

    :param N1_60: SPT number corrected for overburden stress and energy (:math:`(N_1)_{60}`) [:math:`-`] - Suggested range: 0.0 <= N_1_60 <= 100.0
    :param d_50: Median grain size (:math:`d_{50}`) [:math:`mm`] - Suggested range: 0.002 <= d_50 <= 20.0
    :param calibration_factor_1: First calibration factor (:math:``) [:math:`-`] (optional, default= 60.0)
    :param calibration_factor_2: Second calibration factor (:math:``) [:math:`-`] (optional, default= 25.0)
    :param time_since_deposition: Time since deposition (:math:`t`) [:math:`years`] - Suggested range: time_since_deposition >= 1.0 (optional, default= 1.0)
    :param ocr: Overconsolidation ratio (:math:`OCR`) [:math:`-`] - Suggested range: 1.0 <= ocr <= 50.0 (optional, default= 1.0)
    :param ca_override: Direct specification of factor CA (:math:`C_A`) [:math:`-`] - Suggested range: ca_override >= 1.0 (optional, default= np.nan)
    :param cocr_override: Direct specification of factor COCR (:math:`C_{OCR}`) [:math:`-`] - Suggested range: cocr_override >= 1.0 (optional, default= np.nan)

    .. math::
        \\text{Unaged, normally consolidated sand}

        D_r = \\sqrt{\\frac{(N_1)_{60}}{60 + 25 \\cdot \\log_{10} ( d_{50} )}}

        \\text{With corrections for overconsolidation and ageing}

        D_r = \\sqrt{\\frac{(N_1)_{60}}{\\left( 60 + 25 \\cdot \\log d_{50} \\right) \\cdot C_A \\cdot C_{OCR}}}

        C_A = 1.2 + 0.05 \\cdot \\log_{10} \\left( \\frac{t}{100} \\right)

        C_{OCR} = (OCR)^{0.18}

    :returns: Dictionary with the following keys:

        - 'Dr [-]': Relative density (unitless) (:math:`D_r`)  [:math:`-`]
        - 'Dr [pct]': Relative density (percent) (:math:`D_r`)  [:math:`pct`]
        - 'C_A [-]': Correction factor for ageing (:math:`C_A`)  [:math:`-`]
        - 'C_OCR [-]': Correction factor for overconsolidation (:math:`C_{OCR}`)  [:math:`-`]

    Reference - Kulhawy FH, Mayne PW (1990) Manual on estimating soil properties for foundation design. Electric Power Research Institute, Palo Alto

    """
    if np.isnan(ca_override):
        if time_since_deposition == 1:
            _C_A = 1
        else:
            _C_A = 1.2 + 0.05 * np.log10(time_since_deposition / 100)
    else:
        _C_A = ca_override
    if np.isnan(cocr_override):
        _C_OCR = ocr ** 0.18
    else:
        _C_OCR = cocr_override
    _Dr = np.sqrt(N1_60 / ((calibration_factor_1 + calibration_factor_2 * np.log10(d_50)) * _C_A * _C_OCR))
    _Dr_pct = 100 * _Dr


    return {
        'Dr [-]': _Dr,
        'Dr [pct]': _Dr_pct,
        'C_A [-]': _C_A,
        'C_OCR [-]': _C_OCR,
    }


UNDRAINEDSHEARSTRENGTH_SPT_SALGADO = {
    'pi': {'type': 'float', 'min_value': 15.0, 'max_value': 60.0},
    'N_60': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
    'alpha_prime_override': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

UNDRAINEDSHEARSTRENGTH_SPT_SALGADO_ERRORRETURN = {
    'alpha_prime [-]': np.nan,
    'Su [kPa]': np.nan,
}

@Validator(UNDRAINEDSHEARSTRENGTH_SPT_SALGADO, UNDRAINEDSHEARSTRENGTH_SPT_SALGADO_ERRORRETURN)
def undrainedshearstrength_spt_salgado(
        pi, N_60,
        atmospheric_pressure=100.0,alpha_prime_override=np.nan, **kwargs):

    """
    Calculates undrained shear strength based on plasticity index and SPT number (corrected to 60% energy ratio).


    +--------+-------------------------------+
    | PI [%] | :math:`\\alpha^{\prime}` [-]   |
    +========+===============================+
    |   15   |             0.068             |
    +--------+-------------------------------+
    |   20   |             0.055             |
    +--------+-------------------------------+
    |   25   |             0.048             |
    +--------+-------------------------------+
    |   30   |             0.045             |
    +--------+-------------------------------+
    |   40   |             0.044             |
    +--------+-------------------------------+
    |   60   |             0.043             |
    +--------+-------------------------------+


    :param pi: Plasticity index (difference between liquid and plastic limit) (:math:`PI`) [:math:`pct`] - Suggested range: 15.0 <= plasticity_index <= 60.0
    :param N_60: SPT number corrected to 60% energy ratio (:math:`N_{60}`) [:math:`-`] - Suggested range: 0.0 <= N_60 <= 100.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)
    :param alpha_prime_override: Override for direct specification of the alpha prime factor (:math:`\\alpha^{\\prime}`) [:math:`-`] - Suggested range: alpha_prime_override >= 0.0 (optional, default= np.nan)

    .. math::
        \\frac{S_u}{P_a} = \\alpha^{\\prime} \\cdot N_{60}

    :returns: Dictionary with the following keys:

        - 'alpha_prime [-]': Factor based on plasticity index (:math:`\\alpha^{\\prime}`)  [:math:`-`]
        - 'Su [kPa]': Undrained shear strength (:math:`S_u`)  [:math:`kPa`]

    Reference - Salgado R (2008) The engineering of foundations. McGraw-Hill, New York

    """
    if np.isnan(alpha_prime_override):
        _pi = [15, 20, 25, 30, 40, 60]
        _alpha = [0.068,  0.055, 0.048, 0.045, 0.044, 0.043]
        _alpha_prime = np.interp(pi, _pi, _alpha)
    else:
        _alpha_prime = alpha_prime_override

    _Su = atmospheric_pressure * _alpha_prime * N_60

    return {
        'alpha_prime [-]': _alpha_prime,
        'Su [kPa]': _Su,
    }


FRICTIONANGLE_SPT_KULHAWYMAYNE = {
    'N': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': 1000.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_3': {'type': 'float', 'min_value': None, 'max_value': None},
}

FRICTIONANGLE_SPT_KULHAWYMAYNE_ERRORRETURN = {
    'Phi [deg]': np.nan,
}

@Validator(FRICTIONANGLE_SPT_KULHAWYMAYNE, FRICTIONANGLE_SPT_KULHAWYMAYNE_ERRORRETURN)
def frictionangle_spt_kulhawymayne(
        N,sigma_vo_eff,
        atmospheric_pressure=100.0,
        coefficient_1=12.2,coefficient_2=20.3,coefficient_3=0.34, **kwargs):

    """
    Kulhawy and Mayne approximated the chart for friction angle selection from SPT using the formula given below. The friction angle depends on the effective overburden stress and SPT N number.

    :param N: SPT N number (:math:`N`) [:math:`-`] - Suggested range: 0.0 <= N <= 60.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_vo_eff <= 1000.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)
    :param coefficient_1: First calibration coefficient (:math:``) [:math:`-`] (optional, default= 12.2)
    :param coefficient_2: Second  calibration coefficient (:math:``) [:math:`-`] (optional, default= 20.3)
    :param coefficient_3: Third calibration coefficient (:math:``) [:math:`-`] (optional, default= 0.34)

    .. math::
        \\phi = \\tan^{-1} \\left[ \\frac{N}{12.2 +20.3 \\cdot \\left( \\frac{\\sigma_{v0}^{\\prime}}{P_a} \\right)} \\right]^{0.34}

    :returns: Dictionary with the following keys:

        - 'Phi [deg]': Effective internal friction angle of the soil (:math:`\\phi`)  [:math:`deg`]

    Reference - Kulhawy FH, Mayne PW (1990) Manual on estimating soil properties for foundation design. Electric Power Research Institute, Palo Alto

    """

    _phi = np.rad2deg(np.arctan(
        (N / (coefficient_1 + coefficient_2 * (sigma_vo_eff / atmospheric_pressure))) ** coefficient_3
    ))

    return {
        'Phi [deg]': _phi,
    }


RELATIVEDENSITYCLASS_N = {
    'N': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
}

RELATIVEDENSITYCLASS_N_ERRORRETURN = {
    'Dr class': None,
}

@Validator(RELATIVEDENSITYCLASS_N, RELATIVEDENSITYCLASS_N_ERRORRETURN)
def relativedensityclass_spt_terzaghipeck(
    N,
     **kwargs):

    """
    Defines the relative density class for SPT measurements in cohesionless soils based on the uncorrected N-number

    +-----------------+-----------------------------+
    | N (uncorrected) | Relative density category   |
    +=================+=============================+
    |   <= 4          |         Very loose          |
    +-----------------+-----------------------------+
    |   4 < N <= 10   |         Loose               |
    +-----------------+-----------------------------+
    |   10 < N <= 30  |         Medium dense        |
    +-----------------+-----------------------------+
    |   30 < N <= 50  |         Dense               |
    +-----------------+-----------------------------+
    |   N <= 50       |         Very dense          |
    +-----------------+-----------------------------+
    
    :param N: Uncorrected SPT N number (:math:`N`) [-] - Suggested range: 0.0 <= N <= 60.0
    
    :returns: Dictionary with the following keys:
        
        - 'Dr class': Relative density class (:math:`D_r`) 
    
    Reference - Terzaghi K, Peck RB (1967) Soil mechanics in engineering practice, 2nd edn. Wiley, New York

    """
    if N <= 4:
        _Dr_class = 'Very loose'
    elif 4 < N <= 10:
        _Dr_class = 'Loose'
    elif 10 < N <= 30:
        _Dr_class = 'Medium dense'
    elif 30 < N <= 50:
        _Dr_class = 'Dense'
    elif N > 50:
        _Dr_class = 'Very dense'
    else:
        _Dr_class = None
    return {
        'Dr class': _Dr_class,
    }


OVERBURDENCORRECTION_SPT_ISO = {
    'N': {'type': 'int', 'min_value': 0.0, 'max_value': 60.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 25.0, 'max_value': 400.0},
}

OVERBURDENCORRECTION_SPT_ISO_ERRORRETURN = {
    'CN [-]': np.nan,
    'N1 [-]': np.nan,
}

@Validator(OVERBURDENCORRECTION_SPT_ISO, OVERBURDENCORRECTION_SPT_ISO_ERRORRETURN)
def overburdencorrection_spt_ISO(N, sigma_vo_eff, granular=True, **kwargs):

    """
    Corrects the SPT N number or corrected N number (:math:`N_{60}`) for the effect of overburden pressure in granular soils. The multiplier :math:`C_N` is calculated and applied to N or :math:`N_{60}`.
    Note that :math:`C_N` should be limited to 2 and preferably be kept below 1.5. In the function, a lower limit on the vertical effective stress of 25kPa is used in the validation to achieve this.
    
    :param N: Uncorrected or corrected SPT N number (:math:`N or N_{60}`) [-] - Suggested range: 0.0 <= N <= 60.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{v0}^{\\prime}`) [kPa] - Suggested range: 25.0 <= sigma_vo_eff <= 400.0
    :param granular: Boolean defining whether the soil is granular or not. If the soil is not granular, the correction factor is taken equal to 1
    .. math::
        C_N = \\sqrt{\\frac{98}{\\sigma_{v0}^{\\prime}}}
    
    :returns: Dictionary with the following keys:
        
        - 'CN [-]': Multiplier on SPT N number (:math:`C_N`)  [-]
        - 'N1 [-]': Corrected N number (:math:`N_1 or \\left( N_1 \\right)_{60}`)  [-]
    
    Reference - BS EN ISO 22476-3

    """
    
    if granular:
        _CN = np.sqrt(98 / sigma_vo_eff)
    else:
        _CN = 1
    _N1 = N * _CN

    return {
        'CN [-]': _CN,
        'N1 [-]': _N1,
    }


FRICTIONANGLE_SPT_PHT = {
    'N1_60': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
    'intercept': {'type': 'float', 'min_value': 23.0, 'max_value': 35.0},
    'multiplier': {'type': 'float', 'min_value': 0.1, 'max_value': 0.7},
    'multiplier_quadratic': {'type': 'float', 'min_value': 0.0001, 'max_value': 0.001},
}

FRICTIONANGLE_SPT_PHT_ERRORRETURN = {
    'Phi [deg]': np.nan,
}

@Validator(FRICTIONANGLE_SPT_PHT, FRICTIONANGLE_SPT_PHT_ERRORRETURN)
def frictionangle_spt_PHT(N1_60, intercept=27.1, multiplier=0.3, multiplier_quadratic=0.00054, **kwargs):

    """
    Correlation proposed by Peck, Hanson and Thornburn (1974) and mentioned by Wolff (1989)
    
    :param N1_60: Corrected SPT N value (:math:`\\left( N_1 \\right)_{60}`) [-] - Suggested range: 0.0 <= N1_60 <= 60.0
    :param intercept: Intercept at N=0 (:math:`-`) [deg] - Suggested range: 23.0 <= intercept <= 35.0 (optional, default= 27.1)
    :param multiplier: Multiplier on linear term (:math:`-`) [deg/blow] - Suggested range: 0.1 <= multiplier <= 0.7 (optional, default= 0.3)
    :param multiplier_quadratic: Multiplier on the quadratic term (:math:`-`) [deg/blow^2] - Suggested range: 0.0001 <= multiplier_quadratic <= 0.001 (optional, default= 0.00054)
    
    .. math::
        \\varphi^{\\prime} = 27.1 + 0.3 \\cdot \\left( N_1 \\right)_{60} - 0.00054 \\cdot \\left( N_1 \\right)_{60}^2
    
    :returns: Dictionary with the following keys:
        
        - 'Phi [deg]': Friction angle derived from SPT (:math:`\\varphi^{\\prime}`)  [deg]
    
    Reference - Peck, Hanson and Thornburn (1974). Foundation Engineering.

    """
    
    _Phi = intercept + multiplier * N1_60 - multiplier_quadratic * (N1_60 ** 2)

    return {
        'Phi [deg]': _Phi,
    }


YOUNGSMODULUS_SPT_AASHTO = {
    'N1_60': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
    'soiltype': {'type': 'string', 'options': ("Silts", "Clean sands", "Coarse sands", "Gravels"), 'regex': None},
    'multiplier_silts': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_cleansand': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_coarsesand': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_gravel': {'type': 'float', 'min_value': None, 'max_value': None},
}

YOUNGSMODULUS_SPT_AASHTO_ERRORRETURN = {
    'Es [MPa]': np.nan,
}

@Validator(YOUNGSMODULUS_SPT_AASHTO, YOUNGSMODULUS_SPT_AASHTO_ERRORRETURN)
def youngsmodulus_spt_AASHTO(N1_60, soiltype, multiplier_silts=0.4, multiplier_cleansand=0.7, multiplier_coarsesand=1.0, multiplier_gravel=1.1, **kwargs):

    """
    Calculates the Young's modulus based on corrected SPT number for various soil types.

    +-----------------------------------------------------+------------------------+---------------------------+
    | Soil type                                           | Soil type (short name) | :math:`E_s` [MPa]         |
    +=====================================================+========================+===========================+
    | Silts, sandy silts, slightly cohesive mixtures      | Silts                  |  :math:`0.4 ( N_1 )_{60}` |
    +-----------------------------------------------------+------------------------+---------------------------+
    | Clean fine to medium sands and slightly silty sands | Clean sands            |  :math:`0.7 ( N_1 )_{60}` |
    +-----------------------------------------------------+------------------------+---------------------------+
    | Coarse sands and sands with little gravel           | Coarse sands           |  :math:`1.0 ( N_1 )_{60}` |
    +-----------------------------------------------------+------------------------+---------------------------+
    | Sandy gravel and gravels                            | Gravels                |  :math:`1.1 ( N_1 )_{60}` |
    +-----------------------------------------------------+------------------------+---------------------------+
    
    :param N1_60: Corrected SPT N number (:math:`\\left( N_1 \\right)_{60}`) [-] - Suggested range: 0.0 <= N1_60 <= 60.0
    :param soiltype: Soil type - Options: ("Silts", "Clean sands", "Coarse sands", "Gravels")
    :param multiplier_silts: Multiplier on the silty soils (:math:`-`) [-] (optional, default= 0.4)
    :param multiplier_cleansand: Multiplier on the clean find sands (:math:`-`) [-] (optional, default= 0.7)
    :param multiplier_coarsesand: Multiplier on the coarse sands (:math:`-`) [-] (optional, default= 1.0)
    :param multiplier_gravel: Multiplier on the gravels (:math:`-`) [-] (optional, default= 1.1)
    
    :returns: Dictionary with the following keys:
        
        - 'Es [MPa]': Young's modulus (:math:`E_s`)  [MPa]
    
    Reference - AASHTO 1997 - LRFD

    """
    
    if soiltype == "Silts":
        _Es = multiplier_silts * N1_60
    elif soiltype == "Clean sands":
        _Es = multiplier_cleansand * N1_60
    elif soiltype == "Coarse sands":
        _Es = multiplier_coarsesand * N1_60
    elif soiltype == "Gravels":
        _Es = multiplier_gravel * N1_60
    else:
        raise ValueError("Soil type not recognised") 

    return {
        'Es [MPa]': _Es,
    }

UNDRAINEDSHEARSTRENGTHCLASS_N = {
    'N': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
}

UNDRAINEDSHEARSTRENGTHCLASS_N_ERRORRETURN = {
    'Consistency class': None,
    'qu min [kPa]': np.nan,
    'qu max [kPa]': np.nan
}

@Validator(UNDRAINEDSHEARSTRENGTHCLASS_N, UNDRAINEDSHEARSTRENGTHCLASS_N_ERRORRETURN)
def undrainedshearstrengthclass_spt_terzaghipeck(N, **kwargs):

    """
    Defines the relative density class for SPT measurements in cohesionless soils based on the uncorrected N-number

    +-----------------+-------------+-------------------+
    | N (uncorrected) | Consistency | :math:`q_u` [kPa] |
    +=================+=============+===================+
    |   <= 2          | Very soft   | < 25              | 
    +-----------------+-------------+-------------------+
    |   2 < N <= 4    | Soft        | 25 - 50           |
    +-----------------+-------------+-------------------+
    |   4 < N <= 8    | Medium      | 50 - 100          |
    +-----------------+-------------+-------------------+
    |   8 < N <= 15   | Stiff       | 100 - 200         |
    +-----------------+-------------+-------------------+
    |   15 < N <= 30  | Very stiff  | 200 - 400         |
    +-----------------+-------------+-------------------+
    |   N > 30        | Hard        | > 400             |
    +-----------------+-------------+-------------------+

    :param N: Uncorrected SPT N number (:math:`N`) [-] - Suggested range: 0.0 <= N <= 60.0
    
    :returns: Dictionary with the following keys:
        
        - 'Consistency class': Consistency class
        - 'qu min [kPa]': Minimum value for ultimate axial stress in a UCS test
        - 'qu max [kPa]': Maximum value for ultimate axial stress in a UCS test
    
    Reference - Terzaghi K, Peck RB (1967) Soil mechanics in engineering practice, 2nd edn. Wiley, New York

    """
    if N <= 2:
        _consistency_class = 'Very soft'
        _qu_min = np.nan
        _qu_max = 25
    elif 2 < N <= 4:
        _consistency_class = 'Soft'
        _qu_min = 25
        _qu_max = 50
    elif 4 < N <= 8:
        _consistency_class = 'Medium'
        _qu_min = 50
        _qu_max = 100
    elif 8 < N <= 15:
        _consistency_class = 'Stiff'
        _qu_min = 100
        _qu_max = 200
    elif 15 < N <= 30:
        _consistency_class = 'Very stiff'
        _qu_min = 200
        _qu_max = 400
    elif N > 30:
        _consistency_class = 'Hard'
        _qu_min = 400
        _qu_max = np.nan
    else:
        _consistency_class = None
        _qu_min = np.nan
        _qu_max = np.nan
    return {
        'Consistency class': _consistency_class,
        'qu min [kPa]': _qu_min,
        'qu max [kPa]': _qu_max
    }


CORRELATIONS = {
    'Overburden correction Liao and Whitman (1986)': overburdencorrection_spt_liaowhitman,
    'Overburden correction ISO 22476-3': overburdencorrection_spt_ISO,
    'N60 correction': spt_N60_correction,
    'Relative density Kulhawy and Mayne (1990)': relativedensity_spt_kulhawymayne,
    'Relative density class Terzaghi and Peck (1967)': relativedensityclass_spt_terzaghipeck,
    'Undrained shear strength class Terzaghi and Peck (1967)': undrainedshearstrengthclass_spt_terzaghipeck,
    'Undrained shear strength Salgado (2008)': undrainedshearstrength_spt_salgado,
    'Friction angle Kulhawy and Mayne (1990)': frictionangle_spt_kulhawymayne,
    'Friction angle PHT (1974)': frictionangle_spt_PHT
}