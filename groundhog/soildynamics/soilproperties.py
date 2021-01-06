#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

MODULUSREDUCTION_PLASTICITY_ISHIBASHI = {
    'strain': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'PI': {'type': 'float', 'min_value': 0.0, 'max_value': 200.0},
    'sigma_m_eff': {'type': 'float', 'min_value': 0.0, 'max_value': 400.0},
    'multiplier_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_3': {'type': 'float', 'min_value': None, 'max_value': None},
}

MODULUSREDUCTION_PLASTICITY_ISHIBASHI_ERRORRETURN = {
    'G/Gmax [-]': np.nan,
    'K [-]': np.nan,
    'm [-]': np.nan,
    'n [-]': np.nan,
    'dampingratio [pct]': np.nan,
}


@Validator(MODULUSREDUCTION_PLASTICITY_ISHIBASHI, MODULUSREDUCTION_PLASTICITY_ISHIBASHI_ERRORRETURN)
def modulusreduction_plasticity_ishibashi(
        strain, PI, sigma_m_eff,
        multiplier_1=0.000102, exponent_1=0.492, multiplier_2=0.000556, exponent_2=0.4, multiplier_3=-0.0145,
        exponent_3=1.3, **kwargs):
    """
    Calculates the modulus reduction curve (G/Gmax) as a function of shear strain. The curve depends on the plasticity of the material (plasticity index) and the mean effective stress at the depth of interest.

    The curve for cohesionless soils can be established by using a plasticity index of 0. At low plasticity, the effect of confining pressure on the modulus reduction curve is more pronounced.

    Also calculates the damping ratio of plastic and non-plastic soils based on a fit to empirical data.

    :param strain: Strain amplitude (:math:`\\gamma`) [:math:`pct`] - Suggested range: 0.0 <= strain <= 10.0
    :param PI: Plasticity index (:math:`PI`) [:math:`pct`] - Suggested range: 0.0 <= PI <= 200.0
    :param sigma_m_eff: Mean effective pressure (:math:`\\sigma_m^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_m_eff <= 400.0
    :param multiplier_1: Multiplier in equation for K (:math:``) [:math:`-`] (optional, default= 0.000102)
    :param exponent_1: Exponent in equation for K (:math:``) [:math:`-`] (optional, default= 0.492)
    :param multiplier_2: First multiplier in equation for m (:math:``) [:math:`-`] (optional, default= 0.000556)
    :param exponent_2: First exponent in equation for m (:math:``) [:math:`-`] (optional, default= 0.4)
    :param multiplier_3: Second multiplier in equation for m (:math:``) [:math:`-`] (optional, default= -0.0145)
    :param exponent_3: Second exponent in equation for m (:math:``) [:math:`-`] (optional, default= 1.3)

    .. math::
        \\frac{G}{G_{max}} = K \\left( \\gamma, \\text{PI} \\right) \\left( \\sigma_m^{\\prime} \\right)^{m \\left( \\gamma, \\text{PI} \\right) - m_0}

        K \\left( \\gamma, \\text{PI} \\right) = 0.5 \\left[ 1 + \\tanh \\left[ \\ln \\left( \\frac{0.000102 + n ( \\text{PI} )}{\\gamma} \\right)^{0.492} \\right] \\right]

        m \\left( \\gamma, \\text{PI} \\right) - m_0 = 0.272 \\left[ 1 - \\tanh \\left[ \\ln \\left( \\frac{0.000556}{\\gamma} \\right)^{0.4} \\right] \\right] \\exp \\left( -0.0145 \\text{PI}^{1.3} \\right)

        n ( \\text{PI} ) = \\begin{cases}
            0.0       & \\quad \\text{for PI } = 0 \\\\
            3.37 \\times 10^{-6} \\text{PI}^{1.404}  & \\quad \\text{for } 0 < \\text{PI} \\leq 15 \\\\
            7.0 \\times 10^{-7} \\text{PI}^{1.976}  & \\quad \\text{for } 15 < \\text{PI} \\leq 70 \\\\
            2.7 \\times 10^{-5} \\text{PI}^{1.115}  & \\quad \\text{for } \\text{PI} > 70
          \\end{cases}

        \\xi = 0.333 \\frac{1 + \\exp(-0.0145 PI^{1.3})}{2} \\left[ 0.586 \\left( \\frac{G}{G_{max}} \\right)^2 - 1.547 \\frac{G}{G_{max}} + 1 \\right]

    :returns: Dictionary with the following keys:

        - 'G/Gmax [-]': Modulus reduction ratio (:math:`G / G_{max}`)  [:math:`-`]
        - 'K [-]': Factor K in the equation (:math:`K ( \\gamma, \\text{PI} )`)  [:math:`-`]
        - 'm [-]': Exponent m in the equation (:math:`m \\left( \\gamma, \\text{PI} \\right) - m_0`)  [:math:`-`]
        - 'n [-]': Factor n in equations (:math:`n ( \\text{PI} )`)  [:math:`-`]
        - 'dampingratio [pct]': Damping ratio (:math:`\\xi`)  [:math:`pct`]

    Reference - Ishibashi, I., & Zhang, X. (1993). Unified dynamic shear moduli and damping ratios of sand and clay. Soils and foundations, 33(1), 182-191.

    """

    strain = 0.01 * strain

    if PI == 0:
        _n = 0
    elif 0 < PI <= 15:
        _n = 3.37e-6 * (PI ** 1.404)
    elif 15 < PI <= 70:
        _n = 7e-7 * (PI ** 1.976)
    else:
        _n = 2.7e-5 * (PI ** 1.115)

    _m = 0.272 * (1 - np.tanh(np.log((multiplier_2 / strain) ** exponent_2))) * \
        np.exp(multiplier_3 * (PI ** exponent_3))

    _K = 0.5 * (1 + np.tanh(np.log(((multiplier_1 + _n) / strain) ** exponent_1)))

    _G_over_Gmax = min(_K * (sigma_m_eff ** _m), 1)

    _damping = 100 * 0.333 * 0.5 * (1 + np.exp(-0.0145 * (PI ** 1.3))) * \
               (0.586 * (_G_over_Gmax ** 2) - 1.547 * _G_over_Gmax + 1)
    return {
        'G/Gmax [-]': _G_over_Gmax,
        'K [-]': _K,
        'm [-]': _m,
        'n [-]': _n,
        'dampingratio [pct]': _damping,
    }


GMAX_SHEARWAVEVELOCTY = {
    'Vs': {'type': 'float', 'min_value': 0.0, 'max_value': 600.0},
    'gamma': {'type': 'float', 'min_value': 12.0, 'max_value': 22.0},
    'g': {'type': 'float', 'min_value': 9.7, 'max_value': 10.2},
}

GMAX_SHEARWAVEVELOCTY_ERRORRETURN = {
    'rho [kg/m3]': np.nan,
    'Gmax [kPa]': np.nan,
}


@Validator(GMAX_SHEARWAVEVELOCTY, GMAX_SHEARWAVEVELOCTY_ERRORRETURN)
def gmax_shearwavevelocty(
        Vs, gamma,
        g=9.81, **kwargs):
    """
    Calculates the small-strain shear modulus (shear strain < 1e-4%) from the shear wave velocity and the bulk unit weight if the soil based on elastic theory.

    Often, the result of an in-situ or laboratory test will provide the shear wave velocity, which is then converted to the small-strain shear modulus using this function.

    :param Vs: Shear wave velocity (:math:`V_s`) [:math:`m/s`] - Suggested range: 0.0 <= Vs <= 600.0
    :param gamma: Bulk unit weight (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 12.0 <= gamma <= 22.0
    :param g: Acceleration due to gravity (:math:`g`) [:math:`m/s2`] - Suggested range: 9.7 <= g <= 10.2 (optional, default= 9.81)

    .. math::
        G_{max} = \\rho \\cdot V_s^2

        \\rho = \\gamma / g

    :returns: Dictionary with the following keys:

        - 'rho [kg/m3]': Density of the material (:math:`\\rho`)  [:math:`kg/m3`]
        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Robertson, P.K. and Cabal, K.L. (2015). Guide to Cone Penetration Testing for Geotechnical Engineering. 6th edition. Gregg Drilling & Testing, Inc.

    """

    _rho = 1000 * gamma / g
    _Gmax = 1e-3 * _rho * (Vs ** 2)

    return {
        'rho [kg/m3]': _rho,
        'Gmax [kPa]': _Gmax,
    }