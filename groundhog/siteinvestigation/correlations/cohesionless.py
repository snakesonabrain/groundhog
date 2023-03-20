#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


GMAX_SAND_HARDINBLACK = {
    'sigma_m0': {'type': 'float', 'min_value': 0.0, 'max_value': 500.0},
    'void_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'coefficient_B': {'type': 'float', 'min_value': None, 'max_value': None},
    'pref': {'type': 'float', 'min_value': None, 'max_value': None},
}

GMAX_SAND_HARDINBLACK_ERRORRETURN = {
    'Gmax [kPa]': np.nan,
}

@Validator(GMAX_SAND_HARDINBLACK, GMAX_SAND_HARDINBLACK_ERRORRETURN)
def gmax_sand_hardinblack(
        sigma_m0 ,void_ratio,
        coefficient_B=875.0 ,pref=100.0, **kwargs):

    """
    Calculates the small-strain shear modulus of sand based on the correlation proposed with initial void ratio and stress level suggested by Hardin and Black (1968).

    The formulation was developed on a dataset of cohesive soils. However, the correlation was used in the PISA project to predict the small-strain shear modulus of sand.

    The default calibration parameter is taken from the recent study on monopile lateral response for the PISA project (Taborda et al, 2019). This calibration applies for dense marine sand.

    :param sigma_m0: Mean effective stress (:math:`p^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_m0 <= 500.0
    :param void_ratio: In-situ void ratio of the sand (:math:`e_0`) [:math:`-`] - Suggested range: 0.0 <= void_ratio <= 4.0
    :param coefficient_B: Calibration coefficient (:math:`B`) [:math:`-`] (optional, default= 875.0)
    :param pref: Reference pressure (:math:`p_{ref}^{\\prime}`) [:math:`kPa`] (optional, default= 100.0)

    .. math::
        G_{max} = \\frac{B p_{ref}^{\\prime}}{0.3 + 0.7 e_0^2} \\sqrt{\\frac{p^{\\prime}}{p_{ref}^{\\prime}}}

    :returns: Dictionary with the following keys:

        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Hardin, B.O. and Black W.L. 1968. Vibration modulus of normally consolidated clay Journal of Soil Mechanics and Foundations Div, 94(SM2), 353-369.

    Taborda, D.M.G., Zdravković, L., Potts, D.M., Burd, H.J., Byrne, B.W., Gavin, K., Houlsby, G.T., Jardine, R.J., Liu, T., Martin, C.M. and McAdam, R.A. 2018. Finite element modelling of laterally loaded piles in a dense marine sand at Dunkirk. Géotechnique, https://doi.org/10.1680/jgeot.18.pisa.006

    """

    _Gmax = ((coefficient_B * pref) / (0.3 + 0.7 * (void_ratio ** 2))) * np.sqrt(sigma_m0 / pref)

    return {
        'Gmax [kPa]': _Gmax,
    }


PERMEABILITY_D10_HAZEN = {
    'grain_size': {'type': 'float', 'min_value': 0.01, 'max_value': 2.0},
    'coefficient_C': {'type': 'float', 'min_value': None, 'max_value': None},
}

PERMEABILITY_D10_HAZEN_ERRORRETURN = {
    'k [m/s]': np.nan,
}


@Validator(PERMEABILITY_D10_HAZEN, PERMEABILITY_D10_HAZEN_ERRORRETURN)
def permeability_d10_hazen(
        grain_size,
        coefficient_C=0.01, **kwargs):
    """
    Calculates the permeability of a granular soil based on its grain size. Extensive investigation has shown that the fine particles have the greatest influence on permeability since they fill the voids between larger grains. The correlation by Hazen (1892) uses the 10th percentile grain size. Other authors have argues that the 5th percentile would be a better choice.

    :param grain_size: Grain size for which 10% of the particles are finer (:math:`D_{10}`) [:math:`mm`] - Suggested range: 0.01 <= grain_size <= 2.0
    :param coefficient_C: Calibration coefficient containing the effect of the shape of pore channels (:math:`C_{10)`) [:math:`-`] (optional, default= 0.01)

    .. math::
        k = C_{10} \\cdot D_{10}^2

    :returns: Dictionary with the following keys:

        - 'k [m/s]': Permeability of the granular soil (:math:`k`)  [:math:`m/s`]

    .. figure:: images/permeability_d10_hazen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Data supporting the Hazen correlation

    Reference - Terzaghi, K., Peck, R. B., & Mesri, G. (1996). Soil mechanics in engineering practice. John Wiley & Sons.

    """

    _k = coefficient_C * (grain_size ** 2)

    return {
        'k [m/s]': _k,
    }


HSSMALL_PARAMETERS_SAND = {
    'relative_density': {'type': 'float', 'min_value': 10.0, 'max_value': 100.0},
}

HSSMALL_PARAMETERS_SAND_ERRORRETURN = {
    'gamma_unsat [kN/m3]': np.nan,
    'gamma_sat [kN/m3]': np.nan,
    'E50_ref [kPa]': np.nan,
    'Eoed_ref [kPa]': np.nan,
    'Eur_ref [kPa]': np.nan,
    'G0_ref [kPa]': np.nan,
    'm [-]': np.nan,
    'gamma_07 [-]': np.nan,
    'phi_eff [deg]': np.nan,
    'psi [deg]': np.nan,
    'Rf [-]': np.nan,
}

@Validator(HSSMALL_PARAMETERS_SAND, HSSMALL_PARAMETERS_SAND_ERRORRETURN)
def hssmall_parameters_sand(
        relative_density,
        **kwargs):

    """
    Calculates the constitutive parameters for the HS Small model in PLAXIS as a function of relative density.

    The formulae were calibrated against a high-quality laboratory testing dataset on Toyoura, Ham River, Hostun and Ticino sand.

    :param relative_density: Relative density of sand (:math:`D_r`) [:math:`pct`] - Suggested range: 10.0 <= relative_density <= 100.0

    .. math::
        \\gamma_{unsat} = 15 + 4 \\cdot \\frac{D_r}{100}

        \\gamma_{sat} = 19 + 1.6 \\cdot \\frac{D_r}{100}

        E_{50}^{ref} = 6e4 \\cdot \\frac{D_r}{100}

        E_{oed}^{ref} = 6e4 \\cdot \\frac{D_r}{100}

        E_{ur}^{ref} = 18e4 \\cdot \\frac{D_r}{100}

        G_0^{ref} = 6e4 + 6.8e4 \\frac{D_r}{100}

        m = 0.7 - \\frac{D_r}{320}

        \\gamma_{0.7}= 10^{-4} \\cdot \\left( 2 - \\frac{D_r}{100} \\right)

        \\varphi^{\\prime} = 28 + 12.5 \\cdot \\frac{D_r}{100}

        \\psi = -2 + 12.5 \\cdot \\frac{D_r}{100}

        R_f = 1 - \\frac{D_r}{800}

    :returns: Dictionary with the following keys:

        - 'gamma_unsat [kN/m3]': Unsaturated unit weight (:math:`\\gamma_{unsat}`)  [:math:`kN/m3`]
        - 'gamma_sat [kN/m3]': Saturated unit weight (:math:`\\gamma_{sat}`)  [:math:`kN/m3`]
        - 'E50_ref [kPa]': Reference secant stiffness (:math:`E_{50}^{ref}`)  [:math:`kPa`]
        - 'Eoed_ref [kPa]': Reference oedometric stiffness (:math:`E_{oed}^{ref}`)  [:math:`kPa`]
        - 'Eur_ref [kPa]': Reference unloading-reloading stiffness (:math:`E_{ur}^{ref}`)  [:math:`kPa`]
        - 'G0_ref [kPa]': Reference small-strain shear modulus (:math:`G_{0}^{ref}`)  [:math:`kPa`]
        - 'm [-]': Stiffness exponent (:math:`m`)  [:math:`-`]
        - 'gamma_07 [-]': Strain level where shear modulus has reduced to  70 percent of Gmax (:math:`\\gamma_{0.7}`)  [:math:`-`]
        - 'phi_eff [deg]': Effective friction angle (:math:`\\varphi^{\\prime}`)  [:math:`deg`]
        - 'psi [deg]': Dilation angle (:math:`\\psi`)  [:math:`deg`]
        - 'Rf [-]': Failure ratio (:math:`-`)  [:math:`-`]

    .. figure:: images/hssmall_parameters_sand_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        HS Small parameters as a function of relative density

    Reference - Brinkgreve, R. B. J., Engin, E., & Engin, H. K. (2010). Validation of empirical formulas to derive model parameters for sands. Numerical methods in geotechnical engineering, 137- 142.

    """

    _gamma_unsat = 15 + 4 * (relative_density / 100)
    _gamma_sat = 19 + 1.6 * (relative_density / 100)
    _E50_ref = 6e4 * (relative_density / 100)
    _Eoed_ref = 6e4 * (relative_density / 100)
    _Eur_ref = 18e4 * (relative_density / 100)
    _G0_ref = 6e4 + 6.8e4 * (relative_density / 100)
    _m = 0.7 - (relative_density / 320)
    _gamma_07 = 1e-4 * (2 - (relative_density / 100))
    _phi_eff = 28 + 12.5 * (relative_density / 100)
    _psi = -2 + 12.5 * (relative_density / 100)
    _Rf = 1 - (relative_density / 800)

    return {
        'gamma_unsat [kN/m3]': _gamma_unsat,
        'gamma_sat [kN/m3]': _gamma_sat,
        'E50_ref [kPa]': _E50_ref,
        'Eoed_ref [kPa]': _Eoed_ref,
        'Eur_ref [kPa]': _Eur_ref,
        'G0_ref [kPa]': _G0_ref,
        'm [-]': _m,
        'gamma_07 [-]': _gamma_07,
        'phi_eff [deg]': _phi_eff,
        'psi [deg]': _psi,
        'Rf [-]': _Rf,
    }

STRESS_DILATANCY_BOLTON = {
    'relative_density': {'type': 'float', 'min_value': 0.1, 'max_value': 1.0},
    'p_eff': {'type': 'float', 'min_value': 20, 'max_value': 10000},
    'Q': {'type': 'float', 'min_value': 5, 'max_value': 10},
    'R': {'type': 'float', 'min_value': None, 'max_value': None},
    'stress_condition': {'type': 'string', 'options':("triaxial strain","plane strain"),'regex':None},
}

STRESS_DILATANCY_BOLTON_ERRORRETURN = {
    'Ir [-]': np.nan,
    'phi_max - phi_cs [deg]': np.nan,
    'Dilation angle [deg]': np.nan,
    '-depsilon_v/depsilon_1__max [-]': np.nan
}


@Validator(STRESS_DILATANCY_BOLTON, STRESS_DILATANCY_BOLTON_ERRORRETURN)
def stress_dilatancy_bolton(
        relative_density, p_eff, Q=10, R=1,
        stress_condition='triaxial strain', **kwargs):
    """
    Cohesionless soils with sufficiently high relative density will tend to dilate but dilation
    can be suppressed by the stress on the sample. The higher the stress, the lower the amount
    of dilation.
    Bolton (1986) formulated relations to calculated the difference between peak friction angle
    and critical state friction angle based on a series of plane strain and triaxial tests.
    A relative dilatancy index (:math:`I_R`) is defined which has allows prediction of the dilation
    angle for plane strain and triaxial strain.
    This function predicts the value of the relative dilatancy index based on user input and
    applies this to calculate the difference between peak and residual friction angle, 
    dilation angle and the maximum ratio of volumetric strain increment to first principal strain increment
    for a selected stress condition (plane strain or triaxial strain).
    The calibration factors in the equation for relative dilatancy index can be adjusted
    (e.g. for crushable soils).
    Note that the formulae apply for :math:`0 < I_R < 4`.
    
    :param relative_density: Relative density of the material (:math:`D_{r)`) [:math:`-`] - Suggested range: 0.1 <= relative_density <= 1.0
    :param p_eff: Effective pressure (:math:`p_{eff)^{\\prime}`) [kPa] - Suggested range: 20 <= p_eff <= 10000. In the discussion following the paper publication, a remark was made that using a minimum value of 150kPa for the effective pressure is prudent.
    :param Q: First calibration factor in the equation for relative dilatancy index (:math:`Q`) (optional: Default = 10 for quartz and feldspar sands, See Table 2 in Bolton's paper for other grain types)- Suggested range: 5 <= Q <= 10
    :param R: Second calibration factor in the equation for relative dilatancy index (:math:`R`) (optional: Default = 1)
    :param stress_condition: Assumed stress condition: Choose between ``'triaxial strain'`` and ``'plane strain'``

    .. math::
        I_R = D_r \\left( Q - \\ln p^{\\prime} \\right) - R

        \\varphi_{max}^{\\prime} - \\varphi_{crit}^{\\prime} = 0.8 \\phi_{max} = 5 I_R \\ \\ \\text{plane strain}

        \\varphi_{max}^{\\prime} - \\varphi_{crit}^{\\prime} = 3 I_R \\ \\ \\text{triaxial strain}
        
        \\left( - \\frac{d \\epsilon_v}{d \\epsilon_1} \\right)_{max} = 0.3 I_R

    :returns: Dictionary with the following keys:

        - 'Ir [-]': Relative dilatancy index (:math:`I_R`)  [-]
        - 'phi_max - phi_cs [deg]': Difference between peak and critical state friction angle (:math:`\\varphi_{max}^{\\prime} - \\varphi_{crit}^{\\prime}`)  [deg]
        - 'Dilation angle [deg]': Calculated dilation angle for the selected stress condition (:math:`\\psi`)  [deg]
        - '-depsilon_v/depsilon_1__max [-]': Maximum ratio of volumetric to first principal strain increment (:math:`\\left( - \\frac{d \\epsilon_v}{d \\epsilon_1} \\right)_{max}`) [-]

    .. figure:: images/data_bolton.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Stress-dilatancy theory applied to selected tests (Bolton, 1986)

    Reference - Bolton, M. D. "The strength and dilatancy of sands." Geotechnique 36.1 (1986): 65-78.

    """

    _Ir = relative_density * (Q  - np.log(p_eff)) - R

    if _Ir < 0 or _Ir > 4:
        warnings.warn("Outside of valid range for relative dilatancy index")

    if stress_condition == 'plane strain':
        _phi_difference = 5 * _Ir
        _dilationangle = _phi_difference / 0.8
    elif stress_condition == 'triaxial strain':
        _phi_difference = 3 * _Ir
        _dilationangle = _phi_difference
    else:
        raise ValueError("Stress condition %s not recognised" % stress_condition)

    _strainratio = 0.3 * _Ir

    return {
        'Ir [-]': _Ir,
        'phi_max - phi_cs [deg]': _phi_difference,
        'Dilation angle [deg]': _dilationangle,
        '-depsilon_v/depsilon_1__max [-]': _strainratio
    }
