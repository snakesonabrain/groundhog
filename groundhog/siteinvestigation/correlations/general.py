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

SHEARWAVEVELOCITY_COMPRESSIONINDEX_CHA = {
    'Cc': {'type': 'float', 'min_value': 0.005, 'max_value': 1.2},
    'sigma_eff_particle_motion': {'type': 'float', 'min_value': 10, 'max_value': 1200},
    'sigma_eff_wave_propagation': {'type': 'float', 'min_value': 10, 'max_value': 1200},
    'alpha': {'type': 'float', 'min_value': 5, 'max_value': 1000},
    'beta': {'type': 'float', 'min_value': 0.0, 'max_value': 0.6},
    'calibration_factor_alpha_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_alpha_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_beta_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_factor_beta_2': {'type': 'float', 'min_value': None, 'max_value': None}
}

SHEARWAVEVELOCITY_COMPRESSIONINDEX_CHA_ERRORRETURN = {
    'Vs [m/s]': np.nan,
    'alpha [-]': np.nan,
    'beta [-]': np.nan
}

@Validator(SHEARWAVEVELOCITY_COMPRESSIONINDEX_CHA, SHEARWAVEVELOCITY_COMPRESSIONINDEX_CHA_ERRORRETURN)
def shearwavevelocity_compressionindex_cha(
        Cc, sigma_eff_particle_motion, sigma_eff_wave_propagation,
        alpha=np.nan, beta=np.nan, calibration_factor_alpha_1=13.5, calibration_factor_alpha_2=-0.63,
        calibration_factor_beta_1=0.17, calibration_factor_beta_2=0.43, **kwargs):

    """
    Shear wave velocity is dependent on the stiffness of the soil skeleton which is in turn affected by the compression index :math:`C_c`.
    Cha et al (2014) reported a series of oedometer tests with bender elements to establish the coefficients of a power law equation.
    Note that :math:`C_c` itself is also stress-dependent and requires the selection of appropriate points in :math:`e-\\log p^{\\prime}` space.

    The relations proposed by Cha et al (2014) are used here by default, but the user can also enter custom values for :math:`\\alpha` and :math:`\\beta`
    Since porosity is not a parameter which is commonly used, the user can enter bulk unit weight instead which is then converted to porosity for a saturated soil.

    For application to field cases, the effective stress in the direction of particle motion and wave propagation needs to be estimated.
    This usually involves estimation of the coefficient of lateral earth pressure.

    :param Cc: Compression index (:math:`C_c`) [:math:`-`] - Suggested range: 0.005 <= Cc <= 1.2
    :param sigma_eff_particle_motion: Effective stress in the direction of particle motion (:math:`\\sigma_{\\perp}^{\\prime}`) [:math:`kPa`] - Suggested range: 10 <= sigma_eff_particle_motion <= 1200
    :param sigma_eff_wave_propagation: Effective stress in the direction of wave propagation (:math:`\\sigma_{\\parallel}^{\\prime}`) [:math:`kPa`] - Suggested range: 10 <= sigma_eff_wave_propagation <= 1200
    :param alpha: Custom alpha-factor in the power law (:math:`\\alpha`) [:math:`-`] - Suggested range: 5 <= alpha <= 1000 (optional, default=``np.nan``)
    :param beta: Custom beta-factor in the power law (:math:`\\beta`) [:math:`-`] - Suggested range: 0.0 <= beta <= 0.6 (optional, default= ´´np.nan´´)
    :param calibration_factor_alpha_1: First calibration factor for alpha [:math:`-`] (optional, default= 13.5)
    :param calibration_factor_alpha_2: Second calibration factor for alpha [:math:`-`] (optional, default= 0.63)
    :param calibration_factor_beta_1: First calibration factor for beta [:math:`-`] (optional, default= 0.17)
    :param calibration_factor_beta_2: First calibration factor for alpha [:math:`-`] (optional, default= 0.43)

    .. math::
        V_s = \\sqrt{\\frac{G}{\\rho}} = \\alpha \\left( \\frac{\\sigma_{\\perp}^{\\prime} + \\sigma_{\\parallel}^{\\prime}}{2 \\ \\text{kPa}} \\right)^{\\beta}

        \\alpha = 13.5 (\\text{m/s}) \\cdot C_c^{-0.63}

        \\beta = 0.17 \\log_{10} C_c + 0.43

    :returns: Dictionary with the following keys:

        - 'Vs [m/s]': Shear wave velocity (:math:`V_s`)  [:math:`\\text{m/s}`]
        - 'alpha [-]': Alpha-factor (multiplier) (:math:`\\alpha`)  [:math:`-`]
        - 'beta [-]': Beta-factor (exponent) (:math:`\\beta`)  [:math:`-`]

    .. figure:: images/chaetal_data.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Compiled data from Cha et al

    Reference - Cha et al (2014). Small-Strain Stiffness, Shear-Wave Velocity and Soil Compressibilitys. Journal of Geotechnical and Geoenvironmental Engineering.

    """
    if np.isnan(alpha):
        _alpha = calibration_factor_alpha_1 * (Cc ** calibration_factor_alpha_2)
    else:
        _alpha = alpha

    if np.isnan(beta):
        _beta = calibration_factor_beta_1 * np.log10(Cc) + calibration_factor_beta_2
    else:
        _beta = beta
    
    _Vs = _alpha * ((0.5 * (sigma_eff_particle_motion + sigma_eff_wave_propagation)) ** _beta)
    
    return {
        'Vs [m/s]': _Vs,
        'alpha [-]': _alpha,
        'beta [-]': _beta
    }


K0_FRICTIONANGLE_MESRI = {
    'phi_cs': {'type': 'float', 'min_value': 15, 'max_value': 45},
    'ocr': {'type': 'float', 'min_value': 1, 'max_value': 30},
}

K0_FRICTIONANGLE_MESRI_ERRORRETURN = {
    'K0 [-]': np.nan,
}


@Validator(K0_FRICTIONANGLE_MESRI, K0_FRICTIONANGLE_MESRI_ERRORRETURN)
def k0_frictionangle_mesri(
        phi_cs,
        ocr=1, **kwargs):
    """
    Calculates the coefficient of lateral earthpressure at rest for normally and overconsolidated sand and clay.
    Mesri and Hayat (1993) showed that the equation by Jaky (1944) only applied for sedimented, normally consolidated young clays and sands.
    The effect of overconsolidation was captured by multiplying the value for normally consolidated soil with the OCR raised to an exponent.
    This exponent is independent of the soil's initial density and thus needs to be related to the critical state friction angle,
    rather than the peak friction angle of the soil. By adjusting for the effect of overconsolidation, reasonable predictions are obtained
    for overconsolidated and pre-sheared soils.

    :param phi_cs: Critical state friction angle (:math:`\\varphi_{cs}^{\\prime}`) [:math:`deg`] - Suggested range: 0.01 <= grain_size <= 2.0
    :param ocr: Overconsolidation ratio (:math:`\\text{OCR}`) [:math:`-`] (optional, default= 1, suggested range: 1 <= OCR < 30)

    .. math::
        K_0 = \\left( 1 - \\sin \\varphi_{cv}^{\\prime} \\right) \\text{OCR}^{\\sin \\varphi_{cv}^{\\prime}}

    :returns: Dictionary with the following keys:

        - 'K0 [-]': Coefficient of lateral earth pressure at rest (:math:`K_0`)  [:math:`-`]

    Reference - Mesri and Hayat (1993) The coefficient of earth pressure at rest. Canadian Geotechnical Journal. 30(4), 647-666

    """

    _K0 = (1 - np.sin(np.radians(phi_cs))) * (ocr ** (np.sin(np.radians(phi_cs))))

    return {
        'K0 [-]': _K0,
    }