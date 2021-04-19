#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np
from scipy.optimize import brentq

# Project imports
from groundhog.general.validation import Validator


PCPT_KEY_MAPPING = {
    'qc [MPa]': 'qc',
    'fs [MPa]': 'fs',
    'u2 [MPa]': 'u2',
    'qt [MPa]': 'qt',
    'ft [MPa]': 'ft',
    'qnet [MPa]': 'qnet',
    'Vertical total stress [kPa]': 'sigma_vo',
    'Vertical effective stress [kPa]': 'sigma_vo_eff',
    'Ic [-]': 'ic',
    'Dr [-]': 'relative_density',
    'Gmax [kPa]': 'gmax',
    'Qt [-]': 'Qt',
    'Bq [-]': 'Bq',
    'Fr [%]': 'Fr',
    'Rf [%]': 'Rf',
    'K0 [-]': 'K0',
    'Vs [m/s]': 'Vs',
    'gamma [kN/m3]': 'gamma',
    'OCR [-]': 'ocr'
}



PCPT_NORMALISATIONS = {
    'measured_qc': {'type': 'float', 'min_value': 0.0, 'max_value': 150.0},
    'measured_fs': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'measured_u2': {'type': 'float', 'min_value': -10.0, 'max_value': 10.0},
    'sigma_vo_tot': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'depth': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'cone_area_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'start_depth': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'unitweight_water': {'type': 'float', 'min_value': 9.0, 'max_value': 11.0},
}

PCPT_NORMALISATIONS_ERRORRETURN = {
    'qt [MPa]': np.NaN,
    'qc [MPa]': np.NaN,
    'u2 [MPa]': np.NaN,
    'Delta u2 [MPa]': np.NaN,
    'Rf [pct]': np.NaN,
    'Bq [-]': np.NaN,
    'Qt [-]': np.NaN,
    'Fr [-]': np.NaN,
    'qnet [MPa]': np.NaN,
}


@Validator(PCPT_NORMALISATIONS, PCPT_NORMALISATIONS_ERRORRETURN)
def pcpt_normalisations(
        measured_qc, measured_fs, measured_u2, sigma_vo_tot, sigma_vo_eff, depth, cone_area_ratio,
        start_depth=0.0, unitweight_water=10.25, **kwargs):
    """
    Carried out the necessary normalisation and correction on PCPT data to allow calculation of derived parameters and soil type classification.

    For a downhole test, the depth of the test and the unit weight of water can optionally be provided. If no start depth is specified, a continuous test starting from the surface is assumed. The measurements are corrected for this effect.

    Next, the cone resistance is corrected for the unequal area effect using the cone area ratio. The correction for total sleeve friction is not included as it is more uncommon. The procedure assumes that the pore pressure are measured at the shoulder of the cone. If this is not the case, corrections can be used which are not included in this function.

    During normalisation, the friction ratio and pore pressure ratio are calculated. Note that the total cone resistance is used for the friction ratio and pore pressure ratio calculation, the pore pressure ratio calculation also used the total vertical effective stress. The normalised cone resistance and normalised friction ratio are also calculated.

    Finally the net cone resistance is calculated.

    :param measured_qc: Measured cone resistance (:math:`q_c^*`) [:math:`MPa`] - Suggested range: 0.0 <= measured_qc <= 150.0
    :param measured_fs: Measured sleeve friction (:math:`f_s^*`) [:math:`MPa`] - Suggested range: 0.0 <= measured_fs <= 10.0
    :param measured_u2: Pore pressure measured at the shoulder (:math:`u_2^*`) [:math:`MPa`] - Suggested range: -10.0 <= measured_u2 <= 10.0
    :param sigma_vo_tot: Total vertical stress (:math:`\\sigma_{vo}`) [:math:`kPa`] - Suggested range: sigma_vo_tot >= 0.0
    :param sigma_vo_eff: Effective vertical stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param depth: Depth below surface (for saturated soils) where measurement is taken. For onshore tests, use the depth below the watertable. (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param cone_area_ratio: Ratio between the cone rod area and the maximum cone area (:math:`a`) [:math:`-`] - Suggested range: 0.0 <= cone_area_ratio <= 1.0
    :param start_depth: Start depth of the test, specify this for a downhole test. Leave at zero for a test starting from surface (:math:`d`) [:math:`m`] - Suggested range: start_depth >= 0.0 (optional, default= 0.0)
    :param unitweight_water: Unit weight of water, default is for seawater (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.0 <= unitweight_water <= 11.0 (optional, default= 10.25)

    .. math::
        q_c = q_c^* + d \\cdot a \\cdot \\gamma_w

        q_t = q_c + u_2 \\cdot (1 - a)

        u_2 = u_2^* + \\gamma_w \\cdot d

        \\Delta u_2 = u_2 - u_o

        R_f = \\frac{f_s}{q_t}

        B_q = \\frac{\\Delta u_2}{q_t - \\sigma_{vo}}

        Q_t = \\frac{q_t - \\sigma_{vo}}{\\sigma_{vo}^{\\prime}}

        F_r = \\frac{f_s}{q_t - \\sigma_{vo}}

        q_{net} = q_t - \\sigma_{vo}

    :returns: Dictionary with the following keys:

        - 'qt [MPa]': Total cone resistance (:math:`q_t`)  [:math:`MPa`]
        - 'qc [MPa]': Cone resistance corrected for downhole effect (:math:`q_c`)  [:math:`MPa`]
        - 'u2 [MPa]': Pore pressure at the shoulder corrected for downhole effect (:math:`u_2`)  [:math:`MPa`]
        - 'Delta u2 [MPa]': Difference between measured pore pressure at the shoulder and hydrostatic pressure (:math:`\\Delta u_2`)  [:math:`MPa`]
        - 'Rf [pct]': Ratio of sleeve friction to total cone resistance (note that it is expressed as a percentage) (:math:`R_f`)  [:math:`pct`]
        - 'Bq [-]': Pore pressure ratio (:math:`B_q`)  [:math:`-`]
        - 'Qt [-]': Normalised cone resistance (:math:`Q_t`)  [:math:`-`]
        - 'Fr [-]': Normalised friction ratio (:math:`F_r`)  [:math:`-`]
        - 'qnet [MPa]': Net cone resistance (:math:`q_{net}`)  [:math:`MPa`]

    .. figure:: images/pcpt_normalisations_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Pore water pressure effects on measured parameters

    Reference - Lunne, T., Robertson, P.K., Powell, J.J.M., 1997. Cone penetration testing in geotechnical practice. E & FN Spon.

    """

    _qc = measured_qc + 0.001 * start_depth * cone_area_ratio * unitweight_water
    _u2 = measured_u2 + 0.001 * unitweight_water * start_depth
    _qt = _qc + _u2 * (1.0 - cone_area_ratio)
    _Delta_u2 = _u2 - 0.001 * depth * unitweight_water
    _Rf = 100.0 * measured_fs / _qt
    _Bq = _Delta_u2 / (_qt - 0.001 * sigma_vo_tot)
    _Qt = (_qt - 0.001 * sigma_vo_tot) / (0.001 * sigma_vo_eff)
    _Fr = measured_fs / (_qt - 0.001 * sigma_vo_tot)
    _qnet = _qt - 0.001 * sigma_vo_tot

    return {
        'qt [MPa]': _qt,
        'qc [MPa]': _qc,
        'u2 [MPa]': _u2,
        'Delta u2 [MPa]': _Delta_u2,
        'Rf [pct]': _Rf,
        'Bq [-]': _Bq,
        'Qt [-]': _Qt,
        'Fr [-]': _Fr,
        'qnet [MPa]': _qnet,
    }


IC_SOILCLASS_ROBERTSON = {
    'ic': {'type': 'float', 'min_value': 1.0, 'max_value': 5.0},
}

IC_SOILCLASS_ROBERTSON_ERRORRETURN = {
    'Soil type number [-]': np.nan,
    'Soil type': np.nan,
}


@Validator(IC_SOILCLASS_ROBERTSON, IC_SOILCLASS_ROBERTSON_ERRORRETURN)
def ic_soilclass_robertson(
        ic,
        **kwargs):
    """
    Provides soil type classification according to the soil behaviour type index by Robertson and Wride.

    :param ic: Soil behaviour type index (:math:`I_c`) [:math:`-`] - Suggested range: 1.0 <= ic <= 5.0

    :returns: Dictionary with the following keys:

        - 'Soil type number [-]': Number of the soil type in the Robertson chart [:math:`-`]
        - 'Soil type': Description of the soil type in the Robertson chart

    Reference - Fugro guidance on PCPT interpretation

    """

    if ic < 1.31:
        ic_class_number = 7
        ic_class = "Gravelly sand to sand"
    elif 1.31 <= ic < 2.05:
        ic_class_number = 6
        ic_class = "Sands: clean sands to silty sands"
    elif 2.05 <= ic < 2.6:
        ic_class_number = 5
        ic_class = "Sand mixtures: silty sand to sand silty"
    elif 2.6 <= ic < 2.95:
        ic_class_number = 4
        ic_class = "Silt mixtures: clayey silt to silty clay"
    elif 2.95 <= ic < 3.6:
        ic_class_number = 3
        ic_class = "Clays: clay to silty clay"
    else:
        ic_class_number = 2
        ic_class = "Organic soils-peats"

    return {
        'Soil type number [-]': ic_class_number,
        'Soil type': ic_class,
    }


BEHAVIOURINDEX_PCPT_ROBERTSONWRIDE = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'fs': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'ic_min': {'type': 'float', 'min_value': None, 'max_value': None},
    'ic_max': {'type': 'float', 'min_value': None, 'max_value': None},
    'zhang_multiplier_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'zhang_multiplier_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'zhang_subtraction': {'type': 'float', 'min_value': None, 'max_value': None},
    'robertsonwride_coefficient1': {'type': 'float', 'min_value': None, 'max_value': None},
    'robertsonwride_coefficient2': {'type': 'float', 'min_value': None, 'max_value': None},
}

BEHAVIOURINDEX_PCPT_ROBERTSONWRIDE_ERRORRETURN = {
    'exponent_zhang [-]': np.nan,
    'Qtn [-]': np.nan,
    'Fr [%]': np.nan,
    'Ic [-]': np.nan,
    'Ic class number [-]': np.nan,
    'Ic class': None,
}


@Validator(BEHAVIOURINDEX_PCPT_ROBERTSONWRIDE, BEHAVIOURINDEX_PCPT_ROBERTSONWRIDE_ERRORRETURN)
def behaviourindex_pcpt_robertsonwride(
        qt, fs, sigma_vo, sigma_vo_eff,
        atmospheric_pressure=100.0, ic_min=1.0, ic_max=4.0, zhang_multiplier_1=0.381, zhang_multiplier_2=0.05,
        zhang_subtraction=0.15, robertsonwride_coefficient1=3.47, robertsonwride_coefficient2=1.22, **kwargs):
    """
    Calculates the soil behaviour index according to Robertson and Wride (1998). This index is a measure for the behaviour of soils. Soils with a value below 2.5 are generally cohesionless and coarse grained whereas a value above 2.7 indicates cohesive, fine-grained sediments. Between 2.5 and 2.7, partially drained behaviour is expected.
    Because the exponent n is defined implicitly, an iterative approach is required to calculate the soil behaviour type index.

    :param qt: Corrected cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param fs: Sleeve friction (:math:`f_s`) [:math:`MPa`] - Suggested range: fs >= 0.0
    :param sigma_vo: Total vertical stress (:math:`\\sigma_{vo}`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 9.0
    :param atmospheric_pressure: Atmospheric pressure (used for normalisation) (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)
    :param ic_min: Minimum value for soil behaviour type index used in the optimisation routine (:math:`I_{c,min}`) [:math:`-`] (optional, default= 1.0)
    :param ic_max: Maximum value for soil behaviour type index used in the optimisation routine (:math:`I_{c,max}`) [:math:`-`] (optional, default= 4.0)
    :param zhang_multiplier_1: First multiplier in the equation for exponent n (:math:``) [:math:`-`] (optional, default= 0.381)
    :param zhang_multiplier_2: Second multiplier in the equation for exponent n (:math:``) [:math:`-`] (optional, default= 0.05)
    :param zhang_subtraction: Term subtracted in the equation for exponent n (:math:``) [:math:`-`] (optional, default= 0.15)
    :param robertsonwride_coefficient1: First coefficient in the equation by Robertson and Wride (:math:``) [:math:`-`] (optional, default= 3.47)
    :param robertsonwride_coefficient2: Second coefficient in the equation by Robertson and Wride (:math:``) [:math:`-`] (optional, default= 1.22)

    .. math::
        Q_{tn} = \\frac{q_t - \\sigma_{vo}}{P_a} \\left( \\frac{P_a}{\\sigma_{vo}^{\\prime}} \\right)^n
        \\\\
        n = 0.381 \\cdot I_c + 0.05 \\cdot \\frac{\\sigma_{vo}^{\\prime}}{P_a} - 0.15 \\ \\text{where} \\ n \\leq 1
        \\\\
        I_c = \\sqrt{ \\left( 3.47 - \\log_{10} Q_{tn} \\right)^2 + \\left( \\log_{10} F_r + 1.22 \\right)^2 }

    :returns: Dictionary with the following keys:

        - 'exponent_zhang [-]': Exponent n according to Zhang et al (:math:`n`)  [:math:`-`]
        - 'Qtn [-]': Normalised cone resistance (:math:`Q_{tn}`)  [:math:`-`]
        - 'Fr [%]': Normalised friction ratio (:math:`F_r`)  [:math:`%`]
        - 'Ic [-]': Soil behaviour type index (:math:`I_c`)  [:math:`-`]
        - 'Ic class number [-]': Soil behaviour type class number according to the Robertson chart
        - 'Ic class': Soil behaviour type class description according to the Robertson chart

    .. figure:: images/behaviourindex_pcpt_robertsonwride_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Contour lines for soil behaviour type index

    Reference - Fugro guidance on PCPT interpretation

    """

    def Qtn(qt, sigma_vo, sigma_vo_eff, n, pa=0.001 * atmospheric_pressure):
        return ((qt - 0.001 * sigma_vo) / pa) * ((pa / (0.001 * sigma_vo_eff)) ** n)

    def Fr(fs, qt, sigma_vo):
        return 100 * fs / (qt - 0.001 * sigma_vo)

    def exponent_zhang(ic, sigma_vo_eff, pa=atmospheric_pressure):
        return min(1, zhang_multiplier_1 * ic + zhang_multiplier_2 * (sigma_vo_eff / pa) - zhang_subtraction)

    def soilbehaviourtypeindex(qt, fr):
        return np.sqrt((robertsonwride_coefficient1 - np.log10(qt)) ** 2 +
                       (np.log10(fr) + robertsonwride_coefficient2) ** 2)

    def rootfunction(ic, qt, fs, sigma_vo, sigma_vo_eff):
        _fr = Fr(fs, qt, sigma_vo)
        _n = exponent_zhang(ic, sigma_vo_eff)
        _qtn = Qtn(qt, sigma_vo, sigma_vo_eff, _n)
        return ic - soilbehaviourtypeindex(_qtn, _fr)

    _Ic = brentq(rootfunction, ic_min, ic_max, args=(qt, fs, sigma_vo, sigma_vo_eff))
    _exponent_zhang = exponent_zhang(_Ic, sigma_vo_eff)
    _Qtn = Qtn(qt, sigma_vo, sigma_vo_eff, _exponent_zhang)
    _Fr = Fr(fs, qt, sigma_vo)

    if _Ic < 1.31:
        _Ic_class_number = 7,
        _Ic_class = "Gravelly sand to sand"
    elif 1.31 <= _Ic < 2.05:
        _Ic_class_number = 6
        _Ic_class = "Sands: clean sands to silty sands"
    elif 2.05 <= _Ic < 2.6:
        _Ic_class_number = 5
        _Ic_class = "Sand mixtures: silty sand to sand silty"
    elif 2.6 <= _Ic < 2.95:
        _Ic_class_number = 4
        _Ic_class = "Silt mixtures: clayey silt to silty clay"
    elif 2.95 <= _Ic < 3.6:
        _Ic_class_number = 3
        _Ic_class = "Clays: clay to silty clay"
    else:
        _Ic_class_number = 2
        _Ic_class = "Organic soils-peats"

    return {
        'exponent_zhang [-]': _exponent_zhang,
        'Qtn [-]': _Qtn,
        'Fr [%]': _Fr,
        'Ic [-]': _Ic,
        'Ic class number [-]': _Ic_class_number,
        'Ic class': _Ic_class
    }


GMAX_SAND_RIXSTOKOE = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'qc_exponent': {'type': 'float', 'min_value': None, 'max_value': None},
    'stress_exponent': {'type': 'float', 'min_value': None, 'max_value': None},
}

GMAX_SAND_RIXSTOKOE_ERRORRETURN = {
    'Gmax [kPa]': np.nan,
}


@Validator(GMAX_SAND_RIXSTOKOE, GMAX_SAND_RIXSTOKOE_ERRORRETURN)
def gmax_sand_rixstokoe(
        qc, sigma_vo_eff,
        multiplier=1634.0, qc_exponent=0.25, stress_exponent=0.375, **kwargs):
    """
    Calculates the small-strain shear modulus for uncemented silica sand based on cone resistance and vertical effective stress. The correlation is based on calibration chamber tests compared to results from PCPT, S-PCPT and cross-hole tests reported by Baldi et al (1989).

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param multiplier: Multiplier in the correlation equation (:math:``) [:math:`-`] (optional, default= 1634.0)
    :param qc_exponent: Exponent applied on the cone tip resistance (:math:``) [:math:`-`] (optional, default= 0.25)
    :param stress_exponent: Exponent applied on the vertical effective stress (:math:``) [:math:`-`] (optional, default= 0.375)

    .. math::
        G_{max} = 1634 \\cdot (q_c)^{0.25} \\cdot (\\sigma_{vo}^{\\prime})^{0.375}

    :returns: Dictionary with the following keys:

        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Rix, G.J. and Stokoe, K.H. (II) (1991), “Correlation of Initial Tangent Modulus and Cone Penetration Resistance”, in Huang, A.B. (Ed.), Calibration Chamber Testing: Proceedings of the First International Symposium on Calibration Chamber Testing ISOCCTI, Potsdam, New York, 28-29 June 1991, Elsevier Science Publishing Company, New York, pp. 351-362.

    """

    _Gmax = multiplier * ((1000 * qc) ** qc_exponent) * (sigma_vo_eff ** stress_exponent)

    return {
        'Gmax [kPa]': _Gmax,
    }


GMAX_CLAY_MAYNERIX = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent': {'type': 'float', 'min_value': None, 'max_value': None},
}

GMAX_CLAY_MAYNERIX_ERRORRETURN = {
    'Gmax [kPa]': np.nan,
}


@Validator(GMAX_CLAY_MAYNERIX, GMAX_CLAY_MAYNERIX_ERRORRETURN)
def gmax_clay_maynerix(
        qc,
        multiplier=2.78, exponent=1.335, **kwargs):
    """
    Mayne and Rix (1993) determined a relationship between small-strain shear modulus and cone tip resistance by studying 481 data sets from 31 sites all over the world. Gmax ranged between about 0.7 MPa and 800 MPa.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param multiplier: Multiplier in the equation (:math:``) [:math:`-`] (optional, default= 2.78)
    :param exponent: Exponent in the equation (:math:``) [:math:`-`] (optional, default= 1.335)

    .. math::
        G_{max} = 2.78 \\cdot q_c^{1.335}

    :returns: Dictionary with the following keys:

        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Mayne, P.W. and Rix, G.J. (1993), “Gmax-qc Relationships for Clays”, Geotechnical Testing Journal, Vol. 16, No. 1, pp. 54-60.

    """

    _Gmax = multiplier * ((1000 * qc) ** exponent)

    return {
        'Gmax [kPa]': _Gmax,
    }


RELATIVEDENSITY_NCSAND_BALDI = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'coefficient_0': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

RELATIVEDENSITY_NCSAND_BALDI_ERRORRETURN = {
    'Dr [-]': np.nan,
}


@Validator(RELATIVEDENSITY_NCSAND_BALDI, RELATIVEDENSITY_NCSAND_BALDI_ERRORRETURN)
def relativedensity_ncsand_baldi(
        qc, sigma_vo_eff,
        coefficient_0=157.0, coefficient_1=0.55, coefficient_2=2.41, **kwargs):
    """
    Calculates the relative density for normally consolidated sand based on calibration chamber tests on silica sand. It should be noted that this correlation provides an approximative estimate of relative density and the sand at the site should be compared to the sands used in the calibration chamber tests. The correlation will always be sensitive to variations in compressibility and horizontal stress.

    :param qc: Cone tipe resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param coefficient_0: Coefficient C0 (:math:`C_0`) [:math:`-`] (optional, default= 157.0)
    :param coefficient_1: Coefficient C1 (:math:`C_1`) [:math:`-`] (optional, default= 0.55)
    :param coefficient_2: Coefficient C2 (:math:`C_2`) [:math:`-`] (optional, default= 2.41)

    .. math::
        D_r = \\frac{1}{2.41} \\cdot \\ln \\left[ \\frac{q_c}{157 \\cdot \\left( \\sigma_{vo}^{\\prime} \\right)^{0.55} } \\right]

    :returns: Dictionary with the following keys:

        - 'Dr [-]': Relative density as a number between 0 and 1 (:math:`D_r`)  [:math:`-`]

    .. figure:: images/relativedensity_ncsand_baldi_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Relationship between cone tip resistance, vertical effective stress and relative density for normally consolidated Ticino sand

    Reference - Baldi et al 1986.

    """

    _Dr = (1 / coefficient_2) * np.log((1000 * qc) / (coefficient_0 * (sigma_vo_eff ** coefficient_1)))

    return {
        'Dr [-]': _Dr,
    }


RELATIVEDENSITY_OCSAND_BALDI = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'k0': {'type': 'float', 'min_value': 0.3, 'max_value': 5.0},
    'coefficient_0': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

RELATIVEDENSITY_OCSAND_BALDI_ERRORRETURN = {
    'Dr [-]': np.nan,
}


@Validator(RELATIVEDENSITY_OCSAND_BALDI, RELATIVEDENSITY_OCSAND_BALDI_ERRORRETURN)
def relativedensity_ocsand_baldi(
        qc, sigma_vo_eff, k0,
        coefficient_0=181.0, coefficient_1=0.55, coefficient_2=2.61, **kwargs):
    """
    Calculates the relative density for overconsolidated sand based on calibration chamber tests on silica sand. It should be noted that this correlation provides an approximative estimate of relative density and the sand at the site should be compared to the sands used in the calibration chamber tests. The correlation will always be sensitive to variations in compressibility and horizontal stress. Note that this correlation requires an estimate of the coefficient of lateral earth pressure.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param k0: Coefficient of lateral earth pressure (:math:`K_o`) [:math:`-`] - Suggested range: 0.3 <= k0 <= 5.0
    :param coefficient_0: Coefficient C0 (:math:`C_0`) [:math:`-`] (optional, default= 181.0)
    :param coefficient_1: Coefficient C1 (:math:`C_1`) [:math:`-`] (optional, default= 0.55)
    :param coefficient_2: Coefficient C2 (:math:`C_2`) [:math:`-`] (optional, default= 2.61)

    .. math::
        D_r = \\frac{1}{2.61} \\cdot \\ln \\left[ \\frac{q_c}{181 \\cdot \\left( \\sigma_{m}^{\\prime} \\right)^{0.55} } \\right]

        \\sigma_{m}^{\\prime} = \\frac{\\sigma_{vo}^{\\prime} + 2 \\cdot K_o \\ cdot \\sigma_{m}^{\\prime}}{3}

    :returns: Dictionary with the following keys:

        - 'Dr [-]': Relative density as a number between 0 and 1 (:math:`D_r`)  [:math:`-`]

    .. figure:: images/relativedensity_ocsand_baldi_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Relationship between cone tip resistance, vertical effective stress and relative density for overconsolidated Ticino sand

    Reference - Baldi et al 1986.

    """

    _sigma_m_eff = (1 / 3) * (sigma_vo_eff + 2 * k0 * sigma_vo_eff)
    _Dr = (1 / coefficient_2) * np.log((1000 * qc) / (coefficient_0 * (_sigma_m_eff ** coefficient_1)))

    return {
        'Dr [-]': _Dr,
    }

CONERESISTANCE_OCSAND_BALDI = {
    'dr': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'k0': {'type': 'float', 'min_value': 0.3, 'max_value': 5.0},
    'coefficient_0': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

CONERESISTANCE_OCSAND_BALDI_ERRORRETURN = {
    'qc [MPa]': np.nan,
}


@Validator(CONERESISTANCE_OCSAND_BALDI, CONERESISTANCE_OCSAND_BALDI_ERRORRETURN)
def coneresistance_ocsand_baldi(
        dr, sigma_vo_eff, k0,
        coefficient_0=181.0, coefficient_1=0.55, coefficient_2=2.61, **kwargs):
    """
    Calculates the cone resistance for a given relative density for overconsolidated sand based on calibration chamber tests on silica sand.
    It should be noted that this correlation provides an approximative estimate of relative density and the sand at the site should be compared to the sands used in the calibration chamber tests.
    The correlation will always be sensitive to variations in compressibility and horizontal stress.
    Note that this correlation requires an estimate of the coefficient of lateral earth pressure.

    :param dr: Relative density (:math:`D_r`) [:math:`-`] - Suggested range: 0.0 <= dr <= 1.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param k0: Coefficient of lateral earth pressure (:math:`K_o`) [:math:`-`] - Suggested range: 0.3 <= k0 <= 5.0
    :param coefficient_0: Coefficient C0 (:math:`C_0`) [:math:`-`] (optional, default= 181.0)
    :param coefficient_1: Coefficient C1 (:math:`C_1`) [:math:`-`] (optional, default= 0.55)
    :param coefficient_2: Coefficient C2 (:math:`C_2`) [:math:`-`] (optional, default= 2.61)

    .. math::
        D_r = \\frac{1}{2.61} \\cdot \\ln \\left[ \\frac{q_c}{181 \\cdot \\left( \\sigma_{m}^{\\prime} \\right)^{0.55} } \\right]

        \\sigma_{m}^{\\prime} = \\frac{\\sigma_{vo}^{\\prime} + 2 \\cdot K_o \\ cdot \\sigma_{m}^{\\prime}}{3}

    :returns: Dictionary with the following keys:

        - 'qc [MPa]': Cone resistance corresponding to the given relative density (:math:`q_c`)  [:math:`MPa`]

    Reference - Baldi et al 1986.

    """

    _sigma_m_eff = (1 / 3) * (sigma_vo_eff + 2 * k0 * sigma_vo_eff)

    _qc = 0.001 * np.exp(dr / (1 / coefficient_2)) * (coefficient_0 * (_sigma_m_eff ** coefficient_1))

    return {
        'qc [MPa]': _qc,
    }


RELATIVEDENSITY_SAND_JAMIOLKOWSKI = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 50.0, 'max_value': 400.0},
    'k0': {'type': 'float', 'min_value': 0.4, 'max_value': 1.5},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_4': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_5': {'type': 'float', 'min_value': None, 'max_value': None},
}

RELATIVEDENSITY_SAND_JAMIOLKOWSKI_ERRORRETURN = {
    'Dr dry [-]': np.nan,
    'Dr sat [-]': np.nan,
}


@Validator(RELATIVEDENSITY_SAND_JAMIOLKOWSKI, RELATIVEDENSITY_SAND_JAMIOLKOWSKI_ERRORRETURN)
def relativedensity_sand_jamiolkowski(
        qc, sigma_vo_eff, k0,
        atmospheric_pressure=100.0, coefficient_1=2.96, coefficient_2=24.94, coefficient_3=0.46, coefficient_4=-1.87,
        coefficient_5=2.32, **kwargs):
    """
    Jamiolkowksi et al formulated a correlation for the relative density of dry sand based on calibration chamber tests. The correlation can be modified for saturated sands and results in relative densities which can be up to 10% higher.
    Note that calibration chamber testing is carried out on sands with vertical effective stress between 50kPa and 400kPa and coefficients of lateral earth pressure Ko between 0.4 and 1.5. Relative densities for stress conditions outside this range (e.g. shallow soils) should be assessed with care.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: 50.0 <= sigma_vo_eff <= 400.0
    :param k0: Coefficient of lateral earth pressure (:math:`K_o`) [:math:`-`] - Suggested range: 0.4 <= k0 <= 1.5
    :param atmospheric_pressure: Atmospheric pressure used for normalisation (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)
    :param coefficient_1: First calibration coefficient (:math:``) [:math:`-`] (optional, default= 2.96)
    :param coefficient_2: Second calibration coefficient (:math:``) [:math:`-`] (optional, default= 24.94)
    :param coefficient_3: Third calibration coefficient (:math:``) [:math:`-`] (optional, default= 0.46)
    :param coefficient_4: Fourth calibration coefficient (:math:``) [:math:`-`] (optional, default= -1.87)
    :param coefficient_5: Fifth calibration coefficient (:math:``) [:math:`-`] (optional, default= 2.32)

    .. math::
        D_{r,dry} = \\frac{1}{2.96} \\cdot \\ln \\left[ \\frac{q_c / P_a}{24.94 \\cdot \\left( \\frac{\\sigma_{m}^{\\prime}}{P_a} \\right)^{0.46} } \\right]

        D_{r,sat} = \\left(  \\frac{-1.87 + 2.32 \\cdot \\ln \\left[ \\frac{q_c}{\\sqrt{P_a + \\sigma_{vo}^{\\prime}}} \\right] }{100} \\right) \\cdot \\frac{D_{r,dry}}{100}

    :returns: Dictionary with the following keys:

        - 'Dr dry [-]': Relative density for dry sand as a number between 0 and 1 (:math:`D_{r,dry}`)  [:math:`-`]
        - 'Dr sat [-]': Relative density for saturated sand as a number between 0 and 1 (:math:`D_{r,sat}`)  [:math:`-`]

    Reference - Jamiolkowski, M., Lo Presti, D.C.F. and Manassero, M. (2003), "Evaluation of Relative Density and Shear Strength of Sands from CPT and DMT", in Germaine, J.T., Sheahan, T.C. and Whitman, R.V. (Eds.), Soil Behavior and Soft Ground Construction: Proceedings of the Symposium, October 5-6, 2001, Cambridge, Massachusetts, Geotechnical Special Publication, No. 119, American Society of Civil Engineers, Reston, pp. 201-238.

    """
    _sigma_m_eff = (1 / 3) * (sigma_vo_eff + 2 * k0 * sigma_vo_eff)
    _Dr_dry = (1 / coefficient_1) * np.log((1000 * qc / atmospheric_pressure) /
                                           (coefficient_2 * ((_sigma_m_eff / atmospheric_pressure) ** coefficient_3)))
    _Dr_sat = (((coefficient_4 + coefficient_5 * np.log(
        (1000 * qc) / (np.sqrt(atmospheric_pressure + sigma_vo_eff))
    )) / 100) + 1) * (_Dr_dry)

    return {
        'Dr dry [-]': _Dr_dry,
        'Dr sat [-]': _Dr_sat,
    }


FRICTIONANGLE_SAND_KULHAWYMAYNE = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

FRICTIONANGLE_SAND_KULHAWYMAYNE_ERRORRETURN = {
    'Phi [deg]': np.nan,
}


@Validator(FRICTIONANGLE_SAND_KULHAWYMAYNE, FRICTIONANGLE_SAND_KULHAWYMAYNE_ERRORRETURN)
def frictionangle_sand_kulhawymayne(
        qt, sigma_vo_eff,
        atmospheric_pressure=100.0, coefficient_1=17.6, coefficient_2=11.0, **kwargs):
    """
    Determines the friction angle for sand based on calibration chamber tests.

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param atmospheric_pressure: Atmospheric pressure used for normalisation (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)
    :param coefficient_1: First calibration coefficient (:math:``) [:math:`-`] (optional, default= 17.6)
    :param coefficient_2: Second calibration coefficient (:math:``) [:math:`-`] (optional, default= 11.0)

    .. math::
        \\varphi^{\\prime} = 17.6 + 11.0 \\cdot \\log_{10} \\left[  \\frac{q_t / P_a}{ \\sqrt{\\sigma_{vo}^{\\prime} / P_a}} \\right]

    :returns: Dictionary with the following keys:

        - 'Phi [deg]': Effective friction angle for sand (:math:`\\varphi`)  [:math:`deg`]

    Reference - Kulhawy, F.H. and Mayne, P.H. (1990), “Manual on Estimating Soil Properties for Foundation Design”, Electric Power Research Institute EPRI, Palo Alto, EPRI Report, EL-6800.

    """

    _phi = coefficient_1 + coefficient_2 * np.log10(
        (1000 * qt / atmospheric_pressure) / (np.sqrt(sigma_vo_eff / atmospheric_pressure))
    )

    return {
        'Phi [deg]': _phi,
    }


UNDRAINEDSHEARSTRENGTH_CLAY_RADLUNNE = {
    'qnet': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'Nk': {'type': 'float', 'min_value': 8.0, 'max_value': 30.0},
}

UNDRAINEDSHEARSTRENGTH_CLAY_RADLUNNE_ERRORRETURN = {
    'Su [kPa]': np.nan,
}


@Validator(UNDRAINEDSHEARSTRENGTH_CLAY_RADLUNNE, UNDRAINEDSHEARSTRENGTH_CLAY_RADLUNNE_ERRORRETURN)
def undrainedshearstrength_clay_radlunne(
        qnet, Nk,
        **kwargs):
    """
    Calculates the undrained shear strength of clay from net cone tip resistance. The correlation is empirical and the cone factor needs to be adjusted to fit CIU or other high-quality laboratory tests for undrained shear strength.

    :param qnet: Net cone resistance (corrected for area ratio and total stress at the depth of the cone) (:math:`q_{net}`) [:math:`MPa`] - Suggested range: 0.0 <= qnet <= 120.0
    :param Nk: Empirical factor (:math:`N_k`) [:math:`-`] - Suggested range: 8.0 <= Nk <= 30.0

    .. math::
        S_u = \\frac{q_{net}}{N_k}

    :returns: Dictionary with the following keys:

        - 'Su [kPa]': Undrained shear strength inferred from PCPT data (:math:`S_u`)  [:math:`kPa`]

    Reference - Rad, N.S. and Lunne, T. (1988), "Direct Correlations between Piezocone Test Results and Undrained Shear Strength of Clay", in De Ruiter, J. (Ed.), Penetration Testing 1988: Proceedings of the First International Symposium on Penetration Testing, ISOPT-1, Orlando, 20-24 March 1988, Vol. 2, A.A. Balkema, Rotterdam, pp. 911-917.

    """

    _Su = 1000 * qnet / Nk

    return {
        'Su [kPa]': _Su,
    }


FRICTIONANGLE_OVERBURDEN_KLEVEN = {
    'sigma_vo_eff': {'type': 'float', 'min_value': 10.0, 'max_value': 800.0},
    'relative_density': {'type': 'float', 'min_value': 40.0, 'max_value': 100.0},
    'Ko': {'type': 'float', 'min_value': 0.3, 'max_value': 2.0},
    'max_friction_angle': {'type': 'float', 'min_value': None, 'max_value': None},
}

FRICTIONANGLE_OVERBURDEN_KLEVEN_ERRORRETURN = {
    'phi [deg]': np.NaN,
    'sigma_m [kPa]': np.NaN
}

@Validator(FRICTIONANGLE_OVERBURDEN_KLEVEN, FRICTIONANGLE_OVERBURDEN_KLEVEN_ERRORRETURN)
def frictionangle_overburden_kleven(sigma_vo_eff, relative_density, Ko=0.5, max_friction_angle=45.0,
                                    **kwargs):
    """
    This function calculates the friction angle according to the chart proposed by Kleven (1986). The function takes into account the effective confining pressure of the sand and its relative density. The function was calibrated on North Sea sand tests with confining pressures ranging from 10 to 800kPa. Lower confinement clearly leads to higher friction angles. The fit to the data is not excellent and this function should be compared to site-specific testing or other correlations.


    :param sigma_vo_eff: Effective vertical stress (:math:`\\sigma \\prime _{vo}`) [:math:`kPa`]  - Suggested range: 10.0<=sigma_vo_eff<=800.0
    :param relative_density: Relative density of sand (:math:`D_r`) [:math:`Percent`]  - Suggested range: 40.0<=relative_density<=100.0
    :param Ko: Coefficient of lateral earth pressure at rest (:math:`K_o`) [:math:`-`] (optional, default=0.5) - Suggested range: 0.3<=Ko<=2.0
    :param max_friction_angle: The maximum allowable effective friction angle (:math:`\\phi \\prime _{max}`) [:math:`deg`] (optional, default=45.0)

    :returns:   Peak drained friction angle (:math:`\\phi_d`) [:math:`deg`], Mean effective stress (:math:`\\sigma \\prime _m`) [:math:`kPa`]

    :rtype: Python dictionary with keys ['phi [deg]','sigma_m [kPa]']

    .. figure:: images/Phi_Kleven.png
        :figwidth: 500
        :width: 400
        :align: center

        Data and interpretation chart according to Kleven (Lunne et al (1997))

    Reference - Lunne, T., Robertson, P.K., Powell, J.J.M. (1997). Cone penetration testing in geotechnical practice.  SPON press

    Examples:
        .. code-block:: python

            >>>phi = friction_angle_kleven(sigma_vo_eff=100.0,relative_density=60.0,Ko=1.0)['phi [deg]']
            35.8

    """
    sigma_m = ((1.0 + 2.0 * Ko) / 3.0) * sigma_vo_eff

    if relative_density > 100.0:
        relative_density = 100.0

    if sigma_m < 10.0:
        phi = 0.2183 * relative_density + 25.667
    elif sigma_m >= 10.0 and sigma_m < 25.0:
        phi1 = 0.2183 * relative_density + 25.667
        phi2 = 0.2175 * relative_density + 24.75
        phi = phi1 + ((phi2 - phi1) / (25.0 - 10.0)) * (sigma_m - 10.0)
    elif sigma_m >= 25.0 and sigma_m < 50.0:
        phi1 = 0.2175 * relative_density + 24.75
        phi2 = 0.22 * relative_density + 23.5
        phi = phi1 + ((phi2 - phi1) / (50.0 - 25.0)) * (sigma_m - 25.0)
    elif sigma_m >= 50.0 and sigma_m < 100.0:
        phi1 = 0.22 * relative_density + 23.5
        phi2 = 0.2175 * relative_density + 22.75
        phi = phi1 + ((phi2 - phi1) / (100.0 - 50.0)) * (sigma_m - 50.0)
    elif sigma_m >= 100.0 and sigma_m < 200.0:
        phi1 = 0.2175 * relative_density + 22.75
        phi2 = 0.2 * relative_density + 23.0
        phi = phi1 + ((phi2 - phi1) / (200.0 - 100.0)) * (sigma_m - 100.0)
    elif sigma_m >= 200.0 and sigma_m < 400.0:
        phi1 = 0.2 * relative_density + 23
        phi2 = 0.1925 * relative_density + 22.75
        phi = phi1 + ((phi2 - phi1) / (400.0 - 200.0)) * (sigma_m - 200.0)
    elif sigma_m >= 400.0 and sigma_m < 800.0:
        phi1 = 0.1925 * relative_density + 22.75
        phi2 = 0.195 * relative_density + 21.3
        phi = phi1 + ((phi2 - phi1) / (800.0 - 400.0)) * (sigma_m - 400.0)

    phi = min(phi, max_friction_angle)

    return {
        'phi [deg]': phi,
        'sigma_m [kPa]': sigma_m,
    }


OCR_CPT_LUNNE = {
    'Qt': {'type': 'float', 'min_value': 2.0, 'max_value': 34.0},
    'Bq': {'type': 'float', 'min_value': 0.0, 'max_value': 1.4},
}

OCR_CPT_LUNNE_ERRORRETURN = {
    'OCR_Qt_LE [-]': np.nan,
    'OCR_Qt_BE [-]': np.nan,
    'OCR_Qt_HE [-]': np.nan,
    'OCR_Bq_LE [-]': np.nan,
    'OCR_Bq_BE [-]': np.nan,
    'OCR_Bq_HE [-]': np.nan,
}


@Validator(OCR_CPT_LUNNE, OCR_CPT_LUNNE_ERRORRETURN)
def ocr_cpt_lunne(Qt, Bq=np.nan, **kwargs):
    """
    Calculates the overconsolidation ratio (OCR) for clay based on normalised CPT properties. A low estimate, best estimate and high estimate of OCR is provided. The data is based on testing of high-quality undisturbed samples by the Norwegian Geotechnical Institute.

    Both normalised cone resistance Qt and pore pressure ratio Bq can be used as inputs. If only one of the two inputs is specified, NaN is returned for the other.

    The implementation of the formulation is based on digitisation of the graphs.

    :param Qt: Normalised cone resistance (:math:`Q_t`) [:math:`-`] - Suggested range: 2.0 <= Qt <= 34.0
    :param Bq: Pore pressure ratio (:math:`B_q`) [:math:`-`] - Suggested range: 0.0 <= Bq <= 1.4 (optional, default=None)

    :returns: Dictionary with the following keys:

        - 'OCR_Qt_LE [-]': Low estimate OCR based on Qt (:math:`OCR_{Q_t,LE}`)  [:math:`-`]
        - 'OCR_Qt_BE [-]': Best estimate OCR based on Qt (:math:`OCR_{Q_t,BE}`)  [:math:`-`]
        - 'OCR_Qt_HE [-]': High estimate OCR based on Qt (:math:`OCR_{Q_t,HE}`)  [:math:`-`]
        - 'OCR_Bq_LE [-]': Low estimate OCR based on Bq (:math:`OCR_{B_q,LE}`)  [:math:`-`]
        - 'OCR_Bq_BE [-]': Best estimate OCR based on Bq (:math:`OCR_{B_q,BE}`)  [:math:`-`]
        - 'OCR_Bq_HE [-]': High estimate OCR based on Bq (:math:`OCR_{B_q,HE}`)  [:math:`-`]

    .. figure:: images/ocr_cpt_lunne.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Data used for correlations according to Lunne et al

    Reference - Lunne, T., Robertson, P.K., Powell, J.J.M., 1997. Cone penetration testing in geotechnical practice. E & FN Spon.

    """

    _OCR_Qt_HE = 10 ** (np.interp(
        Qt,
        [
            2.118188717, 2.955763555, 4.261775866, 5.212142676, 6.987132136, 9.588363268,
            13.01367738, 18.08607808, 23.27342946, 27.27943334, 31.99025213, 33.87371616],
        [
            0.00599007, 0.161314993, 0.292855171, 0.394488883, 0.508212442, 0.651930234,
            0.801769746, 0.95191644, 1.072244436, 1.138639862, 1.193230681, 1.205518004
        ]))
    _OCR_Qt_BE = 10 ** (np.interp(
        Qt,
        [
            3.294139472, 4.721578548, 6.379458639, 8.389746184, 11.22465638, 14.29432499,
            17.36237457, 20.19404672, 24.08456025, 27.62050763, 31.15537566, 33.86508137
        ],
        [
            0.000241358, 0.173580374, 0.29325012, 0.407017563, 0.532874854, 0.6528079,
            0.754836561, 0.844885083, 0.93513108, 1.007406867, 1.067746398, 1.110027953
        ]))
    _OCR_Qt_LE = 10 ** (np.interp(
        Qt,
        [
            4.705927987, 6.483615819, 8.494443039, 10.50580993, 13.10272367, 16.05258453,
            19.4725019, 22.77369087, 26.42620794, 29.60758917, 34.08526857
        ],
        [
            0.000504658, 0.144068858, 0.263804429, 0.389508129, 0.485480895, 0.581519487,
            0.671677717, 0.749877749, 0.810239222, 0.8645448, 0.942964248
        ]))

    if np.math.isnan(Bq):
        _OCR_Bq_LE = np.nan
        _OCR_Bq_BE = np.nan
        _OCR_Bq_HE = np.nan
    else:
        _OCR_Bq_LE = 10 ** (np.interp(
            Bq,
            [
                0.075352962, 0.188834333, 0.276555837, 0.379904518, 0.46248832,
                0.586554308, 0.715805569, 0.855570091
            ],
            [
                0.692612411, 0.54389548, 0.436844931, 0.347765081, 0.252628686,
                0.157669124, 0.062731666, 0.00364719
            ]
        ))
        _OCR_Bq_BE = 10 ** (np.interp(
            Bq,
            [
                0.102064255, 0.184481558, 0.251580897, 0.354763078, 0.468363377,
                0.602728555, 0.726770757, 0.866392565, 1.00608573
            ],
            [
                0.889671185, 0.752757888, 0.675459566, 0.544602814, 0.425726528,
                0.312906788, 0.211979097, 0.117085847, 0.040096985
            ]
        ))
        _OCR_Bq_HE = 10 ** (np.interp(
            Bq,
            [
                0.164882176, 0.216425695, 0.293705296, 0.438155592, 0.577587115,
                0.717089995, 0.898027489, 1.037673083
            ],
            [
                1.039139658, 0.961775024, 0.83677588, 0.652382801, 0.50974452,
                0.385010626, 0.248517308, 0.159592188
            ]
        ))

    return {
        'OCR_Qt_LE [-]': _OCR_Qt_LE,
        'OCR_Qt_BE [-]': _OCR_Qt_BE,
        'OCR_Qt_HE [-]': _OCR_Qt_HE,
        'OCR_Bq_LE [-]': _OCR_Bq_LE,
        'OCR_Bq_BE [-]': _OCR_Bq_BE,
        'OCR_Bq_HE [-]': _OCR_Bq_HE,
    }


SENSITIVITY_FRICTIONRATIO_LUNNE = {
    'Rf': {'type': 'float', 'min_value': 0.5, 'max_value': 2.2},
}

SENSITIVITY_FRICTIONRATIO_LUNNE_ERRORRETURN = {
    'St LE [-]': np.nan,
    'St BE [-]': np.nan,
    'St HE [-]': np.nan,
}


@Validator(SENSITIVITY_FRICTIONRATIO_LUNNE, SENSITIVITY_FRICTIONRATIO_LUNNE_ERRORRETURN)
def sensitivity_frictionratio_lunne(
        Rf,
        **kwargs):
    """
    Calculates the sensitivity of clay from the friction ratio according to Rad and Lunne (1986). The correlation is derived based on measurements on Norwegian clays.

    Ideally, the sleeve friction corrected for pore pressure effects should be used to calculate the friction ratio but if this is not available (when pore pressures are not measured on both ends of the friction sleeve), the ratio of sleeve friction to cone tip resistance (in percent) can be used.

    The function returns a low estimate, best estimate and high estimate value.

    :param Rf: Friction ratio (:math:`R_f = f_t / q_t`) [:math:`percent`] - Suggested range: 0.5 <= Rf <= 2.2

    :returns: Dictionary with the following keys:

        - 'St LE [-]': Low estimate sensitivity (:math:`S_{t,LE}`)  [:math:`-`]
        - 'St BE [-]': Best estimate sensitivity (:math:`S_{t,BE}`)  [:math:`-`]
        - 'St HE [-]': High estimate sensitivity (:math:`S_{t,HE}`)  [:math:`-`]

    .. figure:: images/sensitivity_frictionratio_lunne_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Data used to derive correlation according to Rad & Lunne (1986)

    Reference - Lunne, T., Robertson, P.K., Powell, J.J.M., 1997. Cone penetration testing in geotechnical practice. E & FN Spon.

    """

    _St_LE = np.interp(
        Rf,
        [
            0.562740471,
            0.611428435,
            0.682345206,
            0.757699952,
            0.850826065,
            0.952873546,
            1.050502516,
            1.152588927,
            1.26358373,
            1.36570907,
            1.490050241,
            1.60551546,
            1.747640974,
            1.858687683,
            1.929747195
        ],
        [
            9.249026219,
            8.570904876,
            7.91500585,
            7.237273803,
            6.537903381,
            5.948152249,
            5.445927855,
            4.987564153,
            4.595023931,
            4.268047658,
            3.919497895,
            3.614614175,
            3.288221847,
            3.070864865,
            2.896719748
        ]
    )
    _St_BE = np.interp(
        Rf,
        [
            0.714072837,
            0.793839606,
            0.878044349,
            0.993334386,
            1.104257818,
            1.224063689,
            1.339464026,
            1.463779243,
            1.605846362,
            1.756841338,
            1.885626972,
            2.045504387
        ],
        [
            9.995760998,
            9.208604309,
            8.399614597,
            7.503487444,
            6.870070268,
            6.214884952,
            5.691022183,
            5.2548808,
            4.731407327,
            4.339451049,
            3.990966168,
            3.577241751
        ]
    )
    _St_HE = np.interp(
        Rf,
        [
            0.825586702,
            0.896490496,
            0.958505363,
            1.042755524,
            1.140358542,
            1.233478166,
            1.353297013,
            1.464226934,
            1.588516198,
            1.72615183,
            1.846016095,
            1.970344289,
            2.099116947,
            2.23678502
        ],
        [
            11.35505317,
            10.65535834,
            9.955533736,
            9.299829359,
            8.710013344,
            7.988745018,
            7.377355512,
            6.76583624,
            6.242103237,
            5.762360691,
            5.30425652,
            4.911910946,
            4.519630255,
            4.149377234
        ]
    )

    return {
        'St LE [-]': _St_LE,
        'St BE [-]': _St_BE,
        'St HE [-]': _St_HE,
    }


UNITWEIGHT_MAYNE = {
    'ft': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': 500.0},
    'unitweight_water': {'type': 'float', 'min_value': 9.0, 'max_value': 11.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

UNITWEIGHT_MAYNE_ERRORRETURN = {
    'gamma [kN/m3]': np.nan,
}


@Validator(UNITWEIGHT_MAYNE, UNITWEIGHT_MAYNE_ERRORRETURN)
def unitweight_mayne(
        ft, sigma_vo_eff,
        unitweight_water=10.25, atmospheric_pressure=100.0, coefficient_1=1.95, exponent_1=0.06, exponent_2=0.06,
        **kwargs):
    """
    Estimates the total unit weight for sand, clay and silt from CPT measurements. A correlation with sleeve friction and vertical effective stress showed the best fit across a range of soil types. The correlation does not apply for cemented soils. An error band of +-2kN/m3 seems to encompass the data rather well.

    For the sake of accuracy, the corrected total sleeve friction is used instead of the uncorrected sleeve friction. PCPT normalisation is required before applying the correlation. If sleeve dimensions are not available, the uncorrected sleeve friction will be used.

    :param ft: Total sleeve friction (:math:`f_t`) [:math:`MPa`] - Suggested range: 0.0 <= ft <= 10.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_vo_eff <= 500.0
    :param unitweight_water: Unit weight of water (:math:`\\gamma_w`) [:math:`kN/m3`] - Suggested range: 9.0 <= unitweight_water <= 11.0 (optional, default= 10.25)
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)
    :param coefficient_1: First coefficient in the calibrated equation (:math:``) [:math:`-`] (optional, default= 1.95)
    :param exponent_1: First exponent in the calibrated equation (:math:``) [:math:`-`] (optional, default= 0.06)
    :param exponent_2: Second exponent in the calibrated equation (:math:``) [:math:`-`] (optional, default= 0.06)

    .. math::
        \\gamma = 1.95 \\cdot \\gamma_w \\cdot \\left( \\frac{\\sigma_{vo}^{\\prime}}{P_a} \\right)^{0.06} \\cdot \\left( \\frac{f_t}{P_a} \\right)^{0.06}

    :returns: Dictionary with the following keys:

        - 'gamma [kN/m3]': Total unit weight (:math:`\\gamma`)  [:math:`kN/m3`]

    .. figure:: images/unitweight_mayne_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Calibration with soil data used

    Reference - P.W. Mayne ; J. Peuchen ; D. Bouwmeester (2010). Soil unit weight estimation from CPTs - 2nd International Symposium on Cone Penetration Testing, Huntington Beach, CA, USA. Volume 2&3: Technical Papers, Session 2: Interpretation, Paper No. 5

    """

    _gamma = coefficient_1 * unitweight_water * \
             ((1000 * ft / atmospheric_pressure) ** exponent_1) * \
             ((sigma_vo_eff / atmospheric_pressure) ** exponent_2)

    return {
        'gamma [kN/m3]': _gamma,
    }


VS_IC_ROBERTSONCABAL = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'ic': {'type': 'float', 'min_value': 1.0, 'max_value': 4.0},
    'sigma_vo': {'type': 'float', 'min_value': 0.0, 'max_value': 800.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
    'gamma': {'type': 'float', 'min_value': 12, 'max_value': 22},
    'g': {'type': 'float', 'min_value': 9.7, 'max_value': 10.2},
    'exponent': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'calibration_coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
}

VS_IC_ROBERTSONCABAL_ERRORRETURN = {
    'alpha_vs [-]': np.nan,
    'Vs [m/s]': np.nan,
    'Gmax [kPa]': np.nan
}


@Validator(VS_IC_ROBERTSONCABAL, VS_IC_ROBERTSONCABAL_ERRORRETURN)
def vs_ic_robertsoncabal(
        qt, ic, sigma_vo,
        atmospheric_pressure=100.0, gamma=19, g=9.81, exponent=0.5, calibration_coefficient_1=0.55, calibration_coefficient_2=1.68,
        **kwargs):
    """
    Calculates shear wave velocity based on a correlation with total cone resistance and soil behaviour type index. Shear wave velocity is sensitive to age and cementation, where older deposits of the same soil have higher shear wave velocity (i.e. higher stiffness) than younger deposits. The correlation is based on measured shear wave velocity data for uncemented Holocene to Pleistocene age soils.
    Since the small-strain shear modulus can be derived from the shear wave velocity and the bulk density of the soil, is it also calculated. The bulk density of the soil can be specified as an optional argument.

    Unfortunately, no plots on the background data to the calibrated equation are available.

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 100.0
    :param ic: Soil behaviour type index according to Robertson and Wride (:math:`I_c`) [:math:`-`] - Suggested range: 1.0 <= ic <= 4.0
    :param sigma_vo: Total vertical stress (:math:`sigma_{vo}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_vo <= 800.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] (optional, default= 100.0)
    :param gamma: Bulk unit weight (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 12.0 <= gamma <= 22.0
    :param g: Acceleration due to gravity (:math:`g`) [:math:`m/s2`] - Suggested range: 9.7 <= g <= 10.2 (optional, default= 9.81)
    :param exponent: Exponent in equation for shear wave velocity (:math:``) [:math:`-`] (optional, default= 0.5)
    :param calibration_coefficient_1: First calibration coefficient in equation for alpha_s (:math:``) [:math:`-`] (optional, default= 0.55)
    :param calibration_coefficient_2: Second calibration coefficient in equation for alpha_s (:math:``) [:math:`-`] (optional, default= 1.68)

    .. math::
        V_s = \\left[ \\alpha_{vs} (q_t - \\sigma_{vo}) / P_a \\right]^{0.5}

        \\alpha_{vs} = 10^{0.55 \\cdot I_c + 1.68}

        G_{max} = \\rho \\cdot V_s^2

        \\rho = \\gamma / g

    :returns: Dictionary with the following keys:

        - 'alpha_vs [-]': Coefficient to the shear wave velocity calculation, capturing the influence of the soil behaviour (:math:`\\alpha_{vs}`)  [:math:`-`]
        - 'Vs [m/s]': Shear wave velocity (:math:`V_s`)  [:math:`m/s`]
        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Robertson, P.K. and Cabal, K.L. (2015). Guide to Cone Penetration Testing for Geotechnical Engineering. 6th edition. Gregg Drilling & Testing, Inc.

    """

    _alpha_vs = 10 ** (calibration_coefficient_1 * ic + calibration_coefficient_2)
    _Vs = (_alpha_vs * ((1000 * qt - sigma_vo) / atmospheric_pressure)) ** exponent
    _rho = 1000 * gamma / g
    _Gmax = 1e-3 * _rho * (_Vs ** 2)

    return {
        'alpha_vs [-]': _alpha_vs,
        'Vs [m/s]': _Vs,
        'Gmax [kPa]': _Gmax
    }


K0_SAND_MAYNE = {
    'qt': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'ocr': {'type': 'float', 'min_value': 1.0, 'max_value': 20.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
    'multiplier': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'friction_angle': {'type': 'float', 'min_value': 25.0, 'max_value': 45.0},
}

K0_SAND_MAYNE_ERRORRETURN = {
    'K0 CPT [-]': np.nan,
    'K0 conventional [-]': np.nan,
    'Kp [-]': np.nan,
}


@Validator(K0_SAND_MAYNE, K0_SAND_MAYNE_ERRORRETURN)
def k0_sand_mayne(
        qt, sigma_vo_eff, ocr,
        atmospheric_pressure=100.0, multiplier=0.192, exponent_1=0.22, exponent_2=0.31, exponent_3=0.27,
        friction_angle=32.0, **kwargs):
    """
    Calculates the lateral coefficient of earth pressure at rest based on calibration chamber tests on clean sands.
    The values calculated from the equation need to be compared to values obtained using friction angle and OCR (see equations).

    :param qt: Total cone resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: 0.0 <= qt <= 100.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param ocr: Overconsolidation ratio (:math:`OCR`) [:math:`-`] - Suggested range: 1.0 <= ocr <= 20.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)
    :param multiplier: Multiplier in equation (:math:``) [:math:`-`] (optional, default= 0.192)
    :param exponent_1: First exponent in equation (:math:``) [:math:`-`] (optional, default= 0.22)
    :param exponent_2: Second exponent in equation (:math:``) [:math:`-`] (optional, default= 0.31)
    :param exponent_3: Third exponent in equation (:math:``) [:math:`-`] (optional, default= 0.27)
    :param friction_angle: Effective friction angle of the sand (:math:`\\varphi^{\\prime}`) [:math:`deg`] - Suggested range: 25.0 <= friction_angle <= 45.0 (optional, default= 32.0)

    .. math::
        K_0 = 0.192 \\cdot \\left( \\frac{q_t}{P_a} \\right)^{0.22} \\cdot \\left( \\frac{P_a}{\\sigma_{vo}^{\\prime}} \\right)^{0.31} \\cdot \\text{OCR}^{0.27}

        \\text{The maximum value for } K_0 \\text{ can be obtained as}:

        K_p = \\tan^2 \\left( \\frac{\\pi}{4} + \\frac{\\varphi^{\\prime}}{2} \\right) = \\frac{1 + \\sin \\varphi^{\\prime}}{1 - \\sin \\varphi^{\\prime}}

        \\text{These values need to be compared to}:

        K_0 = (1 - \\sin \\varphi^{\\prime}) \\cdot \\text{OCR} ^{\\sin \\varphi^{\\prime}}

    :returns: Dictionary with the following keys:

        - 'K0 CPT [-]': Coefficient of lateral earth pressure at rest derived from CPT (:math:`K_{0,CPT}`)  [:math:`-`]
        - 'K0 conventional [-]': Value derived from the conventional equation (:math:`K_{0,\\text{conventional}}`)  [:math:`-`]
        - 'Kp [-]': Limiting value of coefficient of lateral earth pressure based on Rankine passive earth pressure (:math:`K_p`)  [:math:`-`]

    .. figure:: images/k0_sand_mayne_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Dataset used for calibration

    Reference - Mayne (2007) NCHRP SYNTHESIS 368. Cone Penetration Testing. A Synthesis of Highway Practice.

    """

    _K0_CPT = multiplier * ((1000 * qt / atmospheric_pressure) ** exponent_1) * \
              ((atmospheric_pressure / sigma_vo_eff) ** exponent_2) * (ocr ** exponent_3)
    _Kp= (np.tan(0.25 * np.pi + 0.5 * np.radians(friction_angle))) ** 2
    _K0_conventional = (1 - np.sin(np.radians(friction_angle))) * (ocr ** (np.sin(np.radians(friction_angle))))

    return {
        'K0 CPT [-]': _K0_CPT,
        'K0 conventional [-]': _K0_conventional,
        'Kp [-]': _Kp,
    }


GMAX_CPT_PUECHEN = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 70.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'Bq': {'type': 'float', 'min_value': -0.2, 'max_value': 0.5},
    'coefficient_b': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_Bq': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_qc': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'Bq_min': {'type': 'float', 'min_value': None, 'max_value': None},
    'Bq_max': {'type': 'float', 'min_value': None, 'max_value': None},
}

GMAX_CPT_PUECHEN_ERRORRETURN = {
    'Gmax [kPa]': np.nan,
}


@Validator(GMAX_CPT_PUECHEN, GMAX_CPT_PUECHEN_ERRORRETURN)
def gmax_cpt_puechen(
        qc, sigma_vo_eff, Bq,
        coefficient_b=1.0, coefficient_Bq=4.0, multiplier_qc=1.634, exponent_1=0.25, exponent_2=0.375,
        Bq_min=0, Bq_max=0.5, **kwargs):
    """
    Calculates the small-strain modulus based on CPT data. The correlation by Rix and Stokoe is modified to include the importance of the pore pressure ratio.

    The calibration coefficient b has recommended values between 0.5 and 2, with a suggested best estimate of 1.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 70.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param Bq: Pore pressure ratio (:math:`B_q`) [:math:`-`] - Suggested range: -0.2 <= Bq <= 0.5
    :param coefficient_b: Calibration coefficient b (:math:`b`) [:math:`-`] (optional, default= 1.0)
    :param coefficient_Bq: Multiplier on Bq (:math:``) [:math:`-`] (optional, default= 4.0)
    :param multiplier_qc: Multiplier applied on qc (:math:``) [:math:`-`] (optional, default= 1.634)
    :param exponent_1: Exponent on qc (:math:``) [:math:`-`] (optional, default= 0.25)
    :param exponent_2: Exponent on vertical effective stress (:math:``) [:math:`-`] (optional, default= 0.375)
    :param Bq_min: Minimum value of Bq. If Bq is lower than this value, the minimum will be used for the calculation [:math:`-`] (optional, default= 0)
    :param Bq_max: Maximum value of Bq. If Bq is higher than this value, the maximum will be used for the calculation [:math:`-`] (optional, default= 0.5)

    .. math::
        G_{max} = b \\cdot \\left( 1 + 4 \\cdot B_q \\right) \\cdot 1.634 \\cdot q_c^{0.25} \\cdot \\sigma_{vo}^{\\prime \\ 0.375}

    :returns: Dictionary with the following keys:

        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Puechen et al (2020). Characteristic values for geotechnical design of offshore monopiles in sandy soils - Case study. ISFOG2020

    """

    if Bq < Bq_min:
        Bq = Bq_min

    if Bq > Bq_max:
        Bq = Bq_max

    _Gmax = coefficient_b * (1 + coefficient_Bq * Bq) * 1000 * multiplier_qc * \
            ((1000 * qc) ** exponent_1) * (sigma_vo_eff ** exponent_2)

    return {
        'Gmax [kPa]': _Gmax,
    }


BEHAVIOURINDEX_PCPT_NONNORMALISED = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'Rf': {'type': 'float', 'min_value': 0.1, 'max_value': 10.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
}

BEHAVIOURINDEX_PCPT_NONNORMALISED_ERRORRETURN = {
    'Isbt [-]': np.nan,
}


@Validator(BEHAVIOURINDEX_PCPT_NONNORMALISED, BEHAVIOURINDEX_PCPT_NONNORMALISED_ERRORRETURN)
def behaviourindex_pcpt_nonnormalised(
        qc, Rf,
        atmospheric_pressure=100.0, **kwargs):
    """
    Calculates the non-normalised soil behaviour type index. For vertical effective stresses between 50 and 150kPa, the non-normalised index is almost equal to the normalised soil behaviour type index.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 100.0
    :param Rf: Friction rato (:math:`R_f`) [:math:`pct`] - Suggested range: 0.1 <= Rf <= 10.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)

    .. math::
        I_{SBT} = \\sqrt{ \\left( 3.47 - \\log ( q_c / P_a ) \\right)^2 + \\left( \\log R_f + 1.22 \\right)^2}

    :returns: Dictionary with the following keys:

        - 'Isbt [-]': Non-normalised soil behaviour type index (:math:`I_{SBT}`)  [:math:`-`]

    .. figure:: images/behaviourindex_pcpt_nonnormalised_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Contours of non-normalised soil behaviour type index

    Reference - Fugro guidance on PCPT interpretation

    """

    _Isbt = np.sqrt((3.47 - np.log10(1000 * qc / atmospheric_pressure)) ** 2 +
                    (np.log10(Rf) + 1.22) ** 2)

    classes = ic_soilclass_robertson(_Isbt)

    return {
        'Isbt [-]': _Isbt,
        'Isbt class number [-]': classes['Soil type number [-]'],
        'Isbt class': classes['Soil type']
    }



DRAINEDSECANTMODULUS_SAND_BELLOTTI = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 50.0, 'max_value': 300.0},
    'K0': {'type': 'float', 'min_value': 0.5, 'max_value': 2.0},
    'sandtype': {'type': 'string', 'options': ("NC", "Aged NC", "OC"), 'regex': None},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
}

DRAINEDSECANTMODULUS_SAND_BELLOTTI_ERRORRETURN = {
    'qc1 [-]': np.nan,
    'Es_qc [-]': np.nan,
    'Es [kPa]': np.nan,
}

@Validator(DRAINEDSECANTMODULUS_SAND_BELLOTTI, DRAINEDSECANTMODULUS_SAND_BELLOTTI_ERRORRETURN)
def drainedsecantmodulus_sand_bellotti(
        qc, sigma_vo_eff, K0, sandtype,
        atmospheric_pressure=100.0, **kwargs):

    """
    Calculates the drained secant modulus for various types of sand for an average strain of 0.1 percent. This stress range should be representative for well-designed foundations (with sufficient safety against excessive deformations).

    Bands for mean effective stress from 50kPa to 300kPa are provided. Note that the correlation will not return values outside that range.

    Ageing and overconsolidation are beneficial effects, leading to increased stiffness.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 100.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: 50.0 <= sigma_vo_eff <= 300.0
    :param K0: Coefficient of lateral earth pressure at rest (:math:`K_0`) [:math:`-`] - Suggested range: 0.5 <= K0 <= 2.0
    :param sandtype: Type of sand - Options: ("NC", "Aged NC", "OC")
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)

    .. math::
        q_{c1} = \\left( \\frac{q_c}{P_a} \\right) \\cdot \\sqrt{ \\frac{P_a}{\\sigma_{vo}^{\\prime}} }

        \\sigma_{mo}^{\\prime} = \\frac{(1 + 2 \\cdot K_0) \\cdot \\sigma_{vo}^{\\prime}}{3}

    :returns: Dictionary with the following keys:

        - 'qc1 [-]': Normalised cone resistance (:math:`q_{c1}`)  [:math:`-`]
        - 'Es_qc [-]': Ratio of drained secant modulus to cone resistance (:math:`E_s^{\\prime} / q_c`)  [:math:`-`]
        - 'Es [kPa]': Drained secant modulus at strain level of 0.1 percent (:math:`E_s^{\\prime}`)  [:math:`kPa`]

    .. figure:: images/drainedsecantmodulus_sand__1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Visualisation of correlation

    Reference - Bellotti, R., Ghionna, V. N., Jamiolkowski, M., Lancellotta, R., & Robertson, P. K. (1989). Shear strength of sand from CPT. In Congrès international de mécanique des sols et des travaux de fondations. 12 (pp. 179-184).

    """
    _sigma_mo_eff = ((1 + 2 * K0) * sigma_vo_eff) / 3
    _qc1 = (1000 * qc / atmospheric_pressure) * np.sqrt(atmospheric_pressure / sigma_vo_eff)

    qc1__nc_50 = np.array(
        [36.0, 40.60330415823624, 46.020854882827855, 51.41286865966658, 59.40641474746315, 68.97423612037261,
         83.63131157432191, 97.56959774187784, 114.38054759574864, 134.08797382678958, 152.71270994765013,
         173.92441032542308, 200.0])

    E_qc__nc_50 = np.array(
        [3.706708268330736, 3.4820592823712992, 3.2574102964118588, 3.070202808112328, 2.8455538221528904, 2.62090483619345,
         2.3213728549141983, 2.2090483619344816, 2.0218408736349502, 1.8720748829953209, 1.7971918876755095,
         1.7597503900156042, 1.7597503900156042])

    qc1__nc_300 = np.array(
        [36.0, 40.408183360314055, 45.799699828776, 52.66621257255607, 59.693273152946574, 67.33279591108146,
         75.58504107246422, 86.0837562404445, 95.70757527884192, 105.38706697485875, 119.44847879561136, 136.0397981668792,
         152.71270994765013, 173.92441032542308, 200.0])

    E_qc__nc_300 = np.array(
        [5.017160686427459, 4.717628705148208, 4.418096723868956, 4.081123244929799, 3.7815912636505473, 3.519500780031205,
         3.2574102964118588, 3.032761310452422, 2.8455538221528904, 2.6957878315132606, 2.5460218408736357,
         2.4336973478939186, 2.358814352574104, 2.283931357254289, 2.1716068642745725])

    qc1__nc_aged_50 = np.array(
        [36.0, 39.828428929006265, 43.43602549287978, 47.82897607010807, 52.161249633588966, 58.27270013483514,
         64.1661742450743, 72.03032118526116, 80.46972295582609, 90.33201773973744, 103.8750311939775, 117.73469759745085,
         137.35677306263798, 157.94996551160924, 185.16427213057165, 200.0])

    E_qc__nc_aged_50 = np.array(
        [9.734789391575667, 9.210608424336977, 8.611544461778474, 7.975039001560062, 7.375975039001562, 6.70202808112325,
         6.102964118564741, 5.503900156006242, 4.942277691107648, 4.492979719188771, 3.9313572542901767, 3.594383775351016,
         3.2574102964118588, 3.032761310452422, 2.882995319812796, 2.770670826833076])

    qc1__nc_aged_300 = np.array(
        [36.0, 41.79397491103586, 46.46637385138504, 52.920524263063506, 58.27270013483514, 66.36674192014331,
         74.14257284871742, 82.82945568191151, 90.76820798422662, 102.87907878967758, 114.93286204934958,
         134.08797382678958, 154.19109206798834, 180.75775603565143, 200.0])

    E_qc__nc_aged_300 = np.array(
        [11.981279251170047, 11.1201248049922, 10.408736349453976, 9.547581903276138, 8.911076443057727, 8.049921996879878,
         7.3010920436817495, 6.66458658346334, 6.177847113884558, 5.616224648985963, 5.2418096723869, 4.8299531981279245,
         4.492979719188771, 4.156006240249614, 3.968798751950082])

    qc1__oc_50 = np.array(
        [36.0, 38.88059697907535, 41.79397491103586, 46.020854882827855, 49.70813683794835, 54.47239108201235,
         60.27115215047096, 66.68720996800097, 73.43169485579755, 79.31518754764512, 87.3368169432148, 98.0407364118493,
         114.38054759574864, 138.6864973368832, 165.74484792151944, 189.6782103606226, 200.0])

    E_qc__oc_50 = np.array(
        [16.02496099843994, 15.463338533541346, 14.826833073322936, 13.965678627145088, 13.366614664586587,
         12.617784711388456, 11.794071762870518, 10.970358814352577, 10.146645865834637, 9.547581903276138,
         8.836193447737912, 8.049921996879878, 7.151326053042125, 6.2527301092043714, 5.578783151326057, 5.204368174726991,
         5.054602184087365])

    qc1__oc_300 = np.array(
        [36.0, 38.507810337311405, 40.60330415823624, 43.43602549287978, 46.46637385138504, 50.91992276530692,
         54.999727733004406, 58.83682684054747, 66.36674192014331, 72.03032118526116, 78.55471459444749, 86.49943271508333,
         94.78993233218729, 102.87907878967758, 114.38054759574864, 126.55672442978188, 144.13537469613595,
         161.80047284520188, 189.6782103606226, 200.0])

    E_qc__oc_300 = np.array(
        [24.000000000000004, 23.251170046801878, 22.464898595943843, 21.56630265210609, 20.705148205928236,
         19.544461778471145, 18.645865834633387, 17.784711388455538, 16.361934477379094, 15.388455538221534,
         14.377535101404057, 13.366614664586587, 12.393135725429019, 11.606864274570984, 10.745709828393137,
         9.9219968798752, 8.985959438377536, 8.274570982839318, 7.48829953198128, 7.2262090483619374])

    if sandtype == 'NC':
        _qc1_50 = qc1__nc_50
        _qc1_300 = qc1__nc_300
        _Eratio_50 = E_qc__nc_50
        _Eratio_300 = E_qc__nc_300
    elif sandtype == "Aged NC":
        _qc1_50 = qc1__nc_aged_50
        _qc1_300 = qc1__nc_aged_300
        _Eratio_50 = E_qc__nc_aged_50
        _Eratio_300 = E_qc__nc_aged_300
    elif sandtype == 'OC':
        _qc1_50 = qc1__oc_50
        _qc1_300 = qc1__oc_300
        _Eratio_50 = E_qc__oc_50
        _Eratio_300 = E_qc__oc_300
    else:
        raise ValueError("Sand type not recognised, selected from 'NC', 'Aged NC' or 'OC'")

    _Es_qc_50 = np.interp(np.log10(_qc1), np.log10(_qc1_50), _Eratio_50)
    _Es_qc_300 = np.interp(np.log10(_qc1), np.log10(_qc1_300), _Eratio_300)
    _Es_qc = np.interp(_sigma_mo_eff, [50, 300], [_Es_qc_50, _Es_qc_300])
    _Es = _Es_qc * qc * 1000

    return {
        'qc1 [-]': _qc1,
        'Es_qc [-]': _Es_qc,
        'Es [kPa]': _Es,
    }



CORRELATIONS = {
    'Ic Robertson and Wride (1998)': behaviourindex_pcpt_robertsonwride,
    'Isbt Robertson (2010)': behaviourindex_pcpt_nonnormalised,
    'Gmax Rix and Stokoe (1991)': gmax_sand_rixstokoe,
    'Gmax Mayne and Rix (1993)': gmax_clay_maynerix,
    'Gmax Puechen (2020)': gmax_cpt_puechen,
    'Dr Baldi et al (1986) - NC sand': relativedensity_ncsand_baldi,
    'Dr Baldi et al (1986) - OC sand': relativedensity_ocsand_baldi,
    'Dr Jamiolkowski et al (2003)': relativedensity_sand_jamiolkowski,
    'Friction angle Kulhawy and Mayne (1990)': frictionangle_sand_kulhawymayne,
    'Su Rad and Lunne (1988)': undrainedshearstrength_clay_radlunne,
    'Friction angle Kleven (1986)': frictionangle_overburden_kleven,
    'OCR Lunne (1989)': ocr_cpt_lunne,
    'Sensitivity Rad and Lunne (1986)': sensitivity_frictionratio_lunne,
    'Unit weight Mayne et al (2010)': unitweight_mayne,
    'Shear wave velocity Robertson and Cabal (2015)': vs_ic_robertsoncabal,
    'K0 Mayne (2007) - sand': k0_sand_mayne,
    'Es Bellotti (1989) - sand': drainedsecantmodulus_sand_bellotti
}
