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
    'K0 [-]': 'K0'
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

    return {
        'exponent_zhang [-]': _exponent_zhang,
        'Qtn [-]': _Qtn,
        'Fr [%]': _Fr,
        'Ic [-]': _Ic
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

CORRELATIONS = {
    'Ic Robertson and Wride (1998)': behaviourindex_pcpt_robertsonwride,
    'Gmax Rix and Stokoe (1991)': gmax_sand_rixstokoe,
    'Gmax Mayne and Rix (1993)': gmax_clay_maynerix,
    'Dr Baldi et al (1986) - NC sand': relativedensity_ncsand_baldi,
    'Dr Baldi et al (1986) - OC sand': relativedensity_ocsand_baldi,
    'Dr Jamiolkowski et al (2003)': relativedensity_sand_jamiolkowski,
    'Friction angle Kulhawy and Mayne (1990)': frictionangle_sand_kulhawymayne,
    'Su Rad and Lunne (1988)': undrainedshearstrength_clay_radlunne,
    'Friction angle Kleven (1986)': frictionangle_overburden_kleven,
    'OCR Lunne (1989)': ocr_cpt_lunne,
    'Sensitivity Rad and Lunne (1986)': sensitivity_frictionratio_lunne
}
