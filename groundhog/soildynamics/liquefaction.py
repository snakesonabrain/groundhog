#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np
from scipy import interpolate

# Project imports
from groundhog.general.validation import Validator

CYCLICSTRESSRATIO_MOSS = {
    'sigma_vo': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'magnitude': {'type': 'float', 'min_value': 5.5, 'max_value': 8.5},
    'acceleration': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'depth': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'gravity': {'type': 'float', 'min_value': 9.8, 'max_value': 10.0},
    'rd_override': {'type': 'float', 'min_value': None, 'max_value': None},
    'DWF_override': {'type': 'float', 'min_value': None, 'max_value': None},
}

CYCLICSTRESSRATIO_MOSS_ERRORRETURN = {
    'CSR [-]': np.nan,
    'CSR* [-]': np.nan,
    'DWF [-]': np.nan,
    'rd [-]': np.nan,
}


@Validator(CYCLICSTRESSRATIO_MOSS, CYCLICSTRESSRATIO_MOSS_ERRORRETURN)
def cyclicstressratio_moss(
        sigma_vo, sigma_vo_eff, magnitude, acceleration, depth,
        gravity=9.81, rd_override=np.nan, DWF_override=np.nan, **kwargs):
    """
    Calculates the equivalent uniform cyclic stress ratio (CSR) based on the technique by Seed and Idriss (1971). Earthquake events are scaled to an equivalent event with magnitude of 7.5 using a magnitude-correlated duration weighting factor (DWF).

    The formulation of DWF from Cetin et al (2004) is used which is based on a SPT-based liquefaction database with the formulation being valid for magnitudes between 5.5 and 8.5.

    The non-linear shear mass participation factor formulation from Cetin et al (2004) is used. This depends on the depth of the layer of interest, the earthquake magnitude and the maximum horizontal ground acceleration. The formula uses the acceleration in units of gravity.

    The paper on which this equation is based offers a discussion on the variability on each of the terms.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param magnitude: Earthquake magnitude (:math:`M_w`) [:math:`-`] - Suggested range: 5.5 <= magnitude <= 8.5
    :param acceleration: Maximum horizontal acceleration at the soil surface (:math:`a_{max}`) [:math:`m/s2`] - Suggested range: acceleration >= 0.0
    :param depth: Depth at which CSR is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param gravity: Acceleration due to gravity (:math:`g`) [:math:`m/s2`] - Suggested range: 9.8 <= gravity <= 10.0 (optional, default= 9.81)
    :param rd_override: Override for direct specification of rd (:math:`r_d`) [:math:`-`] (optional, default=np.nan)
    :param DWF_override: Override for direct specification of DWF (:math:`DWF`) [:math:`-`] (optional, default=np.nan)

    .. math::
        CSR = \\frac{\\tau_{avg}}{\\sigma_v^{\\prime}} = 0.65 \\cdot \\frac{a_{max}}{g} \\cdot \\frac{\\sigma_v}{\\sigma_v^{\\prime}} \\cdot r_d

        CSR^{*} = CSR_{M_w=7.5}=\\frac{CSR}{DWF_{M_w}}

        DWF_{M_w} = 17.84 \\cdot M_w^{-1.43}

        r_d = \\frac{\\left[ 1 + \\frac{-9.147 - 4.173 \\cdot a_{max} + 0.652 \\cdot M_w}{10.567 + 0.089 \\cdot e^{0.089 \\cdot \\left( -z \\cdot 3.28 - 7.760 \\cdot a_{max} + 78.576 \\right)}} \\right]}{\\left[ 1 + \\frac{-9.147 - 4.173 \\cdot a_{max} + 0.652 \\cdot M_w}{10.567 + 0.089 \\cdot e^{0.089 \\cdot (-7.760 \\cdot a_{max} + 78.576)}} \\right]} \\\\ z \\geq 20m

        r_d = \\frac{\\left[ 1 + \\frac{-9.147 - 4.173 \\cdot a_{max} + 0.652 \\cdot M_w}{10.567 + 0.089 \\cdot e^{0.089 \\cdot \\left( -z \\cdot 3.28 - 7.760 \\cdot a_{max} + 78.576 \\right)}} \\right]}{\\left[ 1 + \\frac{-9.147 - 4.173 \\cdot a_{max} + 0.652 \\cdot M_w}{10.567 + 0.089 \\cdot e^{0.089 \\cdot (-7.760 \\cdot a_{max} + 78.576)}} \\right]} - 0.0014 \\cdot (z \\cdot 3.28 - 65) \\\\ z < 20m

    :returns: Dictionary with the following keys:

        - 'CSR [-]': Uncorrected cyclic stress ratio (:math:`CSR`)  [:math:`-`]
        - 'CSR* [-]': Equivalent uniform cyclic stress ratio (for magnitude 7.5 event) (:math:`CSR^{*}`)  [:math:`-`]
        - 'DWF [-]': Magnitude-correlated duration weighting factor (:math:`DWF`)  [:math:`-`]
        - 'rd [-]': Nonlinear shear mass participation factor (:math:`r_d`)  [:math:`-`]

    Reference - Moss et al (2006) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential. Journal of Geotechnical & Geoenvironmental Engineering, 132(8)

    """

    if np.math.isnan(rd_override):
        if depth >= 20:
            _rd = (1 +
                   ((-9.147 - 4.173 * (acceleration / gravity) + 0.652 * magnitude) /
                    (10.567 + 0.089 * np.exp(0.089 * (-3.28 * depth - 7.760 * (acceleration / gravity) + 78.576)))
                   )
                  ) / \
                  (1 +
                   ((-9.147 - 4.173 * (acceleration / gravity) + 0.652 * magnitude) /
                   (10.567 + 0.089 * np.exp(0.089 * (-7.760 * (acceleration / gravity) + 78.576)))
                    )
                   )
        else:
            _rd = ((1 + ((-9.147 - 4.173 * (acceleration / gravity) + 0.652 * magnitude) /
                        (10.567 + 0.089 * np.exp(
                            0.089 * (-3.28 * depth - 7.760 * (acceleration / gravity) + 78.576)
                        )))) / \
                  (1 + ((-9.147 - 4.173 * (acceleration / gravity) + 0.652 * magnitude) /
                        (10.567 + 0.089 * np.exp(
                            0.089 * (-7.760 * (acceleration / gravity) + 78.576)
                        ))))) - 0.0014 * (depth * 3.28 - 65)
    else:
        _rd = rd_override

    if np.math.isnan(DWF_override):
        _DWF = 17.84 * (magnitude ** (-1.43))
    else:
        _DWF = DWF_override

    _CSR = 0.65 * (acceleration / gravity) * (sigma_vo / sigma_vo_eff) * _rd
    _CSR_star = _CSR / _DWF

    return {
        'CSR [-]': _CSR,
        'CSR* [-]': _CSR_star,
        'DWF [-]': _DWF,
        'rd [-]': _rd,
    }


LIQUEFACTION_ROBERTSONFEAR = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'CSR': {'type': 'float', 'min_value': 0.073, 'max_value': 0.49},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
}

LIQUEFACTION_ROBERTSONFEAR_ERRORRETURN = {
    'qc1 [-]': np.nan,
    'qc1 liquefaction [-]': np.nan,
    'qc liquefaction [MPa]': np.nan,
    'liquefaction': np.nan,
}


@Validator(LIQUEFACTION_ROBERTSONFEAR, LIQUEFACTION_ROBERTSONFEAR_ERRORRETURN)
def liquefaction_robertsonfear(
        qc, sigma_vo_eff, CSR,
        atmospheric_pressure=100.0, **kwargs):
    """
    Calculates whether cyclic liquefaction can be triggered based on the normalised cone tip resistance and the cyclic shear stress ratio imposed on the soil by the earthquake event.

    For earthquake magnitudes different from 7.5, CSR:math:`^*` should be used.

    Note that this correlation was developed for clean sands and does not include any modifications for fines content.

    It should also be noted that the correlation is based on averaged cone resistance values from field cases. So it applied to raw cone resistance data, they might be too conservative.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param CSR: Seismic shear stress ratio (:math:`CSR = \\tau_{avg} / \\sigma_{vo}^{\\prime}`) [:math:`-`] - Suggested range: 0.073 <= CSR <= 0.49
    :param atmospheric_pressure: Atmospheric pressure used for normalisation (:math:`p_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)

    .. math::
        q_{c1} = (q_c / p_a) ( p_a /  \\sigma_{vo}^{\\prime} )^{0.5}

    :returns: Dictionary with the following keys:

        - 'qc1 [-]': Normalised dimensionless cone resistance (:math:`q_{c1}`)  [:math:`-`]
        - 'qc1 liquefaction [-]': Normalised dimensionless cone tip resistance for which liquefaction is just triggered at the given CSR (:math:`q_{c1,liq}`)  [:math:`-`]
        - 'qc liquefaction [MPa]': Cone tip resistance for which liquefaction is just triggered at the given CSR (:math:`q_{c,liq}`)  [:math:`MPa`]
        - 'liquefaction': Liquefaction occurs?

    .. figure:: images/liquefaction_robertsonfear_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Dataset supporting the liquefaction triggering function

    Reference - Robertson, P. K., and C. E. Fear. "Application of CPT to evaluate liquefaction potential." CPT’95, Linkoping (1995): 57-79.

    """

    _trigger_points = (
        (38.72211941170742, 0.07284893347618582),
        (60.5259569494497, 0.10067790006817956),
        (71.97623453783967, 0.11888380247394559),
        (85.61020746079677, 0.14352196357261127),
        (99.25002434985879, 0.17460017531898303),
        (110.16460504529078, 0.20246810168501017),
        (122.72134021622679, 0.23999026005649154),
        (131.46586149800333, 0.27645271257426707),
        (139.6717639037694, 0.3193571637284503),
        (147.34196941657743, 0.37192363884289464),
        (150.63991428849715, 0.40625888769845125),
        (153.95831304178438, 0.4631343138209798),
        (155.6238433817084, 0.4985487484172591)
    )

    _qc1 = np.array(list(map(lambda _x: _x[0], _trigger_points)))
    _csr = np.array(list(map(lambda _x: _x[1], _trigger_points)))

    _func = interpolate.UnivariateSpline(_csr, _qc1)

    _qc1 = (1000 * qc / atmospheric_pressure) * np.sqrt(atmospheric_pressure / sigma_vo_eff)
    _qc1_liquefaction = _func(CSR)
    _qc_liquefaction = (1e-3 * atmospheric_pressure * _qc1_liquefaction) / np.sqrt(atmospheric_pressure / sigma_vo_eff)

    if _qc1 <= _qc1_liquefaction:
        _liquefaction = True
    else:
        _liquefaction = False

    return {
        'qc1 [-]': _qc1,
        'qc1 liquefaction [-]': _qc1_liquefaction,
        'qc liquefaction [MPa]': _qc_liquefaction,
        'liquefaction': _liquefaction,
    }


LIQUEFACTIONPROBABILITY_MOSS = {
    'qc': {'type': 'float', 'min_value': 0.0, 'max_value': 120.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'Rf': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'CSR': {'type': 'float', 'min_value': 0.0, 'max_value': 0.6},
    'CSR_star': {'type': 'float', 'min_value': 0.0, 'max_value': 0.6},
    'Pa': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
    'delta_qc_override': {'type': 'float', 'min_value': None, 'max_value': None},
    'c_override': {'type': 'float', 'min_value': None, 'max_value': None},
    'x1': {'type': 'float', 'min_value': None, 'max_value': None},
    'x2': {'type': 'float', 'min_value': None, 'max_value': None},
    'y1': {'type': 'float', 'min_value': None, 'max_value': None},
    'y2': {'type': 'float', 'min_value': None, 'max_value': None},
    'y3': {'type': 'float', 'min_value': None, 'max_value': None},
    'z1': {'type': 'float', 'min_value': None, 'max_value': None},
}

LIQUEFACTIONPROBABILITY_MOSS_ERRORRETURN = {
    'Pl [pct]': np.nan,
    'qc_5 [MPa]': np.nan,
    'qc_95 [MPa]': np.nan,
    'qc1 [MPa]': np.nan,
    'qc1mod [MPa]': np.nan,
    'c [-]': np.nan
}


@Validator(LIQUEFACTIONPROBABILITY_MOSS, LIQUEFACTIONPROBABILITY_MOSS_ERRORRETURN)
def liquefactionprobability_moss(
        qc, sigma_vo_eff, Rf, CSR, CSR_star,
        Pa=100.0, delta_qc_override=np.nan, c_override=np.nan,
        x1=0.78, x2=-0.33, y1=-0.32, y2=-0.35, y3=0.49, z1=1.21,**kwargs):
    """
    Calculates the probability of liquefaction according to Moss et al. The cone tip resistance is normalised to a standard effective overburden pressure of 100kPa.

    The liquefaction probability is based on a database of case studies using Bayesian updating. The probability contours are digitised from the published figure and interpolation between the different values of normalised tip resistance for a given CSR is performed.

    The calculation of the normalisation exponent :math:`c` is performed iteratively, or an override can be specified. While a normalisation exponent of 0.5 is generally assumed, performing the calculation results in a much better statistical fit.

    Soils with an increased friction ratio shows less potential for liquefaction. This is accounted for by modifying the normalised cone resistance using the friction ratio. The bound for modifying the friction ratio are 0.5 - 5%. Below 0.5%, there is no correction and above 5%, the correction for Rf=5% is applied.

    It should also be noted that the correlation is based on averaged cone resistance values from field cases. So it applied to raw cone resistance data, they might be too conservative.

    :param qc: Cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: 0.0 <= qc <= 120.0
    :param sigma_vo_eff: Vertical effective stress at depth of interest (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param Rf: Friction ratio (:math:`R_f`) [:math:`pct`] - Suggested range: 0.0 <= Rf <= 10.0
    :param CSR: Cyclic shear stress ratio (:math:`CSR`) [:math:`-`] - Suggested range: 0.0 <= CSR <= 0.6
    :param CSR_star: Equivalent uniform cyclic stress ratio (for magnitude 7.5 event) (:math:`CSR^{*}`) [:math:`-`] - Suggested range: 0.0 <= CSR_star <= 0.6
    :param Pa: Atmospheric pressure (:math:`p_a`) [:math:`kPa`] - Suggested range: 90.0 <= Pa <= 110.0 (optional, default= 100.0)
    :param delta_qc_override: Override for the correction to the normalised cone tip resistance (:math:`\\Delta q_c`) [:math:`MPa`] (optional, default=np.nan)
    :param c_override: Override for the normalisation exponent (:math:`c`) [:math:`-`] (optional, default=np.nan)
    :param x1: Factor x1 (:math:`x_1`) [:math:`-`] (optional, default=0.78)
    :param x2: Factor x2 (:math:`x_2`) [:math:`-`] (optional, default=-0.33)
    :param y1: Factor y1 (:math:`y_1`) [:math:`-`] (optional, default=-0.32)
    :param y2: Factor y2 (:math:`y_2`) [:math:`-`] (optional, default=-0.35)
    :param y3: Factor y3 (:math:`y_3`) [:math:`-`] (optional, default=0.49)
    :param z1: Factor z1 (:math:`z_1`) [:math:`-`] (optional, default=1.21)

    .. math::
        q_{c,1} = q_c \\left( \\frac{p_a}{\\sigma_{vo}^{\\prime}} \\right)^{c}

        c = f_1 \\cdot \\left( \\frac{R_f}{f_3} \\right)^{f_2}

        f_1 = x_1 \\cdot q_c^{x_2}

        f_2 = -(y_1 \\cdot q_c^{y_2} + y_3 )

        f_3 = abs( \\log_{10}(10 + q_c )^{z_1}

        \\Delta q_c = \\alpha_1 \\cdot \\ln (CSR) + \\alpha_2

        \\alpha_1 = 0.38 \\cdot R_f - 0.19

        \\alpha_2 = 1.46 \\cdot R_f - 0.73

    :returns: Dictionary with the following keys:

        - 'Pl [pct]': Liquefaction probability (:math:`P_L`)  [:math:`pct`]
        - 'qc_5 [MPa]': Cone tip resistance for 5% probability of liquefaction (:math:`q_{c,P_{L,5\\%}}`)  [:math:`MPa`]
        - 'qc_95 [MPa]': Cone tip resistance for 95% probability of liquefaction (:math:`q_{c,P_{L,95\\%}}`)  [:math:`MPa`]
        - 'qc1 [MPa]': Normalised cone tip resistance (:math:`q_{c,1}`)  [:math:`MPa`]
        - 'qc1mod [MPa]': Modified normalised cone tip resistance (:math:`q_{c,1} + \\Delta q_c`)  [:math:`MPa`]
        - 'c [-]': Normalisation exponent  [:math:`-`]

    .. figure:: images/liquefactionprobability_moss_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Contours of liquefaction probability according to Moss et al (2006)

    Reference - Moss et al (2006) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential. Journal of Geotechnical & Geoenvironmental Engineering, 132(8)

    """

    pct_5 = ((0.9684210526315793, 0.03701578192252508),
             (2.0210526315789474, 0.04390243902439028),
             (3.4526315789473694, 0.055093256814921054),
             (4.842105263157895, 0.07058823529411762),
             (5.85263157894737, 0.0817790530846485),
             (7.0315789473684225, 0.09899569583931134),
             (8.08421052631579, 0.11707317073170731),
             (8.88421052631579, 0.13515064562410328),
             (9.810526315789478, 0.15581061692969872),
             (10.610526315789475, 0.18077474892395984),
             (11.578947368421055, 0.2100430416068867),
             (12.336842105263159, 0.23758967001434722),
             (12.96842105263158, 0.266857962697274),
             (13.600000000000001, 0.29784791965566715),
             (14.189473684210526, 0.327116212338594),
             (14.863157894736844, 0.36413199426111914),
             (15.326315789473686, 0.3942611190817791),
             (15.957894736842107, 0.4441893830703013),
             (16.294736842105262, 0.47259684361549503),
             (16.673684210526318, 0.5001434720229556),
             (16.96842105263158, 0.5276901004304161),
             (17.263157894736846, 0.5560975609756098),
             (17.51578947368421, 0.5802008608321378))
    pct_20 = (
        (0.9684210526315793, 0.04476327116212331),
        (2.189473684210527, 0.05337159253945478),
        (3.326315789473685, 0.0654232424677188),
        (4.757894736842105, 0.08263988522238164),
        (5.894736842105263, 0.09985652797704447),
        (7.11578947368421, 0.12051649928263986),
        (7.831578947368421, 0.13428981348637015),
        (8.96842105263158, 0.16269727403156387),
        (9.600000000000001, 0.18163558106169297),
        (10.484210526315792, 0.20918220946915356),
        (11.115789473684213, 0.23242467718794835),
        (11.663157894736845, 0.25652797704447633),
        (12.294736842105266, 0.28751793400286946),
        (13.010526315789473, 0.32281205164992827),
        (13.600000000000001, 0.357245337159254),
        (14.189473684210526, 0.3942611190817791),
        (14.694736842105264, 0.4278335724533716),
        (15.15789473684211, 0.46140602582496415),
        (15.536842105263162, 0.4932568149210904),
        (15.873684210526317, 0.5216642754662841),
        (16.25263157894737, 0.5586800573888092),
        (16.463157894736845, 0.581061692969871)
    )

    pct_50 = (
        (1.0105263157894733, 0.055093256814921054),
        (2.7368421052631584, 0.07144906743185075),
        (4.042105263157896, 0.08866571018651359),
        (5.473684210526316, 0.11104734576757525),
        (6.610526315789474, 0.13428981348637015),
        (7.621052631578949, 0.15925394548063126),
        (8.336842105263159, 0.17819225251076043),
        (9.094736842105263, 0.20315638450502155),
        (9.894736842105264, 0.22984218077474894),
        (10.652631578947368, 0.2616929698708752),
        (11.2, 0.2901004304160689),
        (11.915789473684214, 0.32625538020086087),
        (12.463157894736845, 0.3563845050215208),
        (13.178947368421053, 0.4011477761836442),
        (13.557894736842105, 0.42869440459110475),
        (14.02105263157895, 0.4596843615494979),
        (14.315789473684212, 0.49153515064562414),
        (14.736842105263158, 0.5173601147776185),
        (15.15789473684211, 0.5543758967001435),
        (15.368421052631579, 0.581061692969871)
    )

    pct_80 = (
        (1.0105263157894733, 0.06628407460545194),
        (2.905263157894738, 0.08952654232424673),
        (4.294736842105263, 0.11104734576757525),
        (5.221052631578947, 0.1308464849354376),
        (6.3157894736842115, 0.15494978479196558),
        (7.4105263157894745, 0.18852223816355812),
        (8.547368421052632, 0.22209469153515066),
        (9.642105263157895, 0.26771879483500716),
        (10.442105263157899, 0.3021520803443329),
        (11.031578947368423, 0.339167862266858),
        (11.747368421052634, 0.38307030129124825),
        (12.378947368421052, 0.4252510760401722),
        (12.800000000000004, 0.45882352941176474),
        (13.642105263157895, 0.5242467718794835),
        (14.231578947368423, 0.581061692969871)
    )

    pct_95 = (
        (0.9684210526315793, 0.07919655667144909),
        (2.3578947368421046, 0.09727403156384506),
        (3.621052631578949, 0.12051649928263986),
        (4.757894736842105, 0.1446197991391679),
        (5.6000000000000005, 0.16614060258249647),
        (6.610526315789474, 0.19368723098995694),
        (7.4105263157894745, 0.22037302725968433),
        (8.25263157894737, 0.2539454806312769),
        (9.094736842105263, 0.2944045911047346),
        (9.810526315789478, 0.3296987087517934),
        (10.400000000000002, 0.3675753228120517),
        (11.242105263157896, 0.42352941176470593),
        (11.705263157894738, 0.45451936872309906),
        (12.25263157894737, 0.5035868005738882),
        (13.010526315789473, 0.5621233859397419),
        (13.094736842105267, 0.5819225251076041)
    )

    pl_array = np.array([95, 80, 50, 20, 5])
    qcpl_array = np.array([])
    for _curve, _pl in zip([pct_95, pct_80, pct_50, pct_20, pct_5], pl_array):
        _qc_array = np.array(list(map(lambda _x: _x[0], _curve)))
        _csr_array = np.array(list(map(lambda _x: _x[1], _curve)))
        _spline = interpolate.UnivariateSpline(_csr_array, _qc_array)
        qcpl_array = np.append(qcpl_array, _spline(CSR_star))

    if np.math.isnan(c_override):
        _qc = qc
        for i in range(10):
            f_1 = x1 * (qc ** x2)
            f_2 = -(y1 * (qc ** y2) + y3)
            f_3 = (abs(np.log10(10 + qc)) ** z1)
            _c = f_1 * ((Rf / f_3) ** f_2)
            _qc = _qc * ((Pa / sigma_vo_eff) ** _c)
        _qc1 = qc * ((Pa / sigma_vo_eff) ** _c)
    else:
        _c = c_override

    if np.math.isnan(delta_qc_override):
        if Rf < 0.5:
            _delta_qc = 0
        else:
            if Rf > 5:
                Rf = 5
            else:
                pass
            x1 = 0.38 * Rf - 0.19
            x2 = 1.46 * Rf - 0.73
            _delta_qc = x1 * np.log(CSR) +  x2
        _qc1mod = _qc1 + _delta_qc
    else:
        _qc1mod = _qc1 + delta_qc_override

    _Pl = np.interp(_qc1mod, qcpl_array, pl_array)
    _qc_5 = max(0, qcpl_array[-1])
    _qc_95 = max(0, qcpl_array[0])

    return {
        'Pl [pct]': _Pl,
        'qc_5 [MPa]': _qc_5,
        'qc_95 [MPa]': _qc_95,
        'qc1 [MPa]': _qc1,
        'qc1mod [MPa]': _qc1mod,
        'c [-]': _c
    }