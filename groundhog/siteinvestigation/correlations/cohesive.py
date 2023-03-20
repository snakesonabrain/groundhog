#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np
from scipy import interpolate

# Project imports
from groundhog.general.validation import Validator

COMPRESSIONINDEX_WATERCONTENT_KOPPULA = {
    'water_content': {'type': 'float', 'min_value': 0.0, 'max_value': 4.0},
    'cc_cr_ratio': {'type': 'float', 'min_value': 5.0, 'max_value': 10.0}
}

COMPRESSIONINDEX_WATERCONTENT_KOPPULA_ERRORRETURN = {
    'Cc [-]': np.nan,
    'Cr [-]': np.nan,
}


@Validator(COMPRESSIONINDEX_WATERCONTENT_KOPPULA, COMPRESSIONINDEX_WATERCONTENT_KOPPULA_ERRORRETURN)
def compressionindex_watercontent_koppula(
        water_content, cc_cr_ratio=7.5,
        **kwargs):
    """
    Based on an evaluation of the compression index of clays and eight other soil mechanics parameters, Koppula (1981) concluded that the best fit was obtain using a direct relation with natural water content.

    The recompression index is also calculated using a user-defined ratio of :math:`C_c` and :math:`C_r`. This ratio is generally between 5 and 10.

    :param water_content: In-situ natural water content of the clay (:math:`w_n`) [:math:`-`] - Suggested range: 0.0 <= water_content <= 4.0
    :param cc_cr_ratio: Ratio of compression index and recompression index (:math:`C_r / C_c`) [:math:`-`] - Suggested range: 5.0 <= cc_cr_ratio <= 10.0 (optional, default= 7.5)

    .. math::
        C_c = w_n

        C_c / C_r = 5 \\ \\text{to} \\ 10

    :returns: Dictionary with the following keys:

        - 'Cc [-]': Compression index (:math:`C_c`)  [:math:`-`]
        - 'Cr [-]': Recompression index (:math:`C_r`)  [:math:`-`]

    Reference - Koppula SD (1981) Statistical evaluation of compression index. Geotech Test J ASTM 4(2):68–73

    """

    _Cc = water_content
    _Cr = _Cc / cc_cr_ratio

    return {
        'Cc [-]': _Cc,
        'Cr [-]': _Cr
    }


FRICTIONANGLE_PLASTICITYINDEX = {
    'plasticity_index': {'type': 'float', 'min_value': 5.0, 'max_value': 1000.0},
}

FRICTIONANGLE_PLASTICITYINDEX_ERRORRETURN = {
    'Effective friction angle [deg]': np.nan,
}


@Validator(FRICTIONANGLE_PLASTICITYINDEX, FRICTIONANGLE_PLASTICITYINDEX_ERRORRETURN)
def frictionangle_plasticityindex(
        plasticity_index,
        **kwargs):
    """
    Based on a dataset of soft to stiff clays, a correlation between plasticity index and drained friction angle of clay is proposed. It should be noted that the friction angle of overconsolidated depends strongly on the in-situ condition. If the overconsolidated is fissured, the available shearing resistance will be lower than the value resulting from the correlation.

    :param plasticity_index: Plasticity index of the clay as determined from Atterberg limit tests (:math:`PI`) [:math:`pct`] - Suggested range: 5.0 <= plasticity_index <= 1000.0

    :returns: Dictionary with the following keys:

        - 'Effective friction angle [deg]': Drained friction angle of the clay (:math:`\\varphi^{\\prime}`)  [:math:`deg`]

    .. figure:: images/frictionangle_plasticityindex_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Dataset used for the correlation

    Reference - Terzaghi, K., Peck, R. B., & Mesri, G. (1996). Soil mechanics in engineering practice. John Wiley & Sons.

    """

    _linear_part = (
        (5.308659949013567, 34.743875278396445),
        (14.43975330745808, 31.848552338530073),
        (24.27554974584225, 29.510022271714927),
        (31.90090171018273, 28.06236080178174),
        (38.836419055171284, 26.94877505567929),
        (45.63471289931884, 25.946547884187087),
        (52.43486530103987, 25.055679287305125),
        (59.51689893473675, 24.387527839643656),
        (69.65502073840494, 23.496659242761694),
        (76.59611375611391, 22.71714922048998),
        (87.85246769981816, 22.160356347438757),
        (96.05087491597772, 21.714922048997774),
        (100, 21.38084632516704)
    )

    _x_lin = np.array(list(map(lambda _t: _t[0], _linear_part)))
    _y_lin = np.array(list(map(lambda _t: _t[1], _linear_part)))

    _log_part = (
        (100, 21.381039380159226),
        (116.32812247687298, 18.4785362796218),
        (137.80050320131895, 15.79773827463427),
        (177.27048394421894, 12.890234307279925),
        (213.06409764740178, 10.876885326789974),
        (276.22510814645017, 8.748516409510977),
        (358.2856380722786, 7.06555803005854),
        (494.34896872144236, 5.156893861602676),
        (772.0057248651214, 3.0195233840532367),
        (1000, 1.7823089335484852)
    )

    _x_log = np.array(list(map(lambda _t: np.log10(_t[0]), _log_part)))
    _y_log = np.array(list(map(lambda _t: _t[1], _log_part)))

    if plasticity_index <= 100:
        _spline = interpolate.UnivariateSpline(_x_lin, _y_lin)
        _effective_friction_angle = _spline(plasticity_index)
    else:
        _spline = interpolate.UnivariateSpline(_x_log, _y_log)
        _effective_friction_angle = _spline(np.log10(plasticity_index))

    return {
        'Effective friction angle [deg]': _effective_friction_angle,
    }


CV_LIQUIDLIMIT_USNAVY = {
    'liquid_limit': {'type': 'float', 'min_value': 20.0, 'max_value': 160.0},
    'trend': {'type': 'string', 'options': ('Remoulded', 'NC', 'OC'), 'regex': None},
}

CV_LIQUIDLIMIT_USNAVY_ERRORRETURN = {
    'cv [m2/yr]': np.nan,
}


@Validator(CV_LIQUIDLIMIT_USNAVY, CV_LIQUIDLIMIT_USNAVY_ERRORRETURN)
def cv_liquidlimit_usnavy(
        liquid_limit,
        trend='NC', **kwargs):
    """
    Calculates an estimate of the coefficient of consolidation based on the liquid limit of a clay. Three trends are available; an upper bound trend for remoulded clays, a trend for normally consolidated and a lower bound trend for undisturbed overconsolidated clay. Note that sample disturbance can lead to a reduced coefficient of consolidation.

    :param liquid_limit: Liquid limit of the clay (:math:`LL`) [:math:`pct`] - Suggested range: 20.0 <= liquid_limit <= 160.0
    :param trend: Choice of trend, choose between trends for remoulded, NC and OC clay (optional, default= 'NC') - Options: ('Remoulded', 'NC', 'OC')

    :returns: Dictionary with the following keys:

        - 'cv [m2/yr]': Coefficient of consolidation (:math:`c_v`)  [:math:`m2/yr`]

    .. figure:: images/cv_liquidlimit_usnavy_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Proposed relation between coefficient of consolidation and liquid limit

    Reference - U.S. Navy (1982) Soil mechanics – design manual 7.1, Department of the Navy, Naval Facilities Engineering Command, U.S. Government Printing Office, Washington, DC

    """
    if trend == "Remoulded":
        ll = np.array(
            [27.02702702702703, 30.13513513513513, 33.37837837837838, 36.8918918918919, 41.08108108108108,
             45.13513513513514, 49.45945945945946, 54.189189189189186, 59.189189189189186, 64.32432432432432,
             70.13513513513513, 76.75675675675676, 81.35135135135134, 86.35135135135134, 92.02702702702706,
             97.83783783783785, 102.97297297297298, 107.83783783783785, 112.43243243243244, 116.75675675675676,
             119.32432432432432])
        cv = np.array(
            [4.630822063309323, 3.8367516748979065, 3.2122391203904126, 2.6614202957569466, 2.114777837261994,
             1.789153093899863, 1.4823579994700684, 1.2027671084175822, 0.986162679019052, 0.8085661992948271,
             0.6560607018262126, 0.5323196107580059, 0.4745129932767461, 0.4274273713329328, 0.37313055744122897,
             0.3291530487025109, 0.3027535600686647, 0.2727115172299714, 0.2534740092387953, 0.23072052590916264,
             0.2212747818867807])
    elif trend == 'NC':
        ll = np.array([27.567567567567572, 30.40540540540541, 34.18918918918919, 37.162162162162154, 41.21621621621622,
                       45.270270270270274, 48.10810810810811, 52.2972972972973, 55.810810810810814, 60.270270270270274,
                       64.5945945945946, 69.5945945945946, 74.18918918918921, 78.91891891891892, 83.51351351351352,
                       87.97297297297298, 93.10810810810813, 97.43243243243244, 101.8918918918919, 106.89189189189187,
                       111.08108108108108, 116.8918918918919, 121.35135135135134, 126.08108108108108,
                       131.35135135135135, 137.56756756756755, 143.24324324324326, 150.13513513513516,
                       157.43243243243245, 160.0])
        cv = np.array([18.983055258835638, 15.56442681131302, 12.238997318607332, 10.354487139495033, 8.227727110118272,
                       6.675876912745551, 5.767242726040353, 4.679470046791455, 3.9177872379240335, 3.113093273947991,
                       2.5792754870822447, 2.071035744516675, 1.6804130140979967, 1.3922641289640214,
                       1.1535255848041792, 0.9759104869460395, 0.8085661992948271, 0.6840665206536173,
                       0.5971684063713107, 0.5105263404735331, 0.4456733255412902, 0.3770503904529313,
                       0.3326108866106533, 0.29960612028550504, 0.26707074860207874, 0.23314430470695946,
                       0.21221574849530087, 0.1872038063002005, 0.17039916941183805, 0.16513979448635985])
    elif trend == 'OC':
        ll = np.array([33.10810810810811, 36.75675675675676, 39.729729729729726, 43.1081081081081, 46.621621621621635,
                       49.86486486486486, 53.918918918918926, 57.43243243243244, 61.62162162162162, 65.4054054054054,
                       69.72972972972973, 73.24324324324327, 78.10810810810811, 82.43243243243244, 86.75675675675676,
                       90.94594594594595, 95.0, 99.72972972972973, 104.5945945945946, 109.72972972972971,
                       114.86486486486488, 119.72972972972973])
        cv = np.array([94.90893726308384, 70.09521157973178, 53.417717287029845, 40.708237493673955, 30.38100372887026,
                       23.152563941389857, 16.745695005475795, 12.895516135304856, 9.72516829490034, 7.647330038378023,
                       5.889051954526701, 4.778304561515615, 3.5290277068786917, 2.7461812536665784, 2.2282178035561744,
                       1.8269415776458084, 1.5136667015367813, 1.2805985642889186, 1.0947988735954193,
                       0.9165971813425068, 0.791841805338592, 0.6769549339857521])
    else:
        raise ValueError("Trend not recognised. Choose from 'Remoulded', 'NC' or 'OC'")

    if liquid_limit < ll.min() or liquid_limit > ll.max():
        _cv = np.nan
    else:
        _cv = 10 ** np.interp(liquid_limit, ll, np.log10(cv))

    return {
        'cv [m2/yr]': _cv,
    }



GMAX_PLASTICITYOCR_ANDERSEN = {
    'pi': {'type': 'float', 'min_value': 0.0, 'max_value': 160.0},
    'ocr': {'type': 'float', 'min_value': 1.0, 'max_value': 40.0},
    'sigma_vo_eff': {'type': 'float', 'min_value': 0.0, 'max_value': 1000.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': 90.0, 'max_value': 110.0},
    'coefficient_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_4': {'type': 'float', 'min_value': None, 'max_value': None},
    'coefficient_5': {'type': 'float', 'min_value': None, 'max_value': None},
}

GMAX_PLASTICITYOCR_ANDERSEN_ERRORRETURN = {
    'sigma_0_ref [kPa]': np.nan,
    'Gmax [kPa]': np.nan,
}

@Validator(GMAX_PLASTICITYOCR_ANDERSEN, GMAX_PLASTICITYOCR_ANDERSEN_ERRORRETURN)
def gmax_plasticityocr_andersen(
        pi, ocr, sigma_vo_eff,
        atmospheric_pressure=100.0,coefficient_1=30.0,coefficient_2=75.0,coefficient_3=0.03,coefficient_4=0.5,coefficient_5=0.9, **kwargs):

    """
    Calculates the small-strain shear modulus for cohesive soils based on plasticity index, effective overburden pressure and OCR. The proposed relation is calibrated on a number of shear wave velocity tests on clay samples with different plasticity index and OCR.

    :param pi: Plasticity index (difference between liquid limit and plastic limit) (:math:`PI`) [:math:`pct`] - Suggested range: 0.0 <= PI <= 160.0
    :param ocr: Overconsolidation ratio of the clay (:math:`OCR`) [:math:`-`] - Suggested range: 1.0 <= OCR <= 40.0
    :param sigma_vo_eff: Vertical effective stress (:math:`\\sigma_{vo}^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_vo_eff <= 1000.0
    :param atmospheric_pressure: Atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: 90.0 <= atmospheric_pressure <= 110.0 (optional, default= 100.0)
    :param coefficient_1: First calibration coefficient (:math:``) [:math:`-`] (optional, default= 30.0)
    :param coefficient_2: Second calibration coefficient (:math:``) [:math:`-`] (optional, default= 75.0)
    :param coefficient_3: Third calibration coefficient (:math:``) [:math:`-`] (optional, default= 0.03)
    :param coefficient_4: Fourth calibration coefficient (exponent for OCR) (:math:``) [:math:`-`] (optional, default= 0.5)
    :param coefficient_5: Fifth calibration coefficient (exponent for sigma_ref) (:math:``) [:math:`-`] (optional, default= 0.9)

    .. math::
        \\frac{G_{max}}{\\sigma_{ref}^{\\prime}} = \\left( 30 + \\frac{75}{\\frac{I_p}{100} + 0.03} \\right) \\cdot OCR^{0.5}

        \\sigma_{ref}^{\\prime} = P_a \\cdot \\left( \\sigma_{0}^{\\prime}  / P_a \\right)^{0.9}

    :returns: Dictionary with the following keys:

        - 'sigma_0_ref [kPa]': Reference stress (:math:`\\sigma_{ref}^{\\prime}`)  [:math:`kPa`]
        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    .. figure:: images/gmax_plasticityocr_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Data used for calibrating the correlation

    Reference - Andersen KH. Cyclic soil parameters for offshore foundation design. The Third ISSMGE McClelland Lecture. In: Meyer V, editor. Proc. Int. Symp. Frontiers in offshore geotechnics, ISFOG 2015. London: Taylor and Francis; 2015. 5–82.

    """

    _sigma_0_ref = atmospheric_pressure * ((sigma_vo_eff / atmospheric_pressure) ** coefficient_5)
    _Gmax = _sigma_0_ref * (coefficient_1 + (coefficient_2 / (0.01 * pi + coefficient_3))) * (ocr ** coefficient_4)

    return {
        'sigma_0_ref [kPa]': _sigma_0_ref,
        'Gmax [kPa]': _Gmax,
    }


K0_PLASTICITY_KENNEY = {
    'pi': {'type': 'float', 'min_value': 5, 'max_value': 80},
    'ocr': {'type': 'float', 'min_value': 1, 'max_value': 30},
    'coeff_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'coeff_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'coeff_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'coeff_4': {'type': 'float', 'min_value': None, 'max_value': None}
}

K0_PLASTICITY_KENNEY_ERRORRETURN = {
    'K0 [-]': np.nan,
}

@Validator(K0_PLASTICITY_KENNEY, K0_PLASTICITY_KENNEY_ERRORRETURN)
def k0_plasticity_kenney(
        pi, ocr=1, coeff_1=0.19, coeff_2=0.233, coeff_3=-281, coeff_4=1.85, **kwargs):
    """
    Calculates the coefficient of lateral earthpressure at rest for normally and overconsolidated clay.
    Kenney (1959) presented a formula for coefficient of lateral earth pressure at rest for normally consolidated clay.
    The plasticity index :math:`\\text{PI}` was used as a basis for this correlation.
    This relation was modified for the effect of overconsolidation as shown by Alpan (1967).
    The exponent on OCR shows a linear variation with plasticity index.

    :param pi: Plasticity index (:math:`\\text{PI}`) [:math:`pct`] - Suggested range: 5 <= PI <= 80
    :param ocr: Overconsolidation ratio (:math:`\\text{OCR}`) [:math:`-`] (optional, default= 1, suggested range: 1 <= OCR < 30)
    :param coeff_1: First calibration coefficient (optional, default=0.19)
    :param coeff_2: Second calibration coefficient (optional, default=0.233)
    :param coeff_3: First calibration coefficient (optional, default=-281)
    :param coeff_4: Second calibration coefficient (optional, default=1.85)

    .. math::
        K_{0,NC} = 0.19 + 0.233 \\log_{10} I_p

        I_p = -281 \\log_{10} \\left( 1.85 \\lambda \\right)

    :returns: Dictionary with the following keys:

        - 'K0 NC [-]': Coefficient of lateral earth pressure at rest for normally consolidated conditions (:math:`K_{0,NC}`)  [:math:`-`]
        - 'K0 [-]': Coefficient of lateral earth pressure at rest (:math:`K_0`)  [:math:`-`]

    Reference - Alpan (1967) THE EMPIRICAL EVALUATION OF THE COEFFICIENT K0 AND K0R. Soils and Foundations. Volume 7, Issue 1

    """
    _K0_NC = coeff_1 + coeff_2 * np.log10(pi)
    _exponent = (1 / coeff_4) * (10 ** (pi / coeff_3))
    _K0 = (_K0_NC) * (ocr ** _exponent)

    return {
        'K0 NC [-]': _K0_NC,
        'K0 [-]': _K0,
    }