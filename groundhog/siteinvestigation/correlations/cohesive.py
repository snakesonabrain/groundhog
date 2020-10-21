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

    _linear_part =  (
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