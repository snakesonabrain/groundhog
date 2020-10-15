#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np

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