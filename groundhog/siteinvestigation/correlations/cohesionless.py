#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

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

    The default calibration parameter is taken from the recent study on monopile lateral response for the PISA project (Taborda et al, 2019). This calibration applies for dense marine sand

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