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