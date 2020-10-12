#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

STRESSES_POINTLOAD = {
    'pointload': {'type': 'float', 'min_value': None, 'max_value': None},
    'z': {'type': 'float', 'min_value': None, 'max_value': None},
    'r': {'type': 'float', 'min_value': None, 'max_value': None},
    'poissonsratio': {'type': 'float', 'min_value': 0.0, 'max_value': 0.5},
}

STRESSES_POINTLOAD_ERRORRETURN = {
    'delta sigma z [kPa]': np.nan,
    'delta sigma r [kPa]': np.nan,
    'delta sigma theta [kPa]': np.nan,
    'delta tau rz [kPa]': np.nan,
}


@Validator(STRESSES_POINTLOAD, STRESSES_POINTLOAD_ERRORRETURN)
def stresses_pointload(
        pointload, z, r, poissonsratio,
        **kwargs):
    """
    Calculates the stresses at a point below a line load according the solution proposed by Boussinesq (1885). The vertical stress increase is calculated as well as the increases in radial and tangential stress

    :param pointload: Magnitude of the point load (:math:`Q`) [:math:`kN`]
    :param z: Vertical distance from the surface to the point where the stresses are calculated (:math:`z`) [:math:`m`]
    :param r: Radial distance from the surface to the point where the stresses are calculated (:math:`r`) [:math:`m`]
    :param poissonsratio: Poisson's ratio (:math:`\\nu`) [:math:`-`] - Suggested range: 0.0 <= poissonsratio <= 0.5

    .. math::
        \\Delta \\sigma_z = \\frac{3Q}{2 \\pi z^2 \\left[ 1 + \\left( \\frac{r}{z} \\right)^2 \\right]^{5/2}}

        \\Delta \\sigma_r = \\frac{Q}{2 \\pi} \\left( \\frac{3 r^2 z}{(r^2 + z^2)^{5/2}} - \\frac{1 - 2 \\nu}{r^2 + z^2 + z \\left( r^2 + z^2 \\right)^{1/2}}\\right)

        \\Delta \\sigma_{\\theta} = \\frac{Q}{2 \\pi} \\left( 1 - 2 \\nu \\right) \\left( \\frac{z}{\\left( r^2 + z^2 \\right)^{3/2}} - \\frac{1}{r^2 + z^2 + z \\left( r^2 + z^2 \\right)^{1/2} } \\right)

        \\Delta \\tau_{rz} = \\frac{3 Q}{2 \\pi} \\left[ \\frac{r z^2}{\\left( r^2 + z^2 \\right)^{5/2}} \\right]

    :returns: Dictionary with the following keys:

        - 'delta sigma z [kPa]': Increase in vertical normal stress (:math:`\\Delta \\sigma_z`)  [:math:`kPa`]
        - 'delta sigma r [kPa]': Increase in radial normal stress (:math:`\\Delta \\sigma_r`)  [:math:`kPa`]
        - 'delta sigma theta [kPa]': Increase in tangential normal stress (:math:`\\Delta \\sigma_{\\theta}`)  [:math:`kPa`]
        - 'delta tau rz [kPa]': Increase in shear stress in the rz plane (:math:`\\Delta \\tau_{rz}`)  [:math:`kPa`]

    .. figure:: images/stresses_pointload_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Nomenclature used for point load stress calculation (Budhu, 2011)

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _delta_sigma_z = (3 * pointload) / (
            (2 * np.pi * z ** 2) *
            (1 + (r / z) ** 2 ) ** (5/2) )
    _delta_sigma_r = (pointload / (2 * np.pi)) * \
                     (
                        ((3 * r ** 2 * z) / ((r ** 2 + z ** 2) ** (5/2))) -
                        (1 - 2 * poissonsratio) / (r ** 2 + z ** 2 + z * np.sqrt(r ** 2 + z ** 2))
                     )
    _delta_sigma_theta = (pointload / (2 * np.pi)) * (1 - 2 * poissonsratio) * (
            (z / ((r ** 2 + z ** 2) ** (3 / 2))) -
            (1 / (r ** 2 + z ** 2 + z * np.sqrt(r ** 2 + z ** 2)))
    )
    _delta_tau_rz = ((3 * pointload) / (2 * np.pi)) * (
        (r * z ** 2) /
        ((r ** 2 + z ** 2) ** (5 / 2))
    )

    return {
        'delta sigma z [kPa]': _delta_sigma_z,
        'delta sigma r [kPa]': _delta_sigma_r,
        'delta sigma theta [kPa]': _delta_sigma_theta,
        'delta tau rz [kPa]': _delta_tau_rz,
    }