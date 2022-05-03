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


STRESSES_STRIPLOAD = {
    'z': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'x': {'type': 'float', 'min_value': None, 'max_value': None},
    'width': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'imposedstress': {'type': 'float', 'min_value': None, 'max_value': None},
    'triangular': {'type': 'bool', },
}

STRESSES_STRIPLOAD_ERRORRETURN = {
    'delta sigma z [kPa]': np.nan,
    'delta sigma x [kPa]': np.nan,
    'delta tau zx [kPa]': np.nan,
}


@Validator(STRESSES_STRIPLOAD, STRESSES_STRIPLOAD_ERRORRETURN)
def stresses_stripload(z, x, width, imposedstress, triangular=False, **kwargs):
    """
    Calculates the stress redistribution at a point in the subsoil due to a strip load with a given width, applied at the surface.

    Two cases can be specified. By default, a uniform load is specified, but the stresses under a triangular load can also be calculated.

    :param z: Vertical distance from the soil surface (:math:`z`) [:math:`m`] - Suggested range: z >= 0.0
    :param x: Horizontal offset from the leftmost corner of the strip footing (:math:`x`) [:math:`m`]
    :param width: Width of the strip footing (:math:`B`) [:math:`m`] - Suggested range: width >= 0.0
    :param imposedstress: Maximum value of the imposed force per unit area (:math:`q_s`) [:math:`kN/m^2`]
    :param triangular: Boolean determining whether a triangular load pattern is applied (optional, default= False)

    .. math::
        R_1 = \\sqrt{x^2 + z^2}

        R_2 = \\sqrt{(x - B)^2 + z^2}

        \\cos \\left(\\alpha + \\beta \\right) = z / R_1

        \\cos \\beta = z / R_2

        \\text{Uniform load}

        \\Delta \\sigma_z = \\frac{q_s}{\\pi} \\left[ \\alpha + \\sin \\alpha \\cos \\left( \\alpha + 2 \\beta \\right) \\right]

        \\Delta \\sigma_x = \\frac{q_s}{\\pi} \\left[ \\alpha - \\sin \\alpha \\cos \\left( \\alpha + 2 \\beta \\right) \\right]

        \\Delta \\tau_{zx} = \\frac{q_s}{\\pi} \\left[ \\sin \\alpha \\sin \\left( \\alpha + 2 \\beta \\right) \\right]

        \\text{Triangular load}

        \\Delta \\sigma_z = \\frac{q_s}{\\pi} \\left( \\frac{x}{B} \\alpha - \\frac{1}{2} \\sin 2 \\beta \\right)

        \\Delta \\sigma_x = \\frac{q_s}{\\pi} \\left( \\frac{x}{B} \\alpha - \\frac{z}{B} \\ln \\frac{R_1^2}{R_2^2} + \\frac{1}{2} \\sin 2 \\beta \\right)

        \\Delta \\tau_zx = \\frac{q_s}{2 \\pi} \\left( 1 + \\cos 2 \\beta - 2 \\frac{z}{B} \\alpha \\right)

    :returns: Dictionary with the following keys:

        - 'delta sigma z [kPa]': Increase in vertical stress due to surface load (:math:`\\Delta \\sigma_z`)  [:math:`kPa`]
        - 'delta sigma x [kPa]': Increase in horizontal stress due to surface load (:math:`\\Delta \\sigma_x`)  [:math:`kPa`]
        - 'delta tau zx [kPa]': Increase in shear stress due to surface load (:math:`\\Delta \\tau_{zx}`)  [:math:`kPa`]

    .. figure:: images/stresses_stripload_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Nomenclature for inputs in stress calculation due to a strip footing

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    R_1 = np.sqrt(x ** 2 + z ** 2)
    R_2 = np.sqrt((x - width) ** 2 + z ** 2)

    _theta1 = np.arccos(z / R_1)
    _theta2 = np.arccos(z / R_2)
    if x < width:
        _theta2 = -_theta2
    beta = _theta2
    alpha = _theta1 - beta

    if triangular:
        # Calculation for triangular stress distribution
        _delta_sigma_z = (imposedstress / np.pi) * (
            (x / width) * alpha -
            0.5 * np.sin(2 * beta)
        )
        _delta_sigma_x = (imposedstress / np.pi) * (
            (x / width) * alpha -
            (z / width) * np.log((R_1 ** 2) / (R_2 ** 2)) +
            0.5 * np.sin(2 * beta)
        )
        _delta_tau_zx = (imposedstress / (2 * np.pi)) * (
            1 +
            np.cos(2 * beta) -
            2 * (z / width) * alpha
        )
    else:
        # Calculation for uniform stress distribution
        _delta_sigma_z = (imposedstress / np.pi) * (
            alpha +
            np.sin(alpha) * np.cos(alpha + 2 * beta)
        )
        _delta_sigma_x = (imposedstress / np.pi) * (
            alpha -
            np.sin(alpha) * np.cos(alpha + 2 * beta)
        )
        _delta_tau_zx = (imposedstress / np.pi) * (
            np.sin(alpha) * np.sin(alpha + 2 * beta)
        )

    return {
        'delta sigma z [kPa]': _delta_sigma_z,
        'delta sigma x [kPa]': _delta_sigma_x,
        'delta tau zx [kPa]': _delta_tau_zx,
    }


STRESSES_CIRCLE = {
    'z': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'footing_radius': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'imposedstress': {'type': 'float', 'min_value': None, 'max_value': None},
    'poissonsratio': {'type': 'float', 'min_value': 0.0, 'max_value': 0.5},
}

STRESSES_CIRCLE_ERRORRETURN = {
    'delta sigma z [kPa]': np.nan,
    'delta sigma r [kPa]': np.nan,
}


@Validator(STRESSES_CIRCLE, STRESSES_CIRCLE_ERRORRETURN)
def stresses_circle(z, footing_radius, imposedstress, poissonsratio, **kwargs):
    """
    Calculates the stress distribution below a uniformly loaded circular foundation. The stresses are calculated below the center of the circular foundation

    :param z: Depth below the base of the foundation (:math:`z`) [:math:`m`] - Suggested range: z >= 0.0
    :param footing_radius: Radius of the circular foundation (:math:`r_0`) [:math:`m`] - Suggested range: footing_radius >= 0.0
    :param imposedstress: Applied uniform stress to the circular footing (:math:`q_s`) [:math:`kPa`]
    :param poissonsratio: Poissons ratio for the soil material (:math:`\\nu`) [:math:`-`] - Suggested range: 0.0 <= poissonsratio <= 0.5

    .. math::
        \\Delta \\sigma_z = q_s \\left[ 1 - \\left( \\frac{1}{1 + (r_0 / z)^2} \\right)^{3/2} \\right]

        \\Delta \\sigma_r = \\Delta \\sigma_{\\theta} = \\frac{q_s}{2} \\left[ (1 + 2 \\nu) - \\frac{4 (1 + \\nu)}{\\sqrt{1 + (r_0 / z)^2}} + \\frac{1}{\\left[ 1 + (r_0 / z)^2 \\right]^{3/2}} \\right]

    :returns: Dictionary with the following keys:

        - 'delta sigma z [kPa]': Vertical stress increase (:math:`\\Delta \\sigma_z`)  [:math:`kPa`]
        - 'delta sigma r [kPa]': Radial stress increase (:math:`\\Delta \\sigma_r`)  [:math:`kPa`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _delta_sigma_z = imposedstress * (
        1 -
        (1 / (1 + ((footing_radius / z) ** 2))) ** (3 / 2)
    )
    _delta_sigma_r = 0.5 * imposedstress * (
        (1 + 2 * poissonsratio) -
        (4 * (1 + poissonsratio)) / np.sqrt(1 + (footing_radius / z) ** 2) +
        (1 / ((1 + ((footing_radius / z) ** 2)) ** (3 / 2)))
    )

    return {
        'delta sigma z [kPa]': _delta_sigma_z,
        'delta sigma r [kPa]': _delta_sigma_r,
    }


STRESSES_RECTANGLE = {
    'imposedstress': {'type': 'float', 'min_value': None, 'max_value': None},
    'length': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'width': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'z': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

STRESSES_RECTANGLE_ERRORRETURN = {
    'delta sigma z [kPa]': np.nan,
    'delta sigma x [kPa]': np.nan,
    'delta sigma y [kPa]': np.nan,
    'delta tau zx [kPa]': np.nan,
}


@Validator(STRESSES_RECTANGLE, STRESSES_RECTANGLE_ERRORRETURN)
def stresses_rectangle(
        imposedstress, length, width, z,
        **kwargs):
    """
    Calculates the stresses under the corner of a uniformly loaded rectangular area. Stresses under other points can be calculated by subdividing the rectangular in smaller sub-rectangles and using superposition stresses (justified because the solution is elastic). E.g. the stresses under the center of a rectangle is calculated by subdividing the rectangle into four equal sub-areas and calculating the stress below the corner of each and summing them.

    :param imposedstress: Stress applied to the uniformly loaded area (:math:`q_s`) [:math:`kPa`]
    :param length: Dimension of the longest edge of the rectangle (:math:`L`) [:math:`m`] - Suggested range: length >= 0.0
    :param width: Dimension of the shortest edge of the rectangle (:math:`B`) [:math:`m`] - Suggested range: width >= 0.0
    :param z: Depth below the footing (:math:`z`) [:math:`m`] - Suggested range: z >= 0.0

    .. math::
        \\Delta \\sigma_z = \\frac{q_s}{2 \\pi} \\left[ \\tan^{-1) \\frac{L B}{z R_3} + \\frac{L B z}{R_3} \\left( \\frac{1}{R_1^2} + \\frac{1}{R_2^2} \\right) \\right]

        \\Delta \\sigma_x = \\frac{q_s}{2 \\pi} \\left[ \\tan^{-1) \\frac{L B}{z R_3} - \\frac{L B z}{R_1^2 R_3} \\right]

        \\Delta \\sigma_y = \\frac{q_s}{2 \\pi} \\left[ \\tan^{-1) \\frac{L B}{z R_3} - \\frac{L B z}{R_2^2 R_3} \\right]

        \\Delta \\tau_{zx} = \\frac{q_s}{2 \\pi} \\left[ \\frac{B}{R_2} - \\frac{z^2 B}{R_1^2 R_3} \\right]

        \\text{where}

        R_1 = \\sqrt{L^2 + z^2}

        R_2 = \\sqrt{B^2 + z^2}

        R_3 = \\sqrt{L^2 + B^2 + z^2}

    :returns: Dictionary with the following keys:

        - 'delta sigma z [kPa]': Increase in vertical stress below the corner of the footing (:math:`\\Delta \\sigma_z`)  [:math:`kPa`]
        - 'delta sigma x [kPa]': Increase in horizontal stress in the width direction below the corner of the footing (:math:`\\Delta \\sigma_x`)  [:math:`kPa`]
        - 'delta sigma y [kPa]': Increase in horizontal stress in the length direction below the corner of the footing (:math:`\\Delta \\sigma_y`)  [:math:`kPa`]
        - 'delta tau zx [kPa]': Increase in shear stress in the zx plane below the corner of the footing (:math:`\\Delta \\tau_{zx}`)  [:math:`kPa`]

    .. figure:: images/stresses_rectangle_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Nomenclature used for calculation of stresses below the corner of a uniformly loaded rectangle

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """
    R_1 = np.sqrt(length ** 2 + z ** 2)
    R_2 = np.sqrt(width ** 2 + z ** 2)
    R_3 = np.sqrt(length ** 2 + width ** 2 + z ** 2)

    _delta_sigma_z = (imposedstress / (2 * np.pi)) * (
        np.arctan((length * width) / (z * R_3)) +
        ((length * width * z) / R_3) * ((1 / R_1 ** 2) + (1 / (R_2 ** 2)))
    )
    _delta_sigma_x = (imposedstress / (2 * np.pi)) * (
        np.arctan((length * width) / (z * R_3)) -
        ((length * width * z) / ((R_1 ** 2) * R_3))
    )
    _delta_sigma_y = (imposedstress / (2 * np.pi)) * (
        np.arctan((length * width) / (z * R_3)) -
        ((length * width * z) / ((R_2 ** 2) * R_3))
    )
    _delta_tau_zx = (imposedstress / (2 * np.pi)) * (
        (width / R_2) -
        ((z ** 2 * width) / (R_1 **2 * R_3))
    )

    return {
        'delta sigma z [kPa]': _delta_sigma_z,
        'delta sigma x [kPa]': _delta_sigma_x,
        'delta sigma y [kPa]': _delta_sigma_y,
        'delta tau zx [kPa]': _delta_tau_zx,
    }