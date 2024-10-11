#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
from copy import deepcopy

# 3rd party packages
import numpy as np
from scipy.optimize import brentq
import pandas as pd
from plotly import tools
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

# Project imports
from groundhog.general.validation import Validator


STRESS_ELASTIC_ISOTROPIC = {
    'internal_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'farfield_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'borehole_radius': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'shear_modulus': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

STRESS_ELASTIC_ISOTROPIC_ERRORRETURN = {
    'radial stress [kPa]': np.nan,
    'tangential stress [kPa]': np.nan,
    'radial displacement [m]': np.nan,
}


@Validator(STRESS_ELASTIC_ISOTROPIC, STRESS_ELASTIC_ISOTROPIC_ERRORRETURN)
def stress_cylinder_elastic_isotropic(
        radius, internal_pressure, farfield_pressure, borehole_radius,
        shear_modulus=np.nan, **kwargs):
    """
    Calculates the radial and tangential stress around a cylindrical borehole under internal pressure, in a soil mass with isotropic virgin stress conditions in the given plane

    :param radius: Radius or radii at which to calculate the stresses (float or NumPy array) (:math:`r`) [:math:`m`] - Suggested range: radius >= 0.0
    :param internal_pressure: Internal pressure on the borehole (:math:`p`) [:math:`kPa`] - Suggested range: internal_pressure >= 0.0
    :param farfield_pressure: Far-field pressure, equal to the virgin horizontal stress (:math:`p_0`) [:math:`kPa`] - Suggested range: farfield_pressure >= 0.0
    :param borehole_radius: Radius of the borehole (:math:`a`) [:math:`m`] - Suggested range: borehole_radius >= 0.0
    :param shear_modulus: Shear modulus used to calculate the radial displacement (:math:`G`) [:math:`kPa`] - Suggested range: shear_modulus >= 0.0 (optional, default=np.nan). If unspecified, radial displacements are not calculated

    .. math::
        \\frac{d \\sigma_{r}}{dr} + \\frac{\\left(\\sigma_{r} - \\sigma_{\\theta} \\right)}{r} = 0

        \\sigma_r | _{r=a} = p \\\\
        \\sigma_r | _{r=a} = p_0

        \\sigma_{r} = p_0 + \\left( p - p_0 \\right) \\cdot \\left( \\frac{a}{r} \\right)^2

        \\sigma_{\\theta} = p_0 - \\left( p - p_0 \\right) \\cdot \\left( \\frac{a}{r} \\right)^2

    :returns: Dictionary with the following keys:

        - 'radial stress [kPa]': Radial stress at the specified radii (:math:`\\sigma_r`)  [:math:`kPa`]
        - 'tangential stress [kPa]': Tangential stress at the specified radii (:math:`\\sigma_{\\theta}`)  [:math:`kPa`]
        - 'radial displacement [m]': Radial displacement (calculated if shear modulus is specified) (:math:`u`)  [:math:`m`]

    .. figure:: images/stress_elastic_isotropic.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Basic sketch of the cavity expansion problem

    Reference - Yu, H.-S., 2000. Cavity Expansion Methods in Geomechanics. Springer-Science+Business Media, B.V.

    """

    _radial_stress = farfield_pressure + (internal_pressure - farfield_pressure) * ((
        borehole_radius / radius) ** 2)
    _tangential_stress = farfield_pressure - (internal_pressure - farfield_pressure) * ((
        borehole_radius / radius) ** 2)
    if np.isnan(shear_modulus):
        try:
            _radial_displacement = list(map(lambda _x: np.nan, _radial_stress))
        except:
            _radial_displacement = np.nan
    else:
        _radial_displacement = ((internal_pressure - farfield_pressure) / (2 * shear_modulus)) * \
            radius * ((borehole_radius / radius) ** 2)

    return {
        'radial stress [kPa]': _radial_stress,
        'tangential stress [kPa]': _tangential_stress,
        'radial displacement [m]': _radial_displacement,
    }


EXPANSION_TRESCA_THICKSPHERE = {
    'undrained_shear_strength': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'internal_radius': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'external_radius': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'internal_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'external_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'youngs_modulus': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'poissons_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 0.5},
    'seed': {'type': 'int', 'min_value': None, 'max_value': None},
}

EXPANSION_TRESCA_THICKSPHERE_ERRORRETURN = {
    'elastic radii [m]': np.nan,
    'elastic_radial_stress [kPa]': np.nan,
    'elastic_tangential_stress [kPa]': np.nan,
    'elastic_radial_displacement [m]': np.nan,
    'yielding_pressure [kPa]': np.nan,
    'plastic_radius [m]': np.nan,
    'elastoplastic radii [m]': np.nan,
    'elastoplastic_radial_stress [kPa]': np.nan,
    'elastoplastic_tangential_stress [kPa]': np.nan,
    'expanded_radius [m]': np.nan,
}


@Validator(EXPANSION_TRESCA_THICKSPHERE, EXPANSION_TRESCA_THICKSPHERE_ERRORRETURN)
def expansion_tresca_thicksphere(
        undrained_shear_strength, internal_radius, external_radius, internal_pressure, external_pressure,
        youngs_modulus, poissons_ratio, seed=100, **kwargs):
    """
    Calculates the stresses for cavity expansion around a thick-walled sphere in Tresca material.

    The plastic radius is first calculated from the pressure boundary conditions. Using this plastic radius, the stresses in the elastic and plastic region are calculated.

    :param undrained_shear_strength: Undrained shear strength of the material surrounding the spherical cavity (:math:`S_u`) [:math:`kPa`] - Suggested range: undrained_shear_strength >= 0.0
    :param internal_radius: Initial internal radius of the spherical cavity (:math:`a_0`) [:math:`m`] - Suggested range: internal_radius >= 0.0
    :param external_radius: Initial external radius of the region (:math:`b_0`) [:math:`m`] - Suggested range: external_radius >= 0.0
    :param internal_pressure: Internal pressure applied on the inside of the sphere (:math:`p`) [:math:`kPa`] - Suggested range: internal_pressure >= 0.0
    :param external_pressure: External pressure on the outside of the sphere (:math:`p_0`) [:math:`kPa`] - Suggested range: external_pressure >= 0.0
    :param youngs_modulus: Young's modulus of the material (:math:`E`) [:math:`kPa`] - Suggested range: youngs_modulus >= 0.0
    :param poissons_ratio: Poisson's ratio of the material (:math:`\\nu`) [:math:`-`] - Suggested range: 0 <= poissons_ratio <= 0.5
    :param seed: Number of radii at which stresses and displacements are calculated (:math:``) [:math:`-`] (optional, default= 100)

    .. math::
        \\text{Elastic solutions}

        \\sigma_r = -p_0 - (p - p_0) \\frac{\\left( \\frac{b_0}{r} \\right)^3 - 1}{\\left( \\frac{b_0}{a_0} \\right)^3 - 1}

        \\sigma_{\\theta} = \\sigma_{\\phi} = -p_0 + (p - p_0) \\frac{\\frac{1}{2} \\left( \\frac{b_0}{r} \\right)^3 - 1}{\\left( \\frac{b_0}{a_0} \\right)^3 - 1}

        u = r - r_0 = \\frac{p - p_0}{E} \\frac{(1 - 2 \\nu) r + \\frac{(1 + \\nu) b_0^3}{2 r^2}}{\\left( \\frac{b_0}{a_0} \\right)^3 - 1}

        \\text{Yield criterion}

        \\sigma_1 - \\sigma_3 = 2 \\cdot S_u

        \\text{Internal pressure for yielding } (\\sigma_1 = \\sigma_{\\theta}, \\sigma_3 = \\sigma_r)

        p = p_{1y} = p_0 + \\frac{4 \\cdot S_u}{3} \\left[ 1 - \\left( \\frac{a_0}{b_0} \\right)^3 \\right]

        \\text{Displacement at internal and external boundaries}

        u |_{r=a_0} = \\frac{2 \\cdot S_u \\cdot a_0}{E} \\left[ \\frac{2 \\cdot (1 - 2 \\nu) \\cdot a_0^3 }{3 \\cdot b_0^3} + \\frac{1 + \\nu}{3} \\right]

        u |_{r=b_0} = \\frac{2 \\cdot S_u \\cdot (1 - \\nu) \\cdot a_0 }{E \\cdot b_0^2}

        \\text{Stresses and displacements in the elastic region after yielding}

        \\sigma_r = - \\frac{4 \\cdot S_u \\cdot c^3}{3 \\cdot b_0^3} \\left[ \\left( \\frac{b_0}{r} \\right)^3 - 1 \\right] - p_0

        \\sigma_{\\theta} = \\sigma_{\\phi} = \\frac{4 \\cdot S_u \\cdot c^3}{3 \\cdot b_0^3} \\left[ \\frac{1}{2} \\left( \\frac{b_0}{r} \\right)^3 + 1 \\right] - p_0

        u = \\frac{4 \\cdot S_u \\cdot c^3}{3 \\cdot E \\cdot b_0^3} \\left[ (1 - 2 \\nu) \\cdot r + \\frac{(1 + \\nu) b_0^3}{2 \\cdot r^2} \\right]

        \\text{Stresses in the plastic region}

        \\sigma_r = - 4 \\cdot S_u \\cdot \\ln \\left( \\frac{c}{r} \\right) - \\frac{4 \\cdot S_u}{3} \\left[ 1 - \\left( \\frac{c}{b_0} \\right)^3 \\right] - p_0

        \\sigma_{\\theta} = 2 \\cdot S_u - 4 \\cdot S_u \\cdot \\ln \\left( \\frac{c}{r} \\right) - \\frac{4 \\cdot S_u}{3} \\cdot \\left[ 1 - \\left( \\frac{c}{b_0} \\right)^3 \\right] - p_0

        \\text{The pressure required to generate a plastic radius is thus}

        p = 4 \\cdot S_u \\cdot \\ln \\left( \\frac{c}{a} \\right) + \\frac{4 \\cdot S_u}{3} \\left[ 1 - \\left( \\frac{c}{b_0} \\right)^3 \\right] + p_0

        \\text{Expansion of the boundary}

        \\left( \\frac{a}{a_0} \\right)^3 = 1 + \\frac{6 (1 - \\nu) S_u c^3}{E \\cdot a_0^3} - \\frac{4 (1 - 2 \\nu) S_u}{E} \\left[ 3 \\ln \\left( \\frac{c}{a_0} \\right) + 1 - \\left( \\frac{c}{b_0} \\right)^3 \\right]

    :returns: Dictionary with the following keys:

        - 'elastic radii [m]': Radii at which elastic stresses are calculated
        - 'elastic_radial_stress [kPa]': Radial stresses for purely elastic deformation (:math:`\\sigma_{r,elastic}`)  [:math:`kPa`]
        - 'elastic_tangential_stress [kPa]': Tangential stresses for purely elastic deformation (:math:`\\sigma_{\\theta,elastic}`)  [:math:`kPa`]
        - 'elastic_radial_displacement [m]': Radial displacement for purely elastic deformation (:math:`u_{elastic}`)  [:math:`m`]
        - 'yielding_pressure [kPa]': Internal pressure which initiates yield (:math:`p_{1y}`)  [:math:`kPa`]
        - 'plastic_radius [m]': Radius of the plastic zone for the given internal and external pressures (:math:`c`)  [:math:`m`]
        - 'elastoplastic_radii [m]': Radii at which elastoplastic stresses are calculated (equal amount of point inside and outside the plastic radius)
        - 'elastoplastic_radial_stress [kPa]': Radial stresses for Tresca soil (:math:`\\sigma_r`)  [:math:`kPa`]
        - 'elastoplastic_tangential_stress [kPa]': Tangential stresses for Tresca soil (:math:`\\sigma_{\\theta}`)  [:math:`kPa`]
        - 'expanded_radius [m]': Radius of the internal wall after expansion (:math:`a`)  [:math:`m`]

    Reference - Yu, H.-S., 2000. Cavity Expansion Methods in Geomechanics. Springer-Science+Business Media, B.V.

    """
    _radii = np.linspace(internal_radius, external_radius, seed)
    _elastic_radial_stress = -external_pressure - (internal_pressure - external_pressure) * \
                             ((((external_radius / _radii) ** 3) - 1) /
                              (((external_radius / internal_radius) ** 3) - 1))
    _elastic_tangential_stress = -external_pressure + (internal_pressure - external_pressure) * \
                                 ((0.5 * ((external_radius / _radii) ** 3) - 1) /
                                  (((external_radius / internal_radius) ** 3) - 1))
    _elastic_radial_displacement = ((internal_pressure - external_pressure) / youngs_modulus) * \
                                   (((1 - 2 * poissons_ratio) * _radii +
                                     (((1 + poissons_ratio) * (external_radius ** 3)) /
                                      (2 * _radii ** 2))) / (((external_radius / internal_radius) ** 3) - 1))
    _yielding_pressure = internal_pressure + (4 * undrained_shear_strength / 3) * \
                         (1 - ((internal_radius / external_radius) ** 3))

    def wall_expansion(c, a0, b0, su, E, nu):
        return a0 * ((
            1 +
            ((6 * (1 - nu) * su * (c ** 3)) / (E * (a0 ** 3))) -
            ((4 * (1 - 2 * nu) * su) / (E)) * (3 * np.log(c / a0) +
                                               1 -
                                               ((c / b0) ** 3))) ** (1 / 3))

    def optimisation_func(c, a0, b0, su, p, p0, E, nu):
        a = wall_expansion(c, a0, b0, su, E, nu)
        return 4 * su * np.log(c / a) + (4 * su / 3) * (1 - ((c / b0) ** 3)) + p0 - p

    _plastic_radius = None
    _external_radius = external_radius

    while _plastic_radius is None:
        try:
            _plastic_radius = brentq(
                f=optimisation_func,
                a=internal_radius,
                b=_external_radius,
                args=(internal_radius, external_radius,
                      undrained_shear_strength, internal_pressure, external_pressure,
                      youngs_modulus, poissons_ratio)
            )
        except Exception as err:
            _external_radius = 0.95 * _external_radius
            if _external_radius <= internal_radius:
                raise ValueError("A feasible plastic radius could not be found. The internal pressure might be too"
                                 "low to cause plastic deformation or too high"
                                 "to find a feasible plastic radius - %s" % (str(err)))

    _expanded_radius = wall_expansion(
        c=_plastic_radius,
        a0=internal_radius,
        b0=external_radius,
        su=undrained_shear_strength,
        E=youngs_modulus,
        nu=poissons_ratio
    )

    _elastoplastic_radii_elastic = np.linspace(_plastic_radius, external_radius, seed)
    _elastoplastic_radii_plastic = np.linspace(_expanded_radius, _plastic_radius, seed)
    _elastoplastic_radii = np.append(
        _elastoplastic_radii_plastic,
        _elastoplastic_radii_elastic)
    _elastoplastic_radial_stress_elastic = \
        -((4 * undrained_shear_strength * (_plastic_radius ** 3)) / (3 * (external_radius ** 3))) * \
        (((external_radius / _elastoplastic_radii_elastic) ** 3) - 1) - external_pressure
    _elastoplastic_radial_stress_plastic = \
        -4 * undrained_shear_strength * np.log(_plastic_radius / _elastoplastic_radii_plastic) - \
        (4 * undrained_shear_strength / 3) * (1 - ((_plastic_radius / external_radius) ** 3)) - \
        external_pressure
    _elastoplastic_tangential_stress_elastic = \
        -((4 * undrained_shear_strength * (_plastic_radius ** 3)) / (3 * (external_radius ** 3))) * \
        (0.5 * ((external_radius / _elastoplastic_radii_elastic) ** 3) + 1) - external_pressure
    _elastoplastic_tangential_stress_plastic = \
        2 * undrained_shear_strength - \
        4 * undrained_shear_strength * np.log(_plastic_radius / _elastoplastic_radii_plastic) - \
        (4 * undrained_shear_strength / 3) * (1 - ((_plastic_radius / external_radius) ** 3)) - \
        external_pressure
    _elastoplastic_radial_stress = np.append(
        _elastoplastic_radial_stress_plastic, _elastoplastic_radial_stress_elastic)
    _elastoplastic_tangential_stress = np.append(
        _elastoplastic_tangential_stress_plastic, _elastoplastic_tangential_stress_elastic)

    return {
        'elastic radii [m]': _radii,
        'elastic_radial_stress [kPa]': _elastic_radial_stress,
        'elastic_tangential_stress [kPa]': _elastic_tangential_stress,
        'elastic_radial_displacement [m]': _elastic_radial_displacement,
        'yielding_pressure [kPa]': _yielding_pressure,
        'plastic_radius [m]': _plastic_radius,
        'elastoplastic radii [m]': _elastoplastic_radii,
        'elastoplastic_radial_stress [kPa]': _elastoplastic_radial_stress,
        'elastoplastic_tangential_stress [kPa]': _elastoplastic_tangential_stress,
        'expanded_radius [m]': _expanded_radius,
    }


EXPANSION_CYLINDER_TRESCA = {
    'insitu_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'borehole_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'undrained_shear_strength': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'shear_modulus': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'poissons_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 0.5},
    'max_radius_multiplier': {'type': 'float', 'min_value': 1.0, 'max_value': None},
    'number_radii': {'type': 'int', 'min_value': None, 'max_value': None},
}

EXPANSION_CYLINDER_TRESCA_ERRORRETURN = {
    'yielding': False,
    'pressure expansion function': dict(),
    'yielding pressure [kPa]': np.nan,
    'radii [m]': np.nan,
    'radial stresses [kPa]': np.nan,
    'tangential stresses [kPa]': np.nan,
    'elastic wall expansion [m]': np.nan,
    'plastic wall expansion [m]': np.nan,
    'plastic radius [m]': np.nan,
}


@Validator(EXPANSION_CYLINDER_TRESCA, EXPANSION_CYLINDER_TRESCA_ERRORRETURN)
def expansion_cylinder_tresca(
        insitu_pressure, borehole_pressure, diameter, undrained_shear_strength, shear_modulus,
        poissons_ratio=0.5, max_radius_multiplier=10.0, number_radii=250, **kwargs):
    """
    Calculates the cavity expansion for a cylinder in Tresca material. The relation between borehole radius + plastic radius and pressure differential is calculated first.

    This relation is then used to evaluate where the imposed pressure lies, whether it causes any plasticity around the borehole and whether it does not cause overall borehole failure.

    The stresses for the given pressure are then calculated.

    :param insitu_pressure: Isotropic horizontal stress in the soil mass before borehole excavation (:math:`p_0`) [:math:`kPa`] - Suggested range: insitu_pressure >= 0.0
    :param borehole_pressure: Pressure on the borehole wall due to drilling fluids or concrete (:math:`p`) [:math:`kPa`] - Suggested range: borehole_pressure >= 0.0
    :param diameter: Borehole initial diameter (equations are formulated in terms of radius but diameter is more convenient as input) (:math:`2 \\cdot a_0`) [:math:`m`] - Suggested range: diameter >= 0.0
    :param undrained_shear_strength: Undrained shear strength of the material surrounding the borehole (:math:`S_u`) [:math:`kPa`] - Suggested range: undrained_shear_strength >= 0.0
    :param shear_modulus: Shear modulus of the material surrounding the borehole (:math:`G`) [:math:`kPa`] - Suggested range: shear_modulus >= 0.0
    :param poissons_ratio: Poissons ratio of the material surrounding the borehole (default for undrained material)) (:math:`\\nu`) [:math:`-`] - Suggested range: 0.0 <= poissons_ratio <= 0.5 (optional, default= 0.5)
    :param max_radius_multiplier: Multiplier on borehole radius for determining the maximum extent of the calculation (:math:``) [:math:`-`] - Suggested range: max_radius_multiplier >= 1.0 (optional, default= 10.0)
    :param number_radii: Number of radii considered for the calculation (:math:``) [:math:`-`] (optional, default= 250)

    .. math::
        \\text{Elastic properties}

        n = \\frac{4 \\cdot S_u \\cdot (1 - \\nu^2) }{E}

        E = 2 \\cdot G \\cdot (1 + \\nu)

        \\text{Plastic radius and wall radius expansion during plastic deformation}

        \\left( \\frac{c}{a} \\right)^2 = \\left( \\frac{a_0}{a} \\right)^2 + \\frac{1}{n} \\cdot \\left[ 1 - \\left( \\frac{a_0}{a} \\right)^2 \\right]

        \\frac{p - p_0}{2 \\cdot S_u} = \\frac{1}{2} + \\frac{1}{2} \\cdot \\ln \\left[ \\frac{G}{S_u} \\cdot \\left( 1 - \\left( \\frac{a_0}{a} \\right)^2 \\right) + \\left( \\frac{a_0}{a} \\right)^2 \\right]

        \\text{Elastic stresses and displacements can be calculated by taking the limit for the outer radius going to infinity}

        \\sigma_r = - p_0 - (p - p_0) \\cdot \\frac{\\frac{b_0^2}{r^2} - 1}{\\frac{b_0^2}{a_0^2} - 1}  \\implies \\sigma_r = -p_0 - (p - p_0) \\cdot \\frac{a_0^2}{r^2}

        \\sigma_{\\theta} = -p_0 + (p - p_0) \\cdot \\frac{\\frac{b_0^2}{r^2} + 1}{\\frac{b_0^2}{a_0^2} - 1} \\implies \\sigma_{\\theta} = -p_0 + (p - p_0) \\cdot \\frac{a_0^2}{r^2}

        u = \\frac{(1 + \\nu) \\cdot (p - p_0)}{E} \\cdot \\frac{a_0^2}{b_0^2 - a_0^2} \\cdot \\left[ (1 - 2 \\cdot \\nu) \\cdot r + \\frac{b_0^2}{r} \\right] \\implies u = \\frac{(1 + \\nu) \\cdot (p - p_0)}{E} \\cdot \\frac{a_0^2}{r}

        \\text{The plasticity critertion can be expressed as:}

        \\sigma_{r,r=a_0} - \\sigma_{theta,r=a_0} = 2 \\cdot S_u \\implies -(p - p_0)  = S_u

        \\text{The elastic stresses and displacements outside of the plastic zone can be written as:}

        \\sigma_r = -\\frac{S_u \\cdot c^2}{b_0^2} \\cdot \\left( \\frac{b_0^2}{r^2} - 1 \\right) - p_0 \\implies \\sigma_r = -\\frac{S_u \\cdot c^2}{r^2} - p_0

        \\sigma_{\\theta} = \\frac{S_u \\cdot c^2}{b_0^2} \\cdot \\left( \\frac{b_0^2}{r^2} + 1 \\right) - p_0 \\implies \\sigma_{\\theta} = \\frac{S_u \\cdot c^2}{r^2} - p_0

        u = \\frac{(1 + \\nu) \\cdot S_u \\cdot c^2}{E \\cdot b_0^2} \\cdot \\left[ (1 - 2 \\cdot \\nu) \\cdot r + \\frac{b_0^2}{r} \\right] \\implies u = \\frac{(1 + \\nu) \\cdot S_u \\cdot c^2}{E} \\cdot \\frac{1}{r}

        \\text{Stresses in the plastic zone:}

        \\sigma_r = -p_0 - S_u - 2 \\cdot S_u \\cdot \\ln \\left( \\frac{c}{r} \\right)

        \\sigma_{\\theta} = -p_0 + S_u - 2 \\cdot S_u \\cdot \\ln \\left( \\frac{c}{r} \\right)

    :returns: Dictionary with the following keys:

        - 'yielding': Boolean determining whether plastic deformation is taking place or not
        - 'pressure expansion function': Dictionary with the pressure-expansion relation with keys `expansion [m]` and `pressure difference [kPa]`
        - 'yielding pressure [kPa]': Borehole pressure at which yield occurs [:math:`kPa`]
        - 'radii [m]': Numpy array with radii used for the stress and displacement calculation [:math:`m`]
        - 'radial stresses [kPa]': Numpy array with the radial stresses around the borehole [:math:`kPa`]
        - 'tangential stresses [kPa]': Numpy array with the tangential stresses around the borehole [:math:`kPa`]
        - 'elastic wall expansion [m]': Borehole elastic wall expansion [:math:`m`]
        - 'plastic wall expansion [m]': Borehole plastic wall expansion (:math:`a - a_0`)  [:math:`m`]
        - 'plastic radius [m]': Radius of the plastic zone (:math:`c`)  [:math:`m`]

    Reference - Yu, H.-S., 2000. Cavity Expansion Methods in Geomechanics. Springer-Science+Business Media, B.V.

    """
    _yielding_pressure = insitu_pressure + undrained_shear_strength
    _youngs_modulus = 2 * shear_modulus * (1 + poissons_ratio)
    if abs(insitu_pressure - borehole_pressure) < undrained_shear_strength:
        _yielding = False
    else:
        _yielding = True
    _borehole_radius = 0.5 * diameter
    _radii = np.linspace(_borehole_radius, max_radius_multiplier * _borehole_radius, number_radii)
    _pressure_expansion_function = {
        'expansion [m]': _radii - _borehole_radius,
        'pressure difference [kPa]': undrained_shear_strength * (
                1 + np.log((shear_modulus/undrained_shear_strength) +
                           (1 - shear_modulus/undrained_shear_strength) * ((_borehole_radius / _radii) ** 2)))
    }

    def _elastoplastic_radialstress(r, c, su, p0):
        if r < c:
            # Plastic zone
            return -p0 - su - 2 * su * np.log(c / r)
        else:
            # Elastic zone
            return -su * ((c ** 2) / (r ** 2)) - p0

    def _elastoplastic_tangentialstress(r, c, su, p0):
        if r < c:
            # Plastic zone
            return -p0 + su - 2 * su * np.log(c / r)
        else:
            # Elastic zone
            return su * ((c ** 2) / (r ** 2)) - p0

    if (borehole_pressure - insitu_pressure) > _pressure_expansion_function['pressure difference [kPa]'].max():
        warnings.warn("Selected borehole pressure leads to excessive deformation. Stresses are not calculated.")
        _radial_stresses = np.array(list(map(lambda _x: np.nan, _radii)))
        _tangential_stresses = np.array(list(map(lambda _x: np.nan, _radii)))
        _elastic_wall_expansion = np.nan
        _plastic_wall_expansion = np.nan
        _plastic_radius = np.nan
    else:
        if _yielding:
            # Calculate the plastic radius
            _first_term = (_borehole_radius ** 2) * (1 - shear_modulus / undrained_shear_strength)
            _second_term = np.exp(((borehole_pressure - insitu_pressure) / undrained_shear_strength) - 1) - \
                           (shear_modulus / undrained_shear_strength)
            _expanded_radius = np.sqrt(_first_term / _second_term)
            _plastic_wall_expansion = _expanded_radius - _borehole_radius
            _n = 4 * (1 + (poissons_ratio ** 2)) * undrained_shear_strength / _youngs_modulus
            _plastic_radius = _expanded_radius * np.sqrt(
                ((_borehole_radius / _expanded_radius) ** 2) +
                (1 / _n) * (1 - ((_borehole_radius / _expanded_radius) ** 2)))

            _radial_stresses = np.array(
                list(map(lambda _r: _elastoplastic_radialstress(
                    r=_r, c=_plastic_radius, su=undrained_shear_strength, p0=insitu_pressure), _radii)))
            _tangential_stresses = np.array(
                list(map(lambda _r: _elastoplastic_tangentialstress(
                    r=_r, c=_plastic_radius, su=undrained_shear_strength, p0=insitu_pressure), _radii)))
        else:
            _radial_stresses = -insitu_pressure - (borehole_pressure - insitu_pressure) * \
                               ((_borehole_radius ** 2) / (_radii ** 2))
            _tangential_stresses = -insitu_pressure + (borehole_pressure - insitu_pressure) * \
                                   ((_borehole_radius ** 2) / (_radii ** 2))
            _plastic_wall_expansion = 0
            _plastic_radius = _borehole_radius
        _elastic_wall_expansion = ((1 + poissons_ratio) * (borehole_pressure - insitu_pressure)) / _youngs_modulus

    return {
        'yielding': _yielding,
        'pressure expansion function': _pressure_expansion_function,
        'yielding pressure [kPa]': _yielding_pressure,
        'radii [m]': _radii,
        'radial stresses [kPa]': _radial_stresses,
        'tangential stresses [kPa]': _tangential_stresses,
        'elastic wall expansion [m]': _elastic_wall_expansion,
        'plastic wall expansion [m]': _plastic_wall_expansion,
        'plastic radius [m]': _plastic_radius,
    }