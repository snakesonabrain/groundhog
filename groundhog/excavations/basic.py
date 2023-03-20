#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
from math import radians
import numpy as np

# Project imports
from groundhog.general.validation import Validator


EARTHPRESSURECOEFFICIENTS_FRICTIONANGLE = {
    'phi_eff': {'type': 'float', 'min_value': 20.0, 'max_value': 50.0},
}

EARTHPRESSURECOEFFICIENTS_FRICTIONANGLE_ERRORRETURN = {
    'Ka [-]': np.nan,
    'Kp [-]': np.nan,
    'theta_a [radians]': np.nan,
    'theta_p [radians]': np.nan,
}

@Validator(EARTHPRESSURECOEFFICIENTS_FRICTIONANGLE, EARTHPRESSURECOEFFICIENTS_FRICTIONANGLE_ERRORRETURN)
def earthpressurecoefficients_frictionangle(
    phi_eff,
     **kwargs):

    """
    Calculates coefficient of active and passive earth pressure based on a construction with Mohr's circle. Wall friction and wall inclination are not taken into account. The angle of the slip plane with the horizontal for the active and passive wedge are also provided.
    
    :param phi_eff: Effective friction of the soil (:math:`\\varphi^{\\prime}`) [:math:`deg`] - Suggested range: 20.0 <= effective_friction_angle <= 50.0
    
    .. math::
        K_a = \\frac{1 - \\sin \\varphi^{\\prime}}{1 + \\sin \\varphi^{\\prime}}
        
        K_p = \\frac{1 + \\sin \\varphi^{\\prime}}{1 - \\sin \\varphi^{\\prime}}
        
        \\theta_a = \\frac{\\pi}{4} + \\frac{\\varphi^{\\prime}}{2}
        
        \\theta_p = \\frac{\\pi}{4} - \\frac{\\varphi^{\\prime}}{2}
    
    :returns: Dictionary with the following keys:
        
        - 'Ka [-]': Coefficient of active earth pressure (:math:`K_a`)  [:math:`-`]
        - 'Kp [-]': Coefficient of passive earth pressure (:math:`K_p`)  [:math:`-`]
        - 'theta_a [radians]': Angle of the active slip plane with the horizontal (:math:`\\theta_a`)  [:math:`radians`]
        - 'theta_p [radians]': Angle of the passive slip plane with the horizontal (:math:`\\theta_p`)  [:math:`radians`]
    
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.

    """
    phi = np.radians(phi_eff)
    _Ka = (1 - np.sin(phi)) / (1 + np.sin(phi))
    _Kp = (1 + np.sin(phi)) / (1 - np.sin(phi))
    _theta_a = (np.pi / 4) + (phi / 2)
    _theta_p = (np.pi / 4) - (phi / 2)

    return {
        'Ka [-]': _Ka,
        'Kp [-]': _Kp,
        'theta_a [radians]': _theta_a,
        'theta_p [radians]': _theta_p,
    }


EARTHPRESSURECOEFFICIENTS_PONCELET = {
    'phi_eff': {'type': 'float', 'min_value': 20.0, 'max_value': 50.0},
    'interface_friction_angle': {'type': 'float', 'min_value': 15.0, 'max_value': 40.0},
    'wall_angle': {'type': 'float', 'min_value': 0.0, 'max_value': 70.0},
    'top_angle': {'type': 'float', 'min_value': 0.0, 'max_value': 70.0},
}

EARTHPRESSURECOEFFICIENTS_PONCELET_ERRORRETURN = {
    'KaC [-]': np.nan,
    'KpC [-]': np.nan,
}

@Validator(EARTHPRESSURECOEFFICIENTS_PONCELET, EARTHPRESSURECOEFFICIENTS_PONCELET_ERRORRETURN)
def earthpressurecoefficients_poncelet(
    phi_eff,interface_friction_angle,wall_angle,top_angle,
     **kwargs):

    """
    Calculates the active and passive earth pressure coefficients for a retaining wall with friction (characterised by an interface friction angle) and an inclination to the vertical. Inclination of the ground surface on top of the retaining wall is also taken into account. Poncelet used Coulombs limit equilibrium approach to obtain expressions for coefficients of active and passive earth pressure. Note that these coefficients are applied to the effective stress and not to total stresses as used in Coulomb's limit equilibrium analysis.
    
    :param phi_eff: Effective friction angle of the soil (:math:`\\varphi^{\\prime}`) [:math:`deg`] - Suggested range: 20.0 <= effective_friction_angle <= 50.0
    :param interface_friction_angle: Interface friction angle of the wall-soil interaction (:math:`\\delta`) [:math:`deg`] - Suggested range: 15.0 <= interface_friction_angle <= 40.0
    :param wall_angle: Angle to the vertical of the portion of the wall in contact with the soil (:math:`\\eta`) [:math:`deg`] - Suggested range: 0.0 <= wall_angle <= 70.0
    :param top_angle: Angle to the horizontal of the slope on top of the wall (:math:`\\beta`) [:math:`deg`] - Suggested range: 0.0 <= top_angle <= 70.0
    
    .. math::
        K_{aC} = \\frac{\\cos ^2 (\\varphi^{\\prime} - \\eta)}{\\cos ^2 \\eta \\cos (\\eta + \\delta) \\left[  1 + \\left( \\frac{\\sin (\\varphi^{\\prime} + \\delta) \\sin (\\varphi^{\\prime} - \\beta)}{\\cos (\\eta + \\delta) \\cos (\\eta - \\beta)}\\right)^{0.5} \\right]^2}
        
        K_{pC} = \\frac{\\cos ^2 (\\varphi^{\\prime} + \\eta)}{\\cos ^2 \\eta \\cos (\\eta - \\delta) \\left[  1 - \\left( \\frac{\\sin (\\varphi^{\\prime} + \\delta) \\sin (\\varphi^{\\prime} + \\beta)}{\\cos (\\eta - \\delta) \\cos (\\eta - \\beta)}\\right)^{0.5} \\right]^2}
    
    :returns: Dictionary with the following keys:
        
        - 'KaC [-]': Poncelet's coefficient of active earth pressure (:math:`K_{aC}`)  [:math:`-`]
        - 'KpC [-]': Poncelet's coefficient of passive earth pressure (:math:`K_{pC}`)  [:math:`-`]
    
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.

    """
    phi = np.radians(phi_eff)
    delta = np.radians(interface_friction_angle)
    eta = np.radians(wall_angle)
    beta = np.radians(top_angle)
    _KaC = (np.cos(phi - eta) ** 2 ) / \
        (np.cos(eta) ** 2 * np.cos(eta + delta) * (
            1 + (
                (np.sin(phi + delta) * np.sin(phi - beta)) /
                (np.cos(eta + delta) * np.cos(eta - beta))) ** 0.5
                )  ** 2)
    _KpC = (np.cos(phi + eta) ** 2 ) / \
        (np.cos(eta) ** 2 * np.cos(eta - delta) * (
            1 - (
                (np.sin(phi + delta) * np.sin(phi + beta)) /
                (np.cos(eta - delta) * np.cos(eta - beta))) ** 0.5
                )  ** 2)

    return {
        'KaC [-]': _KaC,
        'KpC [-]': _KpC,
    }


EARTHPRESSURECOEFFICIENTS_RANKINE = {
    'phi_eff': {'type': 'float', 'min_value': 20.0, 'max_value': 50.0},
    'wall_angle': {'type': 'float', 'min_value': 0.0, 'max_value': 70.0},
    'top_angle': {'type': 'float', 'min_value': 0.0, 'max_value': 70.0},
}

EARTHPRESSURECOEFFICIENTS_RANKINE_ERRORRETURN = {
    'KaR [-]': np.nan,
    'KpR [-]': np.nan,
    'omega_a [-]': np.nan,
    'omega_p [-]': np.nan,
    'theta_a [radians]': np.nan,
    'theta_p [radians]': np.nan,
    'ksi_a [radians]': np.nan,
    'ksi_p [radians]': np.nan,
}

@Validator(EARTHPRESSURECOEFFICIENTS_RANKINE, EARTHPRESSURECOEFFICIENTS_RANKINE_ERRORRETURN)
def earthpressurecoefficients_rankine(
    phi_eff, wall_angle, top_angle,
     **kwargs):

    """
    The expressions for an inclined wall with sloping ground are developed by Rankine (1857) and Chu (1991). The angles of the slip planes to the horizontal can also be calculated as well as the inclinations of the resultant forces to the normal to the inclined face. Note that the wall friction is not taken into account.
    
    :param phi_eff: Effective friction angle of the soil (:math:`\\varphi^{\\prime}`) [:math:`deg`] - Suggested range: 20.0 <= effective_friction_angle <= 50.0
    :param wall_angle: Angle to the vertical of the portion of the wall in contact with the soil (:math:`\\eta`) [:math:`deg`] - Suggested range: 0.0 <= wall_angle <= 70.0
    :param top_angle: Angle to the horizontal of the slope on top of the wall (:math:`\\beta`) [:math:`deg`] - Suggested range: 0.0 <= top_angle <= 70.0
    
    .. math::
        K_{aR} = \\frac{\\cos (\\beta - \\eta) \\sqrt{1 + \\sin ^2 \\varphi^{\\prime} - 2 \\sin \\varphi^{\\prime} \\cos \\omega_a }}{\\cos ^2 \\eta \\left( \\cos \\beta + \\sqrt{\\sin ^2 \\varphi^{\\prime} - \\sin ^2 \\beta} \\right)}
        
        K_{pR} = \\frac{\\cos (\\beta - \\eta) \\sqrt{1 + \\sin ^2 \\varphi^{\\prime} + 2 \\sin \\varphi^{\\prime} \\cos \\omega_p }}{\\cos ^2 \\eta \\left( \\cos \\beta - \\sqrt{\\sin ^2 \\varphi^{\\prime} - \\sin ^2 \\beta} \\right)}
        
        \\omega_a = \\sin ^{-1} \\left( \\frac{\\sin \\beta}{\\sin \\varphi^{\\prime}} \\right) - \\beta + 2 \\eta
        
        \\omega_p = \\sin ^{-1} \\left( \\frac{\\sin \\beta}{\\sin \\varphi^{\\prime}} \\right) + \\beta - 2 \\eta
        
        \\theta_a = \\frac{\\pi}{4} + \\frac{\\varphi^{\\prime}}{2} + \\frac{\\beta}{2} - \\frac{1}{2} \\sin ^{-1} \\left( \\frac{\\sin \\beta}{\\sin \\varphi^{\\prime}} \\right)
        
        \\theta_p = \\frac{\\pi}{4} - \\frac{\\varphi^{\\prime}}{2} + \\frac{\\beta}{2} + \\frac{1}{2} \\sin ^{-1} \\left( \\frac{\\sin \\beta}{\\sin \\varphi^{\\prime}} \\right)
        
        P_a = \\frac{1}{2} K_{aR} \\gamma^{\\prime} H_0^2
        
        P_p = \\frac{1}{2} K_{pR} \\gamma^{\\prime} H_0^2
        
        \\xi_a = \\tan ^{-1} \\left( \\frac{\\sin \\varphi^{\\prime} \\sin \\theta_a}{1 - \\sin \\varphi^{\\prime} \\cos \\theta_a} \\right)
        
        \\xi_p = \\tan ^{-1} \\left( \\frac{\\sin \\varphi^{\\prime} \\sin \\theta_p}{1 + \\sin \\varphi^{\\prime} \\cos \\theta_a} \\right)
    
    :returns: Dictionary with the following keys:
        
        - 'KaR [-]': Rankine coefficient of active earth pressure (:math:`K_{aR}`)  [:math:`-`]
        - 'KpR [-]': Rankine coefficient of passive earth pressure (:math:`K_{pR}`)  [:math:`-`]
        - 'omega_a [-]': Helper variable for active earth pressure (:math:`\\omega_a`)  [:math:`-`]
        - 'omega_p [-]': Helper variable for passive earth pressure (:math:`\\omega_p`)  [:math:`-`]
        - 'theta_a [radians]': Angle of active slip plane to the horizontal (:math:`\\theta_a`)  [:math:`radians`]
        - 'theta_p [radians]': Angle of passive slip plane to the horizontal (:math:`\\theta_p`)  [:math:`radians`]
        - 'ksi_a [radians]': Angle of active resultant to the normal to the wall face (:math:`\\xi_a`)  [:math:`radians`]
        - 'ksi_p [radians]': Angle of passive resultant to the normal to the wall face (:math:`\\xi_p`)  [:math:`radians`]
    
    .. figure:: images/earthpressurecoefficients_rankine_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Sketch of the problem (after Budhu, 2011)
    
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.

    """
    phi = np.radians(phi_eff)
    eta = np.radians(wall_angle)
    beta = np.radians(top_angle)

    _omega_a = np.arcsin(
        (np.sin(beta)) / np.sin(phi))  - beta + 2 * eta
    _omega_p = np.arcsin(
        np.sin(beta) / np.sin(phi)) + beta - 2 * eta
    _KaR = (np.cos(beta - eta) * np.sqrt(
        1 + (np.sin(phi) ** 2) - 2 * np.sin(phi) * np.cos(_omega_a))) / \
            ((np.cos(eta) ** 2) * (
                np.cos(beta) + np.sqrt((np.sin(phi) ** 2) - (np.sin(beta) ** 2))))
    _KpR = (np.cos(beta - eta) * np.sqrt(
        1 + (np.sin(phi) ** 2) + 2 * np.sin(phi) * np.cos(_omega_p))) / \
            ((np.cos(eta) ** 2) * (
                np.cos(beta) - np.sqrt((np.sin(phi) ** 2) - (np.sin(beta) ** 2))))
    _theta_a = 0.25 * np.pi + 0.5 * phi + 0.5 * beta - 0.5 * np.arcsin(np.sin(beta) / np.sin(phi))
    _theta_p = 0.25 * np.pi - 0.5 * phi + 0.5 * beta + 0.5 * np.arcsin(np.sin(beta) / np.sin(phi))
    _ksi_a = np.arctan(
        (np.sin(phi) * np.sin(_theta_a)) / 
        (1 - np.sin(phi) * np.cos(_theta_a)))
    _ksi_p = np.arctan(
        (np.sin(phi) * np.sin(_theta_p)) / 
        (1 - np.sin(phi) * np.cos(_theta_p)))

    return {
        'KaR [-]': _KaR,
        'KpR [-]': _KpR,
        'omega_a [-]': _omega_a,
        'omega_p [-]': _omega_p,
        'theta_a [radians]': _theta_a,
        'theta_p [radians]': _theta_p,
        'ksi_a [radians]': _ksi_a,
        'ksi_p [radians]': _ksi_p,
    }