#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


class HardeningSoil(object):
    """
    Class for setting up the equations for the hardening soil model
    and performing fitting of drained triaxial tests.

    Reference - Schanz, T., Vermeer, P.A., Bonnier, P.G. (1999). The hardening soil model: Formulation and verification. Beyond 2000 in Computational Geotechnics - 10 years of PLAXIS. 
    """
    def __init__(self, friction_angle, cohesion, Rf=0.9):
        """
        Sets up the material using the strength parameters friction angle (:math:`\\varphi_p`), cohesion (:math:`c`) and failure ratio (:math:`R_f`)
        
        In absence of other guidance, :math:`R_f`=0.9 is a good default setting.

        .. math::
            q_a = \\frac{q_f}{R_f}
        """
        self.friction_angle = friction_angle
        self.cohesion = cohesion
        self.Rf = Rf

    def set_reference_moduli(self, E50_ref, Eur_ref, Eoed_ref, p_ref):
        """
        Sets the stiffnesses for the hardening soil model using moduli identified at a reference pressure (typically 100kPa).

        :param E50_ref: :math:`E_{50}^{ref}` [kPa] is the secant modulus at 50% of the maximum deviator stress for a drained triaxial test carried out at :math:`p_{ref}`
        :param Eur_ref: :math:`E_{ur}^{ref}` [kPa] is the unloading reloading stiffness for a drained triaxial test carried out at :math:`p_{ref}`
        :param Eoef_ref: :math:`E_{oed}^{ref}` [kPa] is the tangent stiffness for primary oedometric loading at :math:`p_{ref}` 
        :param p_ref: :math:`p_{ref}` [kPa] is the reference pressure at which the tests are carried out. Note that the same pressure is used for all tests.
        """
        self.E50_ref = E50_ref
        self.Eur_ref = Eur_ref
        self.Eoed_ref = Eoed_ref
        self.p_ref = p_ref

    @staticmethod
    def calculate_stiffnesses(sigma_3, sigma_1, m, cohesion, friction_angle, E50_ref, Eur_ref, Eoed_ref, p_ref):
        """
        Calculates the stiffnesses :math:`E_{50}`, :math:`E_{ur}` and :math:`E_{oed}` for a given consolidation pressure :math:`\\sigma_{3}`.

        :param sigma_3: Consolidation pressure considered for shearing behaviour
        :param sigma_1: Consolidation pressure considered for volumetric compression behaviour
        :param m: Stress exponent governing the stress dependence of the moduli.
        
        .. math::
            E_{50} = E_{50}^{ref} \\left( \\frac{\\sigma_3 + c \\cot \\varphi}{p_{ref} + c \\cot \\varphi} \\right)^m

            E_{ur} = E_{ur}^{ref} \\left( \\frac{\\sigma_3 + c \\cot \\varphi}{p_{ref} + c \\cot \\varphi} \\right)^m

            E_{oed} = E_{oed}^{ref} \\left( \\frac{\\sigma_1 + c \\cot \\varphi}{p_{ref} + c \\cot \\varphi} \\right)^m
        """
        _E50 = E50_ref * (
            (sigma_3 + cohesion * (1 / np.tan(np.radians(friction_angle)))) /
            (p_ref + cohesion * (1 / np.tan(np.radians(friction_angle))))
        ) ** m
        _Eur = Eur_ref * (
            (sigma_3 + cohesion * (1 / np.tan(np.radians(friction_angle)))) /
            (p_ref + cohesion * (1 / np.tan(np.radians(friction_angle))))
        ) ** m
        _Eoed = Eoed_ref * (
            (sigma_1 + cohesion * (1 / np.tan(np.radians(friction_angle)))) /
            (p_ref + cohesion * (1 / np.tan(np.radians(friction_angle))))
        ) ** m
    
        return _E50, _Eur, _Eoed

    def calculate_drainedtriaxial(self, sigma3, sigma1_0, m, N=100):
        """
        Calculates the response in a drained triaxial tests based on the parameters entered by the user.

        :param sigma3: Radial consolidation stress [kPa]
        :param sigma1_0: Initial vertical consolidation stress [kPa]
        :param m: Stress exponent governing the stress dependency of the stiffness [-]
        :param N: Number of points where :math:`q` is calculated (:math:`\\sigma_1` is spaced evenly between :math:`\\sigma_{1,0}` and :math:`\\sigma_{1,f}`) 
        .. math::
            q_f = \\frac{6 \\sin \\varphi}{3 - \\sin \\varphi} \\left( p + c \cot \\varphi \\right)

            q_a = \\frac{q_f}{R_f}

            \\epsilon_1 = \\frac{q_a}{2 E_{50}} \\frac{(\\sigma_1 - \\sigma_3)}{q_a - (\\sigma_1 - \\sigma_3)} \\\\ \\text{for} \\\\ q \\leq q_f
        """
        _p0 = (1 / 3) * (sigma1_0 + 2 * sigma3)
        _qf = (
            (6 * np.sin(np.radians(self.friction_angle))) /
            (3 - np.sin(np.radians(self.friction_angle)))) * \
                (_p0 + self.cohesion * (1 / np.tan(np.radians(self.friction_angle))))
        _qa = _qf / self.Rf
        _sigma1_f = _qf + sigma3
        _sigma1 = np.linspace(sigma1_0, _sigma1_f, N)
        E50, Eur, Eoed = self.calculate_stiffnesses(
            sigma_3=sigma3,
            sigma_1=sigma1_0,
            m=m,
            cohesion=self.cohesion,
            friction_angle=self.friction_angle,
            E50_ref=self.E50_ref,
            Eur_ref=self.Eur_ref,
            Eoed_ref=self.Eoed_ref,
            p_ref=self.p_ref
        )
        _epsilon1 = (_qa / (2 * E50)) * (
            (_sigma1 - sigma3) /
            (_qa - (_sigma1 - sigma3))
        )
        _q = _sigma1 - sigma3

        return {
            'q [kPa]': _q,
            'epsilon [-]': _epsilon1,
            'E50 [kPa]': E50,
            'qf [kPa]': _qf,
            'qa [kPa]': _qa,
            'p0 [kPa]': _p0
        }