#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruno Stuyts"

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


BENDINGSTIFFNESS_SOILMIX_METHOD1 = {
    'moment_inertia_reinforcement': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'modulus_soilmix': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'height_soilmix': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'reinforcement_offset': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'height_reinforcement': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'flange_thickness': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'connection_thickness': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'flange_width': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'modulus_reinforcement': {'type': 'float', 'min_value': 0.0, 'max_value': 300000000.0},
    'participating_width': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'tensile_strength_soilmix': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

BENDINGSTIFFNESS_SOILMIX_METHOD1_ERRORRETURN = {
    'n [-]': np.nan,
    'Ism [m4]': np.nan,
    'EI-1 [kNm2]': np.nan,
    'Mcr [kNm]': np.nan,
    'c1 [m]': np.nan,
    'c2 [m]': np.nan,
    'd [m]': np.nan,
    'c1b [m]': np.nan,
    'hw [m]': np.nan,
    'Af [m2]': np.nan,
    'rho [-]': np.nan,
    'xi,e [-]': np.nan,
    'xe [m]': np.nan,
    'I2 [m4]': np.nan,
    'EI-2 [kNm2]': np.nan,
    'EI [kNm2]': np.nan,
    'EI-eff/m [kNm2/m]': np.nan
}

@Validator(BENDINGSTIFFNESS_SOILMIX_METHOD1, BENDINGSTIFFNESS_SOILMIX_METHOD1_ERRORRETURN)
def bendingstiffness_soilmix_method1(
    moment_inertia_reinforcement, modulus_soilmix, height_soilmix, reinforcement_offset, height_reinforcement,
    flange_thickness, connection_thickness, flange_width,
    modulus_reinforcement=210000000.0, participating_width=np.nan, tensile_strength_soilmix=np.nan, **kwargs):

    """
    Calculates the bending stiffness of a soilmix wall (for use in geotechnical retaining wall calculations) using the combined stiffness of reinforcements and the soilmix material itself.
    The stiffness is calculated as the average of the cracked and uncracked stiffness of the material according to §5.4.2.3 if EN 1992-1-1. The formulae contain simplications that result in a small error of 1 to 3%.
    Note that all dimensions are in meters and units of force are in kN.
    
    :param moment_inertia_reinforcement: Moment of inertia of the reinforcement (:math:`I_a`) [m4] - Suggested range: moment_inertia_reinforcement >= 0.0
    :param modulus_soilmix: Youngs modulus of the soilmix (:math:`E_{sm}`) [kPa] - Suggested range: modulus_soilmix >= 0.0
    :param height_soilmix: Height of the soilmix material (:math:`h_{sm}`) [m] - Suggested range: height_soilmix >= 0.0
    :param reinforcement_offset: Offset of reinforcement elements (center to center) (:math:`a`) [m] - Suggested range: reinforcement_offset >= 0.0
    :param height_reinforcement: Longest dimension of the reinforcement (:math:`h_a`) [m] - Suggested range: height_reinforcement >= 0.0
    :param flange_thickness: Thickness of the flanges (:math:`t_f`) [m] - Suggested range: flange_thickness >= 0.0
    :param connection_thickness: Thickness of the connection between the flanges (:math:`t_w`) [m] - Suggested range: connection_thickness >= 0.0
    :param flange_width: Width of the flanges (:math:`b_f`) [m] - Suggested range: flange_width >= 0.0
    :param modulus_reinforcement: Youngs modulus of the reinforcement (steel by default) (:math:`E_a`) [kPa] - Suggested range: 0.0 <= modulus_reinforcement <= 300000000.0 (optional, default= 210000000.0)
    :param participating_width: Participating width for bending. For most cases, this is equal to the center-to-center distance of reinforcement (default when np.nan is used) (:math:`b_{c1}`) [m] - Suggested range: participating_width >= 0.0 (optional, default= np.nan)
    :param tensile_strength_soilmix: Tensile strength of soilmix to calculate cracking moment (:math:`f_{sm,t}`) [kPa] - Suggested range: tensile_strength_soilmix >= 0.0 (optional, default= np.nan)
    
    .. math::
        EI\\text{-eff} = \\frac{EI\\text{-1} + EI\\text{-2}}{2}
        
        EI\\text{-1}=E_a I_a + E_{sm}(I_{sm} - I_a)=E_{sm} \\left[ (n-1) I_a + I_{sm}\\right] \\quad \\text{where } n=E_a/E_{sm}
        
        I_{sm} = \\frac{b_{cl} h_{sm}^3}{12}
        
        c_1=c_2=\\frac{h_{sm}-h_a}{2}
        
        d=h_{eq}-c_2-\\frac{t_f}{2} \\quad{where } h_{eq} \\text{ can be taken as } h_{sm}
        
        c_{1,b}=c_1+\\frac{t_f}{2}
        
        h_w=h_a-2 t_f
        
        \\xi_e=-(2n-1) \\rho + \\sqrt{(2n-1)^2 \\rho^2 + 2 \\left[ (n-1) \\frac{h_{sm}}{d} + 1 \\right] \\rho}
        
        \\xi_e=\\frac{x_e}{d}
        
        \\rho=\\frac{A_f}{d b_{cl}}
        
        A_f=t_f b_f
        
        I_2=\\frac{b_{cl} x_e^3}{3} + (n-1) A_f (x_e - c_{1,b})^2 + n A_f (d-x_e)^2 + n t_w \\left( \\frac{(x_e - c_1 - t_f)^3}{3} + \\frac{(h_w - x_e + c_1 + t_f)^3}{3} \\right)
        
        EI\\text{-2} = E_{sm} I_2

        EI\\text{-eff/m} =  EI\\text{-eff} = \\frac{EI\\text{-1} + EI\\text{-2}}{2a}
    
    :returns: Dictionary with the following keys:
        
        - 'n [-]': Ratio of reinforcement and soilmix moduli (:math:`n`)  [-]
        - 'Ism [m4]': Moment of inertia of soilmix (:math:`I_{sm}`)  [m4]
        - 'EI-1 [kNm2]': Bending stiffness of the uncracked section (:math:`EI\\text{-1}`)  [kNm2]
        - 'Mcr [kNm]': Cracking moment (if tensile strength is specified) (:math:`M_{cr}`)  [kNm]
        - 'c1 [m]': Net cover above reinforcement (:math:`c_1`)  [m]
        - 'c2 [m]': Net cover below reinforcement (:math:`c_2`)  [m]
        - 'd [m]': Useful height of lower flange (:math:`d`)  [m]
        - 'c1b [m]': Gross cover on upper flange (:math:`c_{1b}`)  [m]
        - 'hw [m]': Height of the body of the reinforcement (:math:`h_w`)  [m]
        - 'Af [m2]': Flange area (:math:`A_f`)  [m2]
        - 'rho [-]': - (:math:`\\rho`)  [-]
        - 'xi,e [-]': - (:math:`\\xi_e`)  [-]
        - 'xe [m]': Effective height of the section (:math:`x_e`)  [m]
        - 'I2 [m4]': Moment of inertia of the cracked section (:math:`I_2`)  [m4]
        - 'EI-2 [kNm2]': Bending stiffness of the cracked section (:math:`EI\\text{-2}`)  [kNm2]
        - 'EI [kNm2]': Bending stiffness of the combined section (:math:`EI`)  [kNm2]
        - 'EI-eff/m [kNm2]': Bending stiffness of the combined section per unit length (:math:`EI\\text{-eff}`)  [kNm2]
    
    .. figure:: images/IPE_conventions.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Naming conventions for reinforcement geometry

    Reference - Denies, N. and Huybrechts, N. (2016). Handboek soimix-wanden - Ontwerp en uitvoering, SBRCURnet

    """
    if np.isnan(participating_width):
        participating_width = reinforcement_offset
    else:
        pass
    _n = modulus_reinforcement / modulus_soilmix
    _Ism = participating_width * height_soilmix ** 3 / 12
    _EI_1 = modulus_reinforcement * moment_inertia_reinforcement + modulus_soilmix * (_Ism - moment_inertia_reinforcement)
    _Mcr = tensile_strength_soilmix * _Ism / (height_soilmix / 2)
    _c1 = 0.5 * (height_soilmix - height_reinforcement)
    _c2 = 0.5 * (height_soilmix - height_reinforcement)
    _d = height_soilmix - _c2 - 0.5 * flange_thickness
    _c1b = _c1 + 0.5 * flange_thickness
    _hw = height_reinforcement - 2 * flange_thickness
    _Af = flange_thickness * flange_width
    _rho = _Af / (_d * participating_width)
    _xie = -(2 * _n - 1) * _rho + \
        np.sqrt(((2 * _n - 1) ** 2) * _rho **2 + 
                2 * ((_n - 1) * (height_soilmix / _d) + 1) * _rho)
    _xe = _xie * _d
    _I2 = ((participating_width * _xe ** 3) / 3) + \
        (_n - 1) * _Af * ((_xe - _c1b) ** 2) + \
        _n * _Af * ((_d - _xe) ** 2) + \
        _n * connection_thickness * ((((_xe - _c1 - flange_thickness) ** 3) / 3) + 
                             (((_hw - _xe + _c1 + flange_thickness) ** 3) / 3))
    _EI_2 = modulus_soilmix * _I2
    _EI = 0.5 * (_EI_1 + _EI_2)
    _EI_eff = 0.5 * (_EI_1 + _EI_2) / reinforcement_offset

    return {
        'n [-]': _n,
        'Ism [m4]': _Ism,
        'EI-1 [kNm2]': _EI_1,
        'Mcr [kNm]': _Mcr,
        'c1 [m]': _c1,
        'c2 [m]': _c2,
        'd [m]': _d,
        'c1b [m]': _c1b,
        'hw [m]': _hw,
        'Af [m2]': _Af,
        'rho [-]': _rho,
        'xi,e [-]': _xie,
        'xe [m]': _xe,
        'I2 [m4]': _I2,
        'EI-2 [kNm2]': _EI_2,
        'EI [kNm2]': _EI,
        'EI-eff/m [kNm2/m]': _EI_eff
    }


BENDINGSTIFFNESS_SOILMIX_METHOD2 = {
    'bendingstiffness_reinforcement': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'modulus_soilmix': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'height_soilmix': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'reinforcement_offset': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'participating_width': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

BENDINGSTIFFNESS_SOILMIX_METHOD2_ERRORRETURN = {
    'EaIa [kNm2]': np.nan,
    'EI_sm [kNm2]': np.nan,
    'EI-eff [kNm2]': np.nan,
    'EI-eff/m [kNm2/m]': np.nan,
}

@Validator(BENDINGSTIFFNESS_SOILMIX_METHOD2, BENDINGSTIFFNESS_SOILMIX_METHOD2_ERRORRETURN)
def bendingstiffness_soilmix_method2(
    bendingstiffness_reinforcement, modulus_soilmix, height_soilmix, reinforcement_offset,
    participating_width=np.nan, **kwargs):

    """
    Calculates the bending stiffness of a soilmix wall with internal reinforcements according to a simplified method. The effective bending stiffness is calculated as the sum of the stiffness of the reinforcement and the stiffness of the soilmix zone under compression.
    This method is usually on the conservative side, with bending stiffnesses 10 to 20% lower than the ones calculated with the more accurate method 1.
    
    :param bendingstiffness_reinforcement: Bending stiffness of the reinforcement (:math:`E_a I_a`) [m4] - Suggested range: bendingstiffness_reinforcement >= 0.0
    :param modulus_soilmix: Youngs modulus of the soilmix material (:math:`E_{sm}`) [kPa] - Suggested range: modulus_soilmix >= 0.0
    :param height_soilmix: Height of the soilmix material (:math:`h_{sm}`) [m] - Suggested range: height_soilmix >= 0.0
    :param reinforcement_offset: Center-to-center offset of reinforcement elements (:math:`a`) [m] - Suggested range: reinforcement_offset >= 0.0
    :param participating_width: Participating width for bending. For most cases, this is equal to the center-to-center distance of reinforcement (default when np.nan is used) (:math:`b_{c1}`) [m] - Suggested range: participating_width >= 0.0 (optional, default= np.nan)
    
    .. math::
        EI\\text{-eff} = E_a I_a + E_{sm} \\left[ \\frac{b_{c1} \\cdot \\left( \\frac{h_{sm}}{2} \\right)^3}{3} \\right]
        
        EI\\text{-eff} / m^{\\prime} = EI\\text{-eff} / a=\\frac{E_a I_a}{a} + E_{sm} \\left[ \\frac{\\left( \\frac{h_{sm}}{2} \\right)^3}{3} \\right]
    
    :returns: Dictionary with the following keys:
        
        - 'EaIa [kNm2]': Bending stiffness of the reinforcement (:math:`E_a I_a`)  [kNm2]
        - 'EI_sm [kNm2]': Bending stiffness of the soilmix (:math:`EI_{sm}`)  [kNm2]
        - 'EI-eff [kNm2]': Effective bending stiffness of the combined section (:math:`EI\\text{-eff}`)  [kNm2]
        - 'EI-eff/m [kNm2/m]': Effective bending stiffness per unit wall length (:math:`EI\\text{-eff/m}`)  [kNm2/m]
    
    Reference - Denies, N. and Huybrechts, N. (2016). Handboek soimix-wanden - Ontwerp en uitvoering, SBRCURnet

    """
    if np.isnan(participating_width):
        participating_width = reinforcement_offset
    else:
        pass

    _EaIa = bendingstiffness_reinforcement
    _EI_sm = modulus_soilmix * ((participating_width * ((0.5 * height_soilmix) ** 3)) / 3)
    _EI_eff = bendingstiffness_reinforcement + \
        modulus_soilmix * ((participating_width * ((0.5 * height_soilmix) ** 3)) / 3)
    _EI_eff_m = (bendingstiffness_reinforcement / reinforcement_offset) + \
        modulus_soilmix * ((((0.5 * height_soilmix) ** 3)) / 3)

    return {
        'EaIa [kNm2]': _EaIa,
        'EI_sm [kNm2]': _EI_sm,
        'EI-eff [kNm2]': _EI_eff,
        'EI-eff/m [kNm2/m]': _EI_eff_m,
    }