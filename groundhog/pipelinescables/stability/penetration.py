#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruno Stuyts"

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

"""
Implements pipeline penetration formulae
"""


CONTACTWIDTH = {
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
    'penetration': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
}

CONTACTWIDTH_ERRORRETURN = {
    'B [m]': np.nan,
}

@Validator(CONTACTWIDTH, CONTACTWIDTH_ERRORRETURN)
def contactwidth(
    diameter,penetration,
     **kwargs):

    """
    Calculates the contact width depending on the pipeline penetration. The contact width increases until the pipeline penetration reaches half of the diameter
    
    :param diameter: Pipeline or cable diameter (:math:`D`) [m] - Suggested range: 0.0 <= diameter <= 2.0
    :param penetration: Pipeline or cable penetration (:math:`z`) [m] - Suggested range: 0.0 <= penetration <= 2.0
    
    .. math::
        B = 2 \\cdot \\sqrt{D \\cdot z - z^2} \\quad \\text{for } z < D/2
        
        B=D \\quad \\text{for } z \\geq D/2
    
    :returns: Dictionary with the following keys:
        
        - 'B [m]': Contact width of the pipeline or cable with the soil (:math:`B`)  [m]
    
    Reference - DNV-RP-F114

    """
    if penetration < (diameter / 2):
        _B = 2 * np.sqrt(diameter * penetration - penetration ** 2)
    else:
        _B = diameter

    return {
        'B [m]': _B,
    }


PENETRATEDAREA = {
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
    'penetration': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
}

PENETRATEDAREA_ERRORRETURN = {
    'Abm [m2]': np.nan,
    'B [m]': np.nan,
}

@Validator(PENETRATEDAREA, PENETRATEDAREA_ERRORRETURN)
def penetratedarea(
    diameter,penetration,
     **kwargs):

    """
    Calculates the penetrated area of the pipeline below the seabed. Note that for penetrations above half of the diameter, the entire displaced area of soil is counted, not just the submerged pipeline area.
    
    :param diameter: Pipeline or cable diameter (:math:`D`) [m] - Suggested range: 0.0 <= diameter <= 2.0
    :param penetration: Pipeline penetration (:math:`z`) [m] - Suggested range: 0.0 <= penetration <= 2.0
    
    .. math::
        A_{bm} = \\arcsin \\left( \\frac{B}{D} \\right) \\cdot \\frac{D^2}{4} - B \\cdot \\frac{D}{4} \\cdot \\cos \\left( \\arcsin(B/D) \\right) \\quad \\text{for } z<D/2
        
        A_{bm} = \\frac{\\pi \\cdot D^2}{8} + D \\cdot \\left( z - D/2 \\right) \\quad \\text{for } z \\geq D/2
    
    :returns: Dictionary with the following keys:
        
        - 'Abm [m2]': Penetrated cross-sectional area of the pipeline or cable (:math:`A_{bm}`)  [m2]
        - 'B [m]': Contact width (intermediate output of the calculation) (:math:`B`)  [m]
    
    Reference - DNV-RP-F114

    """
    _B = contactwidth(diameter=diameter, penetration=penetration)['B [m]']

    if penetration < (0.5 * diameter):
        _Abm = np.arcsin(_B / diameter) * ((diameter ** 2) / 4) - \
            _B * (diameter / 4) * np.cos(np.arcsin(_B / diameter))
    else:
        _Abm = (np.pi * (diameter ** 2) / 8) + \
                diameter * (penetration - 0.5 * diameter)
    
    return {
        'Abm [m2]': _Abm,
        'B [m]': _B,
    }

EMBEDMENT_UNDRAINED_METHOD1 = {
    'diameter': {'type': 'float', 'min_value': 0.01, 'max_value': 2.0},
    'undrained_shear_strength': {'type': 'float', 'min_value': 0.0, 'max_value': 500.0},
    'k_su': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'gamma_eff': {'type': 'float', 'min_value': 2.0, 'max_value': 12.0},
    'penetration': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
    'Nc': {'type': 'float', 'min_value': 4.0, 'max_value': 9.0},
    'roughness': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
}

EMBEDMENT_UNDRAINED_METHOD1_ERRORRETURN = {
    'F [-]': np.nan,
    'z_su0 [m]': np.nan,
    'B [m]': np.nan,
    'su0 [kPa]': np.nan,
    'Abm [m2]': np.nan,
    'Qv0 [kN/m]': np.nan,
    'Qv [kN/m]': np.nan,
}

@Validator(EMBEDMENT_UNDRAINED_METHOD1, EMBEDMENT_UNDRAINED_METHOD1_ERRORRETURN)
def embedment_undrained_method1(
    diameter,undrained_shear_strength,k_su,gamma_eff,penetration,
    Nc=5.14,roughness=0.67, **kwargs):

    """
    Calculates pipeline embedment in soil which behaves in an undrained manner. 

    Method 1 uses bearing capacity theory to calculate the embedment where the submerged weight of the pipeline is in equilibrium with the vertical bearing capacity of the soil. The method works in soil with linearly increasing undrained strength. The method includes a depth correction factor and a buoyancy term.

    Note that the method assumes quasi-static pipeline penetration. In reality load concentration effects at the touchdown point and dynamic lay effects will lead to increased penetration.

    This formula calculates the bearing capacity for a specified depth. For determining the pipeline penetration, a root finding routine needs to be applied to find the penetration where the bearing capacity is in equilbrium with the submerged weight of the pipeline.
    
    :param diameter: Pipeline diameter (:math:`D`) [m] - Suggested range: 0.01 <= diameter <= 2.0
    :param undrained_shear_strength: Undrained shear strength at the seabed (:math:`s_(u,z=0}`) [kPa] - Suggested range: 0.0 <= undrained_shear_strength <= 500.0
    :param k_su: Linear rate of undrained shear strength increase (:math:`\\rho`) [kPa/m] - Suggested range: 0.0 <= k_su <= 10.0
    :param gamma_eff: Submerged unit weight (:math:`kN/m3`) [kN/m3] - Suggested range: 2.0 <= gamma_eff <= 12.0
    :param penetration: Penetration depth for which the pipeline penetration is calculated (:math:`m`) [m] - Suggested range: 0.0 <= penetration <= 2.0
    :param Nc: Bearing capacity factor (:math:`N_c`) [-] - Suggested range: 4.0 <= Nc <= 9.0 (optional, default= 5.14)
    :param roughness: Measure for the roughness of the pipe of cable (:math:`r`) [-] - Suggested range: 0.0 <= roughness <= 1.0 (optional, default= 0.67)
    
    .. math::
        Q_v = Q_{v0} \\cdot \\left( 1 + d_{ca} \\right) + \\gamma^{\\prime} \\cdot A_{bm}
        
        Q_{v0} = F \\cdot \\left( N_c \\cdot s_{u,0} + \\rho \\cdot B/4 \\right) \\cdot B
        
        z_{su,0} = 0 \\quad \\text{for } z < \\frac{D}{2} \\cdot \\left( 1 - \\frac{\\sqrt{2}}{2} \\right)
        
        z_{su,0} = z + \\frac{D}{2} \\cdot \\left( \\sqrt{2} - 1 \\right) - \\frac{B}{2} \\quad \\text{for } z \\geq \\frac{D}{2} \\cdot \\left( 1 - \\frac{\\sqrt{2}}{2} \\right)
        
        s_{u,0} = s_{u,z=0} + \\rho \\cdot z_{s_{u,0}}
        
        d_{ca} = 0.3 \\cdot \\frac{s_{u,1}}{s_{u,2}} \\cdot \\arctan \\left( \\frac{z_{s_{u,0}}}{B} \\right)
        
        s_{u,1} = \\frac{s_{u,z=0} + s_{u,0}}{2}
        
        s_{u,2} = \\frac{Q_{v0}}{B \\cdot N_c}
    
    :returns: Dictionary with the following keys:
        
        - 'F [-]': Roughness correction factor (:math:`F`)  [-]
        - 'z_su0 [m]': Reference z-level for depth effects (:math:`z_(s_{u,0}}`)  [m]
        - 'B [m]': Contact width (:math:`B`)  [m]
        - 'su0 [kPa]': Undrained shear strength at the reference z-level (:math:`s_{u,0}`)  [kPa]
        - 'Abm [m2]': Penetrated cross-sectional area of the pipe (:math:`A_{bm}`)  [m2]
        - 'Qv0 [kN/m]': Bearing capacity (excluding depth effects and buoyancy) (:math:`Q_{v0}`)  [kN/m]
        - 'Qv [kN/m]': Bearing capacity (including depth and buoyancy effect) (:math:`Q_v`)  [kN/m]
    
    Reference - DNV-RP-F114

    """
    _B = contactwidth(diameter=diameter, penetration=penetration)['B [m]']
    _Abm = penetratedarea(diameter=diameter, penetration=penetration)['Abm [m2]']
    
    _z_check = 0.5 * diameter * (1 - 0.5 * np.sqrt(2))
    if penetration < _z_check:
        _z_su0 = 0
    else:
        _z_su0 = penetration + 0.5 * diameter * (np.sqrt(2) - 1) - 0.5 * _B

    _su0 = undrained_shear_strength + k_su * _z_su0

    _dimensionless_group_su = k_su * _B / _su0

    _F_rough_x = [
        0.0, 0.0660278490866588, 0.2710676104073792, 0.453409862867181, \
        0.6357036715829709, 0.8406756116620753, 1.0682191068676683, \
        1.340936162679618, 1.681531439178196, 2.0446576703320916, \
        2.4529415448621683, 2.9290378530550796, 3.4275973785469627, \
        3.9259748730007415, 4.424223184137157, 4.922342311956207, \
        5.420390976147605, 5.9183633047423765, 6.416259297740526, \
        6.914102443017933, 7.411886868605631, 7.909647806317443, \
        8.40736764024646, 8.905058114330618, 9.40271335660095, \
        9.90035098296437, 10.397982737358818, 10.89557925993944, \
        11.39316403858212, 11.890719457379944, 12.38828074814674, \
        12.88581855103765, 13.383350481959589, 13.880864796974617, \
        14.378384983958613, 14.875881683066726, 15.373372510205863, \
        15.87085746537604]
    _F_rough_y = [
        1.0, 1.0204138439974173, 1.0651991164612953, 1.107413353835584, \
        1.1482517888799422, 1.1911109380819176, 1.232964036315663, \
        1.2736698226489054, 1.3160024268008903, 1.3561759266033704, \
        1.39475006516837, 1.433082670076358, 1.467332429714635, \
        1.496412507870749, 1.5218237798137153, 1.5435662455435335, \
        1.5633075442479982, 1.58088091200833, 1.596286348824529, \
        1.610190910371713, 1.622427832731102, 1.6339976994153738, \
        1.6444002186681892, 1.6539689183271071, 1.6625370344733486, \
        1.6706048588632516, 1.6785059193343752, 1.685406396292822, \
        1.6919733454137098, 1.6977064749407005, 1.7036063683864706, \
        1.708839206157123, 1.7139052800089958, 1.7184710621045305, \
        1.7232036081188442, 1.7272690984580403, 1.731167824878457, \
        1.7348997873800942]
    _F_smooth_x = [
        0.0, 0.230576714370986, 0.1107941746639902, 0.6331200052356838, \
        1.041484619339113, 1.517563311625111, 2.0160758613652257, \
        2.5144122520362067, 3.0126312033277665, 3.510726843270932, \
        4.008740275648503, 4.506671500460477, 5.004520517706856, \
        5.502346047077351, 6.000106984789165, 6.497844434625094, \
        6.995540780678224, 7.493207766886499, 7.990839521280947, \
        8.488465403706424, 8.986061926287046, 9.483658448867669, \
        9.981219739634463, 10.478751670556402, 10.976289473447313, \
        11.473827276338223, 11.971329847415308, 12.468838290461363, \
        12.966334989569477, 13.463831688677589, 13.961316643847764, \
        14.458789855079983, 14.956257194343248, 15.453712789668558, \
        15.905945089742392]
    _F_smooth_y = [
        1.0, 1.038824548008619, 1.0077010572726093, 1.0748756646169828, \
        1.1157428070651996, 1.1535751202168492, 1.1864907685048909, \
        1.2144034992295487, 1.2389809515786177, 1.260056361633318, \
        1.278797076825106, 1.2952030971539816, 1.309274422619945, \
        1.3226786924107905, 1.334248559095062, 1.3451513701042157, \
        1.354886833681913, 1.3637884776657136, 1.3716895381368372, \
        1.3794238346891812, 1.386324311647628, 1.393224788606075, \
        1.399124682051845, 1.404190755903718, 1.40942359367437, \
        1.4146564314450225, 1.418888685702998, 1.4232877038797531, \
        1.4273531942189492, 1.4314186845581454, 1.4351506470597826, \
        1.4385490817238609, 1.4417807524691597, 1.4446788953768996, \
        1.4473118862635443]
    _F_smooth = np.interp(_dimensionless_group_su, _F_smooth_x, _F_smooth_y)
    _F_rough = np.interp(_dimensionless_group_su, _F_rough_x, _F_rough_y)
    _F = np.interp(roughness, [0, 1], [_F_smooth, _F_rough])
    
    _Qv0 = _F * (Nc * _su0 + 0.25 * k_su * _B) * _B
    _su1 = 0.5 * (undrained_shear_strength + _su0)
    _su2 = _Qv0 / (_B * Nc)
    _dca = 0.3 * (_su1 / _su2) * np.arctan(_z_su0 / _B)    
    _Qv = _Qv0 * (1 + _dca) + gamma_eff * _Abm

    return {
        'F [-]': _F,
        'z_su0 [m]': _z_su0,
        'B [m]': _B,
        'su0 [kPa]': _su0,
        'Abm [m2]': _Abm,
        'Qv0 [kN/m]': _Qv0,
        'Qv [kN/m]': _Qv,
    }