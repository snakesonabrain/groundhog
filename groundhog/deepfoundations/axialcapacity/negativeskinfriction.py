#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np
from scipy.integrate import cumulative_trapezoid

# Project imports
from groundhog.general.validation import Validator


NEGATIVESKINFRICTION_PILEGROUP_ZEEVAERTDEBEER = {
    'depths': {'type': 'list', 'elementtype': 'float', 'order': 'ascending', 'unique': True, 'empty_allowed': False},
    'effective_unit_weights': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False,
                               'empty_allowed': False},
    'lateral_earth_pressure_coefficients': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False,
                                            'empty_allowed': False},
    'interface_friction_angles': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False,
                                  'empty_allowed': False},
    'surcharge': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'diameter': {'type': 'float', 'min_value': 0.01, 'max_value': 10.0},
    'diameter_influence': {'type': 'float', 'min_value': 0.0, 'max_value': None},
}

NEGATIVESKINFRICTION_PILEGROUP_ZEEVAERTDEBEER_ERRORRETURN = {
    'virgin_effective_stress [kPa]': np.nan,
    'group_effective_stress [kPa]': np.nan,
    'negative_skin_friction_profile_single [kN]': np.nan,
    'negative_skin_friction_profile_group [kN]': np.nan,
    'negative_skin_friction [kN]': np.nan,
    'negative_skin_friction_group [kN]': np.nan,
}


@Validator(NEGATIVESKINFRICTION_PILEGROUP_ZEEVAERTDEBEER, NEGATIVESKINFRICTION_PILEGROUP_ZEEVAERTDEBEER_ERRORRETURN)
def negativeskinfriction_pilegroup_zeevaertdebeer(
        depths, effective_unit_weights, lateral_earth_pressure_coefficients, interface_friction_angles, surcharge,
        diameter, diameter_influence,
        **kwargs):
    """
    Calculates the negative skin friction for a pile in a pile group according to the method of Zeevaert en De Beer. To allow any unit weight profile, the differential equation is solved in finite difference form.

    :param depths: Array with depths used for the calculation (:math:`z`) [:math:`m`] - Elementtype: float, order: ascending, unique: True, empty entries allowed: False
    :param effective_unit_weights: Array with effective unit weights used for the calculation (:math:`\\gamma^{\\prime}`) [:math:`kN/m3`] - Elementtype: float, order: None, unique: False, empty entries allowed: False
    :param lateral_earth_pressure_coefficients: Array with lateral earth pressure coefficient at each depth (:math:`K_0`) [:math:`-`] - Elementtype: float, order: None, unique: False, empty entries allowed: False
    :param interface_friction_angles: Array with interface friction angles (:math:`\\delta^{\\prime}`) [:math:`deg`] - Elementtype: float, order: None, unique: False, empty entries allowed: False
    :param surcharge: Amount of stress applied on top of the soil mass (:math:`p_0^{\\prime}`) [:math:`kPa`] - Suggested range: surcharge >= 0.0
    :param diameter: Pile diameter (:math:`D_p`) [:math:`m`] - Suggested range: 0.01 <= diameter <= 10.0
    :param diameter_influence: Diameter of the zone of influence for negative skin friction (:math:`D_n`) [:math:`m`] - Suggested range: diameter_influence >= 0.0

    .. math::
        \\frac{d \\sigma_v^{\\prime}}{dz} = \\gamma^{\\prime} - \\tau \\cdot \\frac{O_s}{A}

        m = K_0 \\cdot \\tan \\delta^{\\prime} \\cdot \\frac{O_s}{A}

        \\frac{\\Delta \\sigma_{v,i+1}^{\\prime}}{\\Delta z} - \\gamma^{\\prime} + m \\cdot \\sigma_{v,i}^{\\prime} = 0

        \\sigma_{v,0}^{\\prime} = p_0^{\\prime}

        O_s = \\pi \\cdot D_p

        A = \\frac{\\pi}{4} \\cdot \\left( D_n^2 - D_p^2 \\right)

    :returns: Dictionary with the following keys:

        - 'virgin_effective_stress [kPa]': Effective stress profile in the absence of surcharge (:math:`\\sigma_{vo}^{\\prime}`)  [:math:`kPa`]
        - 'group_effective_stress [kPa]': Effective stress accounting for the effect of soil hanging on the pile (:math:`\\sigma_{v}^{\\prime}`)  [:math:`kPa`]
        - 'negative_skin_friction_profile_single [kN]': Cumulative negative skin friction for a single pile (:math:`F_{n}`)  [:math:`kN`]
        - 'negative_skin_friction_profile_group [kN]': Cumulative negative skin friction for pile in the group (:math:`F_{n, group}`)  [:math:`kN`]
        - 'negative_skin_friction [kN]': Total value of negative skin friction for a single pile (:math:`F_{n,tot}`)  [:math:`kN`]
        - 'negative_skin_friction_group [kN]': Total value of negative skin friction for a pile in a group (:math:`F_{n,tot,group}`)  [:math:`kN`]

    .. figure:: images/negativeskinfriction_pilegroup_zeevaertdebeer_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Pile group with contributing soil and equilibrium of infinitesimal soil slice

    Reference - Zeevaert - De Beer (1966)

    """
    # Validation
    if diameter_influence < diameter:
        raise ValueError("Diameter of the zone of influence should be greated than the pile diameter")
    if len(depths) != len(effective_unit_weights) or len(depths) != len(lateral_earth_pressure_coefficients) or \
        len(depths) != len(interface_friction_angles):
        raise ValueError("All array inputs should have equal lengths")

    area = 0.25 * np.pi * ((diameter_influence) ** 2 - diameter ** 2)
    circumference = np.pi * diameter

    m_values = np.array(lateral_earth_pressure_coefficients) * \
               np.tan(np.radians(np.array(interface_friction_angles))) * \
               (circumference / area)

    _virgin_effective_stress = np.append(0, cumulative_trapezoid(
        y=np.array(effective_unit_weights),
        x=np.array(depths))) + surcharge

    sigma_v_neg_fd = np.zeros(len(depths))
    sigma_v_neg_fd[0] = surcharge
    for i, z in enumerate(depths):
        if i > 0:
            sigma_v_neg_fd[i] = np.diff(depths)[0] * (
                    effective_unit_weights[i] -
                    m_values[i] * sigma_v_neg_fd[i - 1] +
                    (sigma_v_neg_fd[i - 1] / np.diff(depths)[0]))
    _group_effective_stress = sigma_v_neg_fd
    _negative_skin_friction_profile_single = \
        np.append(0, cumulative_trapezoid(_virgin_effective_stress, depths)) * \
        np.pi * diameter * np.array(lateral_earth_pressure_coefficients) * \
        np.tan(np.radians(np.array(interface_friction_angles)))
    _negative_skin_friction_profile_group = \
        np.append(0, cumulative_trapezoid(sigma_v_neg_fd, depths)) * \
        np.pi * diameter * np.array(lateral_earth_pressure_coefficients) * \
        np.tan(np.radians(np.array(interface_friction_angles)))

    _negative_skin_friction = _negative_skin_friction_profile_single[-1]
    _negative_skin_friction_group = _negative_skin_friction_profile_group[-1]

    return {
        'virgin_effective_stress [kPa]': _virgin_effective_stress,
        'group_effective_stress [kPa]': _group_effective_stress,
        'negative_skin_friction_profile_single [kN]': _negative_skin_friction_profile_single,
        'negative_skin_friction_profile_group [kN]': _negative_skin_friction_profile_group,
        'negative_skin_friction [kN]': _negative_skin_friction,
        'negative_skin_friction_group [kN]': _negative_skin_friction_group,
    }