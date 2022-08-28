#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator


PILE_SETTLEMENT_CURVES = {
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'shaft_resistance': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'base_resistance': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'pile_type': {'type': 'string', 'options': ('driven', 'CFA', 'bored'), 'regex': None},
}

PILE_SETTLEMENT_CURVES_ERRORRETURN = {
    'shaft_normalised': None,
    'shaft_denormalised': None,
    'base_normalised': None,
    'base_denormalised': None,
    'total': None
}


@Validator(PILE_SETTLEMENT_CURVES, PILE_SETTLEMENT_CURVES_ERRORRETURN)
def pile_settlement_curves(
        diameter, shaft_resistance, base_resistance, pile_type,
        **kwargs):
    """
    Calculates the pile settlement curve for pile shaft and pile base from empirical trends established based on axial pile load tests.
    These curves take into account the pile type (driven, bored or CFA) and the fact that shaft resistance mobilisation distance does not appear to be proportional to the pile diameter, while the base resistance mobilisation distance is.
    The empirical curves are approximated by a mathematical relation. The function returns both the normalised and denormalised curves. Not that the function returns the pile settlement in m which the empirical shaft mobilisation curves mention mm.

    Finally, the overall curve is calculated by summing both curves

    :param diameter: Pile diameter (:math:`D`) [:math:`m`] - Suggested range: diameter >= 0.0
    :param shaft_resistance: Shaft resistance used to denormalise the curve (:math:`F_s`) [:math:`kN`] - Suggested range: shaft_resistance >= 0.0
    :param base_resistance: Base resistance used to denormalise the curve (:math:`F_b`) [:math:`kN`] - Suggested range: base_resistance >= 0.0
    :param pile_type: Pile type - Options: ('driven', 'CFA', 'bored')

    .. math::
        F_{mob} = a + \\frac{b-a}{1 + \\left( \\frac{\\delta_{pile} \\ \\text{or} \\ \\delta_{pile}/D}{c} \\right)^d}

    :returns: Dictionary with the following keys:

        - 'shaft_normalised': Normalised shaft mobilisation curve as a dictionary with keys 'w [m]' and 'Fs/Fsmax [-]'
        - 'shaft_denormalised': Shaft mobilisation curve as a dictionary with keys 'w [m]' and 'Fs [kN]'
        - 'base_normalised': Normalised base mobilisation curve as a dictionary with keys 'w/D [-]' and 'Fb/Fbmax [-]'
        - 'base_denormalised': Base mobilisation curve as a dictionary with keys 'w [m]' and 'Fb [kN]'
        - 'total': Pile tip settlement vs total load with keys 'w [m]' and 'F [kN]'

    .. figure:: images/pile_settlement_curves_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Normalised shaft and base resistance mobilisation curves

    Reference - Syllabus geotechnics

    """

    coeff = {
        'Toe 1': {'a': 145.4, 'b': 0.58, 'c': 3.21, 'd': 0.70},
        'Toe 2': {'a': 137.57, 'b': 0.06, 'c': 5.63, 'd': 0.78},
        'Toe 3': {'a': 139.37, 'b': 0.0, 'c': 8.01, 'd': 1.0},
        'Shaft 1': {'a': 134.41, 'b': 3.49, 'c': 2.8, 'd': 0.82},
        'Shaft 2+3': {'a': 118.64, 'b': 2.8, 'c': 5.91, 'd': 1.23}
    }

    displ = np.linspace(0.0, 25.0, 250)

    # Driven
    fmob_toe_1 = np.minimum(coeff['Toe 1']['a'] + ((coeff['Toe 1']['b'] - coeff['Toe 1']['a']) /
                                                   (1 + ((displ / coeff['Toe 1']['c']) ** coeff['Toe 1']['d']))),
                            100.0 * np.ones(len(displ)))
    # CFA
    fmob_toe_2 = np.minimum(coeff['Toe 2']['a'] + ((coeff['Toe 2']['b'] - coeff['Toe 2']['a']) /
                                                   (1 + ((displ / coeff['Toe 2']['c']) ** coeff['Toe 2']['d']))),
                            100.0 * np.ones(len(displ)))
    # Bored
    fmob_toe_3 = np.minimum(coeff['Toe 3']['a'] + ((coeff['Toe 3']['b'] - coeff['Toe 3']['a']) /
                                                   (1 + ((displ / coeff['Toe 3']['c']) ** coeff['Toe 3']['d']))),
                            100.0 * np.ones(len(displ)))
    # Driven
    fmob_shaft_1 = np.minimum(coeff['Shaft 1']['a'] + ((coeff['Shaft 1']['b'] - coeff['Shaft 1']['a']) /
                                                          (1 + ((displ / coeff['Shaft 1']['c']) ** coeff['Shaft 1'][
                                                              'd']))),
                               100.0 * np.ones(len(displ)))
    # CFA + Bored
    fmob_shaft_23 = np.minimum(coeff['Shaft 2+3']['a'] + ((coeff['Shaft 2+3']['b'] - coeff['Shaft 2+3']['a']) /
                                                       (1 + ((displ / coeff['Shaft 2+3']['c']) ** coeff['Shaft 2+3'][
                                                           'd']))),
                              100.0 * np.ones(len(displ)))


    if pile_type == 'driven':
        fmob_toe = fmob_toe_1
        fmob_shaft = fmob_shaft_1
    elif pile_type == 'CFA':
        fmob_toe = fmob_toe_2
        fmob_shaft = fmob_shaft_23
    elif pile_type == 'bored':
        fmob_toe = fmob_toe_3
        fmob_shaft = fmob_shaft_23
    else:
        raise ValueError("Pile type not recognised")

    _shaft_normalised = {
        'w [m]': displ * 0.001,
        'Fs/Fsmax [-]': 0.01 * fmob_shaft
    }
    _shaft_denormalised = {
        'w [m]': displ * 0.001,
        'Fs [kN]': 0.01 * fmob_shaft * shaft_resistance
    }
    _base_normalised = {
        'w/D [-]': 0.01 * displ,
        'Fb/Fbmax [-]': 0.01 * fmob_toe
    }
    _base_denormalised = {
        'w [m]': 0.01 * displ * diameter,
        'Fb [kN]': 0.01 * fmob_toe * base_resistance
    }

    _w_max = max(_base_denormalised['w [m]'].max(), _shaft_denormalised['w [m]'].max())
    _w_total = np.linspace(0, _w_max, 1000)
    _fs_interp = np.interp(_w_total, _shaft_denormalised['w [m]'], _shaft_denormalised['Fs [kN]'])
    _fb_interp = np.interp(_w_total, _base_denormalised['w [m]'], _base_denormalised['Fb [kN]'])
    _f_total = _fs_interp + _fb_interp

    return {
        'shaft_normalised': _shaft_normalised,
        'shaft_denormalised': _shaft_denormalised,
        'base_normalised': _base_normalised,
        'base_denormalised': _base_denormalised,
        'total': {
            'w [m]': _w_total,
            'F [kN]': _f_total
        }
    }