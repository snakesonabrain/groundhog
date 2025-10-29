#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruno Stuyts"

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator



REINFORCEMENT_INERTIA = {
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'n_bars': {'type': 'int', 'min_value': 1.0, 'max_value': None},
    'offset': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'rebar_diameter': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'maximum_resistance': {'type': 'bool',},
}

REINFORCEMENT_INERTIA_ERRORRETURN = {
    'Start angle [deg]': np.nan,
    'Rebar angles [deg]': None,
    'Offsets [m]': None,
    'Rebar inertia [m4]': np.nan,
    'Ic [m4]': np.nan,
    'I combined [m4]': np.nan
}

@Validator(REINFORCEMENT_INERTIA, REINFORCEMENT_INERTIA_ERRORRETURN)
def reinforced_circularsection_inertia(diameter, n_bars, offset, rebar_diameter, maximum_resistance=True, **kwargs):

    """
    Calculates the combined inertia of a circular section, reinforced with rebar rods at equal center-to-center distance from the concrete section center.
    Steiner's theorem is applied using the center-to-center distance between the center of the concrete section and the center of the rebar rod.
    The positioning of the rebar rods for maximum or minimum bending resistance can be taken.
    Their offset from the bending axis can then be derived.
    
    :param diameter: Diameter of the concrete section (:math:`D`) [m] - Suggested range: offset >= 0.0
    :param n_bars: Number of rebar rods (:math:`N`) [-] - Suggested range: n >= 1.0
    :param offset: Center-to-center distance between rebar rods and concrete section center (:math:`r`) [m] - Suggested range: offset >= 0.0
    :param rebar_diameter: Diameter of the rebar rods (:math:`d`) [m] - Suggested range: rebar_diameter >= 0.0
    :param maximum_resistance: Determines whether the rebar rods should be positioned for maximum bending resistance (if true) (optional, default= True)
    
    .. math::
        I_s = \\frac{\\pi d^4}{64}
        
        A_s = \\frac{\\pi d^2}{4}
        
        I_c = \\frac{\\pi D^4}{64}
        
        I_{\\text{combined}} = I_c + \\sum_{i=1}^N \\left( I_s + A_s r^2 \\right)
    
    :returns: Dictionary with the following keys:
        
        - 'Start angle [deg]': Angle between the bending axis and the first rebar rod [deg]
        - 'Rebar angles [deg]': Angles between rebar rods and the bending axis [deg]
        - 'Offsets [m]': Offsets of the rebar rods to the bending axis [m]
        - 'Rebar inertia [m4]': Total inertia of the rebar (:math:`\\sum_{i=1}^N \\left( I_s + A_s r^2 \\right)`) [m4]
        - 'Ic [m4]': Concrete section inertia [m4]
        - 'I combined [m4]': Combined inertia of the reinforced section [m4]
    
    """

    _Ic = np.pi * (diameter ** 4) / 64

    def total_offsets(n, deltatheta):
        _sum = 0
        for i in range(n):
            _sum += np.abs(np.sin(deltatheta + i * (2 * np.pi / n)))
        return _sum
    
    if n_bars % 2 == 0:
        angles = np.linspace(0, 360 / n_bars, 1000)
    else:
        angles = np.linspace(0, 360 / (2 * n_bars), 1000)
    offsets = np.array(list(map(lambda _theta: total_offsets(n_bars, np.radians(_theta)), angles)))
    
    if maximum_resistance:
        start_angle = angles[np.argmax(offsets)]
    else:
        start_angle = angles[np.argmin(offsets)]
    
    _angles = []
    for i in range(n_bars):
        _angles.append(np.rad2deg(np.radians(start_angle) + i * (2 * np.pi / n_bars)))
    
    _offset_list = []
    for i in range(n_bars):
        _offset_list.append(offset * np.abs(np.sin(np.radians(start_angle) + i * (2 * np.pi / n_bars))))
    
    _single_rod_inertia = (np.pi * rebar_diameter ** 4) / 64
    _single_rod_area = 0.25 * np.pi * rebar_diameter ** 2
    _total_rebar_inertia = 0
    
    for _offset in _offset_list:
        _total_rebar_inertia += _single_rod_inertia + (_offset ** 2) * _single_rod_area

    return {
        'Start angle [deg]': start_angle,
        'Rebar angles [deg]': _angles,
        'Offsets [m]': _offset_list,
        'Rebar inertia [m4]': _total_rebar_inertia,
        'Ic [m4]': _Ic,
        'I combined [m4]': _Ic + _total_rebar_inertia
    }
