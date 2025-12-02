#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruno Stuyts"

# Native Python packages

# 3rd party packages
import numpy as np
from plotly import tools, subplots
import plotly.graph_objs as go

# Project imports
from groundhog.general.validation import Validator



REINFORCEMENT_INERTIA = {
    'diameter': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'modulus_ratio': {'type': 'float', 'min_value': 1.0, 'max_value': None},
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
def reinforced_circularsection_inertia(diameter, modulus_ratio, n_bars, offset, rebar_diameter, maximum_resistance=True, **kwargs):

    """
    Calculates the combined inertia of a circular section, reinforced with rebar rods at equal center-to-center distance from the concrete section center.
    Steiner's theorem is applied using the center-to-center distance between the center of the concrete section and the center of the rebar rod.
    The positioning of the rebar rods for maximum or minimum bending resistance can be taken.
    Their offset from the bending axis can then be derived.
    
    :param diameter: Diameter of the concrete section (:math:`D`) [m] - Suggested range: diameter >= 0.0
    :param modulus_ratio: Ratio of Young's modulus of steel to Young's modulus of concrete (:math:`n`) [-] - Suggested range: modulus_ratio >= 0.0
    :param n_bars: Number of rebar rods (:math:`N`) [-] - Suggested range: n_bars >= 1.0
    :param offset: Center-to-center distance between rebar rods and concrete section center (:math:`r`) [m] - Suggested range: offset >= 0.0
    :param rebar_diameter: Diameter of the rebar rods (:math:`d`) [m] - Suggested range: rebar_diameter >= 0.0
    :param maximum_resistance: Determines whether the rebar rods should be positioned for maximum bending resistance (if true) (optional, default= True)
    
    .. math::
        I_s = \\frac{\\pi d^4}{64}
        
        A_s = \\frac{\\pi d^2}{4}

        A_{\\text{s,transformed}} = n \\cdot A_s

        I_c = \\frac{\\pi D^4}{64}
        
        I_{\\text{combined}} = I_c + \\sum_{i=1}^N \\left( I_s + A_{\\text{s,transformed}} r^2 \\right)
    
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
    _single_rod_area_transformed = modulus_ratio * _single_rod_area
    _total_rebar_inertia = 0
    
    for _offset in _offset_list:
        _total_rebar_inertia += _single_rod_inertia + (_offset ** 2) * _single_rod_area_transformed

    return {
        'Start angle [deg]': start_angle,
        'Rebar angles [deg]': _angles,
        'Offsets [m]': _offset_list,
        'Rebar inertia [m4]': _total_rebar_inertia,
        'Ic [m4]': _Ic,
        'I combined [m4]': _Ic + _total_rebar_inertia
    }

PILEGROUPEFFECT_REESEVANIMPE = {
    'pile_x': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'pile_x': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'pile_diameters': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'load_x': {"type": "float", "min_value": None, "max_value": None},
    'load_y': {"type": "float", "min_value": None, "max_value": None}
}

PILEGROUPEFFECT_REESEVANIMPE_ERRORRETURN = {
    'efficiency_matrix': None,
    'efficiencies': None,
    'pile_fig': None
}

@Validator(PILEGROUPEFFECT_REESEVANIMPE, PILEGROUPEFFECT_REESEVANIMPE_ERRORRETURN)
def pilegroupeffect_reesevanimpe(pile_x, pile_y, pile_diameters, load_x, load_y, show_fig=True, plot_height=600, plot_width=400, **kwargs):
    """
    When piles are arranged in a group, they influence one another and the lateral reaction for a given displacement can be less than that for a single pile.
    Reese and Van Impe suggest a method for calculating the efficiency of each pile. A distinction is made between in-line leading piles, in-line trailing piles and side-by-side piles.
    Based on the direction of loading and inter-pile distance, a reduction factor (p-multiplier) is calculated for each pile pair.
    The efficiency factors for each pile pair are multiplied to provide the overall efficiency for the pile considered.
    In this function, the centers of the piles are defined using their (X,Y) coordinates.
    The direction of loading is defined using the X- and Y-component of the loading vector. Note that the magnitude of this loading vector (norm) does not play a role in the calculation.
    The diameter of the pile is also required to determine the normalised pile spacing.
    For piles which is neither perfectly inline or side-by-side, the inline and side-by-side efficiencies are combined using the angle to the loading direction.

    :param pile_x: List of X-coordinates of the pile centers (:math:`x`) [m]
    :param pile_y: List of Y-coordinates of the pile centers (:math:`y`) [m]
    :param pile_diameters: List of pile diameters (:math:`y`) [m]
    :param load_x: X-component of the load vector (:math:`x_{\\text{load}}`)
    :param load_y: Y-component of the load vector (:math:`x_{\\text{load}}`)
    :param show_fig: Boolean determining whether the figures for the Chin-Kondler construction need to be plotted (default behaviour) or returned in the output dictionary.

    .. math::
        \\text{Side by side piles: } e = 0.64 \\left( \\frac{s}{D} \\right)^{0.34} \\text{ for } 1 \\leq \\frac{s}{D} \\leq 3.75, e=1 \\text{ for } \\frac{s}{D} > 3.75

        \\text{In-line leading piles: } e = 0.70 \\left( \\frac{s}{D} \\right)^{0.26} \\text{ for } 1 \\leq \\frac{s}{D} \\leq 4 , e=1 \\text{ for } \\frac{s}{D} > 4

        \\text{In-line trailing piles: } e = 0.48 \\left( \\frac{s}{D} \\right)^{0.38} \\text{ for } 1 \\leq \\frac{s}{D} \\leq 7 , e=1 \\text{ for } \\frac{s}{D} > 7

        \\text{Oblique orientation: } e = \\sqrt{e_{\\text{inline}}^2 \\cos^2 \\varphi + e_{\\text{side-by-side}}^2 \\sin^2 \\varphi} 

        p_{\\text{group}} = p_{\\text{single}} e_{\\text{combined}} = p_{\\text{single}} \Pi_j e_j

    :returns: Dictionary with the following keys:

        - 'efficiency_matrix': Matrix with the efficiency of each pile vis-à-vis the others (row i, column j quantifies the influence of pile j on pile i)
        - 'efficiencies': List with the combined efficiencies of each pile (list with an element for each pile)
        - 'pile_fig': Figure with the dimensions of the piles.
    """
    if (pile_x.__len__() != pile_y.__len__()) or (pile_x.__len__() != pile_diameters.__len__()):
        raise ValueError("Lists of pile x- and y-coordinates and diameters need to be the same length.")

    def efficiency_sideside(spacing, diameter):
        if (spacing / diameter) < 3.75:
            return 0.64 * ((spacing / diameter) ** 0.34)
        else:
            return 1
    def efficiency_inline_leading(spacing, diameter):
        if (spacing / diameter) < 4:
            return 0.70 * ((spacing / diameter) ** 0.26)
        else:
            return 1
    def efficiency_inline_trailing(spacing, diameter):
        if (spacing / diameter) < 7:
            return 0.48 * ((spacing / diameter) ** 0.38)
        else:
            return 1

    load_vector = np.array([load_x, load_y])
    efficiency_factors = np.ones((pile_x.__len__(), pile_y.__len__()))
    combined_efficiency = np.ones(pile_x.__len__())

    for i, (_xi, _yi, _Di) in enumerate(zip(pile_x, pile_y, pile_diameters)):
        # Loop over all other piles
        _combined_efficiency = 1
        for j, (_xj, _yj) in enumerate(zip(pile_x, pile_y)):
            if i == j:
                pass
            else:
                _pile_vector = np.array([_xj - _xi, _yj - _yi])
                if np.dot(_pile_vector, load_vector) > 0:
                    _trailing = True
                else:
                    _trailing = False 
                _cos_theta = np.dot(load_vector, _pile_vector) / (np.linalg.norm(load_vector) * np.linalg.norm(_pile_vector))
                _s_inline = np.abs(np.dot(load_vector, _pile_vector) / np.linalg.norm(load_vector))
                _s_sideside = np.abs(np.sqrt(np.linalg.norm(_pile_vector) ** 2 - _s_inline ** 2))
                _s_pile = np.linalg.norm(_pile_vector)
                # Sideside efficiency
                _e_sideside = efficiency_sideside(_s_sideside, _Di)
                # Inline efficiency
                if _trailing:
                    _e_inline = efficiency_inline_trailing(_s_inline, _Di)
                    _s_D_max = np.sqrt(7 ** 2 * _cos_theta ** 2 + 3.75 ** 2 * (1 - _cos_theta ** 2))
                else:
                    _e_inline = efficiency_inline_leading(_s_inline, _Di)
                    _s_D_max = np.sqrt(4 ** 2 * _cos_theta ** 2 + 3.75 ** 2 * (1 - _cos_theta ** 2))
                
                if _s_pile / _Di < _s_D_max:
                    _efficiency = np.sqrt(_e_inline ** 2 * _cos_theta ** 2 + _e_sideside ** 2 * (1 - _cos_theta ** 2))
                else:
                    _efficiency = 1
                    
                efficiency_factors[i][j] = _efficiency
                _combined_efficiency = _combined_efficiency * _efficiency
        combined_efficiency[i] = _combined_efficiency

    pile_fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)
    for i, (_x, _y, _D) in enumerate(zip(pile_x, pile_y, pile_diameters)):
        _data = go.Scatter(
            x=_x + 0.5 * _D * np.cos(np.linspace(0, 2 * np.pi, 250)),
            y=_y + 0.5 * _D * np.sin(np.linspace(0, 2 * np.pi, 250)),
            showlegend=False, mode='lines',name='Pile %i' % (i+1),
            line=dict(color='red'))
        pile_fig.add_trace(_data, 1, 1)
    _data = go.Scatter(
        x=pile_x,
        y=pile_y,
        text=list(map(lambda _e: "%.3f" % _e, combined_efficiency)),
        showlegend=False, mode='text',name='Pile %i' % (i+1),
        line=dict(color='red'))
    pile_fig.add_trace(_data, 1, 1)
    arrow = go.layout.Annotation(dict(
        x=load_x,
        y=load_y,
        xref="x", yref="y",
        text="",
        showarrow=True,
        axref="x", ayref='y',
        ax=0,
        ay=0,
        arrowhead=3,
        arrowwidth=1.5,
        arrowcolor='black',)
    )
    pile_fig['layout']['xaxis1'].update(title=r'$ X \ \text{[m]}$')
    pile_fig['layout']['yaxis1'].update(title=r'$ Y \ \text{[m]}$', scaleanchor='x', scaleratio=1.0)
    pile_fig['layout'].update(height=plot_height, width=plot_width, annotations=[arrow,])
    if show_fig:
        pile_fig.show()

    return {
        'efficiency_matrix': efficiency_factors,
        'efficiencies': combined_efficiency,
        'pile_fig': pile_fig
    }