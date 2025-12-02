#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruno Stuyts"

# Native Python packages

# 3rd party packages
import numpy as np
from scipy.stats import linregress
import warnings
from plotly import subplots
import plotly.graph_objs as go

# Project imports
from groundhog.general.validation import Validator

PILETEST_CHINKONDLER = {
    'loads': {'type': 'list', 'elementtype': 'float', 'order': 'ascending', 'unique': False, 'empty_allowed': False},
    'settlements': {'type': 'list', 'elementtype': 'float', 'order': 'ascending', 'unique': True, 'empty_allowed': False},
    'no_discard_points': {'type': 'int', 'min_value': 0, 'max_value': None},
    'max_settlement': {"type": "float", "min_value": 0.0, "max_value": 1000.0},
    'selected_settlement': {"type": "float", "min_value": 0.0, "max_value": 1000.0},
}

PILETEST_CHINKONDLER_ERRORRETURN = {
    'intercept [mm/kN]': np.nan,
    'slope [1/kN]': np.nan,
    'Correlation coefficient [-]': np.nan,
    'Qmax [kN]': np.nan,
    'Qdisp [kN]': None,
    'Settlements [mm]': None,
    'Q [kN]': None,
    'construction_fig': None,
    'extrapolation_fig': None
}

@Validator(PILETEST_CHINKONDLER, PILETEST_CHINKONDLER_ERRORRETURN)
def piletest_chinkondler(loads, settlements, no_discard_points=1, max_settlement=50, selected_settlement=40, show_fig=True, **kwargs):
    """
    Extrapotates a pile head load-settlement curve based on the procedure by Chin-Kondler.
    The settlements are divided by the corresponding loads. This yields a straight line in a graph of this fraction vs settlement.
    The user selects how many points to discard and the fitting happens using ``np.polyfit``.
    An extrapolated load-settlement curve is calculated and plotted for the user to check the results.

    :param loads: List of loads recorded during the load test. Note that unload-reload loops should be removed before running the algorithm (:math:`Q`) [kN]
    :param settlements: List of settlements recorded during the load test. Should be the same length as the list with loads (:math:`s`) [mm]
    :param no_discard_points: Number of points at the start of the curve to discard for the fitting of the straight line.
    :param max_settlement: Maximum settlement used for plotting the reconstructed pile head load-settlement curve (:math:`s_{max}`) [mm]. Optional, default=50mm.
    :param selected_settlement: Settlement as which pile capacity is calculated (e.g. 10% of OD) (:math:`Q_{s=s_{\\text{selected}}}`) [kN]. Optional, default=40mm.
    :param show_fig: Boolean determining whether the figures for the Chin-Kondler construction need to be plotted (default behaviour) or returned in the output dictionary.

    .. math::
        \\frac{s}{Q} = a + b \\cdot s

        Q_{\\text{ult}} = 1 / b

    :returns: Dictionary with the following keys:

        - 'intercept [mm/kN]': Coefficient :math:`a` from the linear regression [mm/kN]
        - 'slope [1/kN]': Slope :math:`b` for the linear regression [1/kN]
        - 'Correleation coefficient [-]': Pearson correlation coefficient for the points used for the construction.
        - 'Qmax [kN]': Ultimate pile resistance (fully mobilised shaft and base) (:math:`Q_{\\text{ult}}`) [kN]
        - 'Qdisp [kN]': Pile resistance at the selected displacement level (e.g. 10% of OD) (:math:`Q_{s=s_\\text{selected}}`) [kN]
        - 'Settlements [mm]': List with settlements for the load-displacement reconstruction [mm]
        - 'Q [kN]': List with loads for the load-displacement reconstruction [kN]
        - 'construction_fig': Figure with the linear regression construction (for inspection of the goodness-of-fit)
    """
    if loads.__len__() != settlements.__len__():
        raise ValueError("Array of loads and settlements need to be the same length.")
    else:
        pass

    _loads = np.array(loads)
    _settlements = np.array(settlements)
    _s_Q = _settlements / _loads

    
    _result = linregress(_settlements[no_discard_points:], _s_Q[no_discard_points:])
    
    _reconstruction_settlements = np.linspace(0, max_settlement, 250)
    _reconstruction_s_Q = _result.intercept + _result.slope * _reconstruction_settlements
    _reconstruction_Q = _reconstruction_settlements / _reconstruction_s_Q

    construction_fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)
    _data = go.Scatter(x=_settlements, y=_s_Q, showlegend=False, mode='markers',name='Test')
    construction_fig.add_trace(_data, 1, 1)
    _data = go.Scatter(x=_reconstruction_settlements, y=_reconstruction_s_Q, showlegend=True, mode='lines',name='Fit', line=dict(color='red', dash='dot'))
    construction_fig.add_trace(_data, 1, 1)
    construction_fig['layout']['xaxis1'].update(title=r'$ s \ \text{[mm]}$')
    construction_fig['layout']['yaxis1'].update(title=r'$ s/Q \ \text{[mm/kN]}$')
    construction_fig['layout'].update(height=500, width=700)
    if show_fig:
        construction_fig.show()

    extrapolation_fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)
    trace1 = go.Scatter(x=_loads, y=_settlements, showlegend=True, mode='markers', name='Test')
    extrapolation_fig.append_trace(trace1, 1, 1)
    _data = go.Scatter(x=_reconstruction_Q, y=_reconstruction_settlements, showlegend=True, mode='lines',name='Fit', line=dict(color='red', dash='dot'))
    extrapolation_fig.add_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[1 / _result.slope, 1 / _result.slope],
        y=[0, max_settlement], showlegend=True, mode='lines',name=r'$Q_{max}$', line=dict(color='grey', dash='dashdot'))
    extrapolation_fig.add_trace(_data, 1, 1)
    extrapolation_fig['layout']['xaxis1'].update(title=r'$ Q \ \text{[kN]}$', side='top', anchor='y')
    extrapolation_fig['layout']['yaxis1'].update(title=r'$ s \ \text{[mm]}$', autorange='reversed')
    extrapolation_fig['layout'].update(height=600, width=600)
    if show_fig:
        extrapolation_fig.show()

    return {
        'intercept [mm/kN]': _result.intercept,
        'slope [1/kN]': _result.slope,
        'Correlation coefficient [-]': _result.rvalue,
        'Qmax [kN]': 1 / _result.slope,
        'Qdisp [kN]': np.interp(selected_settlement, _reconstruction_settlements, _reconstruction_Q),
        'Settlements [mm]': _reconstruction_settlements,
        'Q [kN]': _reconstruction_Q,
        'construction_fig': construction_fig,
        'extrapolation_fig': extrapolation_fig
    }