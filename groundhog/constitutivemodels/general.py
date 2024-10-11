#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go

# Project imports
from groundhog.general.validation import Validator


MOHRCOULOMB_TRIAXIAL_COMPRESSION = {
    'sigma_3': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'cohesion': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'phi': {'type': 'float', 'min_value': 0.0, 'max_value': 90.0},
}

MOHRCOULOMB_TRIAXIAL_COMPRESSION_ERRORRETURN = {
    'sigma_1_f [kPa]': np.nan,
    'sigma_3_f [kPa]': np.nan,
    'Failure angle [deg]': np.nan,
    'tau_f [kPa]': np.nan,
    'sigma_f [kPa]': np.nan,
    'center [kPa]': np.nan,
    'radius [kPa]': np.nan,
    'Mohr circle': pd.DataFrame(),
    'Plot': None,
}

@Validator(MOHRCOULOMB_TRIAXIAL_COMPRESSION, MOHRCOULOMB_TRIAXIAL_COMPRESSION_ERRORRETURN)
def mohrcoulomb_triaxial_compression(sigma_3, cohesion, phi, latex_titles=True, **kwargs):

    """
    Calculates Mohr's circle and the orientation of the failure plane for a situation in which the radial (effective) stress is kept constant. A Mohr-Coulomb failure criterion with (effective) cohesion and (effective) friction angle is used.
    The user needs to specify the value of the radial stress and the values of cohesion and friction angle. The axial stress at failure and the orientation of the failure plane are then calculated.
    A graphical construction of Mohr's circle is also made as a Plotly plot.
    Note that this construction can be used for both total stress and effective stress analysis, the user needs to make an appropriate choice.
    
    :param sigma_3: Radial stress (total or effective) (:math:`\\sigma_3`) [kPa] - Suggested range: sigma_3 >= 0.0
    :param cohesion: Cohesion (total or effective) (:math:`c`) [kPa] - Suggested range: cohesion >= 0.0
    :param phi: Friction angle (total or effective) (:math:`\\varphi`) [deg] - Suggested range: 0.0 <= phi <= 90.0
    :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
    
    .. math::
        \\tau_f = c + \\sigma \\tan \\varphi
        
        \\text{Failure plane orientation (relative to vertical)} = \\frac{1}{2} \\left( \\frac{\\pi}{2} - \\varphi \\right)
        
        \\tau_f = \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\cos \\varphi
        
        \\sigma_f = \\frac{\\sigma_1 + \\sigma_3}{2} - \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\sin \\varphi
        
        \\implies \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\cos \\varphi = c + \\left( \\frac{\\sigma_1 + \\sigma_3}{2} - \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\sin \\varphi \\right) \\tan \\varphi
    
    :returns: Dictionary with the following keys:
        
        - 'sigma_1_f [kPa]': Axial stress at failure (:math:`\\sigma_{1,f}`)  [kPa]
        - 'sigma_3_f [kPa]': Radial stress at failure as specified by the user (:math:`\\sigma_{3,f}`)  [kPa]
        - 'Failure angle [deg]': Orientation of the failure plane (relative to the vertical) (:math:`\\frac{\\pi}{2} - \\varphi`)  [deg]
        - 'tau_f [kPa]': Shear stress at failure (:math:`\\tau_f`)  [kPa]
        - 'sigma_f [kPa]': Normal stress on the failure plane at failure (:math:`\\sigma_f`)  [kPa]
        - 'center [kPa]': Center of Mohr's circle (:math:`\\frac{\\sigma_1 + \\sigma_3}{2}`)  [kPa]
        - 'radius [kPa]': Radius of Mohr's circle (:math:`\\frac{\\sigma_1 - \\sigma_3}{2}`)  [kPa]
        - 'Mohr circle': Pandas dataframe with normal stresses and shear stresses for Mohrs circle
        - 'Plot': Plotly plot showing Mohr's circle, the failure criterion and the failure point
    
    .. figure:: images/Mohrs_circle
        :figwidth: 500.0
        :width: 450
        :align: center

        Illustration of the concepts of Mohr's circle
    
    Reference - Budhu (2011) Introduction to soil mechanics and foundations.

    """
    _part1 = cohesion + \
        (0.5 * sigma_3 * np.cos(np.radians(phi))) + \
        (0.5 * sigma_3 * np.tan(np.radians(phi))) + \
        (0.5 * sigma_3 * np.sin(np.radians(phi) * np.tan(np.radians(phi))))
    _part2 = 0.5 * np.cos(np.radians(phi)) - \
         0.5 * np.tan(np.radians(phi)) + \
         0.5 * np.sin(np.radians(phi)) * np.tan(np.radians(phi))
    _sigma_1_f = _part1 / _part2

    _center = 0.5 * (sigma_3 + _sigma_1_f)
    _radius = 0.5 * (_sigma_1_f - sigma_3)
     
    _failure_angle = 0.5 * (np.rad2deg(0.5 * np.pi - np.radians(phi)))

    _tau_f = _radius * np.cos(np.radians(phi))
    _sigma_f = _center - _radius * np.sin(np.radians(phi))

    _mohr_circle = pd.DataFrame({
        'tau [kPa]': _radius * np.sin(np.linspace(0, 2 * np.pi, 250)),
        'sigma [kPa]': _center + _radius * np.cos(np.linspace(0, 2 * np.pi, 250))
    })
    fig = subplots.make_subplots(rows=1, cols=2, print_grid=False, column_widths=[0.7, 0.3])
    _data = go.Scatter(
        x=_mohr_circle['sigma [kPa]'], y=_mohr_circle['tau [kPa]'],
        showlegend=True, mode='lines',name='Mohr circle',
        line=dict(color='black'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=np.linspace(0, _sigma_1_f, 250), y=cohesion + np.tan(np.radians(phi)) * np.linspace(0, _sigma_1_f, 250),
        showlegend=True, mode='lines',name='Mohr-Coulomb criterion',
        line=dict(color='red', dash='dot'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[_center, _sigma_f],
        y=[0, _tau_f], showlegend=False, mode='lines',name='Location of stress state', line=dict(color='green'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[_sigma_f, ],
        y=[_tau_f, ], showlegend=False, mode='markers',name='Location of stress state',
        marker=dict(size=7,color='green',line=dict(width=1,color='black')))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[0, 1, 1, 0, 0], y=[0, 0, 1, 1, 0], showlegend=True, mode='lines',
        name='Sample', line=dict(color='black'))
    fig.append_trace(_data, 1, 2)

    _data = go.Scatter(
        x=[0.5 + 0.5 * np.tan(np.radians(_failure_angle)), 0.5 + -0.5 * np.tan(np.radians(_failure_angle))], y=[0, 1],
        showlegend=True, mode='lines', name='Orientation of selected plane', line=dict(dash='dot'))
    fig.append_trace(_data, 1, 2)

    if latex_titles:
        fig['layout']['xaxis1'].update(title=r'$ \sigma \ \text{[kPa]}$')
        fig['layout']['yaxis1'].update(title=r'$ \tau \ \text{[kPa]}$', scaleanchor='x', scaleratio=1.0)
        fig['layout']['xaxis2'].update(title=r'$ X $')
        fig['layout']['yaxis2'].update(title=r'$ Y $', scaleanchor='x2', scaleratio=1.0)
    else:
        fig['layout']['xaxis1'].update(title='sigma [kPa]')
        fig['layout']['yaxis1'].update(title='tau [kPa]', scaleanchor='x', scaleratio=1.0)
        fig['layout']['xaxis2'].update(title='X')
        fig['layout']['yaxis2'].update(title='Y', scaleanchor='x2', scaleratio=1.0)
    fig['layout'].update(height=500, width=900)
    _plot = fig

    return {
        'sigma_1_f [kPa]': _sigma_1_f,
        'sigma_3_f [kPa]': sigma_3,
        'Failure angle [deg]': _failure_angle,
        'tau_f [kPa]': _tau_f,
        'sigma_f [kPa]': _sigma_f,
        'center [kPa]': _center,
        'radius [kPa]': _radius,
        'Mohr circle': _mohr_circle,
        'Plot': _plot,
    }

MOHRCOULOMB_TRIAXIAL_EXTENSION = {
    'sigma_1': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'cohesion': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'phi': {'type': 'float', 'min_value': 0.0, 'max_value': 90.0},
}

MOHRCOULOMB_TRIAXIAL_EXTENSION_ERRORRETURN = {
    'sigma_1_f [kPa]': np.nan,
    'sigma_3_f [kPa]': np.nan,
    'Failure angle [deg]': np.nan,
    'tau_f [kPa]': np.nan,
    'sigma_f [kPa]': np.nan,
    'center [kPa]': np.nan,
    'radius [kPa]': np.nan,
    'Mohr circle': pd.DataFrame(),
    'Plot': None,
}

@Validator(MOHRCOULOMB_TRIAXIAL_EXTENSION, MOHRCOULOMB_TRIAXIAL_EXTENSION_ERRORRETURN)
def mohrcoulomb_triaxial_extension(sigma_1, cohesion, phi, latex_titles=True, **kwargs):

    """
    Calculates Mohr's circle and the orientation of the failure plane for a situation in which the axial (effective) stress is reduced and the radial stress is kept constant, the axial stress then becomes the minor principal stress. A Mohr-Coulomb failure criterion with (effective) cohesion and (effective) friction angle is used.
    The user needs to specify the value of the axial stress and the values of cohesion and friction angle. The radial stress at failure and the orientation of the failure plane are then calculated.
    A graphical construction of Mohr's circle is also made as a Plotly plot.
    Note that this construction can be used for both total stress and effective stress analysis, the user needs to make an appropriate choice.
    
    :param sigma_1: Radial stress (total or effective) (:math:`\\sigma_1`) [kPa] - Suggested range: sigma_1 >= 0.0
    :param cohesion: Cohesion (total or effective) (:math:`c`) [kPa] - Suggested range: cohesion >= 0.0
    :param phi: Friction angle (total or effective) (:math:`\\varphi`) [deg] - Suggested range: 0.0 <= phi <= 90.0
    :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
    
    .. math::
        \\tau_f = c + \\sigma \\tan \\varphi
        
        \\text{Failure plane orientation (relative to vertical)} = \\frac{1}{2} \\left( \\frac{\\pi}{2} - \\varphi \\right)
        
        \\tau_f = \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\cos \\varphi
        
        \\sigma_f = \\frac{\\sigma_1 + \\sigma_3}{2} - \\frac{\\sigma_1 - \\sigma_3}{2} \\cdot \\sin \\varphi
        
        \\implies \\left( \\frac{\\cos \\varphi}{2} - \\frac{\\tan \\varphi}{2} + \\frac{\\sin \\varphi \\tan \\varphi}{2} \\right) \\sigma_1 - c = \\sigma_3 \\left( \\frac{1}{2} \\cos \\varphi + \\frac{\\tan \\varphi}{2} + \\frac{\\sin \\varphi \\tan \\varphi}{2} \\right)
    
    :returns: Dictionary with the following keys:
        
        - 'sigma_1_f [kPa]': Radial stress at failure, as specified by the user (:math:`\\sigma_{1,f}`)  [kPa]
        - 'sigma_3_f [kPa]': Axial stress at failure  (:math:`\\sigma_{3,f}`)  [kPa]
        - 'Failure angle [deg]': Orientation of the failure plane (relative to the vertical) (:math:`\\frac{\\pi}{2} - \\varphi`)  [deg]
        - 'tau_f [kPa]': Shear stress at failure (:math:`\\tau_f`)  [kPa]
        - 'sigma_f [kPa]': Normal stress on the failure plane at failure (:math:`\\sigma_f`)  [kPa]
        - 'center [kPa]': Center of Mohr's circle (:math:`\\frac{\\sigma_1 + \\sigma_3}{2}`)  [kPa]
        - 'radius [kPa]': Radius of Mohr's circle (:math:`\\frac{\\sigma_1 - \\sigma_3}{2}`)  [kPa]
        - 'Mohr circle': Pandas dataframe with normal stresses and shear stresses for Mohrs circle
        - 'Plot': Plotly plot showing Mohr's circle, the failure criterion and the failure point
    
    .. figure:: images/Mohrs_circle
        :figwidth: 500.0
        :width: 450
        :align: center

        Illustration of the concepts of Mohr's circle
    
    Reference - Budhu (2011) Introduction to soil mechanics and foundations.

    """
    _part1 = -cohesion + sigma_1 * (
        0.5 * np.cos(np.radians(phi)) - 0.5 * np.tan(np.radians(phi)) + 0.5 * np.sin(np.radians(phi)) * np.tan(np.radians(phi))
    )
    _part2 = 0.5 * np.cos(np.radians(phi)) + \
         0.5 * np.tan(np.radians(phi)) + \
         0.5 * np.sin(np.radians(phi)) * np.tan(np.radians(phi))
    _sigma_3_f = _part1 / _part2

    _center = 0.5 * (sigma_1 + _sigma_3_f)
    _radius = 0.5 * (sigma_1 - _sigma_3_f)
     
    _failure_angle = 0.5 * (np.rad2deg(0.5 * np.pi - np.radians(phi)))

    _tau_f = _radius * np.cos(np.radians(phi))
    _sigma_f = _center - _radius * np.sin(np.radians(phi))

    _mohr_circle = pd.DataFrame({
        'tau [kPa]': _radius * np.sin(np.linspace(0, 2 * np.pi, 250)),
        'sigma [kPa]': _center + _radius * np.cos(np.linspace(0, 2 * np.pi, 250))
    })
    fig = subplots.make_subplots(rows=1, cols=2, print_grid=False, column_widths=[0.7, 0.3])
    _data = go.Scatter(
        x=_mohr_circle['sigma [kPa]'], y=_mohr_circle['tau [kPa]'],
        showlegend=True, mode='lines',name='Mohr circle',
        line=dict(color='black'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=np.linspace(0, sigma_1, 250), y=cohesion + np.tan(np.radians(phi)) * np.linspace(0, sigma_1, 250),
        showlegend=True, mode='lines',name='Mohr-Coulomb criterion',
        line=dict(color='red', dash='dot'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[_center, _sigma_f],
        y=[0, _tau_f], showlegend=False, mode='lines',name='Location of stress state', line=dict(color='green'))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[_sigma_f, ],
        y=[_tau_f, ], showlegend=False, mode='markers',name='Location of stress state',
        marker=dict(size=7,color='green',line=dict(width=1,color='black')))
    fig.append_trace(_data, 1, 1)
    _data = go.Scatter(
        x=[0, 1, 1, 0, 0], y=[0, 0, 1, 1, 0], showlegend=True, mode='lines',
        name='Sample', line=dict(color='black'))
    fig.append_trace(_data, 1, 2)

    _data = go.Scatter(
        x=[0.5 + 0.5 * np.tan(np.radians(_failure_angle)), 0.5 + -0.5 * np.tan(np.radians(_failure_angle))], y=[0, 1],
        showlegend=True, mode='lines', name='Orientation of selected plane', line=dict(dash='dot'))
    fig.append_trace(_data, 1, 2)

    if latex_titles:
        fig['layout']['xaxis1'].update(title=r'$ \sigma \ \text{[kPa]}$')
        fig['layout']['yaxis1'].update(title=r'$ \tau \ \text{[kPa]}$', scaleanchor='x', scaleratio=1.0)
        fig['layout']['xaxis2'].update(title=r'$ X $')
        fig['layout']['yaxis2'].update(title=r'$ Y $', scaleanchor='x2', scaleratio=1.0)
    else:
        fig['layout']['xaxis1'].update(title='sigma [kPa]')
        fig['layout']['yaxis1'].update(title='tau [kPa]', scaleanchor='x', scaleratio=1.0)
        fig['layout']['xaxis2'].update(title='X')
        fig['layout']['yaxis2'].update(title='Y', scaleanchor='x2', scaleratio=1.0)
    fig['layout'].update(height=500, width=900)
    _plot = fig

    return {
        'sigma_1_f [kPa]': sigma_1,
        'sigma_3_f [kPa]': _sigma_3_f,
        'Failure angle [deg]': _failure_angle,
        'tau_f [kPa]': _tau_f,
        'sigma_f [kPa]': _sigma_f,
        'center [kPa]': _center,
        'radius [kPa]': _radius,
        'Mohr circle': _mohr_circle,
        'Plot': _plot,
    }