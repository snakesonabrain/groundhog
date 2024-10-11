#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import os

# 3rd party packages
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Project imports
from groundhog.general.plotting import GROUNDHOG_PLOTTING_CONFIG


class PlasticityChart(object):
    """
    Class for plasticity chart
    """
    def __init__(self, plot_height=500, plot_width=800, plot_title=None):
        """
        Initiates a plasticity chart
        :param plot_height: Height of the plot in pixels
        :param plot_width: Width of the plot in pixels
        :param plot_title: Title of the plot
        """
        self.fig = make_subplots(rows=1, cols=1, print_grid=False)
        ll = np.linspace(0, 100, 100)
        u_line = 0.9 * (ll - 8)
        a_line = 0.73 * (ll - 20)
        _data = go.Scatter(
            x=ll,
            y=u_line, showlegend=False, mode='lines', name='U-line',
            line=dict(width=1, color='black'))
        self.fig.append_trace(_data, 1, 1)
        _data = go.Scatter(
            x=ll,
            y=a_line, showlegend=False, mode='lines', name='A-line',
            line=dict(width=1, color='black'))
        self.fig.append_trace(_data, 1, 1)
        _data = go.Scatter(
            x=[np.interp(4, u_line, ll), np.interp(4, a_line, ll)],
            y=[4, 4], showlegend=False, mode='lines', name='Lower bound ML',
            line=dict(width=1, color='black'))
        self.fig.append_trace(_data, 1, 1)
        _data = go.Scatter(
            x=[np.interp(7, u_line, ll), np.interp(7, a_line, ll)],
            y=[7, 7], showlegend=False, mode='lines', name='Upper bound ML',
            line=dict(width=1, color='black'))
        self.fig.append_trace(_data, 1, 1)
        _data = go.Scatter(
            x=[50, 50],
            y=[0, 60], showlegend=False, mode='lines', name='Upper bound ML',
            line=dict(width=1, color='black'))
        self.fig.append_trace(_data, 1, 1)

        self.fig['layout']['xaxis1'].update(title='Liquid limit [%]', range=(0, 100), dtick=10)
        self.fig['layout']['yaxis1'].update(title='Plasticity index [%]$', range=(0, 60), dtick=10)
        self.fig['layout'].update(
            height=plot_height, width=plot_width,
            title=plot_title,
            hovermode='closest',
            annotations=[
                dict(x=20, y=5.5, text='CL-ML', showarrow=False),
                dict(x=15, y=2, text='ML', showarrow=False),
                dict(x=15, y=2, text='ML', showarrow=False),
                dict(x=35, y=18, text='CL', showarrow=False),
                dict(x=64, y=38, text='CH', showarrow=False),
                dict(x=40, y=5.5, text='ML or OL', showarrow=False),
                dict(x=70, y=18, text='MH or OH', showarrow=False),
                dict(x=32, y=24, text='U-line: PI=0.9(LL-8)', showarrow=False, textangle=-42),
                dict(x=80, y=46, text='A-line: PI=0.73(LL-20)', showarrow=False, textangle=-37)
            ],
        )

    def add_trace(self, ll, pi, name, **kwargs):
        """
        Adds a trace to the plot. By default, markers are added but optional keyword arguments can be added for go.Scatter as ``**kwargs``
        :param ll: Array with the Liquid Limit values (in percent)
        :param pi: Array with the Plasticity Index values (in percent)
        :param name: Name for the trace (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        _data = go.Scatter(
            x=ll,
            y=pi,
            mode='markers',
            name=name,
            **kwargs)
        self.fig.append_trace(_data, 1, 1)

    def show(self):
        self.fig.show(config=GROUNDHOG_PLOTTING_CONFIG)


class PSDChart(object):
    """
    Class for plotting of grain size distribution data
    """

    def __init__(self, plot_title=None, marginsettings=dict(l=0, r=0, b=100, t=100, pad=0),
                 legendsettings=dict(x=0.1, y=0.9)):
        """
        Initiates a particle size distribution chart
        :param plot_height: Height of the plot in pixels
        :param plot_width: Width of the plot in pixels
        :param plot_title: Title of the plot
        """
        self.fig = make_subplots(rows=1, cols=1, print_grid=False)
        self.fig['layout']['xaxis1'].update(
            title='Particle size [mm]',
            type='log', range=(-3, 2),
            anchor='y',
            tickvals=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=[0.001, 0.002, '', '', '', 0.006, '', '', '',
                      0.01, 0.02, '', '', '', 0.06, '', '', '',
                      0.1, 0.2, '', '', '', 0.6, '', '', '',
                      1, 2, '', '', '', 6, '', '', '',
                      10, 20, '', '', '', 60, '', '', '', 100])
        self.fig['layout']['yaxis1'].update(
            title=r'% finer', range=(-20, 100),
            tickvals=np.linspace(0, 100, 11),
            ticktext=list(map(lambda _x: '%.0f' % _x, np.linspace(0, 100, 11))))
        psd_plot_shapes = [
            # Clay fraction
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.001, y0=-20, x1=0.002, y1=0,
                            line=dict(color="Black", width=2, ), fillcolor="White"),
            # Silt fraction
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.002, y0=-20, x1=0.063, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White"),
            # Sand fraction
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.063, y0=-20, x1=2, y1=-10,
                            line=dict(color="Black", width=2), fillcolor="White"),
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.063, y0=-10, x1=0.2, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White"),
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.2, y0=-10, x1=0.63, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White"),
            go.layout.Shape(type="rect", xref="x", yref="y", x0=0.63, y0=-10, x1=2, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White"),
            # Gravel fraction
            go.layout.Shape(type="rect", xref="x", yref="y", x0=2, y0=-20, x1=63, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White"),
            # Cobble fraction
            go.layout.Shape(type="rect", xref="x", yref="y", x0=63, y0=-20, x1=100, y1=0,
                            line=dict(color="Black", width=2), fillcolor="White")
        ]
        psd_plot_annotations = [
            go.layout.Annotation(x=0.5 * (np.log10(0.001) + np.log10(0.002)), y=-10, xref="x", yref="y",
                                 text="Clay", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(0.002) + np.log10(0.063)), y=-10, xref="x", yref="y",
                                 text="Silt", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(0.063) + np.log10(2)), y=-15, xref="x", yref="y",
                                 text="Sand", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(0.063) + np.log10(0.2)), y=-5, xref="x", yref="y",
                                 text="Fine", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(0.2) + np.log10(0.63)), y=-5, xref="x", yref="y",
                                 text="Medium", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(0.63) + np.log10(2)), y=-5, xref="x", yref="y",
                                 text="Coarse", showarrow=False, arrowhead=0, ax=0, ay=0),
            go.layout.Annotation(x=0.5 * (np.log10(2) + np.log10(63)), y=-10, xref="x", yref="y",
                                 text="Gravel", showarrow=False, arrowhead=0, ax=0, ay=0),
        ]
        self.fig['layout'].update(
            title=plot_title,
            shapes=psd_plot_shapes,
            annotations=psd_plot_annotations,
            autosize=False,
            width=800,
            height=500,
            margin=marginsettings,
            legend=legendsettings
        )

    def add_trace(self, grainsize, pctpassing, name, **kwargs):
        """
        Adds a trace to the plot. By default, lines are added but optional keyword arguments can be added for go.Scatter as ``**kwargs``
        :param grainsize: Array with the grain sizes (in mm)
        :param pctpassing: Array with the percentages passing each sieze opening size (in percent)
        :param name: Name for the trace (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        _data = go.Scatter(
            x=grainsize,
            y=pctpassing,
            mode='lines',
            name=name,
            **kwargs)
        self.fig.append_trace(_data, 1, 1)

    def show(self):
        self.fig.show(config=GROUNDHOG_PLOTTING_CONFIG)