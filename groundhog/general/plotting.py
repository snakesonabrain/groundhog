#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import datetime
import warnings
import traceback

# 3rd party packages
from jinja2 import Environment, BaseLoader
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly import subplots
import numpy as np
import matplotlib.pyplot as plt

# Project imports

GROUNDHOG_PLOTTING_CONFIG = {
    'showLink': True,
    'plotlyServerURL': "https://github.com/snakesonabrain/groundhog",
    'linkText': 'Created by groundhog using Plotly!'
}

PLOTLY_GLOBAL_FONT = dict(family='Century Gothic', size=12, color='#5f5f5f')
C0 = '#1f77b4'; C1 = '#ff7f0e'; C2 = '#2ca02c'; C3 = '#d62728'; C4 = '#9467bd'; C5 = '#8c564b'; C6 = '#e377c2'; C7 = '#7f7f7f'; C8 = '#bcbd22'; C9 = '#17becf'
PLOTLY_COLORS = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9]

BRIGHTCOLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

USCS_HATCHES = {
    'SW': '..',
    'SP': '....',
    'SM': '||...',
    'SC': '///...',
    'ML': '||||',
    'CL': '////',
    'OL': '--',
    'MH': '||',
    'CH': '//'
}

def plot_with_log(x=[[],], z=[[],], names=[[],], showlegends=None, hide_all_legends=False,
                  modes=None, markerformats=None,
                  soildata=None, fillcolordict={'SAND': 'yellow', 'CLAY': 'brown', 'SILT': 'green', 'ROCK': 'grey'},
                  depth_from_key="Depth from [m]", depth_to_key="Depth to [m]",
                  colors=None, logwidth=0.05,
                  xtitles=[], ztitle=None, xranges=None, zrange=None, ztick=None, dticks=None,
                  layout=dict(),
                  showfig=True):
    """
    Plots a given number of traces in a plot with a soil mini-log on the left hand side.
    The traces are given as a list of lists, the traces are grouped per plotting panel.
    For example x=[[np.linspace(0, 1, 100), np.logspace(0,2,100)], [np.linspace(1, 3, 100), ]] leads to the first two
    traces plotted in the first panel and one trace in the second panel. The same goes for the z arrays, trace names, ...

    :param x: List of lists of x-arrays for the traces
    :param z: List of lists of z-arrays for the traces
    :param names: List of lists of names for the traces (used in legend)
    :param showlegends: Array of booleans determining whether or not to show the trace in the legend. Showing/hiding legends can be specified per trace.
    :param hide_all_legends: Boolean indicating whether all legends need to be hidden (default=False).
    :param modes: List of display modes for the traces (select from 'lines', 'markers' or 'lines+markers'
    :param markerformats: List of formats for the markers (see Plotly docs for more info)
    :param soildata: Pandas dataframe with keys 'Soil type': Array with soil type for each layer, 'Depth from [m]': Array with start depth for each layer, 'Depth to [m]': Array with bottom depth for each layer
    :param fillcolordict: Dictionary with fill colours (default yellow for 'SAND', brown from 'CLAY' and grey for 'ROCK')
    :param depth_from_key: Key for the column with start depths of each layer
    :param depth_to_key: Key for the column with end depths of each layer
    :param colors: List of colours to be used for plotting (default = default Plotly colours)
    :param logwidth: Width of the soil width as a ratio of the total plot with (default = 0.05)
    :param xtitles: Array with X-axis titles for the panels
    :param ztitle: Depth axis title (Depth axis is shared between all panels)
    :param xranges: List with ranges to be used for X-axes
    :param zrange: Range to be used for Y-axis
    :param ztick: Tick interval to be used for the Y-axis
    :param dticks: List of tick intervals to be used for the X-axes
    :param layout: Dictionary with the layout settings
    :param showfig: Boolean determining whether the figure needs to be shown
    :return: Plotly figure object which can be further modified
    """

    no_panels = x.__len__()

    panel_widths = list(map(lambda _x: (1 - logwidth) / no_panels, x))

    panel_widths = list(np.append(logwidth, panel_widths))

    _fig = subplots.make_subplots(rows=1, cols=no_panels + 1, column_widths=panel_widths, shared_yaxes=True,
                                  print_grid=False)

    _showlegends = []
    _modes = []
    _markerformats = []
    _colors = []
    for i, _x in enumerate(x):
        _showlegends_panel = []
        _modes_panel = []
        _markerformats_panel = []
        _colors_panel = []
        for j, _trace_x in enumerate(_x):
            _showlegends_panel.append(not(hide_all_legends))
            _modes_panel.append('lines')
            _markerformats_panel.append(dict(size=5))
            _colors_panel.append(DEFAULT_PLOTLY_COLORS[j])
        _showlegends.append(_showlegends_panel)
        _modes.append(_modes_panel)
        _markerformats.append(_markerformats_panel)
        _colors.append(_colors_panel)

    if showlegends is None:
        showlegends = _showlegends
    if modes is None:
        modes = _modes
    if markerformats is None:
        markerformats = _markerformats
    if colors is None:
        colors = _colors

    _traces = []

    log_dummy_trace = go.Scatter(x=[0, 1], y=[np.nan, np.nan], showlegend=False)
    _fig.append_trace(log_dummy_trace, 1, 1)

    for i, _x in enumerate(x):
        for j, _trace_x in enumerate(_x):
            try:
                _trace = go.Scatter(
                    x=x[i][j],
                    y=z[i][j],
                    mode=modes[i][j],
                    name=names[i][j],
                    showlegend=showlegends[i][j],
                    marker=markerformats[i][j],
                    line=dict(color=colors[i][j]))
                _fig.append_trace(_trace, 1, i + 2)
            except Exception as err:
                warnings.warn(
                    "Error during traces creation for trace %s - %s" % (names[i][j], str(traceback.format_exc())))

    _layers = []
    for i, row in soildata.iterrows():
        _fillcolor = fillcolordict[row['Soil type']]
        _y0 = row[depth_from_key]
        _y1 = row[depth_to_key]
        _layers.append(
            dict(type='rect', xref='x1', yref='y', x0=0, y0=_y0, x1=1, y1=_y1, fillcolor=_fillcolor, opacity=1))

    if zrange is None:
        _fig['layout']['yaxis1'].update(title=ztitle, autorange='reversed')
    else:
        _fig['layout']['yaxis1'].update(title=ztitle, range=zrange)

    _fig['layout'].update(layout)
    _fig['layout'].update(shapes=_layers)

    if ztick is not None:
        _fig['layout']['yaxis1'].update(dtick=ztick)

    _fig['layout']['xaxis1'].update(
        anchor='y', title=None, side='top', tickvals=[])
    for i, _x in enumerate(x):
        _fig['layout']['xaxis%i' % (i + 2)].update(
            anchor='y', title=xtitles[i], side='top')
        if dticks is not None:
            _fig['layout']['xaxis%i' % (i + 2)].update(dtick=dticks[i])
        if xranges is not None:
            _fig['layout']['xaxis%i' % (i + 2)].update(range=xranges[i])

    if showfig:
        _fig.show(config=GROUNDHOG_PLOTTING_CONFIG)
    return _fig

class LogPlot(object):
    """
    Class for planneled plots with a minilog on the side.
    """

    def __init__(self, soilprofile, no_panels=1, logwidth=0.05,
                 fillcolordict={"Sand": 'yellow', "Clay": 'brown', 'Rock': 'grey'},
                 soiltypelegend=True,
                 soiltypecolumn="Soil type",
                 line_width=1,
                 **kwargs):
        """
        Initializes a figure with a minilog on the side.
        :param soilprofile: Soilprofile used for the minilog
        :param no_panels: Number of panels
        :param logwidth: Width of the minilog as a ratio to the total width of the figure (default=0.05)
        :param fillcolordict: Dictionary with fill colors for each of the soil types. Every unique ``Soil type`` needs to have a corresponding color. Default: ``{"Sand": 'yellow', "Clay": 'brown', 'Rock': 'grey'}``
        :param soiltypelegend: Boolean determining whether legend entries need to be shown for the soil types in the log
        :param soiltypecolumn: Column name used to identify the soil type. The entries in this column need to correspond to keys in ``fillcolordict``
        :param line_width: Line width for the boundary between layers
        :param kwargs: Optional keyword arguments for the make_subplots method
        """
        self.soilprofile = soilprofile

        # Determine the panel widths
        panel_widths = list(map(lambda _x: (1 - logwidth) / no_panels, range(0, no_panels)))

        panel_widths = list(np.append(logwidth, panel_widths))

        # Set up the figure
        self.fig = subplots.make_subplots(
            rows=1, cols=no_panels+1, column_widths=panel_widths, shared_yaxes=True,
            print_grid=False, **kwargs)
        self.fig['layout']['yaxis1'].update(range=(soilprofile.max_depth, soilprofile.min_depth))

        # Create rectangles for the log plot
        _layers = []
        for i, row in soilprofile.iterrows():
            try:
                _fillcolor = fillcolordict[row[soiltypecolumn]]
            except:
                _fillcolor = DEFAULT_PLOTLY_COLORS[i % 10]
            _y0 = row[self.soilprofile.depth_from_col]
            _y1 = row[self.soilprofile.depth_to_col]
            _layers.append(
                dict(type='rect', xref='x1', yref='y', x0=0, y0=_y0, x1=1, y1=_y1, fillcolor=_fillcolor, opacity=1, line_width=line_width))

        for _soiltype in soilprofile[soiltypecolumn].unique():
            try:
                _fillcolor = fillcolordict[_soiltype]
            except:
                soiltypelegend = False

            try:
                if soiltypelegend:
                    _trace = go.Bar(
                        x=[-10, -10],
                        y=[row['Depth to [m]'], row['Depth to [m]']],
                        name=_soiltype,
                        marker=dict(color=_fillcolor))
                    self.fig.append_trace(_trace, 1, 1)
            except:
                pass

        self.fig['layout'].update(shapes=_layers)
        self.fig['layout']['xaxis1'].update(
            anchor='y', title=None, side='top', tickvals=[], range=(0, 1))
        self.fig['layout']['yaxis1'].update(title='Depth [m]')

        for i in range(0, no_panels):
            _dummy_data = go.Scatter(
                x=[0, 100],
                y=[np.nan, np.nan],
                mode='lines',
                name='Dummy',
                showlegend=False,
                line=dict(color='black'))
            self.fig.append_trace(_dummy_data, 1, i + 2)
            self.fig['layout']['xaxis%i' % (i + 2)].update(
                anchor='y', title='X-axis %i' % (i+1), side='top')

    def add_trace(self, x, z, name, panel_no, resetaxisrange=False, **kwargs):
        """
        Adds a trace to the plot. By default, lines are added but optional keyword arguments can be added for go.Scatter as ``**kwargs``
        :param x: Array with the x-values
        :param z: Array with the z-values
        :param name: Name for the trace (LaTeX allowed, e.g. ``r'$ \\alpha $'``)
        :param panel_no: Panel to plot the trace on (1-indexed)
        :param resetaxisrange: Boolean determining whether the axis range needs to be reset to fit this trace
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        try:
            mode = kwargs['mode']
            kwargs.pop('mode')
        except:
            mode = 'lines'
        _data = go.Scatter(
            x=x,
            y=z,
            mode=mode,
            name=name,
            **kwargs)
        self.fig.append_trace(_data, 1, panel_no + 1)

        if resetaxisrange:
            self.fig['layout']['xaxis%i' % (panel_no + 1)].update(
                range=(np.array(x).min(), np.array(x).max()))

    def add_soilparameter_trace(self, parametername, panel_no, legendname=None, resetaxisrange=False, **kwargs):
        """
        Adds a trace to the plot based on a soil parameter available in the SoilProfile. By default, lines are added but optional keyword arguments can be added for go.Scatter as ``**kwargs``
        :param parametername: Name of the soil parameter (with units) e.g. ``'Su [kPa]'`` when ``'Su from [kPa]'`` and ``'Su to [kPa]'`` are available in the SoilProfile
        :param panel_no: Panel to plot the trace on (1-indexed)
        :param legendname: Name for the trace (LaTeX allowed, e.g. ``r'$ \\alpha $'``), default is None to use ``parametername``
        :param resetaxisrange: Boolean determining whether the axis range needs to be reset to fit this trace
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        if not parametername in self.soilprofile.soil_parameters():
            raise ValueError("Soil parameter %s not encoded in the soil profile. Check soil profile definition and try again" % parametername)
        x = self.soilprofile.soilparameter_series(parametername)[1]
        z = self.soilprofile.soilparameter_series(parametername)[0]

        if legendname is not None:
            name = legendname
        else:
            name = parametername
        self.add_trace(x, z, name, panel_no, resetaxisrange, **kwargs)

    def set_xaxis(self, title, panel_no, **kwargs):
        """
        Changes the X-axis title of a panel
        :param title: Title to be set (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param panel_no: Panel number (1-indexed)
        :param kwargs: Additional keyword arguments for the axis layout update function, e.g. ``range=(0, 100)``
        :return: Adjusts the X-axis of the specified panel
        """
        self.fig['layout']['xaxis%i' % (panel_no + 1)].update(
            title=title, **kwargs)

    def set_zaxis(self, title, **kwargs):
        """
        Changes the Z-axis
        :param title: Title to be set (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param kwargs: Additional keyword arguments for the axis layout update function, e.g. ``range=(0, 100)``
        :return: Adjusts the Z-axis
        """
        self.fig['layout']['yaxis1'].update(
            title=title, **kwargs)

    def set_size(self, width, height):
        """
        Adjust the size of the plot
        :param width: Width of the plot in pixels
        :param height: Height of the plot in pixels
        :return: Adjust the height and width as specified
        """
        self.fig['layout'].update(height=height, width=width)

    def set_title(self, title):
        """
        Set a title for the plot
        :param title: Title for the plot
        :return: Sets the title as specified
        """
        self.fig['layout'].update(title=title)

    def show(self):
        self.fig.show(config=GROUNDHOG_PLOTTING_CONFIG)


class LogPlotMatplotlib(object):
    """
    Class for planneled plots with a minilog on the side, using the Matplotlib plotting backend
    """

    def __init__(self, soilprofile, no_panels=1, logwidth=0.05,
                 fillcolordict={"Sand": 'yellow', "Clay": 'brown', 'Rock': 'grey', 'Silt': 'green'},
                 hatchpatterns={"Sand": "...", "Clay": '////', 'Rock':'oo', 'Silt': '|||'},
                 soiltypelegend=True, soiltypecolumn='Soil type', edgecolor='black',
                 figheight=6, plot_layer_transitions=True, showgrid=True,
                 **kwargs):
        """
        Initializes a figure with a minilog on the side.
        :param soilprofile: Soilprofile used for the minilog
        :param no_panels: Number of panels
        :param logwidth: Width of the minilog as a percentage of the total width (default=0.05)
        :param fillcolordict: Dictionary with fill colors for each of the soil types. Every unique ``Soil type`` needs to have a corresponding color. Default: ``{"Sand": 'yellow', "Clay": 'brown', 'Rock': 'grey'}``
        :param hatchpatterns: Matplotlib letters used for hatching of the soil types
        :param soiltypelegend: Boolean determining whether legend entries need to be shown for the soil types in the log
        :param soiltypecolumn: Column name used to identify the soil type. The entries in this column need to correspond to keys in ``fillcolordict``
        :param edgecolor: Color of the edge of a layer
        :param figheight: Figure height in inches (default=6in)
        :param plot_layer_transitions: Boolean determining whether layer transitions need to be plotted or not
        :param showgrid: Boolean determining whether a grid is shown on the plot panels or not (default=True)
        :param kwargs: Optional keyword arguments for the make_subplots method
        """
        self.soilprofile = soilprofile
        self.no_panels = no_panels
        # Determine the panel widths
        panel_widths = list(map(lambda _x: (1 - logwidth) / no_panels, range(0, no_panels)))

        panel_widths = list(np.append(logwidth, panel_widths))

        # Set up the figure
        self.fig, self.axes = plt.subplots(1, no_panels + 1, figsize=(4 * no_panels, figheight), sharex=False, sharey=True,
                        constrained_layout=False, gridspec_kw={'width_ratios': panel_widths})
        
        self.axes[0].set_ylim([soilprofile.max_depth, soilprofile.min_depth])

        # Create rectangles for the log plot
        _layers = []
        _color_assignment = dict()
        for i, row in soilprofile.iterrows():
            try:
                _fillcolor = fillcolordict[row[soiltypecolumn]]
                _color_assignment[row[soiltypecolumn]] = _fillcolor
            except:
                if row[soiltypecolumn] in _color_assignment.keys():
                    _fillcolor = _color_assignment[row[soiltypecolumn]]
                else:
                    _fillcolor = BRIGHTCOLORS[i % 7]
                    _color_assignment[row[soiltypecolumn]] = _fillcolor
            try:
                _hatch = hatchpatterns[row[soiltypecolumn]]
            except:
                _hatch = None
                
            _y0 = row[self.soilprofile.depth_from_col]
            _y1 = row[self.soilprofile.depth_to_col]
            self.axes[0].fill(
                [0.0,0.0,1.0,1.0],[_y0, _y1, _y1, _y0], fill=True, color=_fillcolor,
                label='_nolegend_', edgecolor=edgecolor, hatch=_hatch)
            
        _legend_handles = []
        for _soiltype in soilprofile[soiltypecolumn].unique():
            try:
                _fillcolor = _color_assignment[_soiltype]
            except:
                soiltypelegend = False

            try:
                if soiltypelegend:
                    _legend_entry, = self.axes[0].fill(
                        [-11.0,-11.0,-10.0,-10.0],[_y0, _y1, _y1, _y0], fill=True, color=_fillcolor,
                        label=_soiltype, edgecolor=edgecolor)
                    _legend_handles.append(_legend_entry)
            except:
                pass

        self._legend_entries = _legend_handles
        
        self.axes[0].set_xlim([0, 1])
        self.axes[0].get_xaxis().set_ticks([])
        self.axes[0].set_ylabel('Depth below mudline [m]',size=15)
        for i in range(0, no_panels):
            _dummy_data = self.axes[i+1].plot([0, 100], [np.nan, np.nan], label='_nolegend_')
            self.axes[i+1].tick_params(labelbottom=False,labeltop=True)
            self.axes[i+1].set_xlabel('X-axis %i' % (i + 1), size=15)
            self.axes[i+1].xaxis.set_label_position('top') 
            self.axes[i+1].set_xlim([0, 1])
            self.axes[i+1].set_ylim([soilprofile.max_depth, soilprofile.min_depth])

        self.plot_layer_transitions = plot_layer_transitions

        if showgrid:
            for i in range(0, no_panels):
                self.axes[i+1].grid()
        else:
            pass

    def add_trace(self, x, z, name, panel_no, resetaxisrange=False, line=True, showlegend=False, **kwargs):
        """
        Adds a trace to the plot. By default, lines are added but optional keyword arguments can be added for plt.plot as ``**kwargs``
        :param x: Array with the x-values
        :param z: Array with the z-values
        :param name: Label for the trace (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param panel_no: Panel to plot the trace on (1-indexed)
        :param resetaxisrange: Boolean determining whether the axis range needs to be reset to fit this trace
        :param line: Boolean determining whether the data needs to be shown as a line or as individual markers
        :param showlegend: Boolean determining whether the trace name needs to be added to the legend entries
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        if line:
            _axes_obj = self.axes[panel_no].plot(x, z,label=name, **kwargs)
        else:
            _axes_obj = self.axes[panel_no].scatter(x, z,label=name, **kwargs)

        if resetaxisrange:
            self.axes[panel_no].set_xlim([x[~np.isnan(x)].min(), x[~np.isnan(x)].max()])
        
        if showlegend:
            if line:
                self._legend_entries.append(_axes_obj[0])
            else:
                self._legend_entries.append(_axes_obj)

    def add_soilparameter_trace(self, parametername, panel_no, legendname=None, resetaxisrange=False, line=True, showlegend=False, **kwargs):
        """
        Adds a trace to the plot based on a soil parameter available in the SoilProfile. By default, lines are added but optional keyword arguments can be added for plt.plot as ``**kwargs``
        :param parametername: Name of the soil parameter (with units) e.g. ``'Su [kPa]'`` when ``'Su from [kPa]'`` and ``'Su to [kPa]'`` are available in the SoilProfile
        :param panel_no: Panel to plot the trace on (1-indexed)
        :param legendname: Label for the trace (LaTeX allowed, e.g. ``r'$ \\alpha $'``)
        :param resetaxisrange: Boolean determining whether the axis range needs to be reset to fit this trace
        :param line: Boolean determining whether the data needs to be shown as a line or as individual markers
        :param showlegend: Boolean determining whether the trace name needs to be added to the legend entries
        :param kwargs: Optional keyword arguments for the ``go.Scatter`` constructor
        :return: Adds the trace to the specified panel
        """
        if not parametername in self.soilprofile.soil_parameters():
            raise ValueError("Soil parameter %s not encoded in the soil profile. Check soil profile definition and try again" % parametername)
        x = self.soilprofile.soilparameter_series(parametername)[1]
        z = self.soilprofile.soilparameter_series(parametername)[0]

        if legendname is not None:
            name = legendname
        else:
            name = parametername
        self.add_trace(x, z, name, panel_no, resetaxisrange, line, showlegend, **kwargs)

    def set_xaxis_title(self, title, panel_no, size=15, **kwargs):
        """
        Changes the X-axis title of a panel
        :param title: Title to be set (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param panel_no: Panel number (1-indexed)
        :param kwargs: Additional keyword arguments for the axis layout update function, e.g. ``range=(0, 100)``
        :return: Adjusts the X-axis of the specified panel
        """
        self.axes[panel_no].set_xlabel(title, size=size)

    def set_xaxis_range(self, min_value, max_value, panel_no, ticks=None, **kwargs):
        """
        Changes the X-axis range of a panel
        :param min_value: Minimum value of the plot panel range
        :param max_value: Maximum value of the plot panel range
        :param panel_no: Panel number (1-indexed)
        :param ticks: List of ticks to set (default=None for Matplotlib defaults)
        :param kwargs: Additional keyword arguments for the ``set_xlim`` method
        :return: Adjusts the X-axis range of the specified panel
        """
        self.axes[panel_no].set_xlim([min_value, max_value])
        if ticks is not None:
            self.axes[panel_no].set_xticks(ticks)

    def set_zaxis_title(self, title, size=15, **kwargs):
        """
        Changes the Z-axis
        :param title: Title to be set (LaTeX allowed, e.g. ``r'$ \alpha $'``)
        :param kwargs: Additional keyword arguments for the ``set_label`` method
        :return: Adjusts the Z-axis title
        """
        self.axes[0].set_ylabel(title, size=size)
        
    def set_zaxis_range(self, min_depth, max_depth, ticks=None, **kwargs):
        """
        Changes the Z-axis
        :param min_depth: Minimum depth of the plot
        :param max_depth: Maximum depth of the plot
        :param ticks: List of ticks to set (default=None for Matplotlib defaults)
        :param kwargs: Additional keyword arguments for the ``set_ylim`` method
        :return: Adjusts the Z-axis range
        """
        self.axes[0].set_ylim([max_depth, min_depth])
        if ticks is not None:
            self.axes[0].set_yticks(ticks)

    def set_size(self, width, height):
        """
        Adjust the size of the plot
        :param width: Width of the plot in inches
        :param height: Height of the plot in inches
        :return: Adjust the height and width as specified
        """
        plt.gcf().set_size_inches(width, height)

    def show_legend(self):
        plt.legend(handles=self._legend_entries, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_layers(self):
        
        for i in range(0, self.no_panels):
            for _y in self.soilprofile.layer_transitions():
                self.axes[i+1].plot(
                    self.axes[i+1].get_xlim(),
                    (_y, _y),
                    color='grey', ls="--"
                )

    def show(self, showlegend=True, showfig=True):
        if self.plot_layer_transitions:
            self.plot_layers()
        else:
            pass

        if showlegend:
            self.show_legend()
        else:
            pass
        
        if showfig:
            plt.show()
        else:
            pass

    def save_fig(self, path, dpi=250, bbox_inches='tight',pad_inches=1):
        """
        Exports the figure to png format

        :param path: Path of the figure (filename ends in .png)
        :param dpi: Output resolution
        :param bbox_inches: Setting for the bounding box
        :param pad_inches: Inches for padding
        """
        plt.savefig(path, dpi=dpi,bbox_inches=bbox_inches, pad_inches=pad_inches)

    def select_additional_layers(self, no_additional_layers, panel_no=1, precision=2):
        """
        Allows for the selection of additional layer transitions for the ``SoilProfile`` object.
        The number of additional transition is controlled by the ``no_additional_layers`` argument.
        Click on the desired layer transition location in the specified panel (default ``panel_no=1``)
        The depth of the layer transition is rounded according to the ``precision`` argument. Default=2
        for cm accuracy."""
        ax = self.axes[panel_no]
        xy = plt.ginput(no_additional_layers, timeout=120)

        x = [p[0] for p in xy]
        y = [round(p[1], precision) for p in xy]
        for _y in y:
            for i in range(self.axes.__len__() - 1):
                line = self.axes[i+1].plot(
                    self.axes[i+1].get_xlim(),
                    (_y, _y), color='grey', ls="--")
            self.soilprofile.insert_layer_transition(_y)
        ax.figure.canvas.draw()

    def select_layering(self, panel_no=1, precision=2, stop_threshold=0):
        """
        Allows for the selection of layer transitions for the ``SoilProfile`` object.
        The number of additional transition is controlled by how often the user clicks.
        Click on the desired layer transition location in the specified panel (default ``panel_no=1``).
        The selection stops when the user clicks on a point with x-coordinate below the ``stop_threshold``.
        The depth of the layer transition is rounded according to the ``precision`` argument. Default=2
        for cm accuracy."""
        ax = self.axes[panel_no]

        final = False

        while not final:
            xy = plt.ginput(1, timeout=120)
            x = [p[0] for p in xy]
            y = [round(p[1], precision) for p in xy]
                
            if x[0] < stop_threshold:
                final = True
            else:
                for _y in y:
                    for i in range(self.axes.__len__() - 1):
                        line = self.axes[i+1].plot(
                            self.axes[i+1].get_xlim(),
                            (_y, _y), color='grey', ls="--")
                    self.soilprofile.insert_layer_transition(_y)
            ax.figure.canvas.draw()
    

    def select_constant(self, panel_no, parametername, units, nan_tolerance=0.1):
        """
        Selects a constant value in each layer. Click the desired value in each layer, working from the top down.
        If a nan value needs to be set in a layer, click sufficiently close to the minimum of the x axis.
        The ``nan_tolerance`` argument determines which values are interpreted as nan.
        The parameter is added to the ``SoilProfile`` object with the ``'parametername [units]'`` key.
        """
        ax = self.axes[panel_no]
        xy = plt.ginput(self.soilprofile.__len__(), timeout=120)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        
        for i, _x in enumerate(x):
            if _x < nan_tolerance:
                x[i] = np.nan

        self.soilprofile["%s [%s]" % (parametername, units)] = x
        self.add_soilparameter_trace(
            parametername="%s [%s]" % (parametername, units),
            panel_no=panel_no)
        ax.figure.canvas.draw()

    def select_linear(self, panel_no, parametername, units, nan_tolerance=0.1):
        """
        Selects a linear variation in each layer. Click the desired value at each layer boundary.
        Note that a value needs to be selected at the top and bottom of each layer (2 x no layers clicks).
        If a nan value needs to be set in a layer, click sufficiently close to the minimum of the x axis.
        The ``nan_tolerance`` argument determines which values are interpreted as nan.
        The parameter is added to the ``SoilProfile`` object with the ``'parametername [units]'`` key.
        """
        ax = self.axes[panel_no]
        xy = plt.ginput(2 * self.soilprofile.__len__(), timeout=120)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        
        for i, _x in enumerate(x):
            if _x < nan_tolerance:
                x[i] = np.nan
                
        self.soilprofile["%s from [%s]" % (parametername, units)] = x[::2]
        self.soilprofile["%s to [%s]" % (parametername, units)] = x[1::2]
        self.add_soilparameter_trace(
            parametername="%s [%s]" % (parametername, units),
            panel_no=panel_no)
        ax.figure.canvas.draw()

def peak_picker(x, y, correct_selected_point=True):
    """
    Generates an interactive Matplotlib plot which allows you to pick the peak from a graph with e.g. load-displacement data

    :param x: Array with x-values
    :param y: Array with y-values
    :param correct_selected_point: Boolean determining whether a correction is applied to interpolate the peak based on the selected value of X for the peak

    :returns: Dictionary with the following keys:

        - 'x100': x-coordinate of the peak
        - 'y100': y-coordinate of the peak
        - 'x50': x-coordinate where y reached 50% of its peak y
        - 'y50': y-coordinate with y equal to 50% of the peak y
    """
    # Coerce to Numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Close all open figures
    plt.close('all')
    # Generate the figure for peak picking
    plt.figure(1, figsize=(8,12))
    plt.plot(x, y)
    plt.xlabel('$ x $', size=15)
    plt.ylabel('$ y $', size=15)

    # Click on the peak
    xy = plt.ginput(1, timeout=120)
    # Calculate derived quantities
    x100 = xy[0][0]
    if correct_selected_point:
        y100 = np.interp(x100, x, y)
    else:
        y100 = xy[0][1]
    prepeak_x = x[np.where(x <= x100)]
    prepeak_y = y[np.where(x <= x100)]
    y50 = 0.5 * y100
    x50 = np.interp(y50, prepeak_y, prepeak_x)

    plt.scatter([x50, x100], [y50, y100], c='red')
    
    return {
        'x100': x100,
        'y100': y100,
        'x50': x50,
        'y50': y50,
        'plot': plt.gcf()
    }