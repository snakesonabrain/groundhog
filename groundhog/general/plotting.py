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
from plotly.offline import plot, iplot
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly import subplots
import numpy as np

# Project imports

GROUNDHOG_PLOTTING_CONFIG = {
    'showLink': True,
    'plotlyServerURL': "https://github.com/snakesonabrain/groundhog",
    'linkText': 'Created by groundhog using Plotly!'
}

PLOTLY_GLOBAL_FONT = dict(family='Century Gothic', size=12, color='#5f5f5f')
C0 = '#1f77b4'; C1 = '#ff7f0e'; C2 = '#2ca02c'; C3 = '#d62728'; C4 = '#9467bd'; C5 = '#8c564b'; C6 = '#e377c2'; C7 = '#7f7f7f'; C8 = '#bcbd22'; C9 = '#17becf'
PLOTLY_COLORS = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9]

PORTRAIT_TEMPLATE = """
<HTML>
    <head>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            @page { margin: 0 }
            body { margin: 0 }
            .sheet {
              margin: 0;
              overflow: hidden;
              position: relative;
              box-sizing: border-box;
              page-break-after: always;
            }

            body.A3           .sheet { width: 297mm; height: 419mm }
            body.A3.landscape .sheet { width: 420mm; height: 296mm }
            body.A4           .sheet { width: 210mm; height: 296mm }
            body.A4.landscape .sheet { width: 297mm; height: 209mm }
            body.A5           .sheet { width: 148mm; height: 209mm }
            body.A5.landscape .sheet { width: 210mm; height: 147mm }

            .sheet.padding-5mm { padding: 5mm}
            .sheet.padding-10mm { padding: 10mm }
            .sheet.padding-15mm { padding: 15mm }
            .sheet.padding-20mm { padding: 20mm }
            .sheet.padding-25mm { padding: 25mm }

            @media screen {
              body { background: #e0e0e0 }
              .sheet {
                background: white;
                box-shadow: 0 .5mm 2mm rgba(0,0,0,.3);
                margin: 5mm;
              }
            }
            @media print {
                       body.A3.landscape { width: 420mm }
              body.A3, body.A4.landscape { width: 297mm }
              body.A4, body.A5.landscape { width: 210mm }
              body.A5                    { width: 148mm }
            }
            table {
                border-spacing: 0;
                border-collapse: collapse;
                width: 100%;
            }
            table td{
                overflow: hidden;
                word-wrap: break-word;
                text-align: center;
            }
            table, th, td {
                border: 3px solid black;
            }
            th, td {
                border: 1px solid black;
                font-family: 'Century Gothic';
                font-size: 10px;
                padding: 3px;
            }
        </style>
    </head>
    <body class="A4 portrait">
        {% for fig in figures %}
        <section class="sheet" style="padding: 5mm;">
            <table class="table">
                <tbody>
                    <tr style="height: 920px">
                        <td colspan=4><div style="align-items: center">{{ fig.path }}</div></td>
                    </tr>
                    <tr>
                        <td style="width: 100px; border-bottom: 0px; padding: 3px">Drawn by</td>
                        <td style="width: 550px; padding: 3px; border-bottom: 0px; font-size: 14px; text-align: left; padding-left: 10px; padding-top: 10px" rowspan=2>
                            {{ fig.title }}<!-- Max chars = 90 --></td>
                        <td colspan=2 style="border-bottom: 0px; padding: 3px">Report no</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px">{{ fig.drawnby }}</td>
                        <td style="padding: 3px" colspan=2>{{ fig.report }}<!-- Max chars = 15 --></td>
                    </tr>
                    <tr>
                        <td style="border-bottom: 0px; padding: 3px">Checked by</td>
                        <td style="width: 550px; padding: 3px; text-align: left; padding-left: 10px" rowspan=4>
                            {% if fig.subtitle1 %}
                                {{ fig.subtitle1 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle2 %}
                                <br>{{ fig.subtitle2 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle3 %}
                                <br>{{ fig.subtitle3 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle4 %}
                                <br>{{ fig.subtitle4 }}
                            {% endif %}
                        </td><!-- Max chars = 120 -->
                        <td style="border-bottom: 0px; padding: 3px">Figure No</td>
                        <td style="border-bottom: 0px; padding: 3px">Rev</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px">{{ fig.checkedby }}</td>
                        <td style="padding: 3px">{{ fig.figno }}<!-- Max chars = 12 --></td>
                        <td style="padding: 3px">{{ fig.rev }}<!-- Max chars = 2 --></td>
                    </tr>
                    <tr>
                        <td rowspan=3><img src="https://en.wikipedia.org/wiki/Ghent_University#/media/File:Ghent_University_logo.svg" width="80px"></td>
                        <td colspan=2 style="border-bottom: 0px; padding: 3px">Date</td>
                    </tr>
                    <tr>
                        <td colspan=2 style="padding: 3px">{{ fig.date }}</td>
                    </tr>
                    <tr>
                        <td colspan=3 style="font-size: 15px">{{ fig.projecttitle }}</td><!-- Max chars = 95-->
                    </tr>

                </tbody>
            </table>
        </section>
        {% endfor %}

    </body>
</HTML>

"""

LANDSCAPE_TEMPLATE = """
<HTML>
    <head>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            @page { margin: 0 }
            body { margin: 0 }
            .sheet {
              margin: 0;
              overflow: hidden;
              position: relative;
              box-sizing: border-box;
              page-break-after: always;
            }

            body.A3           .sheet { width: 297mm; height: 419mm }
            body.A3.landscape .sheet { width: 420mm; height: 296mm }
            body.A4           .sheet { width: 210mm; height: 296mm }
            body.A4.landscape .sheet { width: 297mm; height: 209mm }
            body.A5           .sheet { width: 148mm; height: 209mm }
            body.A5.landscape .sheet { width: 210mm; height: 147mm }

            .sheet.padding-5mm { padding: 5mm}
            .sheet.padding-10mm { padding: 10mm }
            .sheet.padding-15mm { padding: 15mm }
            .sheet.padding-20mm { padding: 20mm }
            .sheet.padding-25mm { padding: 25mm }

            @media screen {
              body { background: #e0e0e0 }
              .sheet {
                background: white;
                box-shadow: 0 .5mm 2mm rgba(0,0,0,.3);
                margin: 5mm;
              }
            }
            @media print {
                       body.A3.landscape { width: 420mm }
              body.A3, body.A4.landscape { width: 297mm }
              body.A4, body.A5.landscape { width: 210mm }
              body.A5                    { width: 148mm }
            }
            table {
                border-spacing: 0;
                border-collapse: collapse;
                width: 100%;
            }
            table td{
                overflow: hidden;
                word-wrap: break-word;
                text-align: center;
            }
            table, th, td {
                border: 3px solid black;
            }
            th, td {
                border: 1px solid black;
                font-family: 'Century Gothic';
                font-size: 10px;
                padding: 3px;
            }
        </style>
    </head>
    <body class="A4 landscape">
        {% for fig in figures %}
        <section class="sheet" style="padding: 5mm;">
            <table class="table">
                <tbody>
                    <tr style="height: 500px">
                        <td colspan=4><div style="align-items: center">{{ fig.path }}</div></td>
                    </tr>
                    <tr>
                        <td style="width: 100px; border-bottom: 0px; padding: 3px">Drawn by</td>
                        <td style="width: 875px; padding: 3px; border-bottom: 0px; font-size: 14px; text-align: left; padding-left: 10px; padding-top: 10px" rowspan=2>
                            {{ fig.title }}<!-- Max chars = 90 --></td>
                        <td colspan=2 style="border-bottom: 0px; padding: 3px">Report no</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px">{{ fig.drawnby }}</td>
                        <td style="padding: 3px" colspan=2>{{ fig.report }}<!-- Max chars = 15 --></td>
                    </tr>
                    <tr>
                        <td style="border-bottom: 0px; padding: 3px">Checked by</td>
                        <td style="width: 875px; padding: 3px; text-align: left; padding-left: 10px" rowspan=4>
                            {% if fig.subtitle1 %}
                                {{ fig.subtitle1 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle2 %}
                                <br>{{ fig.subtitle2 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle3 %}
                                <br>{{ fig.subtitle3 }}<!-- Max chars = 120 -->
                            {% endif %}
                            {% if fig.subtitle4 %}
                                <br>{{ fig.subtitle4 }}
                            {% endif %}
                        </td><!-- Max chars = 120 -->
                        <td style="border-bottom: 0px; padding: 3px">Figure No</td>
                        <td style="border-bottom: 0px; padding: 3px">Rev</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px">{{ fig.checkedby }}</td>
                        <td style="padding: 3px">{{ fig.figno }}<!-- Max chars = 12 --></td>
                        <td style="padding: 3px">{{ fig.rev }}<!-- Max chars = 2 --></td>
                    </tr>
                    <tr>
                        <td rowspan=3><img src="https://en.wikipedia.org/wiki/Ghent_University#/media/File:Ghent_University_logo.svg" width="80px"></td>
                        <td colspan=2 style="border-bottom: 0px; padding: 3px">Date</td>
                    </tr>
                    <tr>
                        <td colspan=2 style="padding: 3px">{{ fig.date }}</td>
                    </tr>
                    <tr>
                        <td colspan=3 style="font-size: 15px">{{ fig.projecttitle }}</td><!-- Max chars = 95-->
                    </tr>

                </tbody>
            </table>
        </section>
        {% endfor %}

    </body>
</HTML>
"""

def generate_html(outpath, figures, titles, drawnby, report,
                           fignos, rev, projecttitle,
                           subtitle1s, subtitle2s=None, subtitle3s=None, subtitle4s=None, checkedby=None,
                           figure_date=datetime.date.today().strftime("%d/%m/%Y"),
                           filename='figures',
                           portraitformat=True,
                           print_message=True):
    """
    Returns HTML for a series of Plotly portrait figures.
    The figures are first coerced to the correct format (portrait: H=870 x W=702, landscape: H=600 x W=1050). Standardized font and colors are also set.

    :param outpath: Path where the output is written. Use absolute paths.
    :param figures: List of Plotly figures
    :param titles: List of titles for the figures (maximum 90 characters)
    :param drawnby: Initials of drawer (maximum 3 characters)
    :param report: Report number (maximum 15 characters)
    :param fignos: List with figure numbers (maximum 12 characters)
    :param rev: Revision of the figures (maximum 2 characters)
    :param date: Date of drawing in %d/%m/%Y format
    :param projecttitle: Title of the project (maximum 95 characters)
    :param subtitle1s: List with first subtitles (maximum 120 characters)
    :param subtitle2s: List with second subtitles (maximum 120 characters) - default is None
    :param subtitle3s: List with third subtitles (maximum 120 characters) - default is None
    :param subtitle4s: List with fourth subtitles (maximum 120 characters) - default is None
    :param checkedby: Initials of the checker (maximum 3 characters)
    :param filename: Filename for the output, 'figures' by default
    :param portraitformat: Boolean determining whether the figure is portrait or landscape format
    :param print_message: Defines whether a message is returned to the user
    :return: Writes a file with HTML code to the specified output path
    """

    # 0. Validation of string lengths
    if len(drawnby) > 3:
        raise ValueError("Initials cannot be longer than 3 characters")
    if checkedby is not None:
        if len(checkedby) > 3:
            raise ValueError("Initials cannot be longer than 3 characters")
    if len(figure_date) > 10:
        raise ValueError("Dates cannot exceed 10 characters")
    if len(report) > 15:
        raise ValueError("Report number cannot exceed 15 characters")
    if len(rev) > 2:
        raise ValueError("Revision number cannot exceed 2 characters")
    if len(projecttitle) > 95:
        raise ValueError("Project title cannot exceed 95 characters")
    for title in titles:
        if len(title) > 90:
            raise ValueError("Figure titles cannot exceed 90 characters")
    for figno in fignos:
        if len(figno) > 12:
            raise ValueError("Figure titles cannot exceed 12 characters")
    for subtitle in subtitle1s:
        if len(subtitle) > 120:
            raise ValueError("Figure subtitles cannot exceed 120 characters")
    if subtitle2s is not None:
        for subtitle in subtitle2s:
            if len(subtitle) > 120:
                raise ValueError("Figure subtitles cannot exceed 120 characters")
    if subtitle3s is not None:
        for subtitle in subtitle3s:
            if len(subtitle) > 120:
                raise ValueError("Figure subtitles cannot exceed 120 characters")
    if subtitle4s is not None:
        for subtitle in subtitle4s:
            if len(subtitle) > 120:
                raise ValueError("Figure subtitles cannot exceed 120 characters")
    if len(titles) == len(fignos) == len(figures) == len(subtitle1s):
        pass
    else:
        raise ValueError('Lists with figures, figure titles, figure numbers, ... all need to be the same length')
    if subtitle2s is not None:
        if len(subtitle1s) != len(subtitle2s):
            raise ValueError("Lists with subtitles need to be the same length")
    if subtitle3s is not None:
        if len(subtitle1s) != len(subtitle3s):
            raise ValueError("Lists with subtitles need to be the same length")
    if subtitle4s is not None:
        if len(subtitle1s) != len(subtitle4s):
            raise ValueError("Lists with subtitles need to be the same length")

    if portraitformat:
        template = PORTRAIT_TEMPLATE
        figure_height=870
        figure_width=702
    else:
        template = LANDSCAPE_TEMPLATE
        figure_height = 600
        figure_width = 1050
    try:
        # 1. Coerce the figure to correct format, set font and colors
        figure_list = []
        for i, fig in enumerate(figures):

            fig['layout'].update(
                height=figure_height,
                width=figure_width,
                font=PLOTLY_GLOBAL_FONT,
                colorway=PLOTLY_COLORS
            )
            div = plot(fig, auto_open=False, output_type='div', show_link=False, include_plotlyjs=False)
            figure = {
                'path': div,
                'title': titles[i],
                'drawnby': drawnby,
                'report': report,
                'subtitle1': subtitle1s[i],
                'checkedby': checkedby,
                'figno': fignos[i],
                'rev': rev,
                'date': figure_date,
                'projecttitle': projecttitle,
            }
            if subtitle2s is not None:
                figure['subtitle2'] = subtitle2s[i]
            if subtitle3s is not None:
                figure['subtitle3'] = subtitle3s[i]
            if subtitle4s is not None:
                figure['subtitle4'] = subtitle4s[i]
            figure_list.append(figure)

        # 2. Render the template
        rtemplate = Environment(loader=BaseLoader).from_string(template)
        html_figures = rtemplate.render(
            figures=figure_list)

        # 3. Write the output
        with open("%s/%s.html" % (outpath, filename), "w+") as renderedhtmlfile:
            renderedhtmlfile.write(html_figures)

        if print_message:
            print("Figures successfully generated. Open the file %s/%s.html to see the output." % (
                outpath, filename))
    except:
        raise


def plot_with_log(x=[[],], z=[[],], names=[[],], showlegends=None,
                  modes=None, markerformats=None,
                  soildata=None, fillcolordict={'SAND': 'yellow', 'CLAY': 'brown', 'SILT': 'green', 'ROCK': 'grey'},
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
    :param showlegends: Array of booleans determining whether or not to show the trace in the legend
    :param modes: List of display modes for the traces (select from 'lines', 'markers' or 'lines+markers'
    :param markerformats: List of formats for the markers (see Plotly docs for more info)
    :param soildata: Pandas dataframe with keys 'Soil type': Array with soil type for each layer, 'Depth from [m]': Array with start depth for each layer, 'Depth to [m]': Array with bottom depth for each layer
    :param fillcolordict: Dictionary with fill colours (default yellow for 'SAND', brown from 'CLAY' and grey for 'ROCK')
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
            _showlegends_panel.append(True)
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
        _y0 = row['Depth from [m]']
        _y1 = row['Depth to [m]']
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
        iplot(_fig, filename='logplot', config=GROUNDHOG_PLOTTING_CONFIG)
    return _fig

