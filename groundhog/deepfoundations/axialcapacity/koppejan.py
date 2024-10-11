#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
from copy import deepcopy

# 3rd party packages
import numpy as np
import pandas as pd
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

# Project imports
from groundhog.general.validation import check_layer_overlap
from groundhog.general.parameter_mapping import map_depth_properties
from groundhog.general.plotting import GROUNDHOG_PLOTTING_CONFIG


class KoppejanCalculation(object):

    def __init__(self, depth, qc, diameter, penetration):
        """
        Initializes a pile calculation according to Koppejan

        :param depth: Array with the depth coordinates (ascending)
        :param qc: Array with corresponding cone resistance values [MPa]
        :param diameter: Pile diameter [m]
        :param penetration: Pile penetration (tip depth below mudline) [m]
        """
        self.data = pd.DataFrame({
            'z [m]': depth,
            'qc [MPa]': qc
        })
        if penetration > (self.data['z [m]'].max() - 4 * diameter):
            raise ValueError("The pile penetration cannot be deeper than the maximum CPT depth - 4D")
        self.diameter = diameter
        self.penetration = penetration

    def set_layer_properties(self, layer_data, waterlevel=0, waterunitweight=10):
        """
        Creates the layering used for further interpretation of the PCPT profile.
        A dataframe with a layering definition needs to be provided
        Typically, total unit weight is provided as a minimum. Linear variations over the depth range are also allowed.
        Other properties may be provided as required by the correlations.
        The water level needs to be defined for calculation of the vertical effective stress profile.

        :param layer_data: Pandas dataframe with layering definition. As a minimum the keys 'Depth from [m]', 'Depth to [m]' and 'Total unit weight [kN/m3]' need to be provided.
        :param waterlevel: Level below soil surface (>=0m) where the watertable starts, default = 0m for fully saturated conditions
        :param waterunitweight: Unit weight of water used for the calculation [kN/m2] - Default = 10kN/m3
        :return: Sets the attribute `layerdata`

        """
        # Validation
        for key in ['Depth from [m]', 'Depth to [m]']:
            if key not in layer_data.columns:
                raise ValueError("Dataframe needs to contain key '%s'" % key)

        for key in ['Total unit weight ____[kN/m3]', ]:
            if (key.replace("____", "") not in layer_data.columns) and \
                    (key.replace("____", "from ") not in layer_data.columns) and \
                    (key.replace("____", "to ") not in layer_data.columns):
                raise ValueError("Dataframe needs to contain key '%s' or a similar range definition" % (
                    key.replace("____", "")))

        layer_data.sort_values('Depth from [m]', inplace=True)
        check_layer_overlap(layer_data, z_from_key="Depth from [m]", z_to_key="Depth to [m]")

        for i, row in layer_data.iterrows():
            layer_data.loc[i, "Layer no"] = i+1
            if row['Depth to [m]'] - row['Depth from [m]'] < 1:
                layer_data.loc[i, "qclim [MPa]"] = 12
            else:
                layer_data.loc[i, "qclim [MPa]"] = 15
        self.waterunitweight = waterunitweight
        self.layerdata = layer_data
        self.waterlevel = waterlevel
        self.map_properties()

    def map_properties(self):
        """
        Maps the soil properties defined in the layering to the grid
        defined by the cone data.

        :return: Expands the dataframe `.data` with additional columns for the soil properties
        """
        # Map cone properties
        # Map soil properties
        self.data = map_depth_properties(
            target_df=self.data,
            layering_df=self.layerdata,
            layering_zfrom_key="Depth from [m]",
            layering_zto_key="Depth to [m]"
        )

    def calculate_side_friction(self, alpha_s):
        """
        Calculates the side friction for a pile according to Koppejan's method.

        The maximum shaft friction is then given by the following formula:

        .. math::
            \\tau_{s,max} = \\alpha_s \\cdot \\min(q_c, q_{c,lim})

        Note that :math:`\\tau_{s,max}` is given in kPa so a conversion factor from MPa to kPa is required.

        The shaft resistance is then obtained by integrating the maximum shaft friction over the pile length and the outer shaft perimeter:

        .. math::
            F_{r,s} = \\int_{0}^{L} dT = \\int_{0}^{L} \\tau_{s,max}(z) \\cdot \\pi \\cdot D \\cdot dz

        This equation can easily be discretised for the numerical calculation of the shaft resistance.

        The value of :math:`\\alpha_s` depends on the pile type and can be read from the table below

        .. figure:: images/koppejan_alphas.png
            :figwidth: 600.0
            :width: 550.0
            :align: center

            :math:`\\alpha_s` and :math:`\\alpha_b` factors for Koppejan calculation

        :param alpha_s: The value of the shaft friction coefficient for the given pile type (as found in the table)
        :return: Expands the ``data`` attribute with columns ``'tau s max [kPa]'``, ``'dT [kN/m]'``, ``'Frs [kN]'``
        """
        # Derive the selected cone resistance (minimum of maximum and actual cone resistance)
        self.data["qc selected [MPa]"] = list(
            map(
                lambda qc, qclim: min(qc, qclim),
                self.data["qc [MPa]"], self.data["qclim [MPa]"]))
        # Determine the grid size increments
        self.data["dz [m]"] = self.data["z [m]"].diff()
        # Calculate unit shaft friction
        self.data["tau s max [kPa]"] = 1000.0 * alpha_s * self.data["qc selected [MPa]"]
        # Calculate shaft friction increments by integrating over the pile circumference
        self.data["dT [kN/m]"] = self.data["tau s max [kPa]"] * np.pi * self.diameter * self.data["dz [m]"]
        # Shaft capacity is the cumulative sum of all increments
        self.data["Frs [kN]"] = self.data["dT [kN/m]"].cumsum()
        # Shaft resistance is obtained at the pile penetration
        self.Frs = np.interp(self.penetration, self.data['z [m]'], self.data['Frs [kN]'])

    def calculate_base_resistance(self, alpha_p, base_coefficient=1, crosssection_coefficient=1,
                                  coring=False, wall_thickness=np.nan):
        """
        Calculates the pile base resistance according to Koppejans method. The unit base resistance is obtained
        using Koppejan's construction.

        .. figure:: images/koppejan_base_theory.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Zone of influence at pile base according to Koppejan

        :math:`q_{cII}` is obtained by taking the average cone resistance in a window varying from
        0.7D to 4D. The size of this window is chosen such that :math:`q_{cII}` is minimized.
        The reason for varying the maximum depth of this window is that weaker layers up to a certain depth
        below the pile tip will have an influence on the pile resistance.
        A weaker layer at a certain depth below the pile tip will lead to a lower value for
        :math:`q_{cII}` as the averaging window expands.

        The value of :math:`q_{c,I}` is obtained by taking the average along the same path as :math:`q_{c,II}`
        but the cone resistance can never increase as one moves up from the cone resistance at the bottom
        of the averaging window.

        :math:`q_{cIII}` accounts for the averaging effect in the soil above the pile tip.
        The averaging window ranges from the pile tip to 8D above the pile tip.

        Similar to :math:`q_{cI}`, the cone resistance should never increase as one moves up through
        the averaging window.

        The unit base resistance is then calculates as:

        .. math::
            q_{c,avg} = \\frac{0.5 \\cdot (q_{cI} + q_{cII}) + q_{cIII}}{2}

        Additional multipliers are applied for the effect of a non-uniform pile profile along the length:

        .. figure:: images/base_increase.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Dimensions considered for non-uniform deepfoundations

        .. figure:: images/base_coefficient.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Chart for coefficient :math:`\\beta` for a non-uniform pile

        and for non-circular cross-sections.

        .. figure:: images/crosssection_coefficient.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Chart for coefficient :math:`s` for a non-circular cross-sections

        A coefficient :math:`\\alpha_p` needs to be selected to account for the pile type.
        The pile base resistance is then given as:

        .. math::
            q_{b,max} = \\alpha_p \\cdot \\beta \\cdot s \\cdot q_{c,avg} \\leq 15 \\text{MPa}

        The base resisance is then obtained by multiplying the maximum unit end bearing by the pile tip area.
        Note that coring tubular deepfoundations can also be calculated. In this case ``coring`` needs to be set to ``True``
        and a ``wall_thickness`` needs to be provided.

        :param alpha_p: Coefficient for the base resistance based on pile type (see table in ``calculate_side_friction`` method
        :param base_coefficient: Coefficient for enlarged bases (default=1 for a uniform pile)
        :param crosssection_coefficient: Coefficient for non-circular cross-sections (default=1 for a circular cross-section
        :param Boolean determining whether the pile behaves in a coring manner (default=False)
        :param Wall thickness [mm]. Only needs to be specified for coring deepfoundations
        :return: Creates the base resistance construction and sets the attribute ``Frb``
        """
        # Calculation of qcII
        qcII_window_min = 0.7 * self.diameter  # Smallest averaging window size
        qcII_window_max = 4.0 * self.diameter  # Largest averaging window size
        qcII_window_sizes = np.linspace(qcII_window_min, qcII_window_max, 50)
        self.qcII_values = np.array([])
        self.qcII_depths = np.array([])
        for i, size in enumerate(qcII_window_sizes):
            window_data = self.data[(self.data["z [m]"] >= self.penetration) & (self.data["z [m]"] <= self.penetration + size)]
            self.qcII_values = np.append(self.qcII_values, window_data["qc [MPa]"].mean())
            self.qcII_depths = np.append(self.qcII_depths, self.penetration + size)
        self.qcII = self.qcII_values.min()
        self.qcIIz = self.qcII_depths[np.where(self.qcII_values == self.qcII)[0][0]]
        # Calculation of qcI
        self.qcI_data = deepcopy(
            self.data[(self.data["z [m]"] <= self.qcIIz) & (self.data["z [m]"] >= self.penetration)])
        for j, (i, row) in enumerate(self.qcI_data.sort_values("z [m]", ascending=False).iterrows()):
            if j == 0:
                self.qcI_data.loc[i, "qc I value [MPa]"] = self.qcI_data.loc[i, "qc [MPa]"]
            else:
                if self.qcI_data.loc[i, "qc [MPa]"] > self.qcI_data.loc[i + 1, "qc I value [MPa]"]:
                    self.qcI_data.loc[i, "qc I value [MPa]"] = self.qcI_data.loc[i + 1, "qc I value [MPa]"]
                else:
                    self.qcI_data.loc[i, "qc I value [MPa]"] = self.qcI_data.loc[i, "qc [MPa]"]
        self.qcI = self.qcI_data["qc I value [MPa]"].mean()
        # Calculation of qcIII
        self.qcIII_data = deepcopy(
            self.data[(self.data["z [m]"] <= self.penetration) &
                      (self.data["z [m]"] >= self.penetration - 8.0 * self.diameter)])
        for j, (i, row) in enumerate(self.qcIII_data.sort_values("z [m]", ascending=False).iterrows()):
            if j == 0:
                if self.qcIII_data.loc[i, "qc [MPa]"] > self.qcI_data["qc I value [MPa]"].min():
                    self.qcIII_data.loc[i, "qc III value [MPa]"] = self.qcI_data["qc I value [MPa]"].min()
                else:
                    self.qcIII_data.loc[i, "qc III value [MPa]"] = self.qcIII_data.loc[i, "qc [MPa]"]
            else:
                if self.qcIII_data.loc[i, "qc [MPa]"] > self.qcIII_data.loc[i + 1, "qc III value [MPa]"]:
                    self.qcIII_data.loc[i, "qc III value [MPa]"] = self.qcIII_data.loc[i + 1, "qc III value [MPa]"]
                else:
                    self.qcIII_data.loc[i, "qc III value [MPa]"] = self.qcIII_data.loc[i, "qc [MPa]"]
        self.qcIII = self.qcIII_data["qc III value [MPa]"].mean()
        # Unit base resistance
        self.qcavg = 0.5 * (0.5 * (self.qcI + self.qcII) + self.qcIII)
        self.qbmax = min(15.0, alpha_p * base_coefficient * crosssection_coefficient * self.qcavg)

        if coring and np.isnan(wall_thickness):
            raise ValueError("For a coring pile, a wall thickness needs to be provided (in mm)")

        if coring:
            self.base_area = 0.25 * np.pi * ((self.diameter ** 2) - ((self.diameter - 2 * 0.001 * wall_thickness) ** 2))
        else:
            self.base_area = 0.25 * np.pi * (self.diameter ** 2)

        self.Frb = 1000 * self.qbmax * self.base_area

    def plot_shaft_resistance(
            self, plot_width=800, plot_height=600, plot_title=None, plot_margin=dict(t=100, l=50, b=50), show_fig=True,
            x_ranges=((0, 50), (0, 250), (0, 50), (0, 2000)), x_ticks=(10, 50, 10, 400), y_range=None, y_tick=2,
            legend_orientation='h', legend_x=0.05, legend_y=-0.05, latex_titles=True):
        """

        :param plot_width: Width of the plot (default = 800px)
        :param plot_height: Height of the plot (default = 600px)
        :param plot_title: Title of the plot (default=None)
        :param plot_margin: Margins for the plot (default=``dict(t=50, l=50, b=50)``
        :param show_fig: Boolean determining whether the plot is shown or not
        :param x_ranges: List of tuples with the ranges for the x-axes of the plot
        :param x_ticks: Tick mark intervals for the x-axes
        :param y_range: Range for the y-axis (default=None which causes ``autorange=reversed`` to be used
        :param y_tick: Tick mark interval for the y-axis
        :param legend_orientation: Orientation of the plot legend (default = ``'h'``)
        :param legend_x: x-coordinate of the legend (plot is between 0 and 1, default=0.05 to start at an offset from the left edge)
        :param legend_y: y-coordinate of the legend (plot is between 0 and 1, default=-0.05 to be below the plot)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Sets the attribute ``shaft_fig`` of the ``KoppejanCalculation`` object
        """
        if latex_titles:
            qc_trace_title = r'$ q_c $'
            qclim_trace_title = r'$ q_{c,lim} $'
            qcselected_trace_title = r'$ q_{c,selected} $'
            tau_s_max_trace_title = r'$ \tau_{s,max} $'
            dT_trace_title = r'$ dT $'
            Frs_trace_title = r'$ F_{rs} $'
            Frs_calc_trace_title = r'$ F_{rs,calc} $'
            qc_axis_title = r'$ q_c \ \text{[MPa]}$'
            tau_s_axis_title = r'$ \tau_{s,max} \ \text{[kPa]} $'
            dT_axis_title = r'$ dT \ \text{[kN/m]} $'
            T_axis_title = r'$ T \ \text{[kN]} $'
            z_axis_title = r'$ z \ \text{[m]} $'
        else:
            qc_trace_title = 'qc'
            qclim_trace_title = 'qc,lim'
            qcselected_trace_title = 'qc,selected'
            tau_s_max_trace_title = 'tau_s,max'
            dT_trace_title = 'dT'
            Frs_trace_title = 'Frs'
            Frs_calc_trace_title = 'Frs,calc'
            qc_axis_title = 'qc [MPa]'
            tau_s_axis_title = 'tau_s,max [kPa]'
            dT_axis_title = 'dT [kN/m]'
            T_axis_title = 'T [kN]'
            z_axis_title = 'z [m]'

        self.shaft_fig = subplots.make_subplots(rows=1, cols=4, print_grid=False, shared_yaxes=True)
        trace1a = go.Scatter(x=self.data["qc [MPa]"], y=self.data["z [m]"],
                             showlegend=True, mode='lines', name=qc_trace_title)
        self.shaft_fig.append_trace(trace1a, 1, 1)
        trace1b = go.Scatter(x=self.data["qclim [MPa]"], y=self.data["z [m]"],
                             showlegend=True, mode='lines', name=qclim_trace_title,
                             line=dict(color='red', dash='dashdot'))
        self.shaft_fig.append_trace(trace1b, 1, 1)
        trace1c = go.Scatter(x=self.data["qc selected [MPa]"], y=self.data["z [m]"],
                             showlegend=True, mode='lines', name=qcselected_trace_title,
                             line=dict(color='green', dash='dot', width=3))
        self.shaft_fig.append_trace(trace1c, 1, 1)
        trace2 = go.Scatter(x=self.data["tau s max [kPa]"], y=self.data["z [m]"],
                            showlegend=False, mode='lines', name=tau_s_max_trace_title)
        self.shaft_fig.append_trace(trace2, 1, 2)
        trace3 = go.Scatter(x=self.data["dT [kN/m]"], y=self.data["z [m]"],
                            showlegend=False, mode='lines', name=dT_trace_title)
        self.shaft_fig.append_trace(trace3, 1, 3)
        trace4 = go.Scatter(x=self.data["Frs [kN]"], y=self.data["z [m]"],
                            showlegend=False, mode='lines', name=Frs_trace_title)
        self.shaft_fig.append_trace(trace4, 1, 4)
        trace5 = go.Scatter(x=[self.Frs,], y=[self.penetration,],
                            showlegend=False, mode='markers', name=Frs_calc_trace_title,
                            marker=dict(size=10,color='red',line=dict(width=2,color='black')))
        self.shaft_fig.append_trace(trace5, 1, 4)
        self.shaft_fig['layout']['xaxis1'].update(
            title=qc_axis_title, side='top', anchor='y',
            range=x_ranges[0], dtick=x_ticks[0])
        self.shaft_fig['layout']['xaxis2'].update(
            title=tau_s_axis_title, side='top', anchor='y',
            range=x_ranges[1], dtick=x_ticks[1])
        self.shaft_fig['layout']['xaxis3'].update(
            title=dT_axis_title, side='top', anchor='y',
            range=x_ranges[2], dtick=x_ticks[2])
        self.shaft_fig['layout']['xaxis4'].update(
            title=T_axis_title, side='top', anchor='y',
            range=x_ranges[3], dtick=x_ticks[3])
        if y_range is None:
            self.shaft_fig['layout']['yaxis1'].update(title=z_axis_title, autorange='reversed', dtick=y_tick)
        else:
            self.shaft_fig['layout']['yaxis1'].update(
                title=z_axis_title, range=y_range, dtick=y_tick)
        self.shaft_fig['layout'].update(height=plot_height, width=plot_width,
            title=plot_title,
            margin=plot_margin,
            legend=dict(orientation=legend_orientation, x=legend_x, y=legend_y))
        if show_fig:
            self.shaft_fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    def plot_baseconstruction(
            self, plot_width=500, plot_height=600, plot_title=None, plot_margin=dict(t=50, l=50, b=50), show_fig=True,
            x_range=(0, 50), y_range=None, latex_titles=True):
        """
        Plots the Koppejan base resistance construction

        :param plot_width: Width of the plot (default=500px)
        :param plot_height: Height of the plot (default=600px)
        :param plot_title: Title of the plot (default=None)
        :param plot_margin: Margins used for the plot (default=dict(t=50, l=50, b=50))
        :param show_fig: Boolean determining whether the plot needs to be displayed (default=True)
        :param x_range: Range of x-values to be used for the plotting (default=0-50)
        :param y_range: Range of y-values to be used for the plotting (default=None which causes ``reversed`` to be used
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Sets the attribute ``base_fig`` of the ``KoppejanCalculation`` object
        """
        if latex_titles:
            qc_trace_title = r'$ q_c $'
            qcII_trace_title = r'$ q_{cII} $'
            qcII_selected_trace_title = r'$ q_{cII,selected} $'
            qcI_values_trace_title = r'$ q_{c,I,values}$'
            qcI_selected_trace_title = r'$ q_{cI,selected} $'
            qcIII_values_trace_title = r'$ q_{cIII,values} $'
            qcIII_selected_trace_title = r'$ q_{cIII,selected} $'
            qc_axis_title = r'$ q_c \ \text{[MPa]} $'
            z_axis_title = r'$ z \ \text{[m]} $'
        else:
            qc_trace_title = 'qc'
            qcII_trace_title = 'qcII'
            qcII_selected_trace_title = 'qcII,selected'
            qcI_values_trace_title = 'qcI,values'
            qcI_selected_trace_title = 'qcI,selected'
            qcIII_values_trace_title = 'qcIII,values'
            qcIII_selected_trace_title = 'qcIII,selected'
            qc_axis_title = 'qc [MPa]'
            z_axis_title = 'z [m]'

        self.base_fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)
        trace1 = go.Scatter(
            x=self.data['qc [MPa]'], y=self.data['z [m]'], showlegend=True, mode='lines',
            name=qc_trace_title)
        self.base_fig.append_trace(trace1, 1, 1)
        trace1c = go.Scatter(
            x=self.qcII_values, y=self.qcII_depths, showlegend=True,
            mode='lines', name=qcII_trace_title, line=dict(color=DEFAULT_PLOTLY_COLORS[1]))
        self.base_fig.append_trace(trace1c, 1, 1)
        # Add the selected qcII value
        trace1d = go.Scatter(
            x=[self.qcII, ], y=[self.qcIIz, ], showlegend=True, mode='markers', name=qcII_selected_trace_title,
            marker=dict(size=5, color=DEFAULT_PLOTLY_COLORS[1], line=dict(width=1, color='black')))
        self.base_fig.append_trace(trace1d, 1, 1)
        # Plot the pile tip depth
        trace1e = go.Scatter(x=[0.0, 200], y=[self.penetration, self.penetration], showlegend=True,
                             mode='lines', name='Tip depth',
                             line=dict(color='red', dash='dot'))
        self.base_fig.append_trace(trace1e, 1, 1)
        traceqcI = go.Scatter(
            x=self.qcI_data["qc I value [MPa]"], y=self.qcI_data["z [m]"], showlegend=True, mode='lines',
            name=qcI_values_trace_title, line=dict(color=DEFAULT_PLOTLY_COLORS[2]))
        self.base_fig.append_trace(traceqcI, 1, 1)
        trace1d = go.Scatter(
            x=[self.qcI, ],
            y=[self.penetration + 0.5 * (self.qcIIz - self.penetration), ],
            showlegend=True, mode='markers', name=qcI_selected_trace_title,
            marker=dict(size=5, color=DEFAULT_PLOTLY_COLORS[2], line=dict(width=1, color='black')))
        self.base_fig.append_trace(trace1d, 1, 1)
        traceqcIII = go.Scatter(
            x=self.qcIII_data["qc III value [MPa]"], y=self.qcIII_data["z [m]"],
            showlegend=True, mode='lines', name=qcIII_values_trace_title,
            line=dict(color=DEFAULT_PLOTLY_COLORS[3]))
        self.base_fig.append_trace(traceqcIII, 1, 1)
        trace1d = go.Scatter(
            x=[self.qcIII, ],
            y=[self.penetration - 4 * (self.diameter), ],
            showlegend=True, mode='markers', name=qcIII_selected_trace_title,
            marker=dict(size=5, color=DEFAULT_PLOTLY_COLORS[3], line=dict(width=1, color='black')))
        self.base_fig.append_trace(trace1d, 1, 1)

        self.base_fig['layout']['xaxis1'].update(title=qc_axis_title, side='top', anchor='y', range=x_range)
        if y_range is None:
            self.base_fig['layout']['yaxis1'].update(title=z_axis_title, autorange='reversed')
        else:
            self.base_fig['layout']['yaxis1'].update(title=z_axis_title, range=y_range)
        self.base_fig['layout'].update(height=plot_height, width=plot_width,
                             title=plot_title,
                             margin=plot_margin,
                             hovermode='closest')
        if show_fig:
            self.base_fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

