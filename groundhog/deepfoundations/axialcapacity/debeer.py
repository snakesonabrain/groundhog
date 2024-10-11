#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
from copy import deepcopy

# 3rd party packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
try:
    from plotly import tools, subplots
    import plotly.graph_objs as go
    from plotly.colors import DEFAULT_PLOTLY_COLORS
except:
    warnings.warn('Plotly could not be imported. Install plotly using conda (conda install plotly) to enable interactive plotting of results')

# Project imports
from groundhog.general.plotting import GROUNDHOG_PLOTTING_CONFIG

# Map Robertson soil classes to De Beer soil types
DEBEER_SOILTYPES_MAPPING = {
    3: "Clay",
    4: "Sandy clay / loam (silt)",
    5: "Clayey sand / loam (silt)",
    6: "Sand",
    7: "Sand"}


class DeBeerCalculation(object):

    def __init__(self, depth, qc, diameter_pile, diameter_cone=0.0357):
        """
        Initializes a pile base resistance calculation according to De Beer's method. Lists or Numpy arrays of depth
        and cone resistance need to be supplied to the routine as well as the diameter of the pile and the cone.
        The depth spacing does not need to be equal to the spacing required by De Beer's method. The `resample_data`
        method is used after calculation initialisation to remap the data onto a 0.2m grid.

        :param depth: List or Numpy array with depths [m]
        :param qc: List or Numpy array with cone resistance values [MPa] - same list length as depth array
        :param diameter_pile: Diameter of the pile [m]
        :param diameter_cone: Diameter of the cone, the cone diameter is 0.0357 for a standard cone [m]
        """

        # Validation

        if depth.__len__() != qc.__len__():
            raise ValueError("depth and qc arrays need to have the same length!")
        if diameter_pile < 0.2:
            raise ValueError("The minimum pile diameter for applying De Beer's method is 0.2m")
        self.depth_raw = np.array(depth)
        self.qc_raw = np.array(qc)
        self.diameter_pile = diameter_pile
        self.diameter_cone = diameter_cone

    def resample_data(self, spacing=0.2):
        """
        Resampling of the data to the required spacing (default=0.2m for mechanical cone) can be performed for e.g.
        piezocone data.

        :param spacing: Spacing used for calculation of De Beer's method
        :return:
        """
        self.h_crit = spacing
        self.depth = np.arange(np.array(self.depth_raw).min(), np.array(self.depth_raw).max(), spacing)
        self.qc_mech = np.interp(self.depth, self.depth_raw, self.qc_raw)

    def set_soil_layers(self, soilprofile, soiltypecolumn="Soil type",
                        tertiaryclaycolumn="Tertiary clay",
                        totalunitweightcolumn="Total unit weight [kN/m3]",
                        water_level=0, total_unit_weight_dry=15.696,
                        total_unit_weight_wet=19.62, **kwargs):
        """
        Sets the soil type for the pile resistance calculation.
        For the shaft resistance calculation, a column 'Soil type' is required.

        The calculation of overburden pressure is included in this routine.
        According to the Belgian practice, a total unit weight of 15.696:math:`kN/m^3` is used for dry soil and
        a unit weight of 19.62:math:`kN/m^3` for wet soil. An array with total unit weights for each soil layer
        can be specified (`total_unit_weight`) to override these defaults. If the water level is not at a layer
        interface, this array need to contain an additional entry to account for the difference between the dry and wet
        unit weight in the layer containing the water table.

        The unit weight used in De Beer's base
        resistance calculation is the total unit weight above the water table
        and the effective unit weight below.

        The soil types need to be specified in accordance with Table 4 of the paper
        Design of deepfoundations - Belgian practice (Huybrechts et al, 2016).

        Finally, it is possible to specify whether each layer is tertiary clay or not using an array of Booleans.
        Occurence of stiff tertiary clays can be taken into account by specifying the array `tertiary_clay`.

        :param soilprofile: SoilProfile object containing the layer definitions. The SoilProfile should con
        :param soiltypecolumn: Name of the column containing the soil type, select from "Clay", "Loam (silt)", "Sandy clay / loam (silt)", "Clayey sand / loam (silt)" and "Sand"
        :param tertiaryclaycolumn: Name of the column containing the booleans determining whether the layer is Tertiary clay or not
        :param water_level: Water level used for the effective stress calculation [m], default = 0 for water level at surface
        :param water_unit_weight: Unit weight of water used for the calculations (default=10:math:`kN/m^3`)
        :param total_unit_weight_dry: Dry unit weight used for all soils (default=15.696:math:`kN/m^3`)
        :param total_unit_weight_wet: Wet unit weight used for all soils (default=19.62:math:`kN/m^3`)
        :param total_unit_weight: Array with total unit weight for each soil layer. Specifying this array will override the defaults (default=`None`)
        """

        self.layering = soilprofile

        # Validate the presence of the column with the Soil Type
        if soiltypecolumn not in soilprofile.string_soil_parameters():
            raise ValueError("SoilProfile does not contain a column with the soil type")

        for i, layer in soilprofile.iterrows():
            if layer[soiltypecolumn] not in ["Clay", "Loam (silt)", "Sandy clay / loam (silt)",
                                  "Clayey sand / loam (silt)", "Sand"]:
                raise ValueError("Soil type %s not recognised. Needs to be one of 'Loam (silt)', "
                                 "'Sandy clay / loam (silt)', 'Clayey sand / loam (silt)' or 'Sand'")

        # Validate the extent of the soil profile
        if soilprofile.min_depth != 0:
            raise ValueError("Layering should start from zero depth")
        if soilprofile.max_depth < self.depth.max():
            raise ValueError("Layering should be defined to the bottom of the CPT")

        # Validation of specified water level
        if water_level < 0:
            raise ValueError("Specified water level should be greater than or equal to zero.")

        # Set whether soil layers are tertiary clay or not
        if tertiaryclaycolumn not in soilprofile.string_soil_parameters():
            self.layering.loc[:, "Tertiary clay"] = False
        else:
            pass

        # Add a layer interface for the water level
        if water_level not in self.layering.layer_transitions():
            self.layering.insert_layer_transition(depth=water_level)

        # Check if a total unit weight has been specified. If not, use the defaults from Belgian practice
        if totalunitweightcolumn not in soilprofile.numerical_soil_parameters():
            for i, layer in self.layering.iterrows():
                if 0.5 * (layer[self.layering.depth_from_col] + layer[self.layering.depth_to_col]) < water_level:
                    self.layering.loc[i, totalunitweightcolumn] = total_unit_weight_dry
                else:
                    self.layering.loc[i, totalunitweightcolumn] = total_unit_weight_wet
        else:
            pass

        # Calculate stresses in the layers
        self.layering.calculate_overburden(
            waterlevel=water_level,
            totalunitweightcolumn=totalunitweightcolumn
        )

        self.calculation_data = self.layering.map_soilprofile(nodalcoords=self.depth)
        self.calculation_data['qc [MPa]'] = self.qc_mech

    @staticmethod
    def phi_func():
        """
        This function implements the Equations 22 and 23 of the original paper by De Beer for the calculation of
        the friction angle from the cone tip resistance and the vertical effective stress level
        :return: An interpolation function providing the friction angle as a function of the ratio of cone tip
        resistance to vertical effective stress.

        .. math::
            V_{b,d} = \\frac{q_c}{p_o}

            V_{b,d} = 1.3 \\cdot \\exp \\left( 2 \\cot \\pi \\cdot \\tan \\varphi \\right) \\tan ^2 \\left( \\frac{\\pi}{4} + \\frac{\\varphi}{2} \\right)
        """
        phi = np.linspace(np.deg2rad(0.01), np.deg2rad(50), 250)

        def v_bd_func(frictionangle):
            return 1.3 * np.exp(2 * np.pi * np.tan(frictionangle)) * ((np.tan(0.25 * np.pi + 0.5 * frictionangle)) ** 2)

        v_bd = v_bd_func(phi)

        return interp1d(v_bd, phi)

    @staticmethod
    def optimisation_func(beta, hd, frictionangle):
        """
        This function is used to find the angle beta for the failure mechanism for CPT and pile.
        The function implements Equation 60 of the original paper by De Beer.
        To obtain the value of beta, the result of evaluating Equation 60 is compared to the actual h/d or h/D
        and the root is found.

        .. math::
            \\left( \\frac{h}{d} \\right)_3 = \\frac{\\tan \\left( \\frac{\\pi}{4} + \\frac{\\varphi}{2} \\right) \\cdot \\exp \\left( \\frac{\\pi}{2} \\cdot \\tan \\varphi \\right) \\cdot \\sin \\left( \\beta \\cdot \\exp(\\beta \\cdot \\tan \\varphi) \\right)}{1 + \\delta \\cdot \\sin(2 \\cdot \\varphi)}

        :param beta: Value of beta to be found
        :param hd: Value of h/D for the pile or h/d for the CPT
        :param frictionangle: Friction angle derived from Equations 22 and 23 of De Beer's paper
        :return: Difference of the calculated and specified value of h/D or h/d
        """
        return ((np.tan(0.25 * np.pi + 0.5 * frictionangle) *
                np.exp(0.5 * np.pi * np.tan(frictionangle)) *
                np.sin(beta) *
                np.exp(beta * np.tan(frictionangle))) / \
               (1 + np.sin(2 * frictionangle))) - hd

    @staticmethod
    def stress_correction(qc, po, diameter_pile, diameter_cone, gamma, hcrit=0.2):
        """
        Calculates the stress correction in De Beer's method

        .. math::
            h^{\\prime}_{crit} = h_{crit} \\cdot D / d

            q_{r,crit} = \\frac{1 + \\frac{\\gamma \\cdot h_{crit}^{\\prime}}{2 \\cdot p_o}}{1 + \\frac{\\gamma \\cdot h_{crit}}{2 \\cdot p_o}} \\cdot q_{c,crit}

        :param qc: Cone tip resistance [MPa]
        :param po: Overburden pressure at the depth of the start of the increase [kPa]
        :param diameter_pile: Diameter of the pile [m]
        :param diameter_cone: Diameter of the cone rod [m]
        :param gamma: Unit weight of the soil (total above the water table, effective below) [kN/m3]
        :param hcrit: Distance to cone tip used in De Beer's method (default=0.2) [m]

        :returns Ultimate bearing resistance for the cone [MPa]
        """
        h_prime_crit = hcrit * (diameter_pile / diameter_cone)
        try:
            return qc * ((1.0 + ((gamma * h_prime_crit) / (2.0 * po))) / (1.0 + ((gamma * hcrit) / (2.0 * po))))
        except:
            return qc

    def calculate_base_resistance(self, vanimpecorrection=False, hcrit=0.2):
        """
        Calculates the base resistance for any pile diameter.

        If the pile diameter is not a multiple of 0.2m, the unit base resistance for diameters which are a
        multiple of 0.2m above and below the given diameter are calculated.
        Linear interpolation is then used to obtain the end result

        :param vanimpecorrection: Boolean determining whether the upward correction according to De Beer's paper (default) or Van Impe (multiplier of 2) needs to be applied.
        :param hcrit: :math:`h_{crit}` adopted for De Beer's calculation (based on the mechanical cone). Default=0.2m
        :return: Sets two dataframes `calc_1` and `calc_2`, one for the multiple of 0.2m lower than the given diameter and one for the multiple of 0.2m higher than the given diameter. Finally, the attribute `qb` is set through linear interpolation.
        """

        if self.diameter_pile % 0.2 == 0:
            # Calculation for pile diameter being a multiple of 0.2m
            self.diameter_1 = self.diameter_pile
            self.diameter_2 = self.diameter_pile
        else:
            self.diameter_1 = np.round(self.diameter_pile - (self.diameter_pile % 0.2), 1)
            self.diameter_2 = np.round(self.diameter_pile + (self.diameter_pile % 0.2), 1)
        self.calc_1 = self.calculate_base_resistance_standard_diameter(
            pile_diameter=self.diameter_1, vanimpecorrection=vanimpecorrection, hcrit=hcrit
        )
        self.calc_2 = self.calculate_base_resistance_standard_diameter(
            pile_diameter=self.diameter_2, vanimpecorrection=vanimpecorrection, hcrit=hcrit
        )
        _qb = []
        for i, row in self.calc_1.iterrows():
            _qb.append(
                np.interp(
                    self.diameter_pile,
                    [self.diameter_1, self.diameter_2],
                    [self.calc_1.loc[i, "qb [MPa]"], self.calc_2.loc[i, "qb [MPa]"]]
                )
            )
        self.qb = np.array(_qb)
        self.depth_qb = self.calc_1['z [m]']

    def calculate_base_resistance_standard_diameter(self, pile_diameter, vanimpecorrection=False, hcrit=0.2):
        """
        Calculates the base resistance according to De Beer's method for a pile diameter which is a multiple of 0.2m. The calculation happens in five steps:

            - Step 1: Correct the cone resistance for the different failure surface for a pile and a CPT using Equation 62 from De Beer's paper. This correction is especially necessary for the shallow layer where the angle beta is lower than 90°
            - Step 2: Apply a correction for the different stress level for a pile compared to a CPT
            - Step 3: Account for the transition from weaker the stronger layers by working downward along the CPT trace. The increase of resistance will be slower for a pile compared to a CPT
            - Step 4: Account for the transition from stronger to weaker layers by working through the CPT trace from the bottom up. A weaker layer will be felt sooner by the model pile than by the CPT
            - Step 5: Take the average unit base resistance for one diameter below the given level. The average value should note be greater than :math:`q_{p,q+1}` at the given level.

        .. math::
            \\text{Step 1}

            q_{p,(1)} = \\frac{q_c}{\\exp \\left( 2 \\cdot ( \\beta_c - \\beta_p ) \\cdot \\tan \\varphi \\right) }

            \\text{Step 2}

            A = \\frac{1 + \\frac{\\gamma \\cdot h_{crit}^{\\prime}}{2 \\cdot p_o}}{1 + \\frac{\\gamma \\cdot h_{crit}}{2 \\cdot p_o}}

            \\text{if } A \\cdot q_{p,(1)} > q_c \\implies A \\cdot q_{p,(1)} = q_c

            \\text{Step 3}

            q_{p,j+1} = q_{p,j} + \\left[ A \\cdot q_{p,(1),j+1} - q_{p,j} \\right] \\cdot \\frac{d}{D}

            \\text{if } q_{p,j+1} > q_{p,(1),j+1} \\implies q_{p,j+1} = q_{p,(1),j+1}

            \\text{Step 4}

            q_{p,q+1} = q_{p,q} + \\left[ (q_{p,j+1})_{q+1} - q_{p,q} \\right] \\cdot \\frac{d}{D}

            \\text{According to Van Impe:}

            q_{p,q+1} = q_{p,q} + 2 \\cdot \\left[ (q_{p,j+1})_{q+1} - q_{p,q} \\right] \\cdot \\frac{d}{D}

            \\text{if } q_{p,q+1} > (q_{p,j+1})_{q+1} \\implies q_{p,q+1} = (q_{p,j+1})_{q+1}


        For numerical stability, rows with zero cone resistance at the top of the cone resistance trace are discarded.

        :param pile_diameter: Diameter of the pile as a multiple of 0.2m
        :param vanimpecorrection: Boolean determining whether the upward correction according to De Beer's original paper (default) or Van Impe (multiplier of 2) needs to be taken into account.
        :param hcrit: :math:`h_{crit}` adopted for De Beer's calculation (based on the mechanical cone). Default=0.2m

        :return: Returns a dataframe `calc` with the different correction stages
        """
        calc = deepcopy(self.calculation_data)
        for i, row in calc.iterrows():
            if row['qc [MPa]'] < 0:
                calc.loc[i, 'qc [MPa]'] = 0

        # ----------------------------------------------------
        # Step 1: Shallow depth failure surface correction
        # ----------------------------------------------------

        # Calculate phi according to Equation 23
        for i, row in calc.iterrows():
            try:
                calc.loc[i, 'phi [deg]'] = np.rad2deg(
                    self.phi_func()(1000 * row['qc [MPa]'] / row['Effective vertical stress [kPa]']))
            except:
                calc.loc[i, 'phi [deg]'] = 45
        # Determine the values of the normalised depths h/d and h/D
        calc['h/d [-]'] = calc['z [m]'] / self.diameter_cone
        calc['h/D [-]'] = calc['z [m]'] / pile_diameter
        # Find values of beta for cone penetration test and pile according to Equation 60
        for i, row in calc.iterrows():
            try:
                root = brentq(
                    f=self.optimisation_func,
                    a=0,
                    b=0.5 * np.pi,
                    args=(row['h/d [-]'], np.deg2rad(row['phi [deg]'])))
            except:
                root = 0.5 * np.pi
            calc.loc[i, 'beta_c [rad]'] = root
            try:
                root = brentq(
                    f=self.optimisation_func,
                    a=0,
                    b=0.5 * np.pi,
                    args=(row['h/D [-]'], np.deg2rad(row['phi [deg]'])))
            except:
                root = 0.5 * np.pi
            calc.loc[i, 'beta_p [rad]'] = root
        # Apply Equation 62 to obtain qp
        calc['qp [MPa]'] = calc['qc [MPa]'] / \
            (np.exp(
              2 *
              (calc['beta_c [rad]'] - calc['beta_p [rad]']) *
              np.tan(np.deg2rad(calc['phi [deg]']))))

        # ----------------------------------------------------
        # Step 2: Stress level correction
        # ----------------------------------------------------

        calc['A qp [MPa]'] = list(map(lambda _qp, _po, _gamma, _qc: min(_qc, self.stress_correction(
            qc=_qp, po=_po, diameter_pile=pile_diameter, diameter_cone=self.diameter_cone, gamma=_gamma, hcrit=hcrit)),
                           calc['qp [MPa]'], calc['Vertical effective stress [kPa]'], calc['Effective unit weight [kN/m3]'], calc['qc [MPa]']))

        # --------------------------------------------------------------------
        # Step 3: Corrections for transition from weaker to stronger layers
        # --------------------------------------------------------------------
        for i, row in calc.iterrows():
            if i > 0:
                calc.loc[i, 'qp,j+1 [MPa]'] = min(row['A qp [MPa]'] ,
                    calc.loc[i-1, 'qp,j+1 [MPa]'] + \
                    (row['A qp [MPa]'] - calc.loc[i-1, 'qp,j+1 [MPa]']) * \
                    (self.diameter_cone / pile_diameter))
            else:
                calc.loc[i, 'qp,j+1 [MPa]'] = 0

        # --------------------------------------------------------------------
        # Step 4: Corrections for transition from stronger to weaker layers
        # --------------------------------------------------------------------
        if vanimpecorrection:
            coefficient = 2.0
        else:
            coefficient = 1.0

        qu = np.zeros(len(calc['z [m]']))
        # Assign the last value of qd as the starting value of qu
        qu[-1] = calc['qp,j+1 [MPa]'].iloc[-1]
        for i, _qd in enumerate(calc['qp,j+1 [MPa]']):
            if i != 0:
                qu[-1 - i] = min(
                    qu[-i] +
                    coefficient * (calc['qp,j+1 [MPa]'].iloc[-1 - i] - qu[-i]) *
                    (self.diameter_cone / pile_diameter),
                    calc['qp,j+1 [MPa]'].iloc[-1 - i]
                )
        calc['qp,q+1 [MPa]'] = qu

        # --------------------------------------------------------------------
        # Step 5: Averaging to 1D below the reference level
        # --------------------------------------------------------------------
        for i, row in calc.iterrows():
            try:
                _window_data = calc[
                    (calc['z [m]'] >= row['z [m]']) &
                    (calc['z [m]'] <= (row['z [m]'] + pile_diameter))]
                calc.loc[i, "qb [MPa]"] = min(
                    row['qp,q+1 [MPa]'],
                    _window_data['qp,q+1 [MPa]'].mean())
            except:
                calc.loc[i, "qb [MPa]"] = row['qp,q+1 [MPa]']

        return calc

    def plot_base_resistance(self, selected_depth=None, show_standard_diameters=False,
                             plot_title=None, plot_height=800, plot_width=600,
                             plot_margin=dict(t=100, l=50, b=50), show_fig=True,
                             x_range=None, x_tick=None, y_range=None, y_tick=None,
                             legend_orientation='h', legend_x=0.05, legend_y=-0.05, latex_titles=True):
        """
        Plots the base resistance construction according to De Beer

        :param selected_depth: Depth at which the base resistance is requested (default=None for no such depth)
        :param show_standard_diameters: Boolean determining whether the traces of De Beer base unit base resistance computed for diameters which are multiples of 0.2m need to be shown or not (default=False)
        :param plot_title: Title of the plot (default=None)
        :param plot_height: Height for the plot (default=800px)
        :param plot_width: Width of the plot (default=600px)
        :param plot_margin: Margin for the plot. (default=``dict(t=100, l=50, b=50)``
        :param show_fig: Boolean determining whether this figure is shown or not (default=``True``)
        :param x_range: Plotting range for the cone resistance (default=None for Plotly defaults)
        :param x_tick: Tick interval for the cone resistance (default=None for Plotly defaults)
        :param y_range: Plotting range for the depth (default=None for Plotly defaults)
        :param y_tick: Tick interval for the depth (default=None for Plotly defaults)
        :param legend_orientation: Orientation of the legend (default=``'h'`` for horizontal)
        :param legend_x: x Position of the legend (default=0.05 for 5% from plot left edge)
        :param legend_y: y Position of the legend (default=-0.05 for 5% below plot bottom)
        :param latex_titles: Boolean determining whether trace names and axis titles are shown as LaTeX (default = True)
        :return: Creates the Plotly figure ``base_plot`` as an attribute of the object
        """
        if latex_titles:
            qc_trace_title = r'$ q_c $'
            qb_trace_title = r'$ q_{b} $'
            qb_diameter1_trace_title = r'$ q_{b,%.1fm} $' % self.diameter_1
            qb_diameter2_trace_title = r'$ q_{b,%.1fm} $' % self.diameter_2
            qc_axis_title = r'$ q_c, \ q_b \ \text{[MPa]} $'
            z_axis_title = r'$ z \ \text{[m]} $'
        else:
            qc_trace_title = 'qc'
            qb_trace_title = 'qb'
            qb_diameter1_trace_title = 'qb,%.1fm' % self.diameter_1
            qb_diameter2_trace_title = 'qb,%.1fm' % self.diameter_2
            qc_axis_title = 'qc, qb [MPa]'
            z_axis_title = 'z [m]'

        self.base_plot = subplots.make_subplots(rows=1, cols=1, print_grid=False)
        traceqc = go.Scatter(x=self.qc_raw, y=self.depth_raw,
                             showlegend=True, mode='lines', name=qc_trace_title)
        self.base_plot.append_trace(traceqc, 1, 1)
        traceqb = go.Scatter(x=self.qb, y=self.depth,
                             showlegend=True, mode='lines', name=qb_trace_title)
        self.base_plot.append_trace(traceqb, 1, 1)
        if show_standard_diameters:
            trace_diameter_1 = go.Scatter(x=self.calc_1['qb [MPa]'], y=self.depth,
                                 showlegend=True, mode='lines', name=qb_diameter1_trace_title)
            self.base_plot.append_trace(trace_diameter_1, 1, 1)
            trace_diameter_2 = go.Scatter(x=self.calc_2['qb [MPa]'], y=self.depth,
                                          showlegend=True, mode='lines', name=qb_diameter2_trace_title)
            self.base_plot.append_trace(trace_diameter_2, 1, 1)
        if selected_depth is not None:
            qb_selected = np.interp(selected_depth, self.depth_qb, self.qb)
            traceqb_selected = go.Scatter(
                x=[qb_selected, ], y=[selected_depth, ], showlegend=False, mode='markers',
                marker=dict(size=10, color='red', line=dict(width=2, color='black')))
            self.base_plot.append_trace(traceqb_selected, 1, 1)

        self.base_plot['layout']['xaxis1'].update(
            title=qc_axis_title, side='top', anchor='y',
            range=x_range, dtick=x_tick)
        if y_range is None:
            self.base_plot['layout']['yaxis1'].update(title=z_axis_title, autorange='reversed', dtick=y_tick)
        else:
            self.base_plot['layout']['yaxis1'].update(
                title=z_axis_title, range=y_range, dtick=y_tick)
        self.base_plot['layout'].update(height=plot_height, width=plot_width,
                                        title=plot_title,
                                        margin=plot_margin,
                                        legend=dict(orientation=legend_orientation, x=legend_x, y=legend_y))
        if show_fig:
            self.base_plot.show(config=GROUNDHOG_PLOTTING_CONFIG)


    def correct_shaft_qc(self, cone_type="E"):
        """
        Corrects the cone resistance for the effect of the cone type according to Belgian practice.
        A correction factor is applied for the mechanical cones M1, M2 and M4

        .. figure:: images/conetype_correction_be.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Correction factors to be used according to Belgian practice

        :param cone_type: Cone type. Select from 'M1', 'M2', 'M4', 'E', 'U'
        """
        if cone_type not in ['M1', 'M2', 'M4', 'E', 'U']:
            raise ValueError("Cone type not recognised. Select from 'M1', 'M2', 'M4', 'E', 'U'")

        for i, row in self.layering.iterrows():

            if (cone_type == "M1" or cone_type == "M2") and row['Tertiary clay']:
                self.qc_corrected = self.qc_raw / 1.3
            elif (cone_type == "M4") and row['Tertiary clay']:
                self.qc_corrected = self.qc_raw / 1.15
            else:
                self.qc_corrected = self.qc_raw

    def calculate_average_qc(self, qc_avg_override=None):
        """
        Calculates the average cone resistance in each layer for shaft resistance calculation.
        If `qc_avg_override` is used, the calculation is discarded and the specified value is set

        :param qc_avg_override: List with average qc values in MPa in the layers with soil types specified in `set_soil_type`
        """
        if qc_avg_override is not None:
            self.layering["qc avg [MPa]"] = qc_avg_override
        else:
            for i, row in self.layering.iterrows():
                self.layering.loc[i, "qc avg [MPa]"] = np.nan_to_num(self.qc_corrected[
                    (self.depth_raw >= row["Depth from [m]"]) &
                    (self.depth_raw <= row["Depth to [m]"])]).mean()

    def calculate_unit_shaft_friction(self):
        """
        Calculates the unit shaft friction according to the Belgian practice.

        Note the importance of using correct units. The cone resistance is provided in MPa whereas the unit shaft
        friction is calculated in kPa

        .. math::
            q_s = 1000 \\cdot \\eta_p^* \\cdot q_{c,avg}

        .. figure:: images/unit_shaft_friction_be.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Unit shaft friction according to Belgian practice
        """

        for i, row in self.layering.iterrows():
            if row['Soil type'] == 'Clay':
                if row['qc avg [MPa]'] <= 4.5:
                    self.layering.loc[i, "qs [kPa]"] = 1000 * (1 / 30) * row['qc avg [MPa]']
                else:
                    self.layering.loc[i, "qs [kPa]"] = 150
            elif row['Soil type'] == 'Loam (silt)':
                if row['qc avg [MPa]'] <= 6:
                    self.layering.loc[i, "qs [kPa]"] = 1000 * (1 / 60) * row['qc avg [MPa]']
                else:
                    self.layering.loc[i, "qs [kPa]"] = 100
            elif (row['Soil type'] == 'Sandy clay / loam (silt)') or (row['Soil type'] == 'Clayey sand / loam (silt)'):
                if row['qc avg [MPa]'] <= 10:
                    self.layering.loc[i, "qs [kPa]"] = 1000 * (1 / 80) * row['qc avg [MPa]']
                else:
                    self.layering.loc[i, "qs [kPa]"] = 125
            elif row['Soil type'] == 'Sand':
                if row['qc avg [MPa]'] <= 10:
                    self.layering.loc[i, "qs [kPa]"] = 1000 * (1 / 90) * row['qc avg [MPa]']
                elif 10 < row['qc avg [MPa]'] <= 20:
                    self.layering.loc[i, "qs [kPa]"] = 110 + 4 * (row['qc avg [MPa]'] - 10)
                else:
                    self.layering.loc[i, "qs [kPa]"] = 150
            else:
                warnings.warn("Unrecognized soil type (%s) for layer %i" % (row['Soil type'], i+1))
                self.layering.loc[i, "qs [kPa]"] = np.nan

    def plot_unit_shaft_friction(self, plot_title=None, plot_height=800, plot_width=600,
                                 plot_margin=dict(t=100, l=50, b=50), show_fig=True,
                                 x_ranges=(None, (0, 160)), x_ticks=(None, None), y_range=None, y_tick=None,
                                 legend_orientation='h', legend_x=0.05, legend_y=-0.05, latex_titles=True):
        """
        Plots the qc averaging and the unit shaft friction following from this

        :param plot_title: Title of the plot (default=None)
        :param plot_height: Height for the plot (default=800px)
        :param plot_width: Width of the plot (default=600px)
        :param plot_margin: Margin for the plot. (default=``dict(t=100, l=50, b=50)``
        :param show_fig: Boolean determining whether this figure is shown or not (default=``True``)
        :param x_ranges: Plotting range for the cone resistance (default=None for Plotly defaults)
        :param x_ticks: Tick interval for the cone resistance (default=None for Plotly defaults)
        :param y_range: Plotting range for the depth (default=None for Plotly defaults)
        :param y_tick: Tick interval for the depth (default=None for Plotly defaults)
        :param legend_orientation: Orientation of the legend (default=``'h'`` for horizontal)
        :param legend_x: x Position of the legend (default=0.05 for 5% from plot left edge)
        :param legend_y: y Position of the legend (default=-0.05 for 5% below plot bottom)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Creates the Plotly figure ``unit_shaft_plot`` as an attribute of the object
        """
        if latex_titles:
            qc_trace_title = r'$ q_c $'
            qc_avg_trace_title = r'$ q_{c,avg} $'
            qs_trace_title = r'$ q_{s} $'
            qc_axis_title = r'$ q_c \ \text{[MPa]} $'
            qs_axis_title = r'$ q_s \ \text{[kPa]} $'
            z_axis_title = r'$ z \ \text{[m]} $'
        else:
            qc_trace_title = 'qc'
            qc_avg_trace_title = 'qc,avg'
            qs_trace_title = 'qs'
            qc_axis_title = 'qc [MPa]'
            qs_axis_title = 'qs [MPa]'
            z_axis_title = 'z [m]'

        self.unit_shaft_plot = subplots.make_subplots(rows=1, cols=2, print_grid=False, shared_yaxes=True)
        trace_qc = go.Scatter(x=self.qc_raw, y=self.depth_raw,
                             showlegend=True, mode='lines', name=qc_trace_title)
        self.unit_shaft_plot.append_trace(trace_qc, 1, 1)
        trace_qc_avg = go.Scatter(
            x=np.insert(np.array(self.layering['qc avg [MPa]']),
                        np.arange(len(self.layering['qc avg [MPa]'])),
                        self.layering['qc avg [MPa]']),
            y=np.insert(np.array(self.layering['Depth to [m]']),
                        np.arange(len(self.layering['Depth from [m]'])),
                        self.layering['Depth from [m]']),
            showlegend=True, mode='lines', name=qc_avg_trace_title)
        self.unit_shaft_plot.append_trace(trace_qc_avg, 1, 1)
        trace_qs = go.Scatter(
            x=np.insert(np.array(self.layering['qs [kPa]']),
                        np.arange(len(self.layering['qs [kPa]'])),
                        self.layering['qs [kPa]']),
            y=np.insert(np.array(self.layering['Depth to [m]']),
                        np.arange(len(self.layering['Depth from [m]'])),
                        self.layering['Depth from [m]']),
            showlegend=True, mode='lines', name=qs_trace_title)
        self.unit_shaft_plot.append_trace(trace_qs, 1, 2)
        self.unit_shaft_plot['layout']['xaxis1'].update(
            title=qc_axis_title, side='top', anchor='y',
            range=x_ranges[0], dtick=x_ticks[0])
        self.unit_shaft_plot['layout']['xaxis2'].update(
            title=qs_axis_title, side='top', anchor='y',
            range=x_ranges[1], dtick=x_ticks[1])
        if y_range is None:
            self.unit_shaft_plot['layout']['yaxis1'].update(
                title=z_axis_title, autorange='reversed', dtick=y_tick)
        else:
            self.unit_shaft_plot['layout']['yaxis1'].update(
                title=z_axis_title, range=y_range, dtick=y_tick)
        self.unit_shaft_plot['layout'].update(height=plot_height, width=plot_width,
                                        title=plot_title,
                                        margin=plot_margin,
                                        legend=dict(orientation=legend_orientation, x=legend_x, y=legend_y))
        if show_fig:
            self.unit_shaft_plot.show(config=GROUNDHOG_PLOTTING_CONFIG)


    def set_shaft_base_factors(self, alpha_b_tertiary_clay, alpha_b_other,
                               alpha_s_tertiary_clay, alpha_s_other):
        """
        Sets the shaft and base factors according to Belgian practice.
        The factors are not determined automatically but need to be specified by the user based on the
        Belgian practice. The factors are obtained by fitting the results from static pile load tests.
        Factors for tertiary OC clay and other soil types need to be defined.

        :param alpha_b_tertiary_clay: Base factor for overconsolidated tertiary clay
        :param alpha_b_other: Base factor for other soil types
        :param alpha_s_tertiary_clay: Shaft factor for overconsolidated tertiary clay
        :param alpha_s_other: Shaft factor for other soil types

        """
        for i, row in self.layering.iterrows():
            if row['Tertiary clay']:
                self.layering.loc[i, "alpha_s"] = alpha_s_tertiary_clay
                self.layering.loc[i, "alpha_b"] = alpha_b_tertiary_clay
            else:
                self.layering.loc[i, "alpha_s"] = alpha_s_other
                self.layering.loc[i, "alpha_b"] = alpha_b_other

    def calculate_pile_resistance(self, pile_penetration, base_area, circumference, beta_base=1, lambda_base=1, epsilon_b=1):
        """
        Calculates the pile capacity for a given penetration. In Eurocode 7 terms, this is the calculated pile resistance.

        The unit base resistance according to De Beer is interpolated from the previous calculation and correction
        factors for material, base shape and enlarged bases are taken into account. Note that the pile base area entered
        by the user determines whether a tubular pile behaves in a coring or plugged manner.
        The shaft resistance is calculated based on the unit shaft resistance calculated previously.


        :param pile_penetration: Pile penetration below the soil surface [m]
        :param base_area: Area of the pile base to be used for the base resistance calculation [m2]
        :param circumference: Circumference of the pile shaft [m]
        :param beta_base: Beta factor taking into account the shape of the pile base (default=1 for circular debeer)
        :param lambda_base: Lambda factor taking into account enlarged pile bases (default=1 for uniform cross-sections)
        :param epsilon_b: Factor taking into account pile bases in stiff tertiary clay (default=1 for pile based not in stiff Tertiary clay)

        .. math::
            R_b = \\alpha_b \\cdot \\epsilon_b \\cdot \\beta \\cdot \\lambda \\cdot A_b \\cdot q_b

            \\epsilon_b =
                  \\begin{cases}
                    \\max \\left( 1 - 0.01 \\cdot \\left( \\frac{D_{b,eq}}{D_{CPT}} - 1 \\right); 0.476 \\right)       & \\quad \\text{in tertiary OC clay}\\\\
                    1  & \\quad \\text{in all other soil types}
                  \\end{cases}

            R_s = \\kappa_s \\cdot \\Sum \\left( \\alpha_{s,i} \\cdot h_i \\cdot q_{s,i} \\right)

            R_c = R_s + R_b

        :returns: Sets the following attributes of the DeBeerCalculation object:

            - 'qb': Unit base resistance (:math:`q_b`)  [:math:`kPa`]
            - 'epsilon_b': Factor epsilon_b used in the base resistance calculation
            - 'Rb': Base resistance (:math:`R_b`)  [:math:`kN`]
            - 'Rs': Shaft resistance (:math:`R_s`)  [:math:`kN`]
            - 'Rc': Calculated pile resistance (:math:`R_c`)  [:math:`kN`]
            - 'capacity_calc': Pandas dataframe with the data used for the calculation, contains the components of shaft resistance

        """
        # Validation
        if pile_penetration > self.depth.max():
            raise ValueError("Pile penetration of %.2fm is greater than maximum CPT depth of %.2fm" % (
                pile_penetration, self.depth.max()
            ))
        # Prepare calculation data
        self.capacity_calc = deepcopy(self.layering[self.layering["Depth from [m]"] < pile_penetration])
        self.capacity_calc["Depth to [m]"].iloc[-1] = pile_penetration
        self.capacity_calc['Layer thickness [m]'] = self.capacity_calc['Depth to [m]'] - self.capacity_calc['Depth from [m]']
        # Base resistance calculation
        self.qb_selected = np.interp(pile_penetration, self.depth_qb, self.qb)
        self.epsilon_b = epsilon_b
        self.Rb = self.capacity_calc['alpha_b'].iloc[-1] * self.epsilon_b * beta_base * \
                          lambda_base * base_area * (1000 * self.qb_selected)

        # Shaft resistance calculation
        self.capacity_calc["Rs,i [kN]"] = circumference * self.capacity_calc['alpha_s'] * \
                                      self.capacity_calc['Layer thickness [m]'] * self.capacity_calc['qs [kPa]']
        self.Rs = self.capacity_calc["Rs,i [kN]"].sum()
        self.Rc = self.Rs + self.Rb
