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
from groundhog.general.plotting import GROUNDHOG_PLOTTING_CONFIG, LogPlot
from groundhog.general.soilprofile import SoilProfile


LCPC_SOILTYPES_DETAIL = [
    "Soft clay and mud",
    "Moderately compact clay", 
    "Silt",
    "Loose sand",
    "Compact to stiff clay",
    "Compact silt",
    "Soft chalk",
    "Moderately compact sand",
    "Moderately compact gravel",
    "Weathered to fragmented chalk",
    "Compact to very compact sand",
    "Compact to very compact gravel"
]

LCPC_SOILTYPES = [
    "Clay",
    "Silt",
    "Sand",
    "Chalk",
    "Gravel",
]

LCPC_SOILPROFILE_TEMPLATE = SoilProfile({
    'Depth from [m]': [0, 3, 6, 15, 20],
    'Depth to [m]': [3, 6, 15, 20, 25],
    "Total unit weight [kN/m3]": [16, 19, 20, 17.5, 21],
    'Soil type': ['Clay', 'Sand', 'Gravel', 'Silt', 'Chalk'],
    "Ignore shaft friction": [True, False, False, False, False]
})

LCPC_FACTORS = {
    'kc': {
        "I": {
            "Soft clay and mud": 0.4,
            "Moderately compact clay": 0.35, 
            "Silt": 0.4,
            "Loose sand": 0.4,
            "Compact to stiff clay": 0.45,
            "Compact silt": 0.45,
            "Soft chalk": 0.2,
            "Moderately compact sand": 0.4,
            "Moderately compact gravel": 0.4,
            "Weathered to fragmented chalk": 0.2,
            "Compact to very compact sand": 0.3,
            "Compact to very compact gravel": 0.3
        },
        "II": {
            "Soft clay and mud": 0.5,
            "Moderately compact clay": 0.45, 
            "Silt": 0.5,
            "Loose sand": 0.5,
            "Compact to stiff clay": 0.55,
            "Compact silt": 0.55,
            "Soft chalk": 0.3,
            "Moderately compact sand": 0.5,
            "Moderately compact gravel": 0.5,
            "Weathered to fragmented chalk": 0.4,
            "Compact to very compact sand": 0.4,
            "Compact to very compact gravel": 0.4
        }
    },
    'alpha': {
        'IA': {
            "Soft clay and mud": 30,
            "Moderately compact clay": 40, 
            "Silt": 60,
            "Loose sand": 60,
            "Compact to stiff clay": 60,
            "Compact silt": 60,
            "Soft chalk": 100,
            "Moderately compact sand": 100,
            "Moderately compact gravel": 100,
            "Weathered to fragmented chalk": 60,
            "Compact to very compact sand": 150,
            "Compact to very compact gravel": 150
        },
        'IB': {
            "Soft clay and mud": 90,
            "Moderately compact clay": 80, 
            "Silt": 150,
            "Loose sand": 150,
            "Compact to stiff clay": 120,
            "Compact silt": 120,
            "Soft chalk": 120,
            "Moderately compact sand": 200,
            "Moderately compact gravel": 200,
            "Weathered to fragmented chalk": 80,
            "Compact to very compact sand": 300,
            "Compact to very compact gravel": 300
        },
        'IIA': {
            "Soft clay and mud": 90,
            "Moderately compact clay": 40, 
            "Silt": 60,
            "Loose sand": 60,
            "Compact to stiff clay": 60,
            "Compact silt": 60,
            "Soft chalk": 100,
            "Moderately compact sand": 100,
            "Moderately compact gravel": 100,
            "Weathered to fragmented chalk": 60,
            "Compact to very compact sand": 150,
            "Compact to very compact gravel": 150
        },
        'IIB': {
            "Soft clay and mud": 30,
            "Moderately compact clay": 80, 
            "Silt": 120,
            "Loose sand": 120,
            "Compact to stiff clay": 120,
            "Compact silt": 120,
            "Soft chalk": 120,
            "Moderately compact sand": 200,
            "Moderately compact gravel": 200,
            "Weathered to fragmented chalk": 80,
            "Compact to very compact sand": 200,
            "Compact to very compact gravel": 200
        }
    },
    'fslim standard': { # kPa
        'IA': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 35, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 35,
            "Compact silt": 35,
            "Soft chalk": 35,
            "Moderately compact sand": 80,
            "Moderately compact gravel": 80,
            "Weathered to fragmented chalk": 120,
            "Compact to very compact sand": 120,
            "Compact to very compact gravel": 120
        },
        'IB': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 35, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 35,
            "Compact silt": 35,
            "Soft chalk": 35,
            "Moderately compact sand": 35,
            "Moderately compact gravel": 35,
            "Weathered to fragmented chalk": 80,
            "Compact to very compact sand": 80,
            "Compact to very compact gravel": 80
        },
        'IIA': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 35, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 35,
            "Compact silt": 35,
            "Soft chalk": 35,
            "Moderately compact sand": 80,
            "Moderately compact gravel": 80,
            "Weathered to fragmented chalk": 120,
            "Compact to very compact sand": 120,
            "Compact to very compact gravel": 120
        },
        'IIB': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 35, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 35,
            "Compact silt": 35,
            "Soft chalk": 35,
            "Moderately compact sand": 80,
            "Moderately compact gravel": 80,
            "Weathered to fragmented chalk": 120,
            "Compact to very compact sand": 120,
            "Compact to very compact gravel": 120
        }
    },
    'fslim careful execution': { # kPa
        'IA': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 80, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 80,
            "Compact silt": 80,
            "Soft chalk": 35,
            "Moderately compact sand": 120,
            "Moderately compact gravel": 120,
            "Weathered to fragmented chalk": 150,
            "Compact to very compact sand": 150,
            "Compact to very compact gravel": 150
        },
        'IB': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 80, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 80,
            "Compact silt": 80,
            "Soft chalk": 35,
            "Moderately compact sand": 80,
            "Moderately compact gravel": 80,
            "Weathered to fragmented chalk": 120,
            "Compact to very compact sand": 120,
            "Compact to very compact gravel": 120
        },
        'IIA': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 80, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 80,
            "Compact silt": 80,
            "Soft chalk": 35,
            "Moderately compact sand": 120,
            "Moderately compact gravel": 120,
            "Weathered to fragmented chalk": 150,
            "Compact to very compact sand": 150,
            "Compact to very compact gravel": 150
        },
        'IIB': {
            "Soft clay and mud": 15,
            "Moderately compact clay": 35, 
            "Silt": 35,
            "Loose sand": 35,
            "Compact to stiff clay": 35,
            "Compact silt": 35,
            "Soft chalk": 35,
            "Moderately compact sand": 80,
            "Moderately compact gravel": 80,
            "Weathered to fragmented chalk": 120,
            "Compact to very compact sand": 120,
            "Compact to very compact gravel": 120
        }
    }
}

class LCPCAxcapCalculation(object):

    def __init__(self, depth, qc, diameter_pile, group_base, group_shaft, diameter_shaft=np.nan):
        """
        Initializes a pile base resistance calculation according to LCPC method Bustamante and Gianeselli (1982). Lists or Numpy arrays of depth
        and cone resistance need to be supplied to the routine as well as the diameter of the pile.
       
        :param depth: List or Numpy array with depths [m]
        :param qc: List or Numpy array with cone resistance values [MPa] - same list length as depth array
        :param diameter_pile: Diameter of the pile [m]
        :param group_base: Pile group used for the base factors (select from ``["I", "II"]``)
        :param group_base: Pile group used for the shaft factors (select from ``["IA", "IB", "IIA", "IIB"]``, factors for micropiles IIIa and IIIb are not encoded)
        :param diameter_shaft: Diameter of the pile shaft, specify this only if it is different from the base diameter [m]
        """

        # Validation

        if depth.__len__() != qc.__len__():
            raise ValueError("depth and qc arrays need to have the same length!")
        if diameter_pile < 0.2:
            raise ValueError("The minimum pile diameter for applying De Beer's method is 0.2m")
        self.depth = np.array(depth)
        self.qc = np.array(qc)
        if np.isnan(diameter_shaft):
            self.diameter_base = diameter_pile
            self.diameter_shaft = diameter_pile
        else:
            self.diameter_base = diameter_pile
            self.diameter_shaft = diameter_shaft
        self.group_base = group_base
        self.group_shaft = group_shaft
        
    @staticmethod
    def soiltype_lcpc(qc, soiltype):
        if soiltype == 'Clay':
            if qc < 1:
                return "Soft clay and mud"
            elif qc <= 1 and qc <5:
                return "Moderately compact clay"
            else:
                return "Compact to stiff clay"
        elif soiltype == 'Silt':
            if qc <= 5:
                return "Silt"
            else:
                return "Compact silt"
        elif soiltype == 'Sand':
            if qc <= 5:
                return "Loose sand"
            elif 5 < qc <= 12:
                return "Moderately compact sand"
            else:
                return "Compact to very compact sand"
        elif soiltype == 'Chalk':
            if qc <= 5:
                return "Soft chalk"
            else:
                return "Weathered to fragmented chalk"
        elif soiltype == 'Gravel':
            if qc <= 5:
                raise ValueError("qc < 5 not defined for gravel")
            elif 5 < qc <= 12:
                return "Moderately compact gravel"
            else:
                return "Compact to very compact gravel"

    def set_soil_layers(self, soilprofile, soiltypecolumn="Soil type", water_level=0, **kwargs):
        """
        Sets the soil type for the pile resistance calculation.
        For the shaft resistance calculation, a column ``"Soil type"`` is required.

        The calculation of overburden pressure is included in this routine.
        
        The soil type should be one of the following:

           - ``'Clay'``
           - ``'Silt'``
           - ``'Sand'``
           - ``'Chalk'``
           - ``'Gravel'``

        The routine checks the cone resistance at a given depth to check for the soil type according to the LCPC method tables.

        :param soilprofile: SoilProfile object containing the layer definitions. The SoilProfile should con
        :param soiltypecolumn: Name of the column containing the soil type
        :param water_level: Water level used for the effective stress calculation [m], default = 0 for water level at surface
        """

        self.layering = soilprofile

        # Validate the presence of the column with the Soil Type
        if soiltypecolumn not in soilprofile.string_soil_parameters():
            raise ValueError("SoilProfile does not contain a column with the soil type")

        for i, layer in soilprofile.iterrows():
            if layer[soiltypecolumn] not in LCPC_SOILTYPES:
                raise ValueError("Soil type %s not recognised. Needs to be one of %s" % (layer[soiltypecolumn], str(LCPC_SOILTYPES)))

        # Validate the extent of the soil profile
        if soilprofile.min_depth != 0:
            raise ValueError("Layering should start from zero depth")
        if soilprofile.max_depth < self.depth.max():
            raise ValueError("Layering should be defined to the bottom of the CPT")

        # Validation of specified water level
        if water_level < 0:
            raise ValueError("Specified water level should be greater than or equal to zero.")

        # Add a layer interface for the water level
        if water_level not in self.layering.layer_transitions():
            self.layering.insert_layer_transition(depth=water_level)
        else:
            pass

        # Calculate stresses in the layers
        self.layering.calculate_overburden(waterlevel=water_level)

        self.calculation_data = self.layering.map_soilprofile(nodalcoords=self.depth)
        self.calculation_data['qc [MPa]'] = self.qc

    def qca_calculation(self):
        """
        Calculates the depth-averaged cone resistance for the LCPC method.

        The average cone resistance in an window 1.5OD above and below the considered is calculated.
        Points which are higher than 1.3 times the average or lower than 0.7 time the average are left out.
        The average is recalculated using only the left-over points

        .. figure:: images/qc_averaging.png
            :figwidth: 350.0
            :width: 300.0
            :align: center

            Cone resistance averaging procedure

        :returns Stored the average cone resistance for the end bearing calculation in the dataframe with calculation data
        """
        for i, row in self.calculation_data.iterrows():
            try:
                _qc_df = deepcopy(self.calculation_data[
                    (self.calculation_data["z [m]"] > row["z [m]"] - 1.5 * self.diameter_base) &
                    (self.calculation_data["z [m]"] < row["z [m]"] + 1.5 * self.diameter_base)])
                _qc_df.reset_index(drop=True, inplace=True)
                _qc_mean = _qc_df["qc [MPa]"].mean()
                self.calculation_data.loc[i, "qca prime [MPa]"] = _qc_mean
                for j, point in _qc_df.iterrows():
                    if (point["qc [MPa]"] < 0.7 * _qc_mean) or (point["qc [MPa]"] > 1.3 * _qc_mean):
                        _qc_df.loc[j, "qc [MPa]"] = np.nan
                self.calculation_data.loc[i, "qca [MPa]"] = _qc_df["qc [MPa]"].mean()
            except:
                self.calculation_data.loc[i, "qca [MPa]"] = np.nan

    def calculate_base_resistance(self):
        """
        Calculates the base resistance.

        .. math::
            q_b = k_c \\cdot q_{ca}

            Q_b = A_b \\cdot q_b

        The factors are taken according to the soil and pile type specified.

        .. figure:: images/base_factors_LCPC.png
            :figwidth: 500.0
            :width: 450.0
            :align: center

            Factors on average cone resistance for base resistance calculation.


        :return: Adds columns ``"qb [MPa]"`` and ``"Qb [kN]"`` to the dataframe with calculation results.
        """
        _area_base = 0.25 * np.pi * (self.diameter_base ** 2)

        for i, row in self.calculation_data.iterrows():
            _soiltype = self.soiltype_lcpc(qc=row["qc [MPa]"], soiltype=row["Soil type"])
            self.calculation_data.loc[i, "qb [MPa]"] = \
                LCPC_FACTORS['kc'][self.group_base][_soiltype] * row["qca [MPa]"]
            self.calculation_data.loc[i, "Qb [kN]"] = \
                1e3 * _area_base * LCPC_FACTORS['kc'][self.group_base][_soiltype] * row["qca [MPa]"]

    def calculate_shaft_resistance(self, careful_execution=False):
        """
        Calculates the shaft resistance. A Boolean ``careful_execution`` can be set to ``True`` to take the factors for careful execution (default=``False``). 

        The user can enter depth ranges where unit skin friction is ignored.

        .. math::
            f_s = \\min( f_{s,lim}, \\frac{q_c}{\\alpha_{\\text{LCPC}}})

            Q_s = \\pi D \\int_{0}^{z} f_s(z) dz

        The factors are taken according to the soil and pile type specified.

        .. figure:: images/shaft_factors_LCPC.png
            :figwidth: 700.0
            :width: 650.0
            :align: center

            Factors on cone resistance for shaft resistance calculation.

        :return: Adds columns ``"fs [kPa]"``, ``"Fs [kN/m]"`` and ``"Qs [kN]"`` to the dataframe with calculation results.
        """
        self.calculation_data["dz [m]"] = self.calculation_data["z [m]"].diff()
        if careful_execution:
            _fslim_key = "fslim careful execution"
        else:
            _fslim_key = "fslim standard"

        _Qs = 0
        for i, row in self.calculation_data.iterrows():
            _soiltype = self.soiltype_lcpc(qc=row["qc [MPa]"], soiltype=row["Soil type"])
            if i > 0:
                if row["Ignore shaft friction"]:
                    _fs = 0
                    _fs_lim = np.nan
                else:
                    _fs_lim = LCPC_FACTORS[_fslim_key][self.group_shaft][_soiltype]
                    _fs = min(
                        1e3 * row["qc [MPa]"] / LCPC_FACTORS['alpha'][self.group_shaft][_soiltype],
                        _fs_lim)
                self.calculation_data.loc[i, "fs [kPa]"] = _fs
                self.calculation_data.loc[i, "fs lim [kPa]"] = _fs_lim
                self.calculation_data.loc[i, "Fs [kN/m]"] = _fs * np.pi * self.diameter_shaft
                _Qs += row["dz [m]"] * _fs * np.pi * self.diameter_shaft
                self.calculation_data.loc[i, "Qs [kN]"] = _Qs

    def get_axialpileresistance(self, pile_penetration):
        """
        Returns a dictionary with shaft resistance, base resistance and total resistance at the selected depth.
        """
        if pile_penetration > self.calculation_data["z [m]"].max():
            raise ValueError("Pile penetration should be less than the maximum CPT depth. Actually, 1.5D of data is required below the tip depth.")
        else:
            _Rs = np.interp(pile_penetration, self.calculation_data["z [m]"], self.calculation_data["Qs [kN]"])
            _Rb = np.interp(pile_penetration, self.calculation_data["z [m]"], self.calculation_data["Qb [kN]"])

            return {
                "Rs [kN]": _Rs,
                "Rb [kN]": _Rb,
                "Rc [kN]": _Rs + _Rb
            }

    def plot_fs_qb(self, return_fig=False, plot_height=600, plot_width=700,
                   fillcolordict={'Sand': 'yellow', 'Clay': 'brown', 'Silt': 'orange', 'Gravel': 'grey', 'Chalk': 'taupe'},
                   fs_lim=(0, 150), fs_tick=20, qb_lim=(0, 30), qb_tick=5):
        """
        Plots the profile of unit skin friction and unit end bearing.
        """
        fs_qb_plot = LogPlot(
            soilprofile=self.layering, no_panels=2, fillcolordict=fillcolordict)
        fs_qb_plot.add_trace(
            x=self.calculation_data["fs [kPa]"],
            z=self.calculation_data["z [m]"],
            name='fs',
            showlegend=False,
            line=dict(color='black'),
            panel_no=1)
        fs_qb_plot.add_trace(
            x=self.calculation_data["qb [MPa]"],
            z=self.calculation_data["z [m]"],
            name='qb',
            showlegend=False,
            line=dict(color='black'),
            panel_no=2)

        fs_qb_plot.set_xaxis(title=r'$ f_s \ \text{[kPa]} $', panel_no=1, range=fs_lim, dtick=fs_tick)
        fs_qb_plot.set_xaxis(title=r'$ q_b \ \text{[MPa]} $', panel_no=2, range=qb_lim, dtick=qb_tick)
        fs_qb_plot.set_zaxis(title=r'$ z \ \text{[m]} $', range=(self.layering.max_depth, 0))
        fs_qb_plot.set_size(height=plot_height, width=plot_width)
        if return_fig:
            return fs_qb_plot
        else:
            fs_qb_plot.show()

    def plot_axcap(self, return_fig=False, plot_height=600, plot_width=900,
                   fillcolordict={'Sand': 'yellow', 'Clay': 'brown', 'Silt': 'orange', 'Gravel': 'grey', 'Chalk': 'taupe'},
                   rs_tick=500, rb_tick=500, rc_tick=500):
        """
        Plots the result of the axial capacity analysis
        """
        axcapplot = LogPlot(
            soilprofile=self.layering,
            no_panels=3,
            fillcolordict=fillcolordict)
        axcapplot.add_trace(
            x=self.calculation_data["Qs [kN]"],
            z=self.calculation_data["z [m]"], name='Shaft', panel_no=1, line=dict(color='black'), showlegend=False)
        axcapplot.add_trace(
            x=self.calculation_data["Qb [kN]"],
            z=self.calculation_data["z [m]"], name='Base', panel_no=2, line=dict(color='black'), showlegend=False)
        axcapplot.add_trace(
            x=self.calculation_data["Qb [kN]"] + self.calculation_data["Qs [kN]"],
            z=self.calculation_data["z [m]"], name='Combined', panel_no=3, line=dict(color='black'), showlegend=False)

        axcapplot.set_xaxis(title='$ R_s $ [kN]', panel_no=1, range=(0, 1.1 * self.calculation_data["Qs [kN]"].max()), dtick=rs_tick)
        axcapplot.set_xaxis(title='$ R_b $ [kN]', panel_no=2, range=(0, 1.1 * self.calculation_data["Qb [kN]"].max()), dtick=rb_tick)
        axcapplot.set_xaxis(
            title='$ R_c $ [kN]', panel_no=3, dtick=rc_tick,
            range=(0, 1.1 * (self.calculation_data["Qb [kN]"].max() + self.calculation_data["Qs [kN]"].max())))
        axcapplot.set_zaxis(title='$ z $ [m]', range=(self.layering.max_depth, 0))
        axcapplot.set_size(height=plot_height, width=plot_width)
        if return_fig:
            return axcapplot
        else:
            axcapplot.show()