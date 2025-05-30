#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings

# 3rd party packages
import pandas as pd
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

# Project imports
from groundhog.general.plotting import plot_with_log, GROUNDHOG_PLOTTING_CONFIG
from groundhog.general.parameter_mapping import SOIL_PARAMETER_MAPPING, merge_two_dicts, reverse_dict
from groundhog.siteinvestigation.insitutests.spt_correlations import *
from groundhog.siteinvestigation.insitutests.pcpt_processing import InsituTestProcessing
from groundhog.general.soilprofile import SoilProfile, plot_fence_diagram
from groundhog.general.parameter_mapping import offsets
from groundhog.general.agsconversion import AGSConverter

DEFAULT_SPT_PROPERTIES = SoilProfile({
    'Depth from [m]': [0, ],
    'Depth to [m]': [20, ],
    'Borehole diameter [mm]': [100, ],
    'Country': ['United States', ],
    'Hammer type': ['Safety', ],
    'Hammer release': ['Rope and pulley',],
    'Sampler type': ['Standard sampler', ],
    'eta H [%]': [np.nan,],
    'eta B [-]': [np.nan,],
    'eta S [-]': [np.nan,],
    'eta R [-]': [np.nan,]
})

class SPTProcessing(InsituTestProcessing):
    """
    The SPTProcessing class implements methods for reading, processing and presentation of Standard Penetration Test (SPT) data.

    `GeoEngineer <https://www.geoengineer.org/education/site-characterization-in-situ-testing-general/standard-penetration-testing-spt/>`_ describes the SPT test as follows:

        Standard Penetration Test (SPT) is a simple and low-cost testing procedure widely used in geotechnical investigation to determine the relative density and angle of shearing resistance of cohesionless soils and also the strength of stiff cohesive soils.

        For this test, a borehole has to be drilled to the desired sampling depth. The split-spoon sampler that is attached to the drill rod is placed at the testing point. A hammer of 63.5 kg (140 lbs) is dropped repeatedly from a height of 76 cm (30 inches) driving the sampler into the ground until reaching a depth of 15 cm (6 inches). The number of the required blows is recorded. This procedure is repeated two more times until a total penetration of 45 cm (18 inches) is achieved. The number of blows required to penetrate the first 15 cm is called “seating drive” and the total number of blows required to penetrate the remaining 30 cm depth is known as the “standard penetration resistance”, or otherwise, the “N-value”. If the N-value exceeds 50 then the test is discontinued and is called a “refusal”. The interpreted results, with several corrections, are used to estimate the geotechnical engineering properties of the soil.

    Check the GeoEngineer website for more useful information and an insightful presentation by Paul W. Mayne on SPT hammer types.

    Data for checking and validation was provided by Ajay Sastri of GeoSyntec and Dennis O'Meara of Foundation Alternatives.

    .. figure:: images/spt_principle.png
        :figwidth: 700.0
        :width: 650.0
        :align: center

        Working principle of the SPT (Mayne, 2016)

    The SPT test is a simple and robust test but it has its drawbacks. For example, measurements are discontinuous, the application of SPT N number in soft clays and silts is not possible, the are energy inefficiency problems, ... The user needs to be aware of these issues at the onset of a SPT processing exercise.
    """
    def __init__(self, title, waterunitweight=10):
        """
        Initialises a SPTProcessing object based on a title. Optionally, a geographical position can be defined. A dictionary for dumping unstructured data (``additionaldata``) is also available.

        An empty dataframe (``.data`) is created for storing the SPT data`

        :param title: Title for the SPT test
        :param waterunitweight: Unit weight of water used for effective stress calculations (default=10.25kN/m3 for seawater)

        """
        super().__init__(title, waterunitweight)
        self.sptprofile = pd.DataFrame()

    # region Utility functions

    def rename_columns(self, z_key=None, N_key=None, fs_key=None):
        if z_key is None:
            z_key = "z [m]"

        if N_key is None:
            N_key = "N [-]"

        self.data.rename(columns={z_key: 'z [m]',
                                  N_key: 'N [-]'}, inplace=True)

    def dropna_rows(self):
        try:
            self.data.dropna(subset=('z [m]', 'N [-]'), how='all', inplace=True)
        except:
            self.data.dropna(how='all', inplace=True)

    # endregion

    # region Data loading

    def load_excel(self, path, z_key=None, N_key=None, z_multiplier=1,**kwargs):
        """
        Loads SPT data from an Excel file. Specific column keys have to be provided for z and SPT N number.
        If column keys are not specified, the following keys are used:

            - 'z [m]' for depth below mudline
            - 'N [-]' for SPT N number

        Note that the SPT N number needs to be the raw field measurement. Further processing can be used for corrections.

        A multiplier can be specified to convert depth in ft to depth in m (`z_multiplier`=0.3048).
        All further SPT processing happens in m so this needs to be specified in case of working with depths in ft or other units.
        Other arguments for the `read_excel` function in Pandas can be specified as `**kwargs`.

        :param path: Path to the Excel file
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param N_key: Column key for SPT N number. Optional, default=None when 'N [-]' is the column key.
        :param z_multiplier: Multiplier applied on depth to convert to meters (e.g. 0.3048 to convert from ft to m)
        :param kwargs: Optional keyword arguments for the read_excel function in Pandas (e.g. sheet_name, header, ...)
        :return: Sets the columns 'z [m]' and 'N [-]' of the `.data` attribute
        """

        try:
            self.data = pd.read_excel(path, **kwargs)
            self.rename_columns(z_key=z_key, N_key=N_key)
            self.data['z [m]'] = self.data['z [m]'] * z_multiplier
        except Exception as err:
            raise ValueError("Error during reading of SPT data. Review the error message and try again. - %s" % (
                str(err)))

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_pandas(self, df, z_key=None, N_key=None, z_multiplier=1,**kwargs):
        """
        Loads SPT data from a Pandas dataframe. Specific column keys have to be provided for z and SPT N number.
        If column keys are not specified, the following keys are used:

            - 'z [m]' for depth below mudline
            - 'N [-]' for SPT N number

        Note that the SPT N number needs to be the raw field measurement. Further processing can be used for corrections.

        A multiplier can be specified to convert depth in ft to depth in m (`z_multiplier`=0.3048).
        All further SPT processing happens in m so this needs to be specified in case of working with depths in ft or other units.
        Other arguments for the `read_excel` function in Pandas can be specified as `**kwargs`.

        :param df: Dataframe containing the SPT data
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param N_key: Column key for SPT N number. Optional, default=None when 'N [-]' is the column key.
        :param z_multiplier: Multiplier applied on depth to convert to meters (e.g. 0.3048 to convert from ft to m)
        :param kwargs: Optional keyword arguments for the read_excel function in Pandas (e.g. sheet_name, header, ...)
        :return: Sets the columns 'z [m]' and 'N [-]' of the `.data` attribute
        """

        try:
            self.data = df
            self.rename_columns(z_key=z_key, N_key=N_key)
            self.data['z [m]'] = self.data['z [m]'] * z_multiplier
        except Exception as err:
            raise ValueError("Error during reading of SPT data. Review the error message and try again. - %s" % (
                str(err)))

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))


    # endregion

    def map_properties(self, layer_profile, spt_profile=DEFAULT_SPT_PROPERTIES,
                       initial_vertical_total_stress=0,
                       vertical_total_stress=None,
                       vertical_effective_stress=None, waterlevel=0,
                       rodlength_abovesoil=1,
                       extend_spt_profile=True, extend_layer_profile=True):
        """
        Maps the soil properties defined in the layering and the cone properties to the grid
        defined by the cone data. The procedure also calculates the total and effective vertical stress.
        Note that pre-calculated arrays with total and effective vertical stress can also be supplied to the routine.
        These needs to have the same length as the array with SPT depth data.

        :param layer_profile: ``SoilProfile`` object with the layer properties (need to contain the soil parameter ``Total unit weight [kN/m3]``
        :param spt_profile: ``SoilProfile`` object with the spt test properties (default=``DEFAULT_SPT_PROPERTIES``)
        :param initial_vertical_total_stress: Initial vertical total stress at the highest point of the soil profile
        :param vertical_total_stress: Pre-calculated total vertical stress at SPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param vertical_effective_stress: Pre-calculated effective vertical stress at SPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param waterlevel: Waterlevel [m] in the soil (measured from soil surface), default = 0m
        :param rodlength_abovesoil: Rod length above soil surface level [m] (used to calculate the rod length to the sampler), default = 1m
        :param extend_spt_profile: Boolean determining whether the cone profile needs to be extended to go to the bottom of the SPT (default = True)
        :param extend_layer_profile: Boolean determining whether the layer profile needs to be extended to the bottom of the SPT (default = True)
        :return: Expands the dataframe `.data` with additional columns for the SPT and soil properties
        """
        super().map_properties(
            layer_profile=layer_profile, initial_vertical_total_stress=initial_vertical_total_stress,
            vertical_total_stress=vertical_total_stress, vertical_effective_stress=vertical_effective_stress,
            waterlevel=waterlevel, extend_layer_profile=extend_layer_profile)

        # Validate that SPT property boundaries fully contain the SPT info
        if extend_spt_profile:
            if spt_profile[spt_profile.depth_to_col].max() < self.data['z [m]'].max():
                warnings.warn("SPT properties extended to bottom of SPT data")
                spt_profile[spt_profile.depth_to_col].iloc[-1] = self.data['z [m]'].max()

        self.sptprofile = spt_profile

        if spt_profile.min_depth > self.data['z [m]'].min():
            raise ValueError(
                "SPT properties starts below minimum SPT depth. " +
                "Ensure that SPT profile fully contains SPT data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))
        if spt_profile.max_depth < self.data['z [m]'].max():
            raise ValueError(
                "SPT profile ends above minimum SPT depth. " +
                "Ensure that cone profile fully contains SPT data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))

        # Map cone properties
        _mapped_spt_props = spt_profile.map_soilprofile(self.data['z [m]'])

        # Join values to the SPT data
        self.data = self.data.join(_mapped_spt_props.set_index('z [m]'), on='z [m]', lsuffix='_left')
        self.data['Rod length [m]'] = self.data['z [m]'] + rodlength_abovesoil

    # region Correlations

    def apply_correlation(self, name, outputs, apply_for_soiltypes='all', **kwargs):
        """
        Applies a correlation to the given SPT data. The name of the correlation needs to be chosen from the following available correlations.
        Each correlation corresponds to a function in the `spt_correlations` module.
        By default, the correlation is applied to the entire depth range. However, a restriction on the soil
        types to which the correlation can be applied can be specified with the `apply_for_soiltypes` keyword argument.
        A list with the soil types for which the correlation needs to be applied can be provided.

            - Overburden correction Liao and Whitman (1986): `overburdencorrection_spt_liaowhitman`,
            - Overburden correction ISO 22476-3: `overburdencorrection_spt_ISO`
            - N60 correction: `spt_N60_correction`
            - Relative density Kulhawy and Mayne (1990): `relativedensity_spt_kulhawymayne`
            - Relative density class Terzaghi and Peck (1967): `relativedensityclass_spt_terzaghipeck`
            - Undrained shear strength class Terzaghi and Peck (1967): `undrainedshearstrengthclass_spt_terzaghipeck`
            - Undrained shear strength Salgado (2008): `undrainedshearstrength_spt_salgado`
            - Friction angle Kulhawy and Mayne (1990): `frictionangle_spt_kulhawymayne`
            - Friction angle PHT (1974): `frictionangle_spt_PHT`
    
        Note that certain correlations require either the application of preceding correlations

        :param name: Name of the correlation according to the list defined above
        :param outputs: a dict of keys and values where keys are the same as the keys in the correlation, values are the table headers you want
        :param apply_for_soiltypes: List with soil types to which the correlation needs the be applied.
        :param kwargs: Optional keyword arguments for the correlation.
        :return: Adds a column with key `outkey` to the dataframe with SPT data
        """

        for resultkey in outputs:
            header = outputs[resultkey]
            if header in self.data.columns:
                self.data.drop(header, axis=1, inplace=True)

        self.data.rename(columns=SOIL_PARAMETER_MAPPING, inplace=True)

        for i, row in self.data.iterrows():
            if apply_for_soiltypes == 'all' or row['Soil type'] in apply_for_soiltypes:
                params = merge_two_dicts(kwargs, dict(row))
                results = CORRELATIONS[name](**params)
                for resultkey in outputs:
                    header = outputs[resultkey]
                    self.data.loc[i, header] = results[resultkey]
            else:
                for resultkey in outputs:
                    header = outputs[resultkey]
                    self.data.loc[i, header] = np.nan
                    
        self.data.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
    # endregion

    def plot_raw_spt(self, n_range=(0, 100), n_tick=10, z_range=None, z_tick=2,
                      plot_height=700, plot_width=700, return_fig=False,
                      plot_title=None, plot_margin=dict(t=100, l=50, b=50), color=None,
                      markersize=5,
                      plot_layers=True):
        """
        Plots the raw SPT data using the Plotly package. This generates an interactive plot.

        :param n_range: Range for the SPT N number (default=(0, 100))
        :param n_tick: Tick interval for the SPT N number (default=10)
        :param z_range: Range for the depth (default=None for plotting from zero to maximum SPT depth)
        :param z_tick: Tick interval for depth (default=2m)
        :param plot_height: Height for the plot (default=700px)
        :param plot_width: Width for the plot (default=700)
        :param return_fig: Boolean determining whether the figure needs to be returned (True) or plotted (False)
        :param plot_title: Plot for the title (default=None)
        :param plot_margin: Margin for the plot (default=dict(t=100, l=50, b=50))
        :param color: Color to be used for plotting (default=None for default plotly colors)
        :param markersize: Size of the markers to be used (default=5)
        :param plot_layers: Boolean determining whether to show the layers (if available)
        :return:
        """

        if z_range is None:
            z_range = (self.data['z [m]'].max(), 0)
        if color is None:
            color = DEFAULT_PLOTLY_COLORS[0]
        fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)

        trace1 = go.Scatter(
            x=self.data['N [-]'],
            y=self.data['z [m]'],
            marker=dict(size=markersize,color=color,line=dict(width=1,color='black')),
            showlegend=False, mode='markers', name='N')
        fig.append_trace(trace1, 1, 1)

        # Plot layers
        try:
            for i, row in self.layerdata.iterrows():
                if i > 0:
                    layer_trace_qc = go.Scatter(
                        x=n_range,
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_qc, 1, 1)
        except:
            pass
        fig['layout']['xaxis1'].update(
            title='N [-]', side='top', anchor='y',
            range=n_range, dtick=n_tick)
        fig['layout']['yaxis1'].update(
            title='z [m]', range=z_range, dtick=z_tick)
        fig['layout'].update(
            title=plot_title, hovermode='closest',
            height=plot_height, width=plot_width,
            margin=plot_margin)
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    def plot_properties_withlog(self, prop_keys, plot_ranges, plot_ticks,
                                legend_titles=None, axis_titles=None, showfig=True, showlayers=True, **kwargs):
        """
        Plots SPT properties vs depth and includes a mini-log on the left-hand side.
        The minilog is composed based on the entries in the ``Soil type`` column of the layering
        :param prop_keys: Tuple of tuples with the keys to be plotted. Keys in the same tuple are plotted on the same panel
        :param plot_ranges: Tuple of tuples with ranges for the panels of the plot
        :param plot_ticks: Tuple with tick intervals for the plot panels
        :param z_range: Range for depths (optional, default is (0, maximum SPT depth)
        :param z_tick: Tick mark distance for SPT depth (optional, default=2)
        :param legend_titles: Tuple with entries to be used in the legend. If left blank, the keys are used
        :param axis_titles: Tuple with entries to be used as axis labels. If left blank, the keys are used
        :param showfig: Boolean determining whether the figure needs to be shown in the notebook (default=True)
        :param showlayers: Boolean determining whether layer positions need to be plotted (default=True)
        :param **kwargs: Specify keyword arguments for the ``general.plotting.plot_with_log`` function
        :return: Plotly figure with mini-log
        """
        # Validate if 'Soil type' column is present in the layering
        if 'Soil type' not in self.layerdata.columns:
            raise ValueError("Layering should contain the column 'Soil type'")

        if legend_titles is None:
            legend_titles = prop_keys
        if axis_titles is None:
            axis_titles = prop_keys

        _x = []
        _z = []
        _modes = []

        for _panel in prop_keys:
            _x_panel = []
            _z_panel = []
            _modes_panel = []
            for _prop in _panel:
                _x_data = self.data[_prop]
                _z_data = self.data['z [m]']
                _x_panel.append(_x_data)
                _z_panel.append(_z_data)
                _modes_panel.append('markers')
            _x.append(_x_panel)
            _z.append(_z_panel)
            _modes.append(_modes_panel)

        fig = plot_with_log(
            x=_x,
            z=_z,
            modes=_modes,
            names=legend_titles,
            soildata=self.layerdata,
            xtitles=axis_titles,
            xranges=plot_ranges,
            dticks=plot_ticks,
            showfig=False,
            **kwargs
        )

        # Plot layers
        if showlayers:
            try:
                for i, row in self.layerdata.iterrows():
                    if i > 0:
                        for j, _range in enumerate(plot_ranges):
                            layer_trace = go.Scatter(
                                x=_range,
                                y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                                line=dict(color='black', dash='dot'),
                                showlegend=False, mode='lines')
                            fig.append_trace(layer_trace, 1, j + 2)
            except Exception as err:
                pass
        else:
            pass

        if showfig:
            fig.show()
        return fig
