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
from plotly.offline import iplot

# Project imports
from groundhog.general.plotting import plot_with_log, GROUNDHOG_PLOTTING_CONFIG
from groundhog.general.parameter_mapping import map_depth_properties, merge_two_dicts, reverse_dict
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
    'eta H [-]': [np.nan,],
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

    def apply_correlation(self, name, outkey, resultkey, apply_for_soiltypes='all', **kwargs):
        """
        Applies a correlation to the given SPT data. The name of the correlation needs to be chosen from the following available correlations.
        Each correlation corresponds to a function in the `spt_correlations` module.
        By default, the correlation is applied to the entire depth range. However, a restriction on the soil
        types to which the correlation can be applied can be specified with the `apply_for_soiltypes` keyword argument.
        A list with the soil types for which the correlation needs to be applied can be provided.

            - Overburden correction Liao and Whitman (1986) (`overburdencorrection_spt_liaowhitman`) - Calculation of overburden correction according to Liao & Whitman (1986)
            - N60 correction (`spt_N60_correction`) - Correction of measured SPT N number to an equivalent N60 at energy ratio of 60%

        Note that certain correlations require either the application of preceding correlations

        :param name: Name of the correlation according to the list defined above
        :param outkey: Key used to the output column
        :param resultkey: Key of the output dictionary of the correlation to be used
        :param apply_for_soiltypes: List with soil types to which the correlation needs the be applied.
        :param kwargs: Optional keyword arguments for the correlation.
        :return: Adds a column with key `outkey` to the dataframe with SPT data
        """

        if outkey in self.data.columns:
            self.data.drop(outkey, axis=1, inplace=True)

        self.data.rename(columns=SPT_KEY_MAPPING, inplace=True)
        for i, row in self.data.iterrows():
            if apply_for_soiltypes == 'all' or row['Soil type'] in apply_for_soiltypes:
                params = merge_two_dicts(kwargs, dict(row))
                self.data.loc[i, outkey] = CORRELATIONS[name](**params)[resultkey]
            else:
                self.data.loc[i, outkey] = np.nan
        self.data.rename(columns=reverse_dict(SPT_KEY_MAPPING), inplace=True)
    # endregion