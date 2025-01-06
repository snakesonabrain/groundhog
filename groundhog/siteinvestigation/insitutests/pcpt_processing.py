#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
from multiprocessing.sharedctypes import Value
import os
import warnings
from copy import deepcopy
import json
import re
import PIL

# 3rd party packages
import pandas as pd
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

# Project imports
from groundhog.general.plotting import plot_with_log, LogPlotMatplotlib, GROUNDHOG_PLOTTING_CONFIG
from groundhog.general.parameter_mapping import SOIL_PARAMETER_MAPPING, merge_two_dicts, reverse_dict
from groundhog.siteinvestigation.insitutests.pcpt_correlations import *
from groundhog.general.soilprofile import SoilProfile, plot_fence_diagram
from groundhog.general.parameter_mapping import offsets, latlon_distance
from groundhog.general.agsconversion import AGSConverter

DEFAULT_CONE_PROPERTIES = SoilProfile({
    'Depth from [m]': [0, ],
    'Depth to [m]': [20, ],
    'area ratio [-]': [0.8, ],
    'Cone type': ['U', ],
    'Cone base area [cm2]': [10, ],
    'Cone sleeve_area [cm2]': [150, ],
    'Sleeve cross-sectional area top [cm2]': [np.nan,],
    'Sleeve cross-sectional area bottom [cm2]': [np.nan,]
})

GEF_COLINFO = {
    '1': 'z [m]',
    '2': 'qc [MPa]',
    '3': 'fs [MPa]',
    '4': 'Rf [%]',
    '5': 'u1 [MPa]',
    '6': 'u2 [MPa]',
    '7': 'u3 [MPa]',
    '8': 'incl [deg]',
    '9': 'incl N-S [deg]',
    '10': 'incl E-W [deg]',
    '11': 'z corrected [m]',
    '12': 'time [s]',
    '13': 'qt [MPa]',
    '14': 'qn [MPa]',
    '15': 'Bq [-]',
    '16': 'Nm [-]',
    '17': 'gamma [kN/m3]',
    '18': 'u0 [MPa]',
    '19': 'sigma_vo [MPa]',
    '20': 'sigma_vo_eff [MPa]'
}

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')


class InsituTestProcessing(object):
    """
    Abstract base class for the processing of in-situ tests.
    Encodes shared functionality between different types of in-situ tests
    """
    def __init__(self, title, waterunitweight=10.25):
        """
        Initialises an in-situ test object
        """
        self.title = title
        self.data = pd.DataFrame()
        self.downhole_corrected = False
        self.waterunitweight=waterunitweight
        self.waterlevel = None
        self.layerdata = pd.DataFrame()
        self.additionaldata = dict()
        self.easting = np.nan
        self.northing = np.nan
        self.elevation = np.nan
        self.srid = None
        self.datum = None
        self.designprofile = None

    def set_position(self, easting, northing, elevation, srid=4326, datum='mLAT'):
        """
        Sets the position of an SPT test in a given coordinate system.

        By default, srid 4326 is used which means easting is longitude and northing is latitude.

        The elevation is referenced to a chart datum for which mLAT (Lowest Astronomical Tide) is the default.

        :param easting: X-coordinate of the SPT position
        :param northing: Y-coordinate of the SPT position
        :param elevation: Elevation of the SPT position
        :param srid: SRID of the coordinate system (see http://epsg.io)
        :param datum: Chart datum used for the elevation
        :return: Sets the corresponding attributes of the ```SPTProcessing``` object
        """
        self.easting = easting
        self.northing = northing
        self.elevation = elevation
        self.srid = srid
        self.datum = datum

    def map_properties(self, layer_profile,
                       initial_vertical_total_stress=0,
                       vertical_total_stress=None,
                       vertical_effective_stress=None, waterlevel=0,
                       extend_layer_profile=True):
        """
        Maps the soil properties defined in the layering to the grid
        defined by the cone data. The procedure also calculates the total and effective vertical stress.
        Note that pre-calculated arrays with total and effective vertical stress can also be supplied to the routine.
        These needs to have the same length as the array with in-situ test depth data.

        :param layer_profile: ``SoilProfile`` object with the layer properties (need to contain the soil parameter ``Total unit weight [kN/m3]``
        :param initial_vertical_total_stress: Initial vertical total stress at the highest point of the soil profile
        :param vertical_total_stress: Pre-calculated total vertical stress at in-situ test depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param vertical_effective_stress: Pre-calculated effective vertical stress at in-situ test depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param waterlevel: Waterlevel [m] in the soil (measured from soil surface), default = 0m
        :param extend_layer_profile: Boolean determining whether the layer profile needs to be extended to the bottom of the CPT (default = True)
        :return: Expands the dataframe `.data` with additional columns for the cone and soil properties
        """
        self.waterlevel = waterlevel
        self.layerdata = layer_profile

        # Validation
        if 'Total unit weight [kN/m3]' not in layer_profile.numerical_soil_parameters():
            raise ValueError("Soil layering profile needs to contain the parameter 'Total unit weight [kN/m3]'")

        # Validate that cone property boundaries fully contain the CPT info

        if extend_layer_profile:
            if layer_profile[layer_profile.depth_to_col].max() < self.data['z [m]'].max():
                warnings.warn("Layering extended to bottom of CPT")
                layer_profile[layer_profile.depth_to_col].iloc[-1] = self.data['z [m]'].max()

        if layer_profile.min_depth > self.data['z [m]'].min():
            raise ValueError(
                "Layering starts below minimum in-situ test depth. " +
                "Ensure that layering fully contains in-situ test data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))
        if layer_profile.max_depth < self.data['z [m]'].max():
            raise ValueError(
                "Layering ends above minimum in-situ test depth. " +
                "Ensure that layering fully contains in-situ test data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))

        # Calculation of total and effective vertical stress
        self.layerdata.calculate_overburden(
            waterlevel=waterlevel,
            waterunitweight=self.waterunitweight,
            initial_vertical_total_stress=initial_vertical_total_stress)

        # Map layer properties
        for i, row in layer_profile.iterrows():
            layer_profile.loc[i, "Layer no"] = i+1
        _mapped_layer_props = layer_profile.map_soilprofile(
            self.data['z [m]'])

        # Join values to the CPT data
        for _key in _mapped_layer_props.columns:
            self.data[_key] = _mapped_layer_props[_key]

        if vertical_total_stress is None:
            pass # Accept overburden calculated from soil profile
        else:
            self.data["Vertical total stress [kPa]"] = vertical_total_stress

        if vertical_effective_stress is None:
            pass # Accept overburden calculated from soil profile
        else:
            self.data["Vertical effective stress [kPa]"] = vertical_effective_stress


class PCPTProcessing(InsituTestProcessing):
    """
    The PCPTProcessing class implements methods for reading, processing and presentation of PCPT data.
    Common correlations are also encoded.
    """

    def __init__(self, title, waterunitweight=10.25):
        """
        Initialises a PCPTProcessing object based on a title. Optionally, a geographical position can be defined. A dictionary for dumping unstructured data (``additionaldata``) is also available.

        An empty dataframe (``.data`) is created for storing the PCPT data`

        :param title: Title for the PCPT test
        :param waterunitweight: Unit weight of water used for effective stress calculations (default=10.25kN/m3 for seawater)

        """
        super().__init__(title, waterunitweight)
        self.downhole_corrected = False
        self.coneprofile = pd.DataFrame()

    # region Utility functions

    def rename_columns(self, z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key=None):
        if z_key is None:
            z_key = "z [m]"

        if qc_key is None:
            qc_key = "qc [MPa]"

        if fs_key is None:
            fs_key = "fs [MPa]"

        if u2_key is None:
            u2_key = "u2 [MPa]"

        if push_key is None:
            push_key = "Push"

        self.data.rename(columns={z_key: 'z [m]',
                                  qc_key: 'qc [MPa]',
                                  fs_key: 'fs [MPa]',
                                  u2_key: 'u2 [MPa]',
                                  push_key: 'Push'}, inplace=True)

    def convert_columns(self, qc_multiplier=1, fs_multiplier=1, u2_multiplier=1):
        self.data['qc [MPa]'] = self.data['qc [MPa]'] * qc_multiplier
        self.data['fs [MPa]'] = self.data['fs [MPa]'] * fs_multiplier
        self.data['u2 [MPa]'] = self.data['u2 [MPa]'] * u2_multiplier

    def add_zerodepth_row(self):
        self.data.loc[self.data.__len__(), "z [m]"] = 0
        self.data.sort_values("z [m]", inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def dropna_rows(self):
        try:
            self.data.dropna(subset=('z [m]', 'qc [MPa]', 'fs [MPa]', 'u2 [MPa]'), how='all', inplace=True)
        except:
            self.data.dropna(how='all', inplace=True)

    # endregion

    # region Data loading

    def load_excel(self, path, z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                   qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True, **kwargs):
        """
        Loads PCPT data from an Excel file. Specific column keys have to be provided for z, qc, fs and u2.
        If column keys are not specified, the following keys are used:

            - 'z [m]' for depth below mudline
            - 'qc [MPa]' for cone tip resistance
            - 'fs [MPa]' for sleeve friction
            - 'u2 [MPa]' for pore pressure at cone shoulder

        Note that cone tip resistance, sleeve friction and pore pressure at the shoulder all need to be converted to MPa.
        Multipliers can be specified if a conversion from kPa to MPa is required. Optional keyword arguments for the
        `read_excel` function in Pandas can be specified as `**kwargs`.

        :param path: Path to the Excel file
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param kwargs: Optional keyword arguments for the read_excel function in Pandas (e.g. sheet_name, header, ...)
        :return: Sets the columns 'z [m]', 'qc [MPa]', 'fs [MPa]' and 'u2 [MPa]' of the `.data` attribute
        """

        try:
            self.data = pd.read_excel(path, **kwargs)
            if push_key not in self.data.columns:
                # No push detected
                self.data.loc[:, "Push"] = 1
            self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
            self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)
        except Exception as err:
            raise ValueError("Error during reading of PCPT data. Review the error message and try again. - %s" % (
                str(err)))
        try:
            if self.data["z [m]"].min() != 0 and add_zero_row:
                # Append a row with zero data for vertical effective stress calculation
                self.add_zerodepth_row()
            else:
                pass
        except Exception as err:
            raise ValueError("Error appending of row with zero depth to PCPT data. Review the error message and try again. - %s" % (
                str(err)))

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_pandas(self, df, z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                    qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True, ):
        """
        Loads PCPT from a Pandas dataframe. Specific column keys have to be provided for z, qc, fs and u2.
        If column keys are not specified, the following keys are used:

            - 'z [m]' for depth below mudline
            - 'qc [MPa]' for cone tip resistance
            - 'fs [MPa]' for sleeve friction
            - 'u2 [MPa]' for pore pressure at cone shoulder

        Note that cone tip resistance, sleeve friction and pore pressure at the shoulder all need to be converted to MPa

        :param df: Pandas dataframe with required column keys
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :return: Sets the columns 'z [m]', 'qc [MPa]', 'fs [MPa]' and 'u2 [MPa]' of the `.data` attribute
        """

        try:
            self.data = df
            if push_key not in self.data.columns:
                self.data.loc[:, "Push"] = 1
            self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
            self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)
        except Exception as err:
            raise ValueError("Error during reading of PCPT data. Review the error message and try again. - %s" % (
                str(err)))
        try:
            if self.data["z [m]"].min() != 0 and add_zero_row:
                # Append a row with zero data for vertical effective stress calculation
                self.add_zerodepth_row()
            else:
                pass
        except Exception as err:
            raise ValueError("Error appending of row with zero depth to PCPT data. Review the error message and try again. - %s" % (
                str(err)))

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_ags(self, path, z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True,
                 ags_group="SCPT", verbose_keys=False, use_shorthands=False, **kwargs):
        """
        Loads PCPT data from an AGS file. Specific column keys have to be provided for z, qc, fs and u2.
        If column keys are not specified, the following keys are used:

            - 'z [m]' for depth below mudline
            - 'qc [MPa]' for cone tip resistance
            - 'fs [MPa]' for sleeve friction
            - 'u2 [MPa]' for pore pressure at cone shoulder

        Note that cone tip resistance, sleeve friction and pore pressure at the shoulder all need to be converted to MPa.
        Multipliers can be specified if a conversion from kPa to MPa is required. Optional keyword arguments for the
        `read_ags` function in Pandas can be specified as `**kwargs`.

        :param path: Path to the ags file
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param ags_group: Name of the AGS group with the CPT data (default= ``"SCPT"``)
        :param verbose_keys: Boolean for using verbose keys in the AGS converter (default=True)
        :param use_shorthands: Boolean for using shorthands in the AGS converter (default=True)
        :param kwargs: Optional keyword arguments for the read_excel function in Pandas (e.g. sheet_name, header, ...)
        :return: Sets the columns 'z [m]', 'qc [MPa]', 'fs [MPa]' and 'u2 [MPa]' of the `.data` attribute
        """

        try:
            converter = AGSConverter(path=path)
            converter.create_dataframes(selectedgroups=[ags_group], verbose_keys=verbose_keys,
                                        use_shorthands=use_shorthands)
        except Exception as err:
            raise ValueError("Error during reading of AGS file. Review the error message and try again. - %s" % (
                str(err)))

        self.load_pandas(
            df=converter.data[ags_group],
            z_key=z_key,
            qc_key=qc_key,
            fs_key=fs_key,
            u2_key=u2_key,
            push_key=push_key,
            qc_multiplier=qc_multiplier,
            fs_multiplier=fs_multiplier,
            u2_multiplier=u2_multiplier,
            add_zero_row=add_zero_row,
            **kwargs)


    def load_asc(self, path, column_widths=[], skiprows=None, custom_headers=None,
                 z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True, **kwargs):
        """
        Reads PCPT data from a Uniplot .asc file
        The widths of the columns for the .asc file need to be specified.

        :param path: Path to the .asc file
        :param column_widths: Column widths to use for the import (compulsory)
        :param skiprows: Number of rows to skip (optional, default=None for auto-detection, depends on the occurence of 'Data table' in the file)
        :param custom_headers: Custom headers to be used (default=None for auto-detection)
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param kwargs: Optional keyword arguments for reading the datafile
        :return:
        """

        # Automatic detection of the number of rows to skip
        if skiprows is None:
            with open(path) as f:
                raw_file = pd.DataFrame(f.readlines(), columns=["asc lines"])
            # Find the line where the data starts
            start_line = raw_file[raw_file["asc lines"].str.contains('Data table')]
            skiprows = start_line.index[0] + 1
        else:
            skiprows = skiprows

        # Read the data
        self.data = pd.read_fwf(
            path,
            widths=column_widths,
            skiprows=skiprows, **kwargs)

        # Convert headers using the two rows with header information in the asc file
        if custom_headers is None:
            header_list = list(map(lambda key, unit: "%s [%s]" % (key, unit),
                list(self.data.columns), list(self.data.loc[0])))
        else:
            header_list = custom_headers

        new_headers = dict()
        for i, _head in enumerate(header_list):
            new_headers[self.data.columns[i]] = _head
        # Rename the headers and drop the first row
        self.data.rename(columns=new_headers, inplace=True)
        self.data = self.data.drop([0])
        self.data = self.data.astype('float')
        self.data.reset_index(drop=True, inplace=True)
        if push_key not in self.data.columns:
            self.data.loc[:, "Push"] = 1
        self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
        self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)

        if self.data["z [m]"].min() != 0 and add_zero_row:
            # Append a row with zero data for vertical effective stress calculation
            self.add_zerodepth_row()
        else:
            pass

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_gef(self, path, inverse_depths=False, override_title=True,
                 z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True,
                 separator=' ', **kwargs):
        """
        Reads PCPT data from a Geotechnical Exchange Format (.gef) file.
        The file is parsed using regular expressions to provide the necessary data.
        If location data is provided, this data is also

        https://publicwiki.deltares.nl/download/attachments/102204314/GEFCR100.pdf?version=1&modificationDate=1411129283000&api=v2

        :param path: Path to the .asc file
        :param inverse_depths: Boolean indicating whether depths need to be inverted. A positive downward z-axis is expected.
        :param override_title: Boolean indicating if the title specified needs to be replaced by the test id (default = True).
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param separator: Separator used for the gef file (default is ``' '``)
        :param kwargs: Optional keyword arguments for reading the datafile (using ``read_csv`` from Pandas)
        :return:
        """

        # Regular expression setup
        colno_pattern = re.compile(r'#COLUMN\s*=\s*(?P<colno>\d+)')
        colinfo_pattern = re.compile(
            r'#COLUMNINFO\s*=\s*(?P<colno>\d+)\s*,\s*(?P<units>.+)\s*,\s*(?P<title>.+)\s*,\s*(?P<quantity>\d+)\s*')
        separator_pattern = re.compile(r'#COLUMNSEPARATOR\s*=\s*(?P<separator>.)')
        test_id_pattern = re.compile(r'#TESTID\s*=\s*(?P<test_id>.*)')
        xy_id_pattern = re.compile(
            r'#XYID\s*=\s*(?P<coordsys>\d*)\s*,\s*(?P<X>\d*.?\d*)\s*,' +
            '\s*(?P<Y>\d*.?\d*)\s*,\s*(?P<dX>\d*.?\d*)\s*,\s*(?P<dY>\d*.?\d*)\s*')
        xy_id_pattern_nodeltas = re.compile(
            r'#XYID\s*=\s*(?P<coordsys>\d*)\s*,\s*(?P<X>\d*.?\d*)\s*,' +
            '\s*(?P<Y>\d*.?\d*)\s*')
        z_id_pattern = re.compile(r'#ZID\s*=\s*(?P<datum>\d*)\s*,\s*(?P<Z>\d*.?\d*)\s*,\s*(?P<dZ>\d*.?\d*)\s*')
        measurementtext_pattern = re.compile(r'#MEASUREMENTTEXT\s*=\s*(?P<number>\d*)\s*,\s*(?P<text>.*)')
        measurementvalue_pattern = re.compile(
            r'#MEASUREMENTVAR\s*=\s*(?P<number>\d*)\s*,\s*(?P<value>\d*.\d*)\s*,\s*(?P<unit>.*)\s*,\s*(?P<text>.*)\s*')
        voidvalue_pattern = re.compile(r'#COLUMNVOID\s*=\s*(?P<colno>\d+)\s*,\s*(?P<voidvalue>-*\d*.\d*)')

        # Read the raw data
        with open(path) as f:
            gef_raw = pd.DataFrame(f.readlines(), columns=["GEF lines"])

        # Process the header
        colinfo_dict = dict()
        measurementtext_dict = dict()
        measurementvalue_dict = dict()
        voidvalue_dict = dict()

        separator = separator
        test_id = None
        easting = None
        northing = None
        srid = None
        dx = None
        dy = None
        z = None
        dz = None
        datum = None


        for i, row in gef_raw.iterrows():
            try:
                match = re.search(colno_pattern, row['GEF lines'])
                colno = match.group('colno')
            except:
                pass

            try:
                match = re.search(colinfo_pattern, row['GEF lines'])
                colinfo_dict[match.group('colno')] = dict(units=match.group('units'), title=match.group('title'),
                                                          quantity=match.group('quantity'))
            except:
                pass

            try:
                match = re.search(voidvalue_pattern, row['GEF lines'])
                voidvalue_dict[match.group('colno')] = match.group('voidvalue')
            except:
                pass

            try:
                match = re.search(separator_pattern, row['GEF lines'])
                separator = match.group('separator')
            except:
                pass

            try:
                match = re.search(test_id_pattern, row['GEF lines'])
                test_id = match.group('test_id')
            except:
                pass

            try:
                try:
                    match = re.search(xy_id_pattern, row['GEF lines'])
                    easting = float(match.group('X'))
                    northing = float(match.group('Y'))
                    srid = match.group('coordsys')
                    dx = float(match.group('dX'))
                    dy = float(match.group('dY'))
                except:
                    match = re.search(xy_id_pattern_nodeltas, row['GEF lines'])
                    easting = float(match.group('X'))
                    northing = float(match.group('Y'))
                    srid = match.group('coordsys')
            except:
                pass

            try:
                match = re.search(z_id_pattern, row['GEF lines'])
                z = float(match.group('Z'))
                dz = float(match.group('dZ'))
                datum = match.group('datum')
            except:
                pass

            try:
                match = re.search(measurementtext_pattern, row['GEF lines'])
                measurementtext_dict[match.group('number')] = dict(text=match.group('text'))
            except:
                pass

            try:
                match = re.search(measurementvalue_pattern, row['GEF lines'])
                measurementvalue_dict[match.group('number')] = dict(
                    value=float(match.group('value')),
                    unit=match.group('unit'),
                    text=match.group('text'))
            except:
                pass

        # Detect the start of the file
        indices = gef_raw[gef_raw["GEF lines"].str.startswith('#EOH')].index

        # Define column keys
        keys = []
        for key, value in colinfo_dict.items():
            try:
                keys.append(GEF_COLINFO[value['quantity']])
            except:
                keys.append('%s [%s]' % (value['title'], value['units']))

        # Read data
        self.data = pd.read_csv(
            path, sep=separator, skiprows=indices[0]+1, names=keys, index_col=False,
            na_values=voidvalue_dict.values(), **kwargs)

        # Dump additional data
        self.additionaldata = {
            "Measurement text": measurementtext_dict,
            "Measurement var": measurementvalue_dict
        }

        # Inverse depth column
        if inverse_depths:
            self.data['z [m]'] = -self.data['z [m]']

        # Coordinates
        if (not easting is None) and (not northing is None) and (not datum is None):
            self.set_position(
                easting=easting,
                northing=northing,
                srid=srid,
                elevation=z,
                datum=datum
            )

        # Set title
        if override_title and test_id is not None:
            self.title = test_id.strip()

        self.data = self.data.astype('float')
        self.data.reset_index(drop=True, inplace=True)
        if push_key not in self.data.columns:
            self.data.loc[:, "Push"] = 1
        try:
            self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
        except Exception as err:
            warnings.warn(
                "Conversion of column names unsuccessful. Review imported data (Error message %s)" % str(err))
        try:
            self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)
        except Exception as err:
            warnings.warn(
                "Application of column multipliers unsuccessful. Review imported data (Error message %s)" % str(err))

        if self.data["z [m]"].min() != 0 and add_zero_row:
            # Append a row with zero data for vertical effective stress calculation
            self.add_zerodepth_row()
        else:
            pass

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_multi_asc(self, path, column_widths=[], skiprows=None, custom_headers=None,
                 z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True,
                 start_string='Data table', end_string='UNICAS data file', **kwargs):
        """
        Reads PCPT data from a Uniplot .asc file with multiple pushes
        The widths of the columns for the .asc file need to be specified.

        :param path: Path to the .asc file
        :param column_widths: Column widths to use for the import (compulsory)
        :param skiprows: Number of rows to skip (optional, default=None for auto-detection, depends on the occurence of 'Data table' in the file)
        :param custom_headers: Custom headers to be used (default=None for auto-detection)
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param start_string: String showing the start of the data
        :param end_string: String showing the end of the data
        :param kwargs: Optional keyword arguments for reading the datafile
        :return:
        """
        # Automatic detection of the number of rows to skip
        with open(path) as f:
            raw_file = pd.DataFrame(f.readlines(), columns=["asc lines"])
        start_lines = list(raw_file[raw_file["asc lines"].str.contains("Data table")].index)
        end_lines = list(raw_file[raw_file["asc lines"].str.startswith(" UNICAS data")].index)[1:]
        end_lines.append(raw_file.__len__())

        self.data = pd.DataFrame()

        colspecs = []
        colpos = 0
        for _w in column_widths:
            colspecs.append((colpos, colpos + _w))
            colpos += _w

        # Read the data
        for i, _start in enumerate(start_lines):

            push_data =  pd.read_fwf(
                path,
                colspecs=colspecs,
                skiprows=start_lines[i]+1,
                nrows=end_lines[i] - start_lines[i] - 3,
                **kwargs)
            if i == 0:
                header_list = list(map(lambda key, unit: "%s [%s]" % (key, unit),
                                       list(push_data.columns), list(push_data.loc[0])))
                new_headers = dict()
                for j, _head in enumerate(header_list):
                    new_headers[push_data.columns[j]] = _head

            push_data.rename(columns=new_headers, inplace=True)
            push_data.reset_index(drop=True, inplace=True)
            push_data = push_data.drop([0])
            push_data = push_data.astype('float')
            push_data.reset_index(drop=True, inplace=True)
            push_data.loc[:, 'Push'] = i+1
            self.data = pd.concat([self.data, push_data]).reset_index(drop=True)

        self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
        self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)

        if self.data["z [m]"].min() != 0 and add_zero_row:
            # Append a row with zero data for vertical effective stress calculation
            self.add_zerodepth_row()
        else:
            pass

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))


    def load_a00(self, path, column_widths=[], skiprows=None, custom_headers=None,
                 z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, add_zero_row=True, **kwargs):
        """
        Reads PCPT data from a Uniplot .a00 file
        The widths of the columns for the .a00 file need to be specified.

        :param path: Path to the .a00 file
        :param column_widths: Column widths to use for the import (compulsory)
        :param skiprows: Number of rows to skip (optional, default=None for auto-detection, depends on the occurence of 'Data table' in the file)
        :param custom_headers: Custom headers to be used (default=None for auto-detection)
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param add_zero_row: Boolean determining whether a datapoint needs to be added at zero depth.
        :param kwargs: Optional keyword arguments for reading the datafile
        :return:
        """

        # Automatic detection of the number of rows to skip
        if skiprows is None:
            with open(path) as f:
                raw_file = pd.DataFrame(f.readlines(), columns=["a00 lines"])
            # Find the line where the data starts
            start_line = raw_file[raw_file["a00 lines"].str.contains('Reading')]
            skiprows = start_line.index[0]
        else:
            skiprows = skiprows

        # Read the data
        self.data = pd.read_fwf(
            path,
            widths=column_widths,
            skiprows=skiprows, **kwargs)

        # Convert headers using the two rows with header information in the asc file
        if custom_headers is None:
            header_list = list(map(lambda key, unit: "%s [%s]" % (key, unit),
                list(self.data.columns), list(self.data.loc[0])))
        else:
            header_list = custom_headers

        new_headers = dict()
        for i, _head in enumerate(header_list):
            new_headers[self.data.columns[i]] = _head
        # Rename the headers and drop the first row
        self.data.rename(columns=new_headers, inplace=True)
        self.data = self.data.drop([0, 1])
        self.data = self.data.astype('float')
        self.data.reset_index(drop=True, inplace=True)
        if push_key not in self.data.columns:
            self.data.loc[:, "Push"] = 1
        self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key, push_key=push_key)
        self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)

        if self.data["z [m]"].min() != 0 and add_zero_row:
            # Append a row with zero data for vertical effective stress calculation
            self.add_zerodepth_row()
        else:
            pass

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def load_multiple_asc(self, folder, column_widths=[], skiprows=None, custom_headers=None,
                 z_key=None, qc_key=None, fs_key=None, u2_key=None, push_key='Push',
                 qc_multiplier=1, fs_multiplier=1, u2_multiplier=1, **kwargs):
        """
        A PCPT can be provided as multiple .asc files in one folder. This method loops over the individual
        files and creates a combined `data` attribute with PCPT data. The method assumes that all .asc files have
        the same format.

        :param folder: Folder with the .asc files
        :param column_widths: Column widths to use for the import (compulsory)
        :param skiprows: Number of rows to skip (optional, default=None for auto-detection, depends on the occurence of 'Data table' in the file)
        :param custom_headers: Custom headers to be used (default=None for auto-detection)
        :param z_key: Column key for depth. Optional, default=None when 'z [m]' is the column key.
        :param qc_key: Column key for cone tip resistance. Optional, default=None when 'qc [MPa]' is the column key.
        :param fs_key: Column key for sleeve friction. Optional, default=None when 'fs [MPa]' is the column key.
        :param u2_key: Column key for pore pressure at shoulder. Optional, default=None when 'u2 [MPa]' is the column key.
        :param push_key: Column key for the current push (for downhole PCPT). Optional, default=None for a continuous push.
        :param qc_multiplier: Multiplier applied on cone tip resistance to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param fs_multiplier: Multiplier applied on sleeve friction to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param u2_multiplier: Multiplier applied on pore pressure at shoulder to convert to MPa (e.g. 0.001 to convert from kPa to MPa)
        :param kwargs: Optional keyword arguments for reading the datafiles
        :return: Sets the `data` attribute of the PCPTProcessing object
        """
        for i, filename in enumerate(os.listdir(folder)):
            if filename.endswith('.asc') or filename.endswith('.ASC'):
                if i > 0:
                    previous_data = deepcopy(self.data)
                    add_zero_row = False
                else:
                    add_zero_row = True
                self.load_asc(
                    self, filename, column_widths=column_widths, skiprows=skiprows,
                    custom_headers=custom_headers, z_key=z_key, qc_key=qc_key, fs_key=fs_key,
                    u2_key=u2_key, push_key=push_key,
                    qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier,
                    add_zero_row=add_zero_row, **kwargs)
                self.data.loc[:, "Push"] = i+1
                self.data = pd.concat([previous_data, self.data])
            else:
                pass

        self.data.sort_values('z [m]', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def load_pydov(self, name, push_key='Push', add_zero_row=True, z_key='diepte', qc_key='qc', fs_key='fs', u2_key='u',
        qc_multiplier=1, fs_multiplier=0.001, u2_multiplier=0.001, **kwargs):
        """
        Load CPT data from Databank Ondergrond Vlaanderen based on the unique CPT name which can be found in DOV
        :param name: Unique identifier of the CPT in pydov
        :return: Sets the `data` attribute of the PCPTProcessing object
        """
        try:
            import pydov
            from owslib.fes2 import PropertyIsEqualTo
            from pydov.search.sondering import SonderingSearch
        except:
            raise IOError("Package pydov not available. Install it first: https://pydov.readthedocs.io/en/stable/")

        sondering = SonderingSearch()
        query = PropertyIsEqualTo(propertyname='sondeernummer',
                                  literal=name)
        self.data = sondering.search(query=query)
        if push_key not in self.data.columns:
            self.data.loc[:, "Push"] = 1
        self.rename_columns(z_key=z_key, qc_key=qc_key, fs_key=fs_key, u2_key=u2_key)
        self.convert_columns(qc_multiplier=qc_multiplier, fs_multiplier=fs_multiplier, u2_multiplier=u2_multiplier)

        try:
            if self.data["z [m]"].min() != 0 and add_zero_row:
                # Append a row with zero data for vertical effective stress calculation
                self.add_zerodepth_row()
            else:
                pass
        except Exception as err:
            raise ValueError("Error appending of row with zero depth to PCPT data. Review the error message and try again. - %s" % (
                str(err)))

        try:
            self.dropna_rows()
        except Exception as err:
            raise ValueError("Error during dropping of empty rows. Review the error message and try again - %s" % str(
                err))

    def combine_pcpt(self, obj, keep="first"):
        """
        Combine PCPT data for two PCPTProcessing objects. The data of the second PCPT (`obj`) will be merged
        with the current object and the `data` attribute will be a combined dataframe.
        Only columns 'z [m]', 'qc [MPa]', 'fs [MPa]', 'u2 [MPa]' and 'Push' will be retained.

        :param obj: PCPTProcessing object containing the second PCPT info
        :param keep: Determines what to do in the area with overlap ("first", "second" or "both"). If "first" is chosen (default), the data from the first PCPT `self` is used in the area with overlap. For "second", data for the second PCPT `obj` is retained. If 'both' is selected, overlapping data will exist which might lead to presentation problems.
        :return: Updates the `data` attribute of the current PCPTProcessing object
        """
        if type(obj) != PCPTProcessing:
            raise TypeError("obj should be a PCPTProcessing object. Instead a %s was provided." % (type(obj)))

        z_min_first = self.data['z [m]'].min()
        z_max_first = self.data['z [m]'].max()
        z_min_second = obj.data['z [m]'].min()
        z_max_second = obj.data['z [m]'].max()

        self.data['Push'] = list(map(lambda _push: "1-%s" % _push, self.data['Push']))
        obj.data['Push'] = list(map(lambda _push: "2-%s" % _push, obj.data['Push']))

        # Second PCPT above first one
        if (z_max_second > z_min_first) and (z_min_second < z_min_first):
            if keep == 'first':
                first_data = self.data
                second_data = obj.data[obj.data["z [m]"] < z_min_first]
            elif keep == "second":
                first_data = self.data[self.data["z [m]"] > z_max_second]
                second_data = obj.data
            elif keep == "both":
                first_data = self.data
                second_data = obj.data
            else:
                raise ValueError("Invalid option for argument 'keep'. Choose from 'first', 'second' or 'both'")

        # Second PCPT below first one
        elif (z_min_second < z_max_first) and (z_max_second > z_max_first):
            if keep == 'first':
                first_data = self.data
                second_data = obj.data[obj.data["z [m]"] > z_max_first]
            elif keep == "second":
                first_data = self.data[self.data["z [m]"] < z_min_second]
                second_data = obj.data
            elif keep == "both":
                first_data = self.data
                second_data = obj.data
            else:
                raise ValueError("Invalid option for argument 'keep'. Choose from 'first', 'second' or 'both'")
        # Other cases
        
        else:
            first_data = self.data
            second_data = obj.data
    
        

        # Combine data
        self.data = pd.concat([first_data, second_data])
        self.data.sort_values('z [m]', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    # endregion

    # region Layer-based processing and correction

    def map_properties(self, layer_profile, cone_profile=DEFAULT_CONE_PROPERTIES,
                       initial_vertical_total_stress=0,
                       vertical_total_stress=None,
                       vertical_effective_stress=None, waterlevel=0,
                       extend_cone_profile=True, extend_layer_profile=True):
        """
        Maps the soil properties defined in the layering and the cone properties to the grid
        defined by the cone data. The procedure also calculates the total and effective vertical stress.
        Note that pre-calculated arrays with total and effective vertical stress can also be supplied to the routine.
        These needs to have the same length as the array with PCPT depth data.

        :param layer_profile: ``SoilProfile`` object with the layer properties (need to contain the soil parameter ``Total unit weight [kN/m3]``
        :param cone_profile: ``SoilProfile`` object with the cone properties (default=``DEFAULT_CONE_PROPERTIES``)
        :param initial_vertical_total_stress: Initial vertical total stress at the highest point of the soil profile
        :param vertical_total_stress: Pre-calculated total vertical stress at PCPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param vertical_effective_stress: Pre-calculated effective vertical stress at PCPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param waterlevel: Waterlevel [m] in the soil (measured from soil surface), default = 0m
        :param extend_cone_profile: Boolean determining whether the cone profile needs to be extended to go to the bottom of the CPT (default = True)
        :param extend_layer_profile: Boolean determining whether the layer profile needs to be extended to the bottom of the CPT (default = True)
        :return: Expands the dataframe `.data` with additional columns for the cone and soil properties
        """
        super().map_properties(
            layer_profile=layer_profile, initial_vertical_total_stress=initial_vertical_total_stress,
            vertical_total_stress=vertical_total_stress, vertical_effective_stress=vertical_effective_stress,
            waterlevel=waterlevel, extend_layer_profile=extend_layer_profile)

        # Validate that cone property boundaries fully contain the CPT info
        if extend_cone_profile:
            if cone_profile[cone_profile.depth_to_col].max() < self.data['z [m]'].max():
                warnings.warn("Cone properties extended to bottom of CPT")
                cone_profile[cone_profile.depth_to_col].iloc[-1] = self.data['z [m]'].max()

        self.coneprofile = cone_profile

        if cone_profile.min_depth > self.data['z [m]'].min():
            raise ValueError(
                "Cone properties starts below minimum CPT depth. " +
                "Ensure that cone profile fully contains CPT data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))
        if cone_profile.max_depth < self.data['z [m]'].max():
            raise ValueError(
                "Cone profile ends above minimum CPT depth. " +
                "Ensure that cone profile fully contains CPT data (%.2fm - %.2fm)" % (
                    self.data['z [m]'].min(), self.data['z [m]'].max()
                ))

        # Map cone properties
        _mapped_cone_props = cone_profile.map_soilprofile(
            self.data['z [m]'], keys_to_map=['area ratio [-]',])

        # Join values to the CPT data
        self.data = self.data.join(_mapped_cone_props.set_index('z [m]'), on='z [m]', lsuffix='_left')

    def downhole_pcpt_corrections(self, area_ratio_override=np.nan):
        """
        Correct the PCPT for downhole effects. Select each push, find the start depth of the push and

        .. math::
            q_c = q_c^* + d \\cdot a \\cdot \\gamma_w

            u_2 = u_2^* + \\gamma_w \\cdot d

        :return:
        """

        if not self.downhole_corrected:
            try:
                self.data['qc raw [MPa]'] = self.data['qc [MPa]']
                self.data['u2 raw [MPa]'] = self.data['u2 [MPa]']
                for _push in self.data['Push'].unique():

                    push_data = self.data[self.data['Push'] == _push]
                    _push_z_min = push_data["z [m]"].min()
                    for i, row in push_data.iterrows():
                        if np.isnan(area_ratio_override):
                            area_ratio = row['area ratio [-]']
                        else:
                            area_ratio = area_ratio_override
                        self.data.loc[i, "qc [MPa]"] = row['qc raw [MPa]'] + \
                            0.001 * _push_z_min * area_ratio * self.waterunitweight
                        self.data.loc[i, "u2 [MPa]"] = row['u2 raw [MPa]'] + \
                            0.001 * self.waterunitweight * _push_z_min
                self.downhole_corrected = True
            except Exception as err:
                raise ValueError("Error during application of downhole corrections - %s" % str(err))
        else:
            warnings.warn("Downhole corrections have already been applied")

    def normalise_pcpt(self, qc_for_rf=False):
        """
        Carries out the necessary normalisation and correction on PCPT data to allow calculation of derived parameters and soil type classification.

        First, the cone resistance is corrected for the unequal area effect using the cone area ratio. The correction for total sleeve friction is not included as it is more uncommon. The procedure assumes that the pore pressure are measured at the shoulder of the cone. If this is not the case, corrections can be used which are not included in this function.

        During normalisation, the friction ratio and pore pressure ratio are calculated. Note that the total cone resistance is used for the friction ratio and pore pressure ratio calculation, the pore pressure ratio calculation also used the total vertical effective stress. The normalised cone resistance and normalised friction ratio are also calculated.

        Finally the net cone resistance is calculated.

        Note that the absence of pore water pressure measurements will lead to NaN values. Reasoning can be used (e.g. presence of rapidly draining layers) to edit the pore pressure data before running this method.

        The total sleeve friction is also calculated. If the cross-sectional area of the friction sleeve is equal at the top and bottom of the sleeve, a correction is not required. This is the default behaviour (cross-sectional areas equal to NaN) but the areas can be entered to calculate the total sleeve friction.

        .. math::
            q_c = q_c^* + d \\cdot a \\cdot \\gamma_w

            q_t = q_c + u_2 \\cdot (1 - a)

            u_2 = u_2^* + \\gamma_w \\cdot d

            \\Delta u_2 = u_2 - u_o

            R_f = \\frac{f_s}{q_t}

            B_q = \\frac{\\Delta u_2}{q_t - \\sigma_{vo}}

            Q_t = \\frac{q_t - \\sigma_{vo}}{\\sigma_{vo}^{\\prime}}

            F_r = \\frac{f_s}{q_t - \\sigma_{vo}}

            q_{net} = q_t - \\sigma_{vo}

            f_t = f_s - u_2 \\frac{A_{sb} - A_{st}}{A_s}

        :param qc_for_rf: Boolean determining whether cone resistance instead of total cone resistance should be used for CPTs where pore pressures are not measured (default=False). If True, :math:`q_t` is replaced by :math:`q_c` in the formula for :math:`R_f`.

        :return: Supplements the PCPT data (`.data`) with the normalised properties (column keys 'qt [MPa]', 'Delta u2 [MPa]', 'Rf [%]', 'Bq [-]', 'Qt [-]', 'Fr [%]', 'qnet [MPa]', 'ft [MPa]'
        """
        try:
            self.data['qt [MPa]'] = self.data['qc [MPa]'] + self.data['u2 [MPa]'] * (1 - self.data['area ratio [-]'])
            self.data['Delta u2 [MPa]'] = self.data['u2 [MPa]'] - 0.001 * self.data["Hydrostatic pressure [kPa]"]
            if qc_for_rf:
                self.data['Rf [%]'] = 100 * self.data['fs [MPa]'] / self.data['qc [MPa]']
            else:
                self.data['Rf [%]'] = 100 * self.data['fs [MPa]'] / self.data['qt [MPa]']
            self.data['Bq [-]'] = self.data['Delta u2 [MPa]'] / (
                    self.data['qt [MPa]'] - 0.001 * self.data["Vertical total stress [kPa]"])
            self.data['Qt [-]'] = (self.data['qt [MPa]'] - 0.001 * self.data["Vertical total stress [kPa]"]) / (
                    0.001 * self.data["Vertical effective stress [kPa]"])
            self.data['Fr [%]'] = 100 * self.data['fs [MPa]'] / (
                    self.data['qt [MPa]'] - 0.001 * self.data["Vertical total stress [kPa]"])
            self.data['qnet [MPa]'] = self.data['qt [MPa]'] - 0.001 * self.data["Vertical total stress [kPa]"]
            for i, row in self.data.iterrows():
                try:
                    self.data.loc[i, "ft [MPa]"] = row['fs [MPa]'] - row['u2 [MPa]'] * (
                        (row['Sleeve cross-sectional area bottom [cm2]'] - row['Sleeve cross-sectional area top [cm2]']) / row['Cone sleeve_area [cm2]']
                    )
                    if np.isnan(self.data.loc[i, "ft [MPa]"]):
                        self.data.loc[i, "ft [MPa]"] = row['fs [MPa]']
                except:
                    self.data.loc[i, "ft [MPa]"] = row['fs [MPa]']

        except Exception as err:
            raise ValueError("Error during calculation of normalised properties."
                             "Review the error message and try again (%s)" % (str(err)))

    # endregion

    # region Data plotting

    def plot_raw_pcpt(self, qc_range=(0, 100), qc_tick=10, fs_range=(0, 1), fs_tick=0.1,
                      u2_range=(-0.5, 2.5), u2_tick=0.5, z_range=None, z_tick=2, rf_range=(0, 5), rf_tick=0.5,
                      show_hydrostatic=True,
                      plot_friction_ratio=False,
                      friction_ratio_panel=3,
                      plot_height=700, plot_width=1000, return_fig=False,
                      plot_title=None, plot_margin=dict(t=100, l=50, b=50), color=None,
                      hydrostaticcolor=None, show_hydrostatic_legend=False, waterlevel_override=0, latex_titles=True):
        """
        Plots the raw PCPT data using the Plotly package. This generates an interactive plot.

        :param qc_range: Range for the cone tip resistance (default=(0, 100MPa))
        :param qc_tick: Tick interval for the cone tip resistance (default=10MPa)
        :param fs_range: Range for the sleeve friction (default=(0, 1MPa))
        :param fs_tick: Tick interval for sleeve friction (default=0.1MPa)
        :param u2_range: Range for the pore pressure at the shoulder (default=(-0.1, 0.5MPa))
        :param u2_tick: Tick interval for the pore pressure at the shoulder (default=0.05MPa)
        :param z_range: Range for the depth (default=None for plotting from zero to maximum cone penetration)
        :param z_tick: Tick interval for depth (default=2m)
        :param show_hydrostatic: Boolean determining whether hydrostatic pressure is shown on the pore pressure plot panel
        :param rf_range: Range for the friction ratio, if used (default=None for plotting from zero to 5%)
        :param rf_tick: Tick interval for friction ratio, if used (default=0.5%)
        :param show_hydrostatic: Boolean determining whether hydrostatic pressure is shown on the pore pressure plot panel
        :param plot_friction_ratio: Boolean determining whether friction ratio needs to be plotted (default=False)
        :param friction_ratio_panel: Panel for plotting friction ratio (default=3 for pore pressure panel). Only used when ``plot_friction_ratio=True``
        :param plot_height: Height for the plot (default=700px)
        :param plot_width: Width for the plot (default=1000px)
        :param return_fig: Boolean determining whether the figure needs to be returned (True) or plotted (False)
        :param plot_title: Plot for the title (default=None)
        :param plot_margin: Margin for the plot (default=dict(t=100, l=50, b=50))
        :param color: Color to be used for plotting (default=None for default plotly colors)
        :param color: Color to be used for plotting the hydrostatic pressure (default=None for default plotly colors)
        :param show_hydrostatic_legend: Boolean determining whether to show the hydrostatic pressure in the legend
        :param waterlevel_override: If a waterlevel is not specified in a soil profile which is mapped to the CPT, this can be used to get the water table at the correct elevation (default=0m for water level at the surface, >0 for water level below groundlevel)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return:
        """
        if latex_titles:
            qc_trace_title = r'$ q_c $'
            fs_trace_title = r'$ f_s $'
            u2_trace_title = r'$ u_2 $'
            Rf_trace_title = r'$ R_f $'
            uhydrostatic_trace_title = r'$ u_{hydrostatic} $'
            qc_axis_title = r'$ q_c \ \text{[MPa]} $'
            fs_axis_title = r'$ f_s \ \text{[MPa]} $'
            u2_axis_title = r'$ u_2 \ \text{[MPa]} $'
            Rf_axis_title = r'$ R_f \ \text{[%]} $'
            z_axis_title = r'$ \text{Depth below mudline, } z \ \text{[m]} $'
        else:
            qc_trace_title = 'qc'
            fs_trace_title = 'fs'
            u2_trace_title = 'u2'
            Rf_trace_title = 'Rf'
            uhydrostatic_trace_title = 'u,hydrostatic'
            qc_axis_title = 'qc [MPa]'
            fs_axis_title = 'fs [MPa]'
            u2_axis_title = 'u2 [MPa]'
            Rf_axis_title = 'Rf [%]'
            z_axis_title = 'z [m]'

        if z_range is None:
            z_range = (self.data['z [m]'].max(), 0)
        if color is None:
            color = DEFAULT_PLOTLY_COLORS[0]
        if hydrostaticcolor is None:
            hydrostaticcolor = DEFAULT_PLOTLY_COLORS[1]

        fig = subplots.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

        for _push in self.data["Push"].unique():
            push_data = self.data[self.data["Push"] == _push]
            trace1 = go.Scatter(
                x=push_data['qc [MPa]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=qc_trace_title)
            
            trace2 = go.Scatter(
                x=push_data['fs [MPa]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=fs_trace_title)
            trace3 = go.Scatter(
                x=push_data['u2 [MPa]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=u2_trace_title)
            trace_rf = go.Scatter(
                    x=100 * push_data['fs [MPa]'] / push_data['qc [MPa]'],
                    y=push_data['z [m]'],
                    line=dict(color=color),
                    showlegend=False, mode='lines', name=Rf_trace_title)

            if plot_friction_ratio:
                if friction_ratio_panel == 1:
                    fig.append_trace(trace_rf, 1, 1)
                    fig.append_trace(trace2, 1, 2)
                    fig.append_trace(trace3, 1, 3)
                elif friction_ratio_panel == 2:
                    fig.append_trace(trace1, 1, 1)
                    fig.append_trace(trace_rf, 1, 2)
                    fig.append_trace(trace3, 1, 3)
                elif friction_ratio_panel == 3:
                    fig.append_trace(trace1, 1, 1)
                    fig.append_trace(trace2, 1, 2)
                    fig.append_trace(trace_rf, 1, 3)
                else:
                    raise ValueError("friction_ratio_panel needs to be equal to 1, 2 or 3")
            else:
                fig.append_trace(trace1, 1, 1)
                fig.append_trace(trace2, 1, 2)
                fig.append_trace(trace3, 1, 3)
        if show_hydrostatic:
            if plot_friction_ratio and friction_ratio_panel == 3:
                pass
            else:
                try:
                    waterlevel = self.waterlevel
                    if waterlevel is None:
                        waterlevel = waterlevel_override
                except:
                    waterlevel = waterlevel_override
                
                self.waterpressures = 0.001 * (self.data['z [m]'] - waterlevel) * self.waterunitweight
                self.positivewaterpressures = np.array([0 if i < 0 else i for i in self.waterpressures])

                trace3 = go.Scatter(
                    x=self.positivewaterpressures,
                    y=self.data['z [m]'],
                    line=dict(color=hydrostaticcolor, dash='dot'),
                    showlegend=show_hydrostatic_legend, mode='lines', name=uhydrostatic_trace_title)
                fig.append_trace(trace3, 1, 3)
        # Plot layers
        try:
            for i, row in self.layerdata.iterrows():
                if i > 0:
                    layer_trace_qc = go.Scatter(
                        x=qc_range,
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_qc, 1, 1)
                    layer_trace_fs = go.Scatter(
                        x=fs_range,
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_fs, 1, 2)
                    layer_trace_u2 = go.Scatter(
                        x=u2_range,
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_u2, 1, 3)
        except:
            pass

        fig['layout']['xaxis1'].update(
            title=qc_axis_title, side='top', anchor='y',
            range=qc_range, dtick=qc_tick)
        fig['layout']['xaxis2'].update(
            title=fs_axis_title, side='top', anchor='y',
            range=fs_range, dtick=fs_tick)
        fig['layout']['xaxis3'].update(
            title=u2_axis_title, side='top', anchor='y',
            range=u2_range, dtick=u2_tick)

        if plot_friction_ratio:
            if friction_ratio_panel == 1:
                fig['layout']['xaxis1'].update(
                    title=Rf_axis_title, range=rf_range, dtick=rf_tick)
            elif plot_friction_ratio and friction_ratio_panel == 2:
                fig['layout']['xaxis2'].update(
                    title=Rf_axis_title, range=rf_range, dtick=rf_tick)
            elif plot_friction_ratio and friction_ratio_panel == 3:
                fig['layout']['xaxis3'].update(
                    title=Rf_axis_title, range=rf_range, dtick=rf_tick)
            else:
                raise ValueError("friction_ratio_panel needs to be equal to 1, 2 or 3")
        else:
            pass
        
        fig['layout']['yaxis1'].update(
            title=z_axis_title, autorange='reversed',
            range=z_range, dtick=z_tick)
        
        fig['layout'].update(
            title=plot_title, hovermode='closest',
            height=plot_height, width=plot_width,
            margin=plot_margin)
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)


    def plot_normalised_pcpt(
            self, qt_range=(0, 3), fr_range=(-1, 1),
            bq_range=(-0.6, 1.4), bq_tick=0.2, z_range=None, z_tick=2,
            plot_height=700, plot_width=1000, color=None, return_fig=False, plot_title=None, latex_titles=True):
        """
        Plots the normalised PCPT properties vs depth.

        :param qt_range: Range for Qt (optional, default is (0, 3) for 1 to 1000 on log scale)
        :param fr_range: Range for Fr (optional, default is (-1, 1) for 0.1 to 10 on log scale)
        :param bq_range: Range for Bq (optional, default is (-0.6, 1.4))
        :param bq_tick: Tick mark interval for Bq (optional, default=0.2)
        :param z_range: Range for depths (optional, default is (0, maximum PCPT depth)
        :param z_tick: Tick mark distance for PCPT (optional, default=2)
        :param plot_height: Height of the plot in pixels
        :param plot_width: Width of the plot in pixels
        :param return_fig: Boolean determining whether the figure is returned or the plot is generated; Default behaviour is to generate the plot.
        :param plot_title: Plot title (optional, default=None for no title)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Returns the figure if return_fig=True. Otherwise the plot is displayed.
        """
        if latex_titles:
            Qt_trace_title = r'$ Q_t $'
            Fr_trace_title = r'$ F_r $'
            Bq_trace_title = r'$ B_q $'
            Qt_axis_title = r'$ Q_t \ \text{[-]} $'
            Fr_axis_title = r'$ F_r \ \text{[%]} $'
            Bq_axis_title = r'$ B_q \ \text{[-]} $'
            z_axis_title = r'$ z \ \text{[m]}$'
        else:
            Qt_trace_title = 'Qt'
            Fr_trace_title = 'Fr'
            Bq_trace_title = 'Bq'
            Qt_axis_title = 'Qt [-]'
            Fr_axis_title = 'Fr [%]'
            Bq_axis_title = 'Bq [-]'
            z_axis_title = 'z [m]'

        if z_range is None:
            z_range = (self.data['z [m]'].max(), 0)
        if color is None:
            color = DEFAULT_PLOTLY_COLORS[0]

        fig = subplots.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)
        for _push in self.data["Push"].unique():
            push_data = self.data[self.data["Push"] == _push]
        
            trace1 = go.Scatter(
                x=push_data['Qt [-]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=Qt_trace_title)
            fig.append_trace(trace1, 1, 1)
            trace2 = go.Scatter(
                x=push_data['Fr [%]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=Fr_trace_title)
            fig.append_trace(trace2, 1, 2)
            trace3 = go.Scatter(
                x=push_data['Bq [-]'],
                y=push_data['z [m]'],
                line=dict(color=color),
                showlegend=False, mode='lines', name=Bq_trace_title)
            fig.append_trace(trace3, 1, 3)
        # Plot layers
        try:
            for i, row in self.layerdata.iterrows():
                if i > 0:
                    layer_trace_qt = go.Scatter(
                        x=(10 ** qt_range[0], 10 ** qt_range[1]),
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_qt, 1, 1)
                    layer_trace_fr = go.Scatter(
                        x=(10 ** fr_range[0], 10 ** fr_range[1]),
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_fr, 1, 2)
                    layer_trace_bq = go.Scatter(
                        x=bq_range,
                        y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                        line=dict(color='black', dash='dot'),
                        showlegend=False, mode='lines')
                    fig.append_trace(layer_trace_bq, 1, 3)
        except:
            pass
        fig['layout']['xaxis1'].update(
            title=Qt_axis_title, side='top', anchor='y', type='log',
            range=qt_range)
        fig['layout']['xaxis2'].update(
            title=Fr_axis_title, side='top', anchor='y', type='log',
            range=fr_range)
        fig['layout']['xaxis3'].update(
            title=Bq_axis_title, side='top', anchor='y',
            range=bq_range, dtick=bq_tick)
        fig['layout']['yaxis1'].update(
            title=z_axis_title, range=z_range, dtick=z_tick)
        fig['layout'].update(
            title=plot_title,
            height=plot_height, width=plot_width,
            margin=dict(t=100, l=50, b=50))
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    def plot_properties(
            self, prop_keys, plot_ranges, plot_ticks, z_range=None, z_tick=2,
            legend_titles=None, axis_titles=None, showlegends=None, plot_layers=True,
            plot_height=700, plot_width=1000, colors=None, return_fig=False, plot_title=None, latex_titles=True):
        """
        Plots the soil and/or PCPT properties vs depth.

        :param prop_keys: Tuple of tuples with the keys to be plotted. Keys in the same tuple are plotted on the same panel
        :param plot_ranges: Tuple of tuples with ranges for the panels of the plot
        :param plot_ticks: Tuple with tick intervals for the plot panels
        :param z_range: Range for depths (optional, default is (0, maximum PCPT depth)
        :param z_tick: Tick mark distance for PCPT (optional, default=2)
        :param legend_titles: Tuple with entries to be used in the legend. If left blank, the keys are used
        :param axis_titles: Tuple with entries to be used as axis labels. If left blank, the keys are used
        :param showlegends: Array of booleans determining whether or not to show the trace in the legend
        :param plot_layers: Boolean determining whether layers are plotted (default=True)
        :param plot_height: Height of the plot in pixels
        :param plot_width: Width of the plot in pixels
        :param return_fig: Boolean determining whether the figure is returned or the plot is generated; Default behaviour is to generate the plot.
        :param plot_title: Plot title (optional, default=None for no title)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Returns the figure if return_fig=True. Otherwise the plot is displayed.
        """
        if z_range is None:
            z_range = (self.data['z [m]'].max(), 0)
        if colors is None:
            colors = DEFAULT_PLOTLY_COLORS
        if legend_titles is None:
            legend_titles = prop_keys
        if axis_titles is None:
            axis_titles = prop_keys

        _showlegends = []
        for i, _x in enumerate(prop_keys):
            _showlegends_panel = []
            for j, _trace_x in enumerate(_x):
                _showlegends_panel.append(True)
        _showlegends.append(_showlegends_panel)

        if showlegends is None:
            showlegends = _showlegends

        fig = subplots.make_subplots(rows=1, cols=prop_keys.__len__(), print_grid=False, shared_yaxes=True)
        for i, _props in enumerate(prop_keys):
            for j, _prop in enumerate(_props):
                for k, _push in enumerate(self.data["Push"].unique()):
                    if k == 0:
                        try:
                            showlegend = showlegends[i][j]
                        except:
                            showlegend = False
                    else:
                        showlegend = False
                    push_data = self.data[self.data["Push"] == _push]
                    trace = go.Scatter(
                        x=push_data[_prop],
                        y=push_data['z [m]'],
                        line=dict(color=colors[j]),
                        showlegend=showlegend, mode='lines', name=legend_titles[i][j])
                    fig.append_trace(trace, 1, i+1)
        # Plot layers
        if plot_layers:
            try:
                for i, row in self.layerdata.iterrows():
                    if i > 0:
                        for j, _range in enumerate(plot_ranges):
                            layer_trace = go.Scatter(
                                x=_range,
                                y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                                line=dict(color='black', dash='dot'),
                                showlegend=False, mode='lines')
                            fig.append_trace(layer_trace, 1, j+1)
            except:
                pass

        for i, _range in enumerate(plot_ranges):
            fig['layout']['xaxis%i' % (i+1)].update(
                title=axis_titles[i], side='top', anchor='y',
                range=_range, dtick=plot_ticks[i])
        
        if latex_titles:
            z_axis_title = r'$ z \ \text{[m]}$'
        else:
            z_axis_title = 'z [m]'

        fig['layout']['yaxis1'].update(
            title=z_axis_title, range=z_range, dtick=z_tick)
        fig['layout'].update(
            title=plot_title,
            height=plot_height, width=plot_width,
            margin=dict(t=100, l=50, b=50))
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    def plot_properties_withlog(self, prop_keys, plot_ranges, plot_ticks,
            legend_titles=None, axis_titles=None, showfig=True, showlayers=True, **kwargs):
        """
        Plots CPT properties vs depth and includes a mini-log on the left-hand side.
        The minilog is composed based on the entries in the ``Soil type`` column of the layering
        :param prop_keys: Tuple of tuples with the keys to be plotted. Keys in the same tuple are plotted on the same panel
        :param plot_ranges: Tuple of tuples with ranges for the panels of the plot
        :param plot_ticks: Tuple with tick intervals for the plot panels
        :param z_range: Range for depths (optional, default is (0, maximum PCPT depth)
        :param z_tick: Tick mark distance for PCPT (optional, default=2)
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

        for _panel in prop_keys:
            _x_panel = []
            _z_panel = []
            for _prop in _panel:
                _x_data = np.array([])
                _z_data = np.array([])
                for _push in self.data["Push"].unique():
                    push_data = self.data[self.data["Push"] == _push]
                    _x_data = np.append(_x_data, np.nan)
                    _z_data = np.append(_z_data, np.nan)
                    _x_data = np.append(_x_data, push_data[_prop])
                    _z_data = np.append(_z_data, push_data['z [m]'])
                _x_panel.append(_x_data)
                _z_panel.append(_z_data)
            _x.append(_x_panel)
            _z.append(_z_panel)

        fig = plot_with_log(
            x=_x,
            z=_z,
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

    def plot_robertson_chart(self, start_depth=None, end_depth=None,
                             qt_range=(0, 3), fr_range=(-1, 1),
            bq_range=(-0.6, 1.4), bq_tick=0.2,
            plot_height=700, plot_width=1000, return_fig=False, plot_title=None,
            backgroundimagedir="", latex_titles=True):
        """
        Plots the normalised PCPT points in the Robertson chart to distinguish the soil type. The display can be limited
        to a specific depth range (by specifying `start_depth` and `end_depth`. The color coding is based on the layer.
        :return: Returns the figure if return_fig=True. Otherwise the plot is displayed.
        """
        frqt_img = PIL.Image.open(os.path.join(IMG_DIR, 'robertsonFr.png'))
        frbq_img = PIL.Image.open(os.path.join(IMG_DIR, 'robertsonBq.png'))

        if start_depth is None:
            start_depth = self.data["z [m]"].min()
        if end_depth is None:
            end_depth = self.data["z [m]"].max()

        selected_data = self.data[(self.data["z [m]"] >= start_depth) & (self.data["z [m]"] <= end_depth)]

        fig = subplots.make_subplots(rows=1, cols=2, print_grid=False)
        for i, layer in enumerate(selected_data["Layer no"].unique()):
            color = DEFAULT_PLOTLY_COLORS[i % DEFAULT_PLOTLY_COLORS.__len__()]
            layer_data = selected_data[selected_data["Layer no"] == layer]
            layer_min_depth = layer_data['z [m]'].min()
            layer_max_depth = layer_data['z [m]'].max()
            friction_trace = go.Scatter(
                x=layer_data['Fr [%]'],
                y=layer_data['Qt [-]'],
                showlegend=True,  # Don't show the legend
                mode='markers',
                name="Layer %i - %.2fm - %.2fm" % (layer, layer_min_depth, layer_max_depth),
                text=layer_data["z [m]"],
                marker=dict(size=10,  # Make markers transparent (last number is 0)
                            opacity=0.5,  # Add some opacity for better display
                            color=color,
                            line=dict(width=1, color=color)))  # Add a line around the markers
            fig.append_trace(friction_trace, 1, 1)
            pressure_trace = go.Scatter(
                x=layer_data['Bq [-]'],
                y=layer_data['Qt [-]'],
                showlegend=False,
                mode='markers',
                text=layer_data["z [m]"],
                marker=dict(size=10,   # Make markers transparent (last number is 0)
                            opacity=0.5,  # Add some opacity for better display
                            color=color,
                            line=dict(width=1, color=color)))  # Add a line around the markers
            fig.append_trace(pressure_trace, 1, 2)

        if latex_titles:
            Fr_axis_title = r'$ F_r \ \text{[%]} $'
            Qt_axis_title = r'$ Q_t \ \text{[-]} $'
            Bq_axis_title = r'$ B_q \ \text{[-]} $'
        else:
            Fr_axis_title = 'Fr [%]'
            Qt_axis_title = 'Qt [-]'
            Bq_axis_title = 'Bq [-]'

        fig['layout']['xaxis1'].update(title=Fr_axis_title, range=fr_range, type='log')
        fig['layout']['yaxis1'].update(title=Qt_axis_title, range=qt_range, type='log')
        fig['layout']['xaxis2'].update(title=Bq_axis_title, range=bq_range, dtick=bq_tick)
        fig['layout']['yaxis2'].update(title=Qt_axis_title, range=qt_range, type='log')
        fig['layout'].update(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            hovermode='closest',
            images=[
                # Image for Qt-Fr relation
                dict(
                    source=frqt_img,
                    xref='x',
                    yref='y',
                    x=-1,
                    y=3,
                    sizex=2,
                    sizey=3,
                    sizing='stretch',
                    opacity=0.7,
                    layer='below',
                ),
                # Image for Qt-Bq relation
                dict(
                    source=frbq_img,
                    xref='x2',
                    yref='y2',
                    x=-0.6,
                    y=3,
                    sizex=2,
                    sizey=3,
                    sizing='stretch',
                    opacity=0.7,
                    layer='below',
                )
            ]
        )
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    # endregion

    # region Correlations
    
    def apply_correlation(self, name, outputs, apply_for_soiltypes='all', **kwargs):
        """
        Applies a correlation to the given PCPT data. The name of the correlation needs to be chosen from the following available correlations.
        Each correlation corresponds to a function in the `pcpt` module.
        By default, the correlation is applied to the entire depth range. However, a restriction on the soil
        types to which the correlation can be applied can be specified with the `apply_for_soiltypes` keyword argument.
        A list with the soil types for which the correlation needs to be applied can be provided.

            - Ic Robertson and Wride (1998) (`behaviourindex_pcpt_robertsonwride`) - Calculation of soil behaviour type index from cone tip resistance and sleeve friction
            - Isbt Robertson (2010) (`behaviourindex_pcpt_nonnormalised`) - Calculation of non-normalised soil behaviour type index from cone tip resistance and friction ratio
            - Gmax Rix and Stokoe (1991) (`gmax_sand_rixstokoe`) - Calculation of small-strain shear modulus for uncemented silica sand from cone tip resistance and vertical effective stress
            - Gmax Mayne and Rix (1993) (`gmax_clay_maynerix`) - Calculation of small-strain shear modulus for clay from cone tip resistance
            - Dr Baldi et al (1986) - NC sand (`relativedensity_ncsand_baldi`) - Calculation of relative density of normally consolidated silica sand
            - Dr Baldi et al (1986) - OC sand (`relativedensity_ocsand_baldi`) - Calculation of relative density of overconsolidated silica sand
            - Dr Jamiolkowski et al (2003) (`relativedensity_sand_jamiolkowski`) - Calculation of relative density dry and saturated silica sand
            - Friction angle Kulhawy and Mayne (1990) (`frictionangle_sand_kulhawymayne`) - Calculation of effective friction angle for sand
            - Su Rad and Lunne (1988): (`undrainedshearstrength_clay_radlunne`) - Calculation of undrained shear strength for clay based on empirical cone factor Nk
            - Friction angle Kleven (1986): (`frictionangle_overburden_kleven`) - Calculation of friction angle for North Sea sands at various stress levels and relative densities
            - OCR Lunne (1989): (`ocr_cpt_lunne`) - Calculation of OCR for clay
            - Sensitivity Rad and Lunne (1986): (`sensitivity_frictionratio_lunne`) - Calculation of sensitivity for clay
            - Unit weight Mayne et al (2010): (`unitweight_mayne`) - Calculation of total unit weight
            - Shear wave velocity Robertson and Cabal (2015): (`vs_ic_robertsoncabal`) - Calculation of shear wave velocity for all soil types
            - K0 Mayne (2007) - sand: (`k0_sand_mayne`) - Calculation of coefficient of lateral earth pressure for sand
            - Es Bellotti (1989) - sand: (`drainedsecantmodulus_sand_bellotti`) - Calculation of drained modulus at average strain of 0.1pct for sand

        Note that certain correlations require either the calculation of normalised properties or application of preceding correlations

        :param name: Name of the correlation according to the list defined above
        :param outputs: a dict of keys and values where keys are the same as the keys in the correlation, values are the table headers you want
        :param apply_for_soiltypes: List with soil types to which the correlation needs the be applied.
        :param kwargs: Optional keyword arguments for the correlation.
        :return: Adds a column with key `outkey` to the dataframe with PCPT data
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

    # region Design lines
    def load_design_profile(self, design_data):
        """
        Loads a design profile for the parameters available in the `data` attribute of the PCPTProcessing object.
        This design profile can then be plotted vs the PCPT data.

        :param design_data: Pandas dataframe with design soil profile. Linear variation of numerical parameters is expected so for example, use columns `Su from [kPa]` and `Su to [kPa]`
        :return: Sets the `designprofile` attribute of the PCPTProcessing object
        """
        warnings.warn("Deprecated: Will be removed in v0.15. Use SoilProfiles and LogPlot instead.")
        self.designprofile = design_data

    def plot_design_profile(self, prop_keys, design_keys, plot_ranges, plot_ticks, z_range=None, z_tick=2,
            legend_titles=None, axis_titles=None,
            plot_height=700, plot_width=1000, colors=None, design_color='red', design_dash='dot',
                            return_fig=False, plot_title=None, latex_titles=True):
        """
        Plots the soil and/or PCPT properties vs depth.

        :param prop_keys: Tuple of tuples with the keys to be plotted. Keys in the same tuple are plotted on the same panel
        :param design_keys: Tuple of tuples with the design profile keys to be plotted. Keys in the same tuple are plotted on the same panel (note that `()` is an empty tuple). The design keys need to be specified as for example `Gmax [kPa]` if the `designprofile` attribute contains keys `Gmax from [kPa]` and `Gmax to [kPa]`
        :param plot_ranges: Tuple of tuples with ranges for the panels of the plot
        :param plot_ticks: Tuple with tick intervals for the plot panels
        :param z_range: Range for depths (optional, default is (0, maximum PCPT depth)
        :param z_tick: Tick mark distance for PCPT (optional, default=2)
        :param legend_titles: Tuple with entries to be used in the legend. If left blank, the keys are used
        :param axis_titles: Tuple with entries to be used as axis labels. If left blank, the keys are used
        :param plot_height: Height of the plot in pixels
        :param plot_width: Width of the plot in pixels
        :param colors: Color list to be used in the plotting panels
        :param design_color: Color to be used for the design lines
        :param design_dash: Dash style to be used for the design lines
        :param return_fig: Boolean determining whether the figure is returned or the plot is generated; Default behaviour is to generate the plot.
        :param plot_title: Plot title (optional, default=None for no title)
        :param latex_titles: Boolean determining whether axis titles should be shown as LaTeX (default = True)
        :return: Returns the figure if return_fig=True. Otherwise the plot is displayed.
        """
        warnings.warn("Deprecated: Will be removed in v0.15. Use SoilProfiles and LogPlot instead.")
        if z_range is None:
            z_range = (self.data['z [m]'].max(), 0)
        if colors is None:
            colors = DEFAULT_PLOTLY_COLORS
        if legend_titles is None:
            legend_titles = prop_keys
        if axis_titles is None:
            axis_titles = prop_keys

        fig = subplots.make_subplots(rows=1, cols=prop_keys.__len__(), print_grid=False, shared_yaxes=True)
        for i, _props in enumerate(prop_keys):
            for j, _prop in enumerate(_props):
                trace = go.Scatter(
                    x=self.data[_prop],
                    y=self.data['z [m]'],
                    line=dict(color=colors[j]),
                    showlegend=False, mode='lines', name=legend_titles[i][j])
                fig.append_trace(trace, 1, i+1)

        for i, _props in enumerate(design_keys):
            for j, _prop in enumerate(_props):
                try:
                    keybase = re.split(" \[", _prop)
                    from_key = "%s from [%s" % (keybase[0], keybase[1])
                    to_key = "%s to [%s" % (keybase[0], keybase[1])
                    design_trace = go.Scatter(
                        x=np.insert(
                            np.array(self.designprofile[to_key]),
                            np.arange(len(self.designprofile)),
                            np.array(self.designprofile[from_key])),
                        y=np.insert(
                            np.array(self.designprofile["z to [m]"]),
                            np.arange(len(self.designprofile)),
                            np.array(self.designprofile["z from [m]"])),
                        line=dict(color=design_color, dash=design_dash),
                        showlegend=False, mode='lines', name=keybase[0])
                    fig.append_trace(design_trace, 1, i+1)
                except:
                    pass

        # Plot layers
        try:
            for i, row in self.layerdata.iterrows():
                if i > 0:
                    for i, _range in enumerate(plot_ranges):
                        layer_trace = go.Scatter(
                            x=_range,
                            y=(row[self.layerdata.depth_from_col], row[self.layerdata.depth_from_col]),
                            line=dict(color='black', dash='dot'),
                            showlegend=False, mode='lines')
                        fig.append_trace(layer_trace, 1, i+1)
        except:
            pass
        for i, _range in enumerate(plot_ranges):
            fig['layout']['xaxis%i' % (i+1)].update(
                title=axis_titles[i], side='top', anchor='y',
                range=_range, dtick=plot_ticks[i])
            
        if latex_titles:
            z_axis_title = r'$ z \ \text{[m]}$'
        else:
            z_axis_title = 'z [m]'

        fig['layout']['yaxis1'].update(
            title=z_axis_title, range=z_range, dtick=z_tick)
        fig['layout'].update(
            title=plot_title,
            height=plot_height, width=plot_width,
            margin=dict(t=100, l=50, b=50))
        if return_fig:
            return fig
        else:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    def select_layering(
            self, default_soil_type="Unknown", default_unit_weight=20,
            qc_range=(-10, 100), fs_range=(0, 1), u2_range=(-0.5, 2.5),
            waterlevel=0, **kwargs):
        """
        Selects the layering for a CPT trace. The routine creates a LogPlotMatplotlib with which the
        layering can be selected interactively. 
        Ensure that the plotting option is set to ``%matplotlib qt`` in Jupyter.
        Clicking below 0MPa in the cone tip resistance panel stops the selection.
        The SoilProfile with the layering is returned. The ``Soil type`` and ``Total unit weight [kN/m3]`` columns
        still need to be fine-tuned by the user after the selection process. Default values are assigned according to
        the ``default_soil_type`` and ``default_unit_weight`` arguments.
        The plotting ranges can be finetuned using the ``qc_range``, ``fs_range`` and ``u2_range`` arguments.
        A hydrostatic line is plotted in the pore pressure panel. A water level (default=0m) can be added for plotting this line.

        Keyword arguments for ``LogPlotMatplotlib`` can be specified by the user and are passed as **kwargs.
        """
        # Create the default SoilProfile for parameter selection
        sp = SoilProfile({
            "Depth from [m]": [self.data['z [m]'].min(),],
            "Depth to [m]": [self.data['z [m]'].max(),],
            "Soil type": [default_soil_type,],
            "Total unit weight [kN/m3]": [default_unit_weight,]
        })
        # Create the plot for parameter selection
        selection_plot = LogPlotMatplotlib(
            sp, no_panels=3, **kwargs)
        selection_plot.add_trace(
            x=self.data['qc [MPa]'],
            z=self.data['z [m]'],
            name='qc',
            panel_no=1
        )
        selection_plot.add_trace(
            x=self.data['fs [MPa]'],
            z=self.data['z [m]'],
            name='fs',
            panel_no=2
        )
        selection_plot.add_trace(
            x=self.data['u2 [MPa]'],
            z=self.data['z [m]'],
            name='u2',
            panel_no=3
        )
        selection_plot.add_trace(
            x=[waterlevel, (self.data['z [m]'].max() - waterlevel) * 0.01],
            z=[waterlevel, self.data['z [m]'].max()],
            name='u2',
            panel_no=3,
            ls='--',
        )
        selection_plot.set_xaxis_title(title='qc [MPa]', panel_no=1)
        selection_plot.set_xaxis_title(title='fs [MPa]', panel_no=2)
        selection_plot.set_xaxis_title(title='u2 [MPa]', panel_no=3)
        selection_plot.set_xaxis_range(min_value=qc_range[0], max_value=qc_range[1], panel_no=1)
        selection_plot.set_xaxis_range(min_value=fs_range[0], max_value=fs_range[1], panel_no=2)
        selection_plot.set_xaxis_range(min_value=u2_range[0], max_value=u2_range[1], panel_no=3)
        selection_plot.set_zaxis_title(title='z [m]')
        selection_plot.show()
        selection_plot.select_layering()

        return sp

    # endregion

    # region Data export

    def to_json(self, write_file=False, output_path=None):
        """
        Write the PCPT object to a JSON file. JSON can either be returned or a JSON file can be written.

        :param write_file: Boolean determining whether a file is written or not (default=False). If True, a file is written to `output_path`
        :param output_path: A valid path to the output .json file (include the file suffix).
        :return: If no file is returned, the JSON containing the location and data of the PCPT is returned.
        """
        dict_pcpt = {
            'location': {
                'easting': self.easting,
                'northing': self.northing,
                'waterdepth': self.elevation,
                'srid': self.srid,
                'datum': self.datum
            },
            'waterlevel': self.waterlevel,
            'coneprops': self.coneprofile.to_json(),
            'layering': self.layerdata.to_json(),
            'data': self.data.to_json()
        }
        if write_file:
            with open(output_path, 'w') as outfile:
                json.dump(dict_pcpt, outfile)
        else:
            return json.dumps(dict_pcpt)

    def to_excel(self, output_path):
        """
        Write the PCPT object to an Excel file.
        The Excel file contains multiple sheet for the location, layer data, cone properties and raw data.

        :param output_path: A valid path to the output .xlsx file (include the file suffix).
        :return: The file is written to the specified location
        """
        # Create some Pandas dataframes from some data.
        loc_df = pd.DataFrame(
            {'Easting': [self.easting,],
             'Northing': [self.northing,],
             'Elevation [m]': [self.elevation,],
             'Water level [m]': [self.waterlevel,],
             'srid': [self.srid,],
             'datum': [self.datum,]
            })

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path, engine='openpyxl')

        # Write each dataframe to a different worksheet.
        loc_df.to_excel(writer, sheet_name='Location data', index=False)
        self.coneprofile.to_excel(writer, sheet_name='Cone properties', index=False)
        self.layerdata.to_excel(writer, sheet_name='Layering', index=False)
        self.data.to_excel(writer, sheet_name='Data', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
    # endregion


def plot_longitudinal_profile(
    cpts=[], latlon=False,
    option='name', start=None, end=None, band=1000, extend_profile=False,
    plotmap=False,
    uniformcolor=None,
    prop='qc [MPa]',
    distance_unit='m', scale_factor=0.001,
    showfig=True, xaxis_layout=None, yaxis_layout=None, general_layout=None, legend_layout=None,
    show_annotations=True, mapbox_zoom=10):
    """
    Creates a longitudinal profile along selected CPTs. A line is drawn from the first (smallest distance from origin)
    to the last location (greatest distance from origin) and the plot of the selected parameter (``prop``) vs depth
    is projected onto this line.

    :param cpts: List with PCPTProcessing objects to be plotted
    :param latlon: Boolean determining whether latitude and longitude are used or easting and northing in m (default=False for easting and northing in m)
    :param option: Determines whether CPT names (``option='name'``) or tuples with coordinates (``option='coords'``) are used for the ``start`` and ``end`` arguments
    :param start: CPT name for the starting point or tuple of coordinates. If a CPT name is used, the selected CPT must be contained in ``cpts``.
    :param end: CPT name for the end point or tuple of coordinates. If a CPT name is used, the selected CPT must be contained in ``cpts``.
    :param band: Offset from the line connecting start and end points in which CPT are considered for plotting (default=1000m)
    :param extend_profile: Boolean determining whether the profile needs to be extended beyond the start and end points (default=False)
    :param plotmap: Boolean determining whether a map of locations needs to be plotted next to the profile (default=False)
    :param uniformcolor: Uniform color to use for all CPT traces (default=None for different color for each trace)
    :param prop: Selected property for plotting (default='qc [MPa]')
    :param distance_unit: Unit for coordinates and elevation (default='m')
    :param scale_factor: Scale factor for the property (default=0.001)
    :param showfig: Boolean determining whether the figure is shown (default=True)
    :param xaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param yaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param general_layout: Dictionary with general layout options (default=None)
    :param legend_layout: Dictionary with legend layout options (default=None)
    :param show_annotations: Boolean determining whether annotations need to be shown (default=True)
    :param mapbox_zoom: Zoom factor for map (if plotted, default=10)
    :return: Plotly figure object
    """

    cpt_names = []
    x_coords = []
    y_coords = []
    elevations = []
    for _cpt in cpts:
        try:
            x_coords.append(_cpt.easting)
            y_coords.append(_cpt.northing)
            elevations.append(_cpt.elevation)
            cpt_names.append(_cpt.title)
        except Exception as err:
            warnings.warn(
                "CPT %s - Error during processing for profile - %s" % (_cpt.title, str(err)))
            raise

    if option == 'name':
        start_point = (x_coords[cpt_names.index(start)], y_coords[cpt_names.index(start)])
        end_point = (x_coords[cpt_names.index(end)], y_coords[cpt_names.index(end)])
    elif option == 'coords':
        if start.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        start_point = start
        if end.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        end_point = end
    else:
        raise ValueError("option should be 'name' or 'coords'")

    cpt_df = pd.DataFrame({
        'CPT objects': cpts,
        'CPT titles': cpt_names,
        'X': x_coords,
        'Y': y_coords,
        'Z': elevations
    })
    for i, row in cpt_df.iterrows():
        if row['X'] == start_point[0] and row['Y'] == start_point[1]:
            cpt_df.loc[i, "Offset"] = 0
            cpt_df.loc[i, "Projected offset"] = 0
            cpt_df.loc[i, "Before start"] = False
            cpt_df.loc[i, "Behind end"] = False
        elif row['X'] == end_point[0] and row['Y'] == end_point[1]:
            cpt_df.loc[i, "Offset"] = 0
            if latlon:
                cpt_df.loc[i, "Projected offset"] = latlon_distance(
                    start_point[0], start_point[1], end_point[0], end_point[1])
            else:
                cpt_df.loc[i, "Projected offset"] = np.sqrt(
                    (start_point[0] - end_point[0]) ** 2 +
                    (start_point[1] - end_point[1]) ** 2)
            cpt_df.loc[i, "Before start"] = False
            cpt_df.loc[i, "Behind end"] = False
        else:
            result = offsets(start_point, end_point, (row['X'], row['Y']), latlon=latlon)
            cpt_df.loc[i, "Offset"] = result['offset to line']
            cpt_df.loc[i, "Projected offset"] = result['offset to start projected']
            cpt_df.loc[i, "Before start"] = result['before start']
            cpt_df.loc[i, "Behind end"] = result['behind end']

    if extend_profile:
        selected_cpts = deepcopy(cpt_df[cpt_df['Offset'] <= band])
    else:
        selected_cpts = deepcopy(cpt_df[
            (cpt_df['Offset'] <= band) &
            (cpt_df['Before start'] == False) &
            (cpt_df['Behind end'] == False)])

    selected_cpts.sort_values('Projected offset', inplace=True)

    if plotmap:
        fig = subplots.make_subplots(rows=1, cols=2, print_grid=False, column_widths=[0.7, 0.3],
                                     specs=[[{'type': 'xy'}, {'type': 'mapbox'},]])
    else:
        fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)

    _annotations = []

    k = 0

    for i, row in selected_cpts.iterrows():
        try:
            if uniformcolor is None:
                _tracecolor = DEFAULT_PLOTLY_COLORS[i % 10]
            else:
                _tracecolor = uniformcolor
            for _push in row['CPT objects'].data["Push"].unique():

                push_data = row['CPT objects'].data[row['CPT objects'].data["Push"] == _push]
                _push_trace = go.Scatter(
                    x=scale_factor * np.array(push_data[prop]) + row['Projected offset'],
                    y=-np.array(push_data['z [m]']) + row['Z'],
                    line=dict(color=_tracecolor),
                    showlegend=False,
                    mode='lines',
                    name='qc')
                fig.append_trace(_push_trace, 1, 1)

            _backbone = go.Scatter(
                x=[row['Projected offset'], row['Projected offset']],
                y=[(-np.array(row['CPT objects'].data['z [m]']) + row['Z']).min(),
                   (-np.array(row['CPT objects'].data['z [m]']) + row['Z']).max()],
                showlegend=True,
                mode='lines',
                line=dict(color=_tracecolor, dash='dot'),
                name="%s - %.0f%s offset" % (row['CPT titles'], row['Offset'], distance_unit)
            )
            fig.append_trace(_backbone, 1, 1)

            if k % 2 == 0:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=row['Z'],
                        text=row['CPT titles'])
                )
            else:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=-np.array(row['CPT objects'].data['z [m]']).max() + row['Z'],
                        text=row['CPT titles'],
                        ay=30)
                )

            k += 1

        except Exception as err:
            warnings.warn("Trace not created for %s - %s" % (row['CPT titles'], str(err)))

    if plotmap:
        mapbox_points = go.Scattermapbox(
            lat=y_coords, lon=x_coords, showlegend=False,
            mode='markers', name='Locations', hovertext=cpt_names, marker=dict(color='black'))
        fig.append_trace(mapbox_points, 1, 2)
        mapbox_profileline = go.Scattermapbox(
            lat=[start_point[1], end_point[1]],
            lon=[start_point[0], end_point[0]],
            showlegend=False, mode='lines', name='Profile', line=dict(color='red'))
        fig.append_trace(mapbox_profileline, 1, 2)

    if xaxis_layout is None:
        fig['layout']['xaxis1'].update(title='Projected distance [%s]' % (distance_unit))
    else:
        fig['layout']['xaxis1'].update(xaxis_layout)
    if yaxis_layout is None:
        fig['layout']['yaxis1'].update(title='Level [%s]' % (distance_unit))
    else:
        fig['layout']['yaxis1'].update(yaxis_layout)
    if general_layout is None:
        fig['layout'].update(height=600, width=900,
             title='Longitudinal profile from %s to %s' % (str(start), str(end)),
             hovermode='closest')
    else:
        fig['layout'].update(general_layout)

    if legend_layout is None:
        fig['layout'].update(legend=dict(orientation='h', x=0, y=-0.2))
    else:
        fig['layout'].update(legend=legend_layout)

    if show_annotations:
        fig['layout'].update(annotations=_annotations)

    if plotmap:
        fig.update_layout(
            mapbox_style='open-street-map', mapbox_zoom=10,
            mapbox_center={'lat': cpt_df['Y'].mean(), 'lon': cpt_df['X'].mean()}
        )

    if showfig:
        fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    return fig

def plot_combined_longitudinal_profile(
    cpts=[],
    profiles=[], latlon=False,
    option='name', start=None, end=None, band=1000, extend_profile=False,
    plotmap=False,
    fillcolordict={'SAND': 'yellow', 'CLAY': 'brown', 'SILT': 'green', 'ROCK': 'grey'},
    uniformcolor=None,
    opacity=1, logwidth=1,
    prop='qc [MPa]',
    distance_unit='m', scale_factor=0.001,
    showfig=True, xaxis_layout=None, yaxis_layout=None, general_layout=None, legend_layout=None,
    show_annotations=True):
    """
    Creates a longitudinal profile along selected CPTs and ``SoilProfile`` objects. A line is drawn from the first
    to the last location and the plot of the selected parameter (``prop``) vs depth
    is projected onto this line.

    This function also adds ``SoilProfile`` objects to the plot through mini-logs.

    :param cpts: List with PCPTProcessing objects to be plotted
    :param latlon: Boolean determining whether latitude and longitude are used or easting and northing in m (default=False for easting and northing in m)
    :param profiles: List with SoilProfile objects for which a log needs to be plotted
    :param option: Determines whether CPT names (``option='name'``) or tuples with coordinates (``option='coords'``) are used for the ``start`` and ``end`` arguments
    :param start: CPT name for the starting point or tuple of coordinates. If a CPT name is used, the selected CPT must be contained in ``cpts``.
    :param end: CPT name for the end point or tuple of coordinates. If a CPT name is used, the selected CPT must be contained in ``cpts``.
    :param band: Offset from the line connecting start and end points in which CPT are considered for plotting (default=1000m)
    :param extend_profile: Boolean determining whether the profile needs to be extended beyond the start and end points (default=False)
    :param plotmap: Boolean determining whether a map of locations needs to be plotted next to the profile (default=False)
    :param fillcolordict: Dictionary with fill colours (default yellow for 'SAND', brown from 'CLAY' and grey for 'ROCK')
    :param uniformcolor: Uniform color to use for all CPT traces (default=None for different color for each trace)
    :param opacity: Opacity of the layers (default = 1 for non-transparent behaviour)
    :param logwidth: Width of the soil logs as an absolute value (default = 1)
    :param prop: Selected property for plotting (default='qc [MPa]')
    :param distance_unit: Unit for coordinates and elevation (default='m')
    :param scale_factor: Scale factor for the property (default=0.001)
    :param showfig: Boolean determining whether the figure is shown (default=True)
    :param xaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param yaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param general_layout: Dictionary with general layout options (default=None)
    :param legend_layout: Dictionary with legend layout options (default=None)
    :param show_annotations: Boolean determining whether annotations need to be shown (default=True)
    :return: Plotly figure object
    """

    cpt_names = []
    x_coords = []
    y_coords = []
    elevations = []
    for _cpt in cpts:
        try:
            x_coords.append(_cpt.easting)
            y_coords.append(_cpt.northing)
            elevations.append(_cpt.elevation)
            cpt_names.append(_cpt.title)
        except Exception as err:
            warnings.warn(
                "CPT %s - Error during processing for profile - %s" % (_cpt.title, str(err)))
            raise

    if option == 'name':
        start_point = (x_coords[cpt_names.index(start)], y_coords[cpt_names.index(start)])
        end_point = (x_coords[cpt_names.index(end)], y_coords[cpt_names.index(end)])
    elif option == 'coords':
        if start.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        start_point = start
        if end.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        end_point = end
    else:
        raise ValueError("option should be 'name' or 'coords'")

    _layers_profile, _annotations_profile, _backbone_profile_traces, _soilcolors = plot_fence_diagram(
        profiles=profiles,
        latlon=latlon,
        option='coords',
        start=start_point,
        end=end_point,
        band=band,
        extend_profile=extend_profile,
        fillcolordict=fillcolordict,
        opacity=opacity,
        logwidth=logwidth,
        distance_unit=distance_unit,
        return_layers=True)

    cpt_df = pd.DataFrame({
        'CPT objects': cpts,
        'CPT titles': cpt_names,
        'X': x_coords,
        'Y': y_coords,
        'Z': elevations
    })
    for i, row in cpt_df.iterrows():
        if row['X'] == start_point[0] and row['Y'] == start_point[1]:
            cpt_df.loc[i, "Offset"] = 0
            cpt_df.loc[i, "Projected offset"] = 0
            cpt_df.loc[i, "Before start"] = False
            cpt_df.loc[i, "Behind end"] = False
        elif row['X'] == end_point[0] and row['Y'] == end_point[1]:
            cpt_df.loc[i, "Offset"] = 0
            if latlon:
                cpt_df.loc[i, "Projected offset"] = latlon_distance(
                    start_point[0], start_point[1], end_point[0], end_point[1])
            else:
                cpt_df.loc[i, "Projected offset"] = np.sqrt(
                    (start_point[0] - end_point[0]) ** 2 +
                    (start_point[1] - end_point[1]) ** 2)
            cpt_df.loc[i, "Before start"] = False
            cpt_df.loc[i, "Behind end"] = False
        else:
            result = offsets(start_point, end_point, (row['X'], row['Y']), latlon=latlon)
            cpt_df.loc[i, "Offset"] = result['offset to line']
            cpt_df.loc[i, "Projected offset"] = result['offset to start projected']
            cpt_df.loc[i, "Before start"] = result['before start']
            cpt_df.loc[i, "Behind end"] = result['behind end']

    if extend_profile:
        selected_cpts = deepcopy(cpt_df[cpt_df['Offset'] <= band])
    else:
        selected_cpts = deepcopy(cpt_df[
            (cpt_df['Offset'] <= band) &
            (cpt_df['Before start'] == False) &
            (cpt_df['Behind end'] == False)])

    selected_cpts.sort_values('Projected offset', inplace=True)

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)

    for _trace in _backbone_profile_traces:
        fig.append_trace(_trace, 1, 1)

    for _trace in _soilcolors:
        fig.append_trace(_trace, 1, 1)

    _annotations = []

    k = 0

    for i, row in selected_cpts.iterrows():
        try:
            if uniformcolor is None:
                _tracecolor = DEFAULT_PLOTLY_COLORS[i % 10]
            else:
                _tracecolor = uniformcolor
            for _push in row['CPT objects'].data["Push"].unique():
                push_data = row['CPT objects'].data[row['CPT objects'].data["Push"] == _push]
                _push_trace = go.Scatter(
                    x=scale_factor * np.array(push_data[prop]) + row['Projected offset'],
                    y=-np.array(push_data['z [m]']) + row['Z'],
                    line=dict(color=_tracecolor),
                    showlegend=False, mode='lines', name='qc')
                fig.append_trace(_push_trace, 1, 1)

            _backbone = go.Scatter(
                x=[row['Projected offset'], row['Projected offset']],
                y=[(-np.array(row['CPT objects'].data['z [m]']) + row['Z']).min(),
                   (-np.array(row['CPT objects'].data['z [m]']) + row['Z']).max()],
                showlegend=False,
                mode='lines',
                line=dict(color=_tracecolor, dash='dot')
            )
            fig.append_trace(_backbone, 1, 1)

            if k % 2 == 0:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=row['Z'],
                        text=row['CPT titles'],
                        ay=-60)
                )
            else:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=-np.array(row['CPT objects'].data['z [m]']).max() + row['Z'],
                        text=row['CPT titles'],
                        ay=60)
                )

            k += 1

        except Exception as err:
            warnings.warn("Trace not created for %s - %s" % (row['CPT titles'], str(err)))

    _annotations = _annotations + _annotations_profile

    if xaxis_layout is None:
        fig['layout']['xaxis1'].update(title='Projected distance [%s]' % (distance_unit))
    else:
        fig['layout']['xaxis1'].update(xaxis_layout)
    if yaxis_layout is None:
        fig['layout']['yaxis1'].update(title='Level [%s]' % (distance_unit))
    else:
        fig['layout']['yaxis1'].update(yaxis_layout)
    if general_layout is None:
        fig['layout'].update(height=600, width=900,
             title='Longitudinal profile from %s to %s' % (str(start), str(end)),
             hovermode='closest')
    else:
        fig['layout'].update(general_layout)

    if legend_layout is None:
        fig['layout'].update(legend=dict(orientation='h', x=0, y=-0.2))
    else:
        fig['layout'].update(legend=legend_layout)

    if show_annotations:
        fig['layout'].update(annotations=_annotations)

    fig['layout'].update(shapes=_layers_profile)

    if showfig:
        fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

    return fig

