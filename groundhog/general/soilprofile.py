#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
import re
from copy import deepcopy

# 3rd party packages
import pandas as pd
import numpy as np
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import matplotlib.pyplot as plt

# Project imports
from groundhog.general.plotting import plot_with_log, LogPlotMatplotlib, GROUNDHOG_PLOTTING_CONFIG
from groundhog.general.parameter_mapping import offsets, latlon_distance


class SoilProfile(pd.DataFrame):
    """
    A SoilProfile object is a Pandas dataframe with specific functionality for geotechnical calculations.
    There is a column syntax requirement in which the columns with the top and bottom depth need to be defined for
    each layer. By default 'Depth from [m]' and 'Depth to [m]' are expected but this can be customised.
    """

    def __init__(self, *args, **kwargs):
        """
        Overrides the init method of a dataframe to check the correctness of the layering and to set the depth
        column names.
        """
        super().__init__(*args, **kwargs)
        self.set_depthcolumn_name()
        self.check_profile()
        self.title = None

    def set_depthcolumn_name(self, name="Depth", unit='m'):
        self.depth_from_col = "%s from [%s]" % (name, unit)
        self.depth_to_col = "%s to [%s]" % (name, unit)

    def convert_depth_reference(self, newname="Depth", newunit='m', multiplier=1):
        """
        Converts the depth reference for a soil profile between one set of units (e.g. ft) to another (e.g. m)
        :param newname: New name for the depth reference (default='Depth')
        :param newunit: Name of the new unit (default=m)
        :param multiplier: Multiplier to go from old to new depth unit (e.g. 0.3 to go from ft to m)
        :return:
        """
        self[self.depth_from_col] = self[self.depth_from_col] * multiplier
        self[self.depth_to_col] = self[self.depth_to_col] * multiplier
        self.rename(columns={
            self.depth_from_col: "%s from [%s]" % (newname, newunit),
            self.depth_to_col: "%s to [%s]" % (newname, newunit)
        }, inplace=True)
        self.depth_from_col = "%s from [%s]" % (newname, newunit)
        self.depth_to_col = "%s to [%s]" % (newname, newunit)

    def check_profile(self):
        """
        Check if the SoilProfile meets the requirements for calculations
        """
        if (not self.depth_from_col in self.columns):
            raise IOError("SoilProfile does not contain the required '%s' column" % self.depth_from_col)

        if (not self.depth_to_col in self.columns):
            raise IOError("SoilProfile does not contain the required '%s' column" % self.depth_to_col)

        for i, row in self.iterrows():
            if i > 0:
                if row[self.depth_from_col] != self.loc[i - 1, self.depth_to_col]:
                    raise IOError(
                        "Invalid layer transition at layer %i, continuous layer transitions are required" % i)

        for _col in self.columns:
            if " from [" in _col and _col.replace(" from [", " to [") not in self.columns:
                raise IOError("""
                Incomplete linear parameter variation for column %s. Column %s not found.
                """ % (_col, _col.replace(" from [", " to [")))
            if " to [" in _col and _col.replace(" to [", " from [") not in self.columns:
                raise IOError("""
                Incomplete linear parameter variation for column %s. Column %s not found.
                """ % (_col, _col.replace(" to [", " from [")))

    @property
    def min_depth(self):
        """
        Returns the minimum depth of the soil profile
        """
        return self[self.depth_from_col].min()

    @property
    def max_depth(self):
        """
        Returns the maximum depth of the soil profile
        """
        return self[self.depth_to_col].max()

    def set_position(self, easting, northing, elevation, srid=4326, datum='mLAT'):
        """
        Sets the position of a soil profile top in a given coordinate system.

        By default, srid 4326 is used which means easting is longitude and northing is latitude.

        The elevation is referenced to a chart datum for which mLAT (Lowest Astronomical Tide) is the default.

        :param easting: X-coordinate of the CPT position
        :param northing: Y-coordinate of the CPT position
        :param elevation: Elevation of the CPT position
        :param srid: SRID of the coordinate system (see http://epsg.io)
        :param datum: Chart datum used for the elevation
        :return: Sets the corresponding attributes of the ```PCPTProcessing``` object
        """
        self.easting = easting
        self.northing = northing
        self.elevation = elevation
        self.srid = srid
        self.datum = datum

    def layer_transitions(self, include_top=False, include_bottom=False):
        """
        Returns a Numpy array with the layer transition depths.
        Use the booleans include_top and include_bottom to in/exclude the top and bottom of the profile
        """
        transitions = np.array(self[self.depth_from_col][1:])
        if include_top:
            transitions = np.append(self[self.depth_from_col].iloc[0], transitions)
        if include_bottom:
            transitions = np.append(transitions, self[self.depth_to_col].iloc[-1])
        return transitions
    
    def adjust_layertransition(self, currentdepth, newdepth, tolerance=1e-3):
        """
        Adjusts the depth of a layer transition.
        
        :param currentdepth: Current depth of the layer transition
        :param newdepth: Desired new depth of the layer transition
        :param tolerance: Offset above and below ``currentdepth`` in which a layer transition is sought (to cope with number precision issues)
        """
        layer_above_index = self[(self[self.depth_to_col] > (currentdepth - tolerance)) & 
                           (self[self.depth_to_col] < (currentdepth + tolerance))].index
        layer_below_index = self[(self[self.depth_from_col] > (currentdepth - tolerance)) & 
                           (self[self.depth_from_col] < (currentdepth + tolerance))].index
        self.loc[layer_above_index, self.depth_to_col] = newdepth
        self.loc[layer_below_index, self.depth_from_col] = newdepth

    def calculate_layerthickness(self, layerthicknesscol="Layer thickness [m]"):
        """
        Adds a column with the layer thickness to the soil profile
        :param layerthicknesscol: Name of the column with the layer thickness
        :return: Adds a column with the layer thickness
        """
        self[layerthicknesscol] = self[self.depth_to_col] - self[self.depth_from_col]

    def calculate_center(self, layercentercol="Depth center [m]"):
        """
        Adds a column with the layer center depth to the soil profile
        :param layercentercol: Name of the column with the layer center
        :return: Adds a column with the layer center
        """
        self[layercentercol] = 0.5 * (self[self.depth_to_col] + self[self.depth_from_col])

    def soil_parameters(self, condense_linear=True):
        """
        Returns a list of soil parameters available in the soil profile.
        Soil parameters with linear variations are returned as a single soil parameter
        when the boolean ``condense_linear`` is set to True
        :return: Returns a list of the soil parameters in the SoilProfile
        """
        _parameters = []
        for _col in self.columns:
            if _col == self.depth_from_col or _col == self.depth_to_col:
                pass
            else:
                if condense_linear:
                    if " to [" in _col:
                        pass
                    elif " from [" in _col:
                        _parameters.append(_col.replace(" from [", " ["))
                    else:
                        _parameters.append(_col)
                else:
                    _parameters.append(_col)
        return _parameters

    def numerical_soil_parameters(self, condense_linear=True):
        """
        Returns a list of numerical soil parameters available in the soil profile.
        Numerical soil parameters have units between square brackets.
        Soil parameters with linear variations are returned as a single soil parameter
        when the boolean ``condense_linear`` is set to True
        :return: Returns a list with the numerical soil parameters in the SoilProfile
        """
        _parameters = []
        for _col in self.columns:
            if _col == self.depth_from_col or _col == self.depth_to_col:
                pass
            else:
                if re.match(r".+ \[.+\]", _col):
                    # Numerical parameter
                    if condense_linear:
                        if " to [" in _col:
                            pass
                        elif " from [" in _col:
                            _parameters.append(_col.replace(" from [", " ["))
                        else:
                            _parameters.append(_col)
                    else:
                        _parameters.append(_col)
                else:
                    pass
        return _parameters

    def string_soil_parameters(self):
        """
        Returns a list of string soil parameters available in the soil profile.
        String soil parameters have no square brackets.
        :return: Returns a list with the string soil parameters in the SoilProfile
        """
        _parameters = []
        for _col in self.columns:
            if _col == self.depth_from_col or _col == self.depth_to_col:
                pass
            else:
                if re.match(r".+ \[.+\]", _col):
                    # Numerical parameter
                    pass
                else:
                    _parameters.append(_col)
        return _parameters

    def check_linear_variation(self, parameter):
        """
        Check if a soil parameter varies linearly or not.
        :return: Boolean determining whether a soil parameter has a linear variation or not
        """
        if not parameter in self.numerical_soil_parameters():
            raise ValueError("Parameter %s not defined, choose one of %s." % (
                parameter, self.numerical_soil_parameters()
            ))
        else:
            if (parameter.replace(' [', ' from [') in self.columns) and \
               (parameter.replace(' [', ' to [') in self.columns):
               return True
            else:
                return False

    def insert_layer_transition(self, depth):
        """
        Inserts a layer transition in the soil profile at a specific depth
        The profile layer is simply split and the properties of the given layer are assigned
        to the new layers above and below the transition.
        For linearly varying parameters, an interpolation is performed.
        """
        if depth < self.min_depth:
            raise ValueError("Selected depth should be below minimum soil profile depth")

        if depth > self.max_depth:
            raise ValueError("Selected depth should be above maximum soil profile depth")

        if depth in self.layer_transitions(include_top=True, include_bottom=True):
            warnings.warn("Specified depth is already at a layer transition, it will be ignored")
        else:
            row = self[(self[self.depth_from_col] < depth) &
                         (self[self.depth_to_col] > depth)]
            self.loc[row.index[0], self.depth_to_col] = depth
            new_index = self.__len__()

            for _param in self.string_soil_parameters():
                self.loc[new_index, _param] = row[_param].iloc[0]
            for _param in self.numerical_soil_parameters():
                # Interpolate for linearly varying properties
                if self.check_linear_variation(_param):
                    self.loc[new_index, _param.replace(' [', ' from [')] = \
                        np.interp(
                            depth,
                            (row[self.depth_from_col].iloc[0],
                             row[self.depth_to_col].iloc[0]),
                            (row[_param.replace(' [', ' from [')].iloc[0],
                             row[_param.replace(' [', ' to [')].iloc[0])
                        )
                    self.loc[row.index[0], _param.replace(' [', ' to [')] = self.loc[
                        new_index, _param.replace(' [', ' from [')]
                    self.loc[new_index, _param.replace(' [', ' to [')] = \
                        row[_param.replace(' [', ' to [')].iloc[0]
                else:
                    self.loc[new_index, _param] = row[_param].iloc[0]
            self.loc[new_index, self.depth_from_col] = depth
            self.loc[new_index, self.depth_to_col] = row[self.depth_to_col].iloc[0]
            self.sort_values(self.depth_from_col, inplace=True)
            self.reset_index(drop=True, inplace=True)

    def convert_depth_sign(self):
        """
        Inverts the sign of the depth from and depth to column.
        This function is useful when the profiles needs to be examined relative to LAT
        """
        self[self.depth_from_col] = -self[self.depth_from_col]
        self[self.depth_to_col] = -self[self.depth_to_col]

    def shift_depths(self, offset):
        """
        Shifts all layer coordinates downward or upward, depending on the sign of the given offset.
        The offset is added to the given coordinates
        """
        self[self.depth_from_col] = self[self.depth_from_col] + offset
        self[self.depth_to_col] = self[self.depth_to_col] + offset

    def soilparameter_series(self, parameter):
        """
        Returns two lists (depths and corresponding parameter values) for plotting
        of a soil parameter vs depth.
        The routine first checks whether a valid parameter is provided.
        The lists are formatted such that variations at a layer interface are
        adequately plotted.
        :param parameter: A valid soil parameter with units
        :return: Two lists (depths and corresponding parameter values)
        """
        if not parameter in self.numerical_soil_parameters():
            raise ValueError("Parameter %s not defined, choose one of %s." % (
                parameter, self.numerical_soil_parameters()
            ))
        if self.check_linear_variation(parameter):
            first_array_x = np.array(self[parameter.replace(' [', ' from [')])
            second_array_x = np.array(self[parameter.replace(' [', ' to [')])
        else:
            first_array_x = np.array(self[parameter])
            second_array_x = np.array(self[parameter])
        first_array_z = np.array(self[self.depth_from_col])
        second_array_z = np.array(self[self.depth_to_col])
        z = np.insert(second_array_z, np.arange(len(first_array_z)), first_array_z)
        x = np.insert(second_array_x, np.arange(len(first_array_x)), first_array_x)
        return z, x

    def map_soilprofile(self, nodalcoords, target_depthkey='z [m]', keys_to_map=None, invert_sign=False, offset=0,
        include_layertransitions=False):
        """
        Maps the soilprofile to a grid. The depth coordinates to the grid are specified
        in a list or Numpy array (``nodalcoords``).
        The depth coordinates should be strictly ascending (no duplicates) and the minimum and
        maximum should be contained inside the soil profile bounds.
        Layer transitions can be included in the grid when ``include_layertransitions=True``.
        All soil parameters are interpolated onto this grid.
        :param nodalcoords: List or Numpy array with the nodal coordinates of the grid
        :param target_depthkey: Name of the depth key in the resulting dataframe
        :param keys_to_map: List with the soilprofile keys to map
        :param invert_sign: Boolean determining whether to invert the sign after interpolation
        :param offset: Offset by which the depth is shifted (added to the depth after sign conversion)
        :param include_layertransitions: Boolean to determine whether the layer transitions needs to be included in the grid (default=False)
        :return: Returns a dataframe with the full grid with soil parameters
        """
        # 1. Convert nodalcoords to Numpy array
        try:
            z = np.array(nodalcoords)
        except Exception as err:
            raise ValueError("Could not convert nodal coords to Numpy array (%s)" % str(err))
        # Merge layer transitions
        if include_layertransitions:
            z = np.unique((np.append(z, self.layer_transitions())))
        else:
            pass

        # Create a target dataframe with the depth column
        target_df = pd.DataFrame({
            target_depthkey: z,
        })

        # 2. Validate if nodal coordinates array is sorted and strictly ascending
        if np.all(np.diff(z) >= 0):
            pass
        else:
            raise ValueError("Nodal coordinates are not ascending. Please check ")

        if keys_to_map is None:
            # Map all keys
            string_params = self.string_soil_parameters()
            numerical_params = self.numerical_soil_parameters()
        else:
            # Map only specific keys
            string_params = []
            numerical_params = []
            for _key in keys_to_map:
                if _key in self.string_soil_parameters():
                    string_params.append(_key)
                elif _key in self.numerical_soil_parameters():
                    numerical_params.append(_key)
                else:
                    raise ValueError(
                        "Key %s is not present in the numerical or string parameters of the soil profile" % _key)

        # 3. Map string parameters
        for _param in string_params:
            target_df[_param] = list(map(lambda _z: self[
                (self[self.depth_from_col] <= _z) & (self[self.depth_to_col] >= _z)][_param].iloc[-1],
                                      target_df[target_depthkey]))

        # 4. Map numerical parameters
        for _param in numerical_params:
            z, x = self.soilparameter_series(_param)

            target_df[_param] = np.interp(target_df[target_depthkey], z, x)

        # 5. Convert sign and apply offset
        if invert_sign:
            target_df[target_depthkey] = -target_df[target_depthkey]

        target_df[target_depthkey] = target_df[target_depthkey] + offset

        return target_df

    def plot_profile(self, parameters, soiltypecolumn='Soil type', **kwargs):
        """
        Generates a Plotly plot of the soil parameters vs depth.
        The panel on which the parameter is plotted is determined by how the parameters are passed to the function.

        ``parameters=(('qc [MPa]',), ('Dr [%]',))`` will plot cone resistance in panel 1 and relative density in panel 2

        ``parameters=(('qc [MPa]', 'qt [MPa]'), ('Dr [%]'))`` will plot cone resistance and total cone resistance in panel 1 and relative density in panel 2

        A column wihh the soil type is expected for the plotting of the log

        :param parameters: List of parameters tuples for plotting
        :param kwargs: Additional keyword arguments for the ``plot_with_log`` function in the ``general.plotting`` module
        :return: Plotly plot with mini-log and parameter values
        """
        if soiltypecolumn not in self.string_soil_parameters():
            raise ValueError("Column %s not recognised for soil type, needs to be one of %s" % (
                soiltypecolumn, self.string_soil_parameters()))

        plotting_z = []
        plotting_x = []
        names = []
        x_titles = []
        for _paramset in parameters:
            _plotting_z = []
            _plotting_x = []
            _names = []
            for _param in _paramset:
                if _param not in self.numerical_soil_parameters():
                    raise ValueError("Soil parameter %s not in SoilProfile, must be one of %s" % (
                        _param, self.numerical_soil_parameters()
                    ))
                else:
                    _z, _x = self.soilparameter_series(_param)
                    _plotting_z.append(_z)
                    _plotting_x.append(_x)
                    _names.append(_param)
            plotting_z.append(_plotting_z)
            plotting_x.append(_plotting_x)
            names.append(_names)
            x_titles.append('no title')

        try:
            names = kwargs['names']
            del kwargs['names']
        except:
            pass

        try:
            x_titles = kwargs['xtitles']
            del kwargs['xtitles']
        except:
            pass

        try:
            z_title = kwargs['ztitle']
            del kwargs['ztitle']
        except:
            z_title = "Depth [m]"

        if soiltypecolumn != 'Soil type':
            self.rename({soiltypecolumn: 'Soil type'}, inplace=True)

        return plot_with_log(
            x=plotting_x,
            z=plotting_z,
            names=names,
            soildata=self,
            xtitles=x_titles,
            ztitle=z_title,
            depth_from_key=self.depth_from_col,
            depth_to_key=self.depth_to_col,
            **kwargs
        )

    def selection_soilparameter(self, parameter, depths, values, rule='mean', linearvariation=False):
        """
        Function for automatic selection of a soil parameters in the layers of a soil profile
        based an a list of provided values. The selection can either be done for a constant value in the layer or a linear variation
        over the layer.
        Selection of a minimum, mean or average trend can be performed using the ``rule`` keyword

        :param parameter: Name of the parameter being selected (including unit in square brackets)
        :param depths: Depths at which a measurement is available (list or Numpy array)
        :param values: Values corresponding to the given depths (list or Numpy array)
        :param rule: Which rule to use for the selection (select from ``min``, ``mean`` and ``max``
        :param linearvariation: Boolean determining whether a linear variation needs to happen over the layer or not
        :return: Adds one (constant value) or two (linear variation) columns to the dataframe
        """
        # Validate the length of depths and values lists
        if depths.__len__() != values.__len__():
            raise ValueError("Lists with depths and values must be of equal length")

        # Validate the specified parameter
        if not re.match(r".+ \[.+\]", parameter):
            raise ValueError(
                "Soil parameter incorrectly specified, parameter should be in the format 'parameter [unit]'")

        if rule not in ['min', 'mean', 'max']:
            raise ValueError("rule should be one of 'min', 'mean' or 'max'")

        for i, row in self.iterrows():
            # Loop over every layer and select provided values in each layer
            selection = np.logical_and(
                np.array(depths) >= row[self.depth_from_col],
                np.array(depths) <= row[self.depth_to_col])
            selected_depths = np.array(depths)[selection]
            selected_values = np.array(values)[selection]

            if selected_depths.__len__() == 0:
                # No samples in selected layer
                if linearvariation:
                    self.loc[i, parameter.replace(' [', ' from [')] = np.nan
                    self.loc[i, parameter.replace(' [', ' to [')] = np.nan
                else:
                    self.loc[i, parameter] = np.nan
            elif selected_depths.__len__() == 1:
                # Single value in the selected layer
                if linearvariation:
                    self.loc[i, parameter.replace(' [', ' from [')] = selected_values[0]
                    self.loc[i, parameter.replace(' [', ' to [')] = selected_values[0]
                else:
                    self.loc[i, parameter] = selected_values[0]
            else:
                # Multiple values in each layer
                if linearvariation:
                    fit_coeff = np.polyfit(
                        x=selected_depths[~np.isnan(selected_values)],
                        y=selected_values[~np.isnan(selected_values)],
                        deg=1)
                    fit_func = np.poly1d(fit_coeff)
                    residuals = np.array(selected_values[~np.isnan(selected_values)]) - fit_func(selected_depths[~np.isnan(selected_values)])
                    sorted_residuals = np.sort(residuals)
                    if rule == 'min':
                        # TODO: Double-check routine for trend selection
                        selected_points = (
                            np.array(residuals) <= sorted_residuals[1])
                        zs = selected_depths[~np.isnan(selected_values)][selected_points]
                        xs = selected_values[~np.isnan(selected_values)][selected_points]
                        self.loc[i, parameter.replace(' [', ' from [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_from_col] - zs[0])
                        self.loc[i, parameter.replace(' [', ' to [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_to_col] - zs[0])
                    elif rule == 'mean':
                        self.loc[i, parameter.replace(' [', ' from [')] = fit_func(row[self.depth_from_col])
                        self.loc[i, parameter.replace(' [', ' to [')] = fit_func(row[self.depth_to_col])
                    elif rule == 'max':
                        selected_points = (
                                np.array(residuals) >= sorted_residuals[-2])
                        zs = selected_depths[~np.isnan(selected_values)][selected_points]
                        xs = selected_values[~np.isnan(selected_values)][selected_points]
                        self.loc[i, parameter.replace(' [', ' from [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_from_col] - zs[0])
                        self.loc[i, parameter.replace(' [', ' to [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_to_col] - zs[0])
                    else:
                        raise ValueError("Rule %s unknown. Rule should be one of 'min', 'mean' or 'max'" % rule)
                else:
                    if rule == 'min':
                        self.loc[i, parameter] = selected_values[~np.isnan(selected_values)].min()
                    elif rule == 'mean':
                        self.loc[i, parameter] = selected_values[~np.isnan(selected_values)].mean()
                    elif rule == 'max':
                        self.loc[i, parameter] = selected_values[~np.isnan(selected_values)].max()
                    else:
                        raise ValueError("Rule %s unknown. Rule should be one of 'min', 'mean' or 'max'" % rule)

    def merge_layers(self, layer_ids, keep='top'):
        """
        Merges two layers. Depending on the ``keep`` keyword, the top or bottom
        properties are retained for the merged layer

        :param layer_ids: List with the 2 IDs of the layers to be merged
        :param keep: String determining whether to retain the parameters of the top or bottom layer for the merged layer (select 'top' or 'bottom')
        :return: Reduces the number of layers of the ``SoilProfile`` object
        """
        # TODO: Implement intermediate behaviour where properties of top and bottom layer are averaged

        # Validate the IDs
        if layer_ids.__len__() != 2:
            raise ValueError("A tuple with two layer indices should be specified")
        try:
            for _id in layer_ids:
                self.loc[_id, self.depth_from_col]
        except:
            raise ValueError("Layer IDs not valid. Valid layer IDs are %s" % list(self.index))

        if keep == 'top':
            self.loc[layer_ids[1], self.depth_from_col] = np.nan
            self.loc[layer_ids[0], self.depth_to_col] = self.loc[layer_ids[1], self.depth_to_col]
        elif keep == 'bottom':
            self.loc[layer_ids[0], self.depth_to_col] = np.nan
            self.loc[layer_ids[1], self.depth_from_col] = self.loc[layer_ids[0], self.depth_from_col]
        self.dropna(subset=(self.depth_from_col, self.depth_to_col), inplace=True)
        self.reset_index(drop=True, inplace=True)

    def remove_parameter(self, parameter):
        """
        Removes a soil parameter from the dataframe
        :param parameter: Soil parameter to remove. For linear variations, simple use the expression ``parameter [unit]``. The routine takes care of removing both the 'to' and 'from' columns
        :return: Removes the requested soil parameter from the ``SoilProfile`` objec
        """
        if (parameter not in self.numerical_soil_parameters()) and (parameter not in self.string_soil_parameters()):
            raise ValueError("Soil parameter %s not in dataframe" % parameter)

        if self.check_linear_variation(parameter):
            self.drop(parameter.replace(' [', ' from ['), axis=1, inplace=True)
            self.drop(parameter.replace(' [', ' to ['), axis=1, inplace=True)
        else:
            self.drop(parameter, axis=1, inplace=True)

    def convert_to_constant(self, parameter, rule='mean'):
        """
        Converts a linearly varying soil parameter to a parameter with constant value.
        :param parameter: Soil parameter to convert. Specify ``Total unit weight [kN/m3]`` if the ``SoilProfile`` contains columns ``Total unit weight from [kN/m3]`` and ``Total unit weight to [kN/m3]``.
        :param rule: Select from ``['min', 'mean', 'max']`` to convert the linear variation to a constant value (default='mean')
        :return: Creates a column for the soil parameter with constant value and removes the column with the linear variation
        """

        # Validate presence of linearly varying soil parameter
        if not parameter in self.numerical_soil_parameters():
            raise ValueError("SoilProfile does not contain the parameter %s" % parameter)
        else:
            if self.check_linear_variation(parameter):
                pass
            else:
                raise ValueError("Soil parameter %s does not show a linear variation")

        # Copy the values at top and bottom of each layer to arrays and drop the columns from the SoilProfile
        _param_from = self[parameter.replace(' [', ' from [')]
        _param_to = self[parameter.replace(' [', ' to [')]
        self.remove_parameter(parameter)

        # Add the values for the constant soil parameter according to the selected rule
        if rule == 'min':
            self[parameter] = list(map(lambda _x, _y: min(_x, _y), _param_from, _param_to))
        elif rule == 'mean':
            self[parameter] = list(map(lambda _x, _y: 0.5 * (_x + _y), _param_from, _param_to))
        elif rule == 'max':
            self[parameter] = list(map(lambda _x, _y: max(_x, _y), _param_from, _param_to))
        else:
            raise ValueError("Rule should be 'min', 'mean' or 'max'")

    def calculate_parameter_center(self, parameter, suffix="center"):
        """
        Calculates the value of a soil parameter at the center. The soil parameter needs to be a linearly varying numerical soil parameter.
        :param parameter: Numerical soil parameter for which the value at the center needs to be computed.
        :param suffix: Suffix to use instead of ``from`` or ``to``
        :return: Adds an extra column with ``from`` or ``to`` in the column name replaced by the chosen suffix
        """
        if parameter not in self.numerical_soil_parameters():
            raise ValueError("Chosen parameter should be a numerical parameter")
        else:
            pass

        if self.check_linear_variation(parameter):
            self[parameter.replace(' [', ' %s [' % suffix)] = 0.5 * (
                self[parameter.replace(' [', ' from [')] + self[parameter.replace(' [', ' to [')]
            )

    def cut_profile(self, top_depth, bottom_depth):
        """
        Returns a deep copy of the ``SoilProfile`` between the specified bounds
        :param top_depth: Top depth for cutting
        :param bottom_depth: Bottom depth for cutting
        :return: Deep copy of the ``SoilProfile`` between the specified bounds
        """
        if top_depth < self.min_depth:
            warnings.warn("Top depth of %.2f is smaller than minimum depth of the soil profile." % top_depth)
            top_depth = self.min_depth

        if bottom_depth > self.max_depth:
            warnings.warn("Bottom depth of %.2f is greater than maximum depth of the soil profile." % bottom_depth)
            bottom_depth = self.max_depth

        # Make a deep copy of the soil profile
        _profile = deepcopy(self)

        # Drop layers outside the bounds
        _profile = _profile[(_profile[self.depth_from_col] < bottom_depth) &
                            (_profile[self.depth_to_col] > top_depth)]

        # Interpolate linearly varying parameters
        for _param in self.numerical_soil_parameters():
            if self.check_linear_variation(_param):
                _from_param = _param.replace(' [', ' from [')
                _to_param = _param.replace(' [', ' to [')
                _profile.loc[_profile.index[0], _from_param] = np.interp(
                    top_depth,
                    [_profile[self.depth_from_col].iloc[0],
                     _profile[self.depth_to_col].iloc[0]],
                    [_profile[_from_param].iloc[0],
                     _profile[_to_param].iloc[0]]
                )
                _profile.loc[_profile.index[-1], _to_param] = np.interp(
                    bottom_depth,
                    [_profile[self.depth_from_col].iloc[-1], _profile[self.depth_to_col].iloc[-1]],
                    [_profile[_from_param].iloc[-1],
                     _profile[_to_param].iloc[-1]]
                )

        # Adjust bounds
        _profile.loc[_profile.index[0], self.depth_from_col] = top_depth
        _profile.loc[_profile.index[-1], self.depth_to_col] = bottom_depth

        # Reset the indices
        _profile.reset_index(drop=True, inplace=True)

        # Make sure the returned object is also a SoilProfile object
        _profile.__class__ = SoilProfile
        pattern = re.compile(r'(?P<depth_key>.+) from \[(?P<unit>.+)\]')
        match = re.search(pattern, self.depth_from_col)
        _profile.set_depthcolumn_name(name=match.group('depth_key'), unit=match.group('unit'))

        return _profile

    def depth_integration(self, parameter, outputparameter, start_value=0):
        """
        Integrate a certain parameter vs depth (e.g. unit weight to obtain vertical stress)
        Note: This routine is only implemented for parameters with a constant value in the layer.
        :param parameter: Parameter to be integrated
        :param outputparameter: Name of the output parameter (with units)
        :param start_value: Value at the top of the profile (default=0)
        :return: Adds a column to the ``SoilProfile`` object for the integrated parameter
        """
        # Validate that the parameter to be integrated is in the list of numerical parameters with contstant value
        if (not parameter in self.numerical_soil_parameters()):
            raise ValueError("Selected parameter is not in the list of numerical parameters")
        else:
            if self.check_linear_variation(parameter):
                raise ValueError("Integration only works for parameter with a constant value in each layer")
            else:
                pass

        # Validate name of the output parameter
        if not re.match(r".+ \[.+\]", outputparameter):
            raise ValueError("Output parameter not propertly formatted, needs to be 'parameter [unit]'")

        # Check that there a no NaN values
        if self[parameter].__len__() != self.dropna(subset=(parameter,)).__len__():
            raise ValueError("Parameter integrated vs depth should not contain nan values")

        for i, row in self.iterrows():
            if i == 0:
                self.loc[i, outputparameter.replace(' [', ' from [')] = start_value
            else:
                self.loc[i, outputparameter.replace(' [', ' from [')] = \
                    self.loc[i - 1, outputparameter.replace(' [', ' to [')]

            self.loc[i, outputparameter.replace(' [', ' to [')] = \
                self.loc[i, outputparameter.replace(' [', ' from [')] + \
                row[parameter] * (row[self.depth_to_col] - row[self.depth_from_col])

    def calculate_overburden(self, waterlevel=0, waterunitweight=10, initial_vertical_total_stress=0,
                             totalunitweightcolumn="Total unit weight [kN/m3]",
                             effectiveunitweightcolumn="Effective unit weight [kN/m3]",
                             waterunitweightcolumn="Water unit weight [kN/m3]",
                             totalverticalstresscolumn="Vertical total stress [kPa]",
                             effectiveverticalstresscolumn="Vertical effective stress [kPa]",
                             hydrostaticpressurecolumn="Hydrostatic pressure [kPa]"):
        """
        Calculates the overburden pressure (total and effective) for a ``SoilProfile`` object.
        The ``SoilProfile`` object needs to contain a column with the total unit weight.
        By default, this is ``Total unit weight [kN/m3]``.
        If the water level does not correspond with a layer interface, an additional layer interface is created.
        Total and effective unit weights are calculated for each layer and the method ``depth_integration`` is used
        to calculate the total vertical stress, effective vertical stress and hydrostatic pressure.

        Note that vertical stress calculations in other units are possible, but the water unit weight then needs
        to be specified in consistent units.

        An initial value can be added to the total vertical stress to simulate the effect of e.g. surcharging.

        :param waterlevel: Water level [m] (default 0m)
        :param waterunitweight: Unit weight of the pore water [kN/m3] (default=10kN/m3)
        :param initial_vertical_total_stress: Initial value of vertical total stress [kPa] (default=0kPa)
        :param totalunitweightcolumn: Column name containing total unit weights (default='Total unit weight [kN/m3]'
        :param waterunitweightcolumn: Output column with the effective unit weight (default='Effective unit weight [kN/m3]')
        :param waterunitweightcolumn: Output column with the water unit weight (default='Water unit weight [kN/m3]')
        :param totalverticalstresscolumn: Total vertical stress column name (default='Total vertical stress [kPa]')
        :param effectiveverticalstresscolumn: Effective vertical stress column name (default='Vertical effective stress [kPa]')
        :param hydrostaticpressurecolumn: Hydrostatic pressure column name (default='Hydrostatic pressure [kPa]')
        :return: Adds the column names from ``totalverticalstresscolumn``, ``effectiveverticalstresscolumn`` and ``hydrostaticpressurecolumn`` to the ``SoilProfile`` object
        """

        # Validate the presence of a column with the total unit weight
        if not totalunitweightcolumn in self.numerical_soil_parameters():
            raise ValueError("SoilProfile should contain a column %s" % totalunitweightcolumn)
        else:
            if self.check_linear_variation(totalunitweightcolumn):
                raise ValueError("Constant unit weight in each layer is required, use the method 'convert_to_constant'")
            else:
                pass

        # Check position of water level
        if waterlevel <= self.min_depth:
            waterlevel = self.min_depth
            self.loc[:, waterunitweightcolumn] = waterunitweight
        elif waterlevel >= self.max_depth:
            waterlevel = self.max_depth
            self.loc[:, waterunitweightcolumn] = 0
        else:
            if waterlevel in self.layer_transitions():
                pass
            else:
                self.insert_layer_transition(depth=waterlevel)

        for i, layer in self.iterrows():
            if 0.5 * (layer[self.depth_from_col] + layer[self.depth_to_col]) < waterlevel:
                self.loc[i, waterunitweightcolumn] = 0
                self.loc[i, effectiveunitweightcolumn] = layer[totalunitweightcolumn]
            else:
                self.loc[i, waterunitweightcolumn] = waterunitweight
                self.loc[i, effectiveunitweightcolumn] = layer[totalunitweightcolumn] - waterunitweight

        self.waterlevel = waterlevel

        # Calculate hydrostatic pressure
        self.depth_integration(parameter=waterunitweightcolumn, outputparameter=hydrostaticpressurecolumn)

        # Calculate effective vertical stress
        self.depth_integration(parameter=effectiveunitweightcolumn, outputparameter=effectiveverticalstresscolumn,
                               start_value=initial_vertical_total_stress)

        # Calculate total vertical stress
        self.depth_integration(parameter=totalunitweightcolumn, outputparameter=totalverticalstresscolumn,
                               start_value=initial_vertical_total_stress)


    def applyfunction(self, function, resultkey, outputkey, parametermapping=dict(), **kwargs):
        """
        Applies a groundhog function to a soil profile. The function is applied to each row of the soilprofile.
        The result is stored in a column with name ``output``. ``resultkey`` determines which key of the function
        output dictionary is used as the result.

        The parameters of the function are mapped to columns
        of the soil profile using the parametermapping dictionary. The keys of this dictionary are the function arguments,
        the values are the corresponding columns of the soilprofile. For parameters with linear variation, this
        method only needs to be applied once and the soil parameter name (without from or to) needs to be supplied in
        the ``parametermapping`` dictionary.

        :param function: Function to be applied
        :param resultkey: Column name for the result (for parameters with linear variation, two result columns are created)
        :param outputkey: The key of the function output dictionary to be used for the result
        :param parametermapping: Dictionary mapping parameters of the function to column names
        :param applyatcenter: Boolean determining whether the function needs to be applied at the center of the layer for a linearly varying parameter. A single output column will then be returned (default=False).
        :param kwargs: Additional keyword arguments of the function which are not mapped to soil profile columns
        :return:
        """
        # Validate the column names in parametermapping
        for key, value in parametermapping.items():
            if value not in self.numerical_soil_parameters():
                raise ValueError(
                    "Column %s does not exist in the soil profile, check your soil profile" % value)

        # Apply the function to each row
        for i, row in self.iterrows():
            function_dict = dict()
            for key, value in parametermapping.items():
                if self.check_linear_variation(value):
                    # Apply function for linear parameter variation
                    function_dict[key] = row[value.replace(' [', ' from [')]
                    self.loc[i, outputkey.replace(' [', ' from [')] = \
                        function(**{**function_dict, **kwargs})[resultkey]
                    function_dict[key] = row[value.replace(' [', ' to [')]
                    self.loc[i, outputkey.replace(' [', ' to [')] = \
                        function(**{**function_dict, **kwargs})[resultkey]
                else:
                    function_dict[key] = row[value]
                    self.loc[i, outputkey] = \
                        function(**{**function_dict, **kwargs})[resultkey]

    def parameter_at_depth(self, depth, parameter, shallowest=True):
        """
        Calculates the value for one of the ``SoilProfile`` parameters at a selected depth
        :param depth: Selected depth [m]
        :param parameter: String or Numerical soil parameter (linear interpolation for linearly varying parameters)
        :param shallowest: Boolean determining whether at a layer interface, the shallowest value needs to be used (default=True).
        :return: The value of the parameter at the selected depth
        """
        if depth > self.max_depth:
            raise ValueError("Selected depth is greater than the maximum depth of the soil profile")

        if depth < self.min_depth:
            raise ValueError("Selected depth is lower than the minimum depth of the soil profile")

        if (not parameter in self.numerical_soil_parameters()) and \
                (not parameter in self.string_soil_parameters()):
            raise ValueError("Selected parameter is not in the list of available parameters")

        _layers = self[
             (self[self.depth_from_col] <= depth) &
             (self[self.depth_to_col] >= depth)
        ]

        if shallowest:
            _selected_layer = _layers.iloc[0]
        else:
            _selected_layer = _layers.iloc[-1]

        if parameter in self.numerical_soil_parameters():
            if self.check_linear_variation(parameter):
                # Linear interpolation for a linearly varying parameter
                return np.interp(
                    depth,
                    [_selected_layer[self.depth_from_col], _selected_layer[self.depth_to_col]],
                    [_selected_layer[parameter.replace(' [', ' from [')],
                     _selected_layer[parameter.replace(' [', ' to [')]]
                )
            else:
                # Constant value for a parameter with constant value in the layer
                return _selected_layer[parameter]
        else:
            # String in the selected layer
            return _selected_layer[parameter]

def create_blank_soilprofile(max_depth, min_depth=0, soiltype='Unknown', bulkunitweight=20):
    """
    Creates a SoilProfile object with a single layer. By default the soil type is set to ``'Unknown'`` and the bulk unit weight to 20kN/m3.
    :param max_depth: Maximum depth for the SoilProfile object [m]
    :param min_depth: Minimum depth for the SoilProfile object [m] (optional, default=0m)
    :param soiltype: Soil type for the layer (optional, default="Unknown")
    :param bulkunitweight: Bulk unit weight for the layer [kN/m3] (optional, default= 20kN/m3)
    """
    if min_depth >= max_depth:
        raise ValueError("The maximum depth needs to be strictly larger than the minimum depth.")
    else:
        _sp = SoilProfile({
            'Depth from [m]': [min_depth,],
            'Depth to [m]': [max_depth],
            'Soil type': [soiltype,],
            'Total unit weight [kN/m3]': [bulkunitweight,]
        })
        return _sp

def read_excel(path, title='', depth_key='Depth', unit='m', column_mapping={}, depth_multiplier=1, **kwargs):
    """
    The method to read from Excel needs to be redefined for SoilProfile objects.
    The method allows for different depth keys (using the 'depth_key' and 'unit' keyword arguments
    Columns can also be renamed using the 'column_mapping' dictionary. The keys in this dictionary are
    the old column names and the values are the new column names.
    """
    sp = pd.read_excel(path, **kwargs)
    sp.__class__ = SoilProfile
    sp.title = title
    sp.set_depthcolumn_name(name=depth_key, unit=unit)
    sp.rename(columns = column_mapping, inplace = True)
    sp.check_profile()
    return sp


def profile_from_dataframe(df, title='', depth_key='Depth', unit='m', column_mapping={}):
    """
    Creates a soil profile from a Pandas dataframe
    :param df: Dataframe to be converted
    :param depth_key: Column key to be used for depth (default = 'Depth')
    :param unit: Unit for the depth (default = 'm')
    :param column_mapping: Dictionary for renaming columns. The keys in this dictionary are the old column names and the values are the new column names.
    :return: ``SoilProfile`` object created as a deep copy of the dataframe
    """
    sp = deepcopy(df.reset_index(drop=True))
    sp.__class__ = SoilProfile
    sp.title = title
    sp.set_depthcolumn_name(name=depth_key, unit=unit)
    sp.rename(columns=column_mapping, inplace=True)
    sp.check_profile()
    return sp


def plot_fence_diagram(
    profiles=[], latlon=False,
    option='name', start=None, end=None, band=1000, extend_profile=False,
    soiltypekey="Soil type",
    plotmap=False,
    fillcolordict={'SAND': 'yellow', 'CLAY': 'brown', 'SILT': 'green', 'ROCK': 'grey'},
    opacity=1, logwidth=1, distance_unit='m', return_layers=False,
    showfig=True, xaxis_layout=None, yaxis_layout=None, general_layout=None,
    show_annotations=True):
    """
    Creates a longitudinal profile along selected soil profiles. A line is drawn from the first (smallest distance from origin)
    to the last location (greatest distance from origin) and the plot of the mini-logs with soil types
    is projected onto this line.

    :param profiles: List with SoilProfile objects for which a log needs to be plotted
    :param latlon: Boolean defining whether coordinates are specified in latitude and longitude (defaulte=False). If this is the case, offsets are calculated using the ``pyproj`` package
    :param option: Determines whether soil profile names (``option='name'``) or tuples with coordinates (``option='coords'``) are used for the ``start`` and ``end`` arguments
    :param start: Soil profile name for the starting point or tuple of coordinates. If a SoilProfile name is used, the selected SoilProfile must be contained in ``profiles``.
    :param end: Soil profile name for the end point or tuple of coordinates. If a SoilProfile name is used, the selected SoilProfile must be contained in ``profiles``.
    :param band: Offset from the line connecting start and end points in which soil profiles are considered for plotting (default=1000m)
    :param extend_profile: Boolean determining whether the profile needs to be extended beyond the start and end points (default=False)
    :param plotmap: Boolean determining whether a map of locations needs to be plotted next to the profile (default=False)
    :param soiltypekey: Key for the soil type in the dataframes with layering (default="Soil type")
    :param fillcolordict: Dictionary with fill colours (default yellow for 'SAND', brown from 'CLAY' and grey for 'ROCK')
    :param opacity: Opacity of the layers (default = 1 for non-transparent behaviour)
    :param logwidth: Width of the soil logs as an absolute value (default = 1)
    :param distance_unit: Unit for coordinates and elevation (default='m')
    :param return_layers: Boolean determining whether layers need to be returned. These layers can be used to updata another plot (e.g. CPT longitudinal profile) (default=False)
    :param showfig: Boolean determining whether the figure is shown (default=True)
    :param xaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param yaxis_layout: Dictionary with layout for the xaxis (default=None)
    :param general_layout: Dictionary with general layout options
    :param show_annotations: Boolean determining whether annotations need to be shown (default=True)
    :return: Plotly figure object
    """

    profile_names = []
    x_coords = []
    y_coords = []
    elevations = []
    for _profile in profiles:
        try:
            x_coords.append(_profile.easting)
            y_coords.append(_profile.northing)
            elevations.append(_profile.elevation)
            profile_names.append(_profile.title)
        except Exception as err:
            warnings.warn(
                "Profile %s - Error during processing for profile - %s" % (_profile.title, str(err)))
            raise

    if option == 'name':
        if start not in profile_names:
            raise ValueError('The soil profile used as starting point should be included in the list with titles')
        if end not in profile_names:
            raise ValueError('The soil profile used as end point should be included in the list with titles')

        start_point = (x_coords[profile_names.index(start)], y_coords[profile_names.index(start)])
        end_point = (x_coords[profile_names.index(end)], y_coords[profile_names.index(end)])
    elif option == 'coords':
        if start.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        start_point = start
        if end.__len__() != 2:
            raise ValueError("If option 'coords' is selected, start should contain an x,y pair")
        end_point = end
    else:
        raise ValueError("option should be 'name' or 'coords'")


    profile_df = pd.DataFrame({
        'Soil profiles': profiles,
        'Titles': profile_names,
        'X': x_coords,
        'Y': y_coords,
        'Z': elevations
    })

    # Calculate offsets from profile line
    for i, row in profile_df.iterrows():
        if row['X'] == start_point[0] and row['Y'] == start_point[1]:
            profile_df.loc[i, "Offset"] = 0
            profile_df.loc[i, "Projected offset"] = 0
            profile_df.loc[i, "Before start"] = False
            profile_df.loc[i, "Behind end"] = False
        elif row['X'] == end_point[0] and row['Y'] == end_point[1]:
            profile_df.loc[i, "Offset"] = 0
            if latlon:
                profile_df.loc[i, "Projected offset"] = latlon_distance(
                    start_point[0], start_point[1], end_point[0], end_point[1])
            else:
                profile_df.loc[i, "Projected offset"] = np.sqrt(
                    (start_point[0] - end_point[0]) ** 2 +
                    (start_point[1] - end_point[1]) ** 2)
            profile_df.loc[i, "Before start"] = False
            profile_df.loc[i, "Behind end"] = False
        else:
            result = offsets(start_point, end_point, (row['X'], row['Y']), latlon=latlon)
            profile_df.loc[i, "Offset"] = result['offset to line']
            profile_df.loc[i, "Projected offset"] = result['offset to start projected']
            profile_df.loc[i, "Before start"] = result['before start']
            profile_df.loc[i, "Behind end"] = result['behind end']

    # Determine which soil profiles need to be plotted
    if extend_profile:
        selected_profiles = deepcopy(profile_df[profile_df['Offset'] <= band])
    else:
        selected_profiles = deepcopy(profile_df[
            (profile_df['Offset'] <= band) &
            (profile_df['Before start'] == False) &
            (profile_df['Behind end'] == False)])

    selected_profiles.sort_values('Projected offset', inplace=True)

    _layers = []
    _backbone_traces = []
    _annotations = []

    for i, row in selected_profiles.iterrows():

        for j, _layer in row["Soil profiles"].iterrows():
            _fillcolor = fillcolordict[_layer[soiltypekey]]
            _y0 = row['Z'] - _layer['Depth from [m]']
            _y1 = row['Z'] - _layer['Depth to [m]']
            _x0 = row['Projected offset'] - 0.5 * logwidth
            _x1 = row['Projected offset'] + 0.5 * logwidth
            _layers.append(
                dict(type='rect', xref='x1', yref='y', x0=_x0, y0=_y0, x1=_x1, y1=_y1,
                     fillcolor=_fillcolor, opacity=opacity))
            if i % 2 == 0:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=row['Z'],
                        text="%s - Offset %.0f%s" % (
                            row['Titles'], row['Offset'], distance_unit
                    ))
                )
            else:
                _annotations.append(
                    dict(
                        x=row['Projected offset'],
                        y=-np.array(row['Soil profiles']['Depth to [m]']).max() + row['Z'],
                        text="%s - Offset %.0f%s" % (
                            row['Titles'], row['Offset'], distance_unit
                        ),
                        ay=30
                    )
                )


        try:
            _trace = go.Scatter(
                x=[row['Projected offset'], row['Projected offset']],
                y=[-row['Soil profiles']['Depth from [m]'].min() + row['Z'],
                   -row['Soil profiles']['Depth to [m]'].max() + row['Z']],
                showlegend=False,
                mode='lines',
                line=dict(color='black', width=0))
            _backbone_traces.append(_trace)
        except:
            pass

    mean_profile_depth = selected_profiles['Z'].mean()

    _soilcolors = []
    for key, value in fillcolordict.items():
        try:
            _trace = go.Bar(
                x=[0, 0],
                y=[mean_profile_depth, mean_profile_depth],
                name=key,
                marker=dict(color=value))
            _soilcolors.append(_trace)
        except:
            pass

    if return_layers:
        return _layers, _annotations, _backbone_traces, _soilcolors
    else:
        if plotmap:
            fig = subplots.make_subplots(rows=1, cols=2, print_grid=False, column_widths=[0.7, 0.3],
                                         specs=[[{'type': 'xy'}, {'type': 'mapbox'},]])
        else:
            fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)

        for _trace in _backbone_traces:
            fig.append_trace(_trace, 1, 1)

        for _trace in _soilcolors:
            fig.append_trace(_trace, 1, 1)

        if plotmap:
            mapbox_points = go.Scattermapbox(
                lat=y_coords, lon=x_coords, showlegend=False,
                mode='markers', name='Locations', hovertext=profile_names, marker=dict(color='black'))
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
            fig['layout']['yaxis1'].update(
                title='Level [%s]' % (distance_unit))
        else:
            fig['layout']['yaxis1'].update(yaxis_layout)
        if general_layout is None:
            fig['layout'].update(height=600, width=900,
                 title='Longitudinal profile from %s to %s' % (str(start), str(end)),
                 hovermode='closest',
                 legend=dict(orientation='h', x=0, y=-0.2))
        else:
            fig['layout'].update(general_layout)

        fig['layout'].update(shapes=_layers)

        if plotmap:
            fig.update_layout(
                mapbox_style='open-street-map', mapbox_zoom=10,
                mapbox_center={'lat': profile_df['Y'].mean(), 'lon': profile_df['X'].mean()}
            )

        if show_annotations:
            fig['layout'].update(annotations=_annotations)
        if showfig:
            fig.show(config=GROUNDHOG_PLOTTING_CONFIG)

        return fig


class CalculationGrid(object):
    """
    A ``CalculationGrid`` is an object which consist of a dataframe with nodes ``.nodes``
    and a dataframe with elements ``.elements``.
    Properties of the soil profile are mapped to the nodes and the elements to allow subsequent calculation.
    At layer transition nodes, the properties of the layer below are always assigned to the node.
    """
    
    def __init__(self, soilprofile, dz, custom_nodes=None, include_layertransitions=True):
        """
        Initializes the ``CalculationGrid`` object from a ``SoilProfile`` object.
        A nodes offset ``dz`` needs to be specified. Additional nodes are inserted
        at layer transitions by default (``include_layertransitions=True``).
        The user can also specify a NumPy array with custom_nodes (None by default).
        After initialization, the dataframes of nodal and elemental information are created.
        For properties varying linearly across a layer, the value of the property at the element center
        is also calculated and is given the name of the parameter (without from and to).
        """
        self.soilprofile = soilprofile
        self.set_nodes(dz=dz, custom_nodes=custom_nodes, include_layertransitions=include_layertransitions)
        self.set_elements()
        
    def set_nodes(self, dz, custom_nodes=None, include_layertransitions=True, **kwargs):
        if custom_nodes is None:
            no_nodes = int(np.ceil(1 + (self.soilprofile.max_depth - self.soilprofile.min_depth) / dz))
            self.nodes = self.soilprofile.map_soilprofile(
                nodalcoords=np.linspace(self.soilprofile.min_depth, self.soilprofile.max_depth, no_nodes),
                include_layertransitions=include_layertransitions, **kwargs)
        else:
            self.nodes = self.soilprofile.map_soilprofile(
                nodalcoords=custom_nodes,
                include_layertransitions=include_layertransitions,
                **kwargs)
            
    def set_elements(self):
        """
        Create a dataframe with elements. Each element has a top and bottom node. The soil property value at bottom and
        top are calculated, as well as the values at the center of the element.
        """
        self.elements = pd.DataFrame({
            "Depth from [m]": list(self.nodes['z [m]'][:-1]),
            "Depth to [m]": list(self.nodes['z [m]'][1:])
        })
        self.elements['z [m]'] = 0.5 * (
            self.elements['Depth from [m]'] +
            self.elements['Depth to [m]']
            )
        self.elements['dz [m]'] = list(self.nodes['z [m]'].diff()[1:])

        for i, _layer in self.soilprofile.iterrows():
            # Loop over layers in the soil profile and set parameters
            for j, _z in enumerate(self.elements['z [m]']):
                if _z < _layer["Depth from [m]"] or _z > _layer["Depth to [m]"]:
                    pass
                else:
                    for _param in self.soilprofile.numerical_soil_parameters():
                        if self.soilprofile.check_linear_variation(_param):
                            # Linearly varying parameters
                            self.elements.loc[j, "%sfrom [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])] = \
                                np.interp(
                                    self.elements.loc[j, "Depth from [m]"],
                                    [_layer["Depth from [m]"], _layer["Depth to [m]"]],
                                    [_layer["%sfrom [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])],
                                     _layer["%sto [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])]])
                            self.elements.loc[j, "%sto [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])] = \
                                np.interp(
                                    self.elements.loc[j, "Depth to [m]"],
                                    [_layer["Depth from [m]"], _layer["Depth to [m]"]],
                                    [_layer["%sfrom [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])],
                                     _layer["%sto [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])]])
                            self.elements.loc[j, _param] = \
                               np.interp(
                                    _z,
                                    [_layer["Depth from [m]"], _layer["Depth to [m]"]],
                                    [_layer["%sfrom [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])],
                                     _layer["%sto [%s" % (re.split('\[', _param)[0], re.split('\[', _param)[1])]])
                        else:
                            self.elements.loc[j, _param] = _layer[_param]
                    for _param in self.soilprofile.string_soil_parameters():
                        self.elements.loc[j, _param] = _layer[_param]
        self.elements = profile_from_dataframe(self.elements)

    def soilparameter_series(self, parameter, ignore_linearvariation=False):
        """
        Returns two lists (depths and corresponding parameter values) for plotting
        of a soil parameter vs depth.
        The routine first checks whether a valid parameter is provided.
        The lists are formatted such that variations at a layer interface are
        adequately plotted.
        :param parameter: A valid soil parameter with units
        :param ignore_linearvariation: Boolean determining if linear variations need to be ignored
        :return: Two lists (depths and corresponding parameter values)
        """
        if not parameter in self.elements.numerical_soil_parameters():
            raise ValueError("Parameter %s not defined, choose one of %s." % (
                parameter, self.soilprofile.numerical_soil_parameters()
            ))
        if self.elements.check_linear_variation(parameter) and not ignore_linearvariation:
            first_array_x = np.array(self.elements[parameter.replace(' [', ' from [')])
            second_array_x = np.array(self.elements[parameter.replace(' [', ' to [')])
        else:
            first_array_x = np.array(self.elements[parameter])
            second_array_x = np.array(self.elements[parameter])
        first_array_z = np.array(self.elements['Depth from [m]'])
        second_array_z = np.array(self.elements['Depth to [m]'])
        z = np.insert(second_array_z, np.arange(len(first_array_z)), first_array_z)
        x = np.insert(second_array_x, np.arange(len(first_array_x)), first_array_x)
        return z, x
