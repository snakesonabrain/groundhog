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

# Project imports
from groundhog.general.plotting import plot_with_log

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

    def set_depthcolumn_name(self, name="Depth", unit='m'):
        self.depth_from_col = "%s from [%s]" % (name, unit)
        self.depth_to_col = "%s to [%s]" % (name, unit)

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

    def map_soilprofile(self, nodalcoords, target_depthkey='z [m]', invert_sign=False, offset=0):
        """
        Maps the soilprofile to a grid. The depth coordinates to the grid are specified
        in a list or Numpy array (``nodalcoords``).
        The depth coordinates should be strictly ascending (no duplicates) and the minimum and
        maximum should be contained inside the soil profile bounds.
        All soil parameters are interpolated onto this grid.
        :param nodalcoords: List or Numpy array with the nodal coordinates of the grid
        :param target_depthkey: Name of the depth key in the resulting dataframe
        :param invert_sign: Boolean determining whether to invert the sign after interpolation
        :param offset: Offset by which the depth is shifted (added to the depth after sign conversion)
        :return: Returns a dataframe with the full grid with soil parameters
        """
        # 1. Convert nodalcoords to Numpy array
        try:
            z = np.array(nodalcoords)
        except Exception as err:
            raise ValueError("Could not convert nodal coords to Numpy array (%s)" % str(err))
        # Create a target dataframe with the depth column
        target_df = pd.DataFrame({
            target_depthkey: z,
        })

        # 2. Validate if nodal coordinates array is sorted and strictly ascending
        if np.all(np.diff(z) >= 0):
            pass
        else:
            raise ValueError("Nodal coordinates are not ascending. Please check ")

        # 3. Map string parameters
        for _param in self.string_soil_parameters():
            target_df[_param] = list(map(lambda _z: self[
                (self[self.depth_from_col] <= _z) & (self[self.depth_to_col] >= _z)][_param].iloc[-1],
                                      target_df[target_depthkey]))

        # 4. Map numerical parameters
        for _param in self.numerical_soil_parameters():
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
                        x=selected_depths,
                        y=selected_values,
                        deg=1)
                    fit_func = np.poly1d(fit_coeff)
                    residuals = np.array(selected_values) - fit_func(selected_depths)
                    sorted_residuals = np.sort(residuals)
                    if rule == 'min':
                        # TODO: Double-check routine for trend selection
                        selected_points = (
                            np.array(residuals) <= sorted_residuals[1])
                        zs = selected_depths[selected_points]
                        xs = selected_values[selected_points]
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
                        zs = selected_depths[selected_points]
                        xs = selected_values[selected_points]
                        self.loc[i, parameter.replace(' [', ' from [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_from_col] - zs[0])
                        self.loc[i, parameter.replace(' [', ' to [')] = \
                            xs[0] + ((xs[1] - xs[0]) / (zs[1] - zs[0])) * (row[self.depth_to_col] - zs[0])
                    else:
                        raise ValueError("Rule %s unknown. Rule should be one of 'min', 'mean' or 'max'" % rule)
                else:
                    if rule == 'min':
                        self.loc[i, parameter] = selected_values.min()
                    elif rule == 'mean':
                        self.loc[i, parameter] = selected_values.mean()
                    elif rule == 'max':
                        self.loc[i, parameter] = selected_values.max()
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

    def cut_profile(self, top_depth, bottom_depth):
        """
        Returns a deep copy of the ``SoilProfile`` between the specified bounds
        :param top_depth: Top depth for cutting
        :param bottom_depth: Bottom depth for cutting
        :return: Deep copy of the ``SoilProfile`` between the specified bounds
        """
        if top_depth < self.min_depth:
            warnings.warn("Top depth of %.2f is smaller than minimum depth of the soil profile." % bottom_depth)
            top_depth = self.min_depth

        if bottom_depth > self.max_depth:
            warnings.warn("Bottom depth of %.2f is greater than maximum depth of the soil profile." % top_depth)
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
                _profile[_from_param].iloc[0] = np.interp(
                    top_depth,
                    [_profile[self.depth_from_col].iloc[0],
                     _profile[self.depth_to_col].iloc[0]],
                    [_profile[_from_param].iloc[0],
                     _profile[_to_param].iloc[0]]
                )
                _profile[_to_param].iloc[-1] = np.interp(
                    bottom_depth,
                    [_profile[self.depth_from_col].iloc[-1], _profile[self.depth_to_col].iloc[-1]],
                    [_profile[_from_param].iloc[-1],
                     _profile[_to_param].iloc[-1]]
                )

        # Adjust bounds
        _profile[self.depth_from_col].iloc[0] = top_depth
        _profile[self.depth_to_col].iloc[-1] = bottom_depth

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

    def calculate_overburden(self, waterlevel=0, waterunitweight=10, totalunitweightcolumn="Total unit weight [kN/m3]",
                             effectiveunitweightcolumn="Effective unit weight [kN/m3]",
                             waterunitweightcolumn="Water unit weight [kN/m3]",
                             totalverticalstresscolumn="Total vertical stress [kPa]",
                             effectiveverticalstresscolumn="Effective vertical stress [kPa]",
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

        :param waterlevel: Water level [m] (default 0m)
        :param waterunitweight: Unit weight of the pore water [kN/m3] (default=10kN/m3)
        :param totalunitweightcolumn: Column name containing total unit weights (default='Total unit weight [kN/m3]'
        :param waterunitweightcolumn: Output column with the effective unit weight (default='Effective unit weight [kN/m3]')
        :param waterunitweightcolumn: Output column with the water unit weight (default='Water unit weight [kN/m3]')
        :param totalverticalstresscolumn: Total vertical stress column name (default='Total vertical stress [kPa]')
        :param effectiveverticalstresscolumn: Effective vertical stress column name (default='Effective vertical stress [kPa]')
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
        self.depth_integration(parameter=effectiveunitweightcolumn, outputparameter=effectiveverticalstresscolumn)

        # Calculate total vertical stress
        self.depth_integration(parameter=totalunitweightcolumn, outputparameter=totalverticalstresscolumn)


def read_excel(path, depth_key='Depth', unit='m', column_mapping={}, **kwargs):
    """
    The method to read from Excel needs to be redefined for SoilProfile objects.
    The method allows for different depth keys (using the 'depth_key' and 'unit' keyword arguments
    Columns can also be renamed using the 'column_mapping' dictionary. The keys in this dictionary are
    the old column names and the values are the new column names.
    """
    sp = pd.read_excel(path, **kwargs)
    sp.__class__ = SoilProfile
    sp.set_depthcolumn_name(name=depth_key, unit=unit)
    sp.rename(columns = column_mapping, inplace = True)
    sp.check_profile()
    return sp