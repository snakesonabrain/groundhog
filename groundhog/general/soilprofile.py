#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
import re

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
        to the new layers above and below the transition
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
            for col in self.columns:
                self.loc[new_index, col] = row[col].iloc[0]
            self.loc[new_index, self.depth_from_col] = depth
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
        Returns tow lists (depths and corresponding parameter values) for plotting
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
            y=plotting_z,
            names=names,
            soildata=self,
            xtitles=x_titles,
            ztitle=z_title,
            showfig=False,
            **kwargs
        )

    def selection_soilparameter(self, parameter, depths, values, rule='mean', linearvariation=False):
        """
        Function for automatic selection of a soil parameters in the layers of a soil profile
        based an a list of provided values. The selection can either be done for a constant value in the layer or a linear variation
        over the layer.
        Selection of a minimum, mean or average trend can be performed using the ``rule`` keyword

        :param parameter: Name of the parameter being selected (including unit in square brackets)
        :param depths: Depths at which a measurement is available
        :param values: Values corresponding to the given depths
        :param rule: Which rule to use for the selection (select from ``min``, ``mean`` and ``max``
        :param linearvariation: Boolean determining whether a linear variation needs to happen over the layer or not
        :return: Adds one (constant value) or two (linear variation) columns to the dataframe
        """
        pass
        # TODO: Implement selection routine


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