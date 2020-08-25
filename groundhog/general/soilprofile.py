#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports


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