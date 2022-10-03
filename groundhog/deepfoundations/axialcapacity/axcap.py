#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
from logging import warning
import warnings

# 3rd party packages
import numpy as np

# Project imports
from distutils.log import error, warn

from sympy import EX
from groundhog.general.parameter_mapping import SOIL_PARAMETER_MAPPING, reverse_dict
from groundhog.deepfoundations.axialcapacity.skinfriction import SKINFRICTION_METHODS, SKINFRICTION_PARAMETERS
from groundhog.deepfoundations.axialcapacity.endbearing import ENDBEARING_METHODS, ENDBEARING_PARAMETERS
from groundhog.general.soilprofile import CalculationGrid


class AxCapCalculation(object):

    def __init__(self, soilprofile):
        """
        Sets up an axial capacity calculation by taking in a ``SoilProfile`` object.
        The ``SoilProfile`` object needs to have the columns ``Unit skin friction`` and ``Unit end bearing`` specified.
        """
        if not 'Unit skin friction' in soilprofile.string_soil_parameters():
            raise ValueError("Column 'Unit skin friction' is not specified")
        
        if not 'Unit skin friction' in soilprofile.string_soil_parameters():
            raise ValueError("Column 'Unit end bearing' is not specified")
        
        self.sp = soilprofile

    def check_methods(self, raise_errors=False):
        """
        Checks the given ``SoilProfile`` object for the presence of the required soil parameters.
        Errors are printed by default but error raising can also be requested by setting ``raise_errors=True``
        """
        self.checked = True
        for _method in self.sp['Unit skin friction'].unique():
            if not _method in SKINFRICTION_METHODS:
                self.checked = False
                error_message = "Skin friction method %s undefined: Select from %s" % (_method, str(SKINFRICTION_METHODS.keys()))
                if raise_errors:
                    raise ValueError(error_message)
                else:
                    print(error_message)
            else:
                # Check required parameters for unit skin friction
                for _param in SKINFRICTION_PARAMETERS[_method]:
                    _param_name = reverse_dict(SOIL_PARAMETER_MAPPING)[_param]
                    if _param_name in self.sp.string_soil_parameters() or _param_name in self.sp.numerical_soil_parameters():
                        pass
                    else:
                        self.checked = False
                        error_message = "Skin friction method %s: Required parameter %s not found" % (_method, _param_name)
                        if raise_errors:
                            raise ValueError(error_message)
                        else:
                            print(error_message)
        for _method in self.sp['Unit end bearing'].unique():
            if not _method in ENDBEARING_METHODS:
                self.checked = False
                error_message = "End bearing method %s undefined: Select from %s" % (_method, str(ENDBEARING_METHODS.keys()))
                if raise_errors:
                    raise ValueError(error_message)
                else:
                    print(error_message)
            else:
                # Check required parameters for unit skin friction
                for _param in ENDBEARING_PARAMETERS[_method]:
                    _param_name = reverse_dict(SOIL_PARAMETER_MAPPING)[_param]
                    if _param_name in self.sp.string_soil_parameters() or _param_name in self.sp.numerical_soil_parameters():
                        pass
                    else:
                        self.checked = False
                        error_message = "End bearing method %s: Required parameter %s not found" % (_method, _param_name)
                        if raise_errors:
                            raise ValueError(error_message)
                        else:
                            print(error_message)

    def create_grid(self, dz=1, **kwargs):
        """
        Sets the ``CalculationGrid`` for the pile capacity calculation.

        A default node spacing of 1m is used but finer or coarser grids can be specified with the argument ``dz``.
        Layer transitions not coinciding with the grid nodes are added as additional nodes by default.

        For additional options (``**kwargs``), check the documentation of ``CalculationGrid``
        """
        if not self.checked:
            raise ValueError("Calculation cannot continue if the required soil parameters are not specified.")
        else:
            self.grid = CalculationGrid(self.sp, dz=dz)

    def calculate_unitskinfriction(self, **kwargs):
        """
        Calculates unit skin friction for the selected method and soil parameters.
        Note that not all parameter combinations will return valid outputs.
        Therefore a check is performed to see if the output contains NaN values.

        The unit skin friction in compression and tension on the outside and inside (for tubulars) is returned in the ``grid.elements`` dataframe.
        The unit skin friction method used determines whether there are differences in these values.
        """
        self.fs_check = False
        self.grid.elements.rename(columns=SOIL_PARAMETER_MAPPING, inplace=True)
        try:
            for i, _element in self.grid.elements.iterrows():
                _fs_calc = SKINFRICTION_METHODS[_element['Unit skin friction']](**dict(_element))
                self.grid.elements.loc[i, "Unit skin friction outside compression [kPa]"] = _fs_calc['f_s_comp_out [kPa]']
                self.grid.elements.loc[i, "Unit skin friction inside compression [kPa]"] = _fs_calc['f_s_comp_in [kPa]']
                self.grid.elements.loc[i, "Unit skin friction outside tension [kPa]"] = _fs_calc['f_s_tens_out [kPa]']
                self.grid.elements.loc[i, "Unit skin friction inside tension [kPa]"] = _fs_calc['f_s_tens_in [kPa]']
        except Exception as err:
            warnings.warn('Error during calculation of unit skin friction (%s). Check inputs.' % str(err))
        finally:
            self.grid.elements.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        try:
            if np.isnan(np.array(self.grid.elements["Unit skin friction outside compression [kPa]"])).any():
                warnings.warn("NaN found in unit skin friction in output. Check parameter combinations.")
            else:
                self.fs_check = True
        except Exception as err:
            warning.warn("Not possible to check for the presence of NaN values in ouput (%s)" % str(err))

    def calculate_unitendbearing(self, **kwargs):
        """
        Calculates unit end bearing for the selected method and soil parameters.
        Note that not all parameter combinations will return valid outputs.
        Therefore a check is performed to see if the output contains NaN values.

        The unit end bearing for plugged and coring conditions is returned in the ``grid.elements`` dataframe.
        The unit end bearing method used determines whether there are differences in these values.
        """
        self.qb_check = False
        self.grid.elements.rename(columns=SOIL_PARAMETER_MAPPING, inplace=True)
        try:
            for i, _element in self.grid.elements.iterrows():
                _qb_calc = ENDBEARING_METHODS[_element['Unit end bearing']](**dict(_element))
                self.grid.elements.loc[i, "Unit end bearing plugged [kPa]"] = _qb_calc['q_b_plugged [kPa]']
                self.grid.elements.loc[i, "Unit end bearing coring [kPa]"] = _qb_calc['q_b_coring [kPa]']
            self.grid.elements.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        except Exception as err:
            warnings.warn('Error during calculation of unit end bearing (%s). Check inputs.' % str(err))
        finally:
            self.grid.elements.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        try:
            if np.isnan(np.array(self.grid.elements["Unit end bearing plugged [kPa]"])).any():
                warnings.warn("NaN found in unit end bearing in output. Check parameter combinations.")
            else:
                self.qb_check = True
        except Exception as err:
            warning.warn("Not possible to check for the presence of NaN values in ouput (%s)" % str(err))