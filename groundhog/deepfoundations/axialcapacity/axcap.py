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

    def set_pilepenetration(self, pile_penetration):
        """
        Sets a certain pile penetration for calculation. A deep copy of the ``grid.elements`` attribute is created, ``output``
        which can be used for further calculation.
        
        The pile penetration is checked (needs to be above the maximum depth of the ``SoilProfile``)
        
        :param pile_penetration: Pile penetration below surface in meters
        :return: Sets the ``output`` attribute, a deep copy of the ``grid.elements`` attribute between surface (0m) and the selected pile penetration
        """
        if pile_penetration <= 0:
            raise ValueError("Pile penetration cannot be smaller than or equal to 0")
        
        if pile_penetration > self.sp.max_depth:
            raise ValueError("Pile penetration cannot exceed the maximum depth of the soil profile")

        self.output = self.grid.elements.cut_profile(top_depth=0, bottom_depth=pile_penetration)

    def calculate_unitskinfriction(self, **kwargs):
        """
        Calculates unit skin friction for the selected method and soil parameters.
        Note that not all parameter combinations will return valid outputs.
        Therefore a check is performed to see if the output contains NaN values.

        The unit skin friction in compression and tension on the outside and inside (for tubulars) is returned in the ``grid.elements`` dataframe.
        The unit skin friction method used determines whether there are differences in these values.
        """
        self.fs_check = False
        self.output.rename(columns=SOIL_PARAMETER_MAPPING, inplace=True)
        try:
            for i, _element in self.output.iterrows():
                _fs_calc = SKINFRICTION_METHODS[_element['Unit skin friction']](**dict(_element))
                self.output.loc[i, "Unit skin friction outside compression [kPa]"] = _fs_calc['f_s_comp_out [kPa]']
                self.output.loc[i, "Unit skin friction inside compression [kPa]"] = _fs_calc['f_s_comp_in [kPa]']
                self.output.loc[i, "Unit skin friction outside tension [kPa]"] = _fs_calc['f_s_tens_out [kPa]']
                self.output.loc[i, "Unit skin friction inside tension [kPa]"] = _fs_calc['f_s_tens_in [kPa]']
        except Exception as err:
            warnings.warn('Error during calculation of unit skin friction (%s). Check inputs.' % str(err))
        finally:
            self.output.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        try:
            if np.isnan(np.array(self.output["Unit skin friction outside compression [kPa]"])).any():
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
        self.output.rename(columns=SOIL_PARAMETER_MAPPING, inplace=True)
        try:
            for i, _element in self.output.iterrows():
                _qb_calc = ENDBEARING_METHODS[_element['Unit end bearing']](**dict(_element))
                self.output.loc[i, "Unit end bearing plugged [kPa]"] = _qb_calc['q_b_plugged [kPa]']
                self.output.loc[i, "Unit end bearing coring [kPa]"] = _qb_calc['q_b_coring [kPa]']
            self.output.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        except Exception as err:
            warnings.warn('Error during calculation of unit end bearing (%s). Check inputs.' % str(err))
        finally:
            self.output.rename(columns=reverse_dict(SOIL_PARAMETER_MAPPING), inplace=True)
        try:
            if np.isnan(np.array(self.output["Unit end bearing plugged [kPa]"])).any():
                warnings.warn("NaN found in unit end bearing in output. Check parameter combinations.")
            else:
                self.qb_check = True
        except Exception as err:
            warning.warn("Not possible to check for the presence of NaN values in ouput (%s)" % str(err))

    def calculate_pilecapacity(self, circumference, base_area, plugged=True, internal_circumference=np.nan, compression=True,
        pile_weight=np.nan, soilplug_weight=np.nan):
        """
        Calculates the shaft friction of the pile by summing the unit shaft friction over all pile elements.

        The shaft friction can be calculated for the following modes:
            - Plugged compression (``plugged=True`` and ``compression=True``, default): Outside skin friction and end bearing over the full base area of the pile
            - Plugged tension (``plugged=True`` and ``compression=False``): Outside skin friction in tension + weight of the pile material and/or internal soil plug
            - Coring compression (``plugged=False`` and ``compression=True``): Outside skin friction + inside skin friction + annular end bearing.
            - Coring tension: (``plugged=False`` and ``compression=False``) Outside skin friction in tension + inside skin friction in tension.
   
        The pile circumference and base cross-sectional area need to be known.
        For coring conditions, the argument ``internal_circumference`` needs to be specified.
        For plugged conditions ``base_area`` is the full end area, for coring conditions, ``base_area`` is the steel annulus area.
        
        Because pile shapes and sizes may differ, the weight of the pile and/or internal soil plug is not calculated automatically
        but needs to be specified as ``pile_weight`` and ``soilplug_weight``.
        Calculating these components with a separate calculation is straightforward.

        .. math::
            F_{s,outside} = \\sum f_{s,out,i} \\cdot \\chi_{outside} \\cdot \\Delta z \\\\ \\text{for compression and tension}

            F_{s,inside} = \\sum f_{s,in,i} \\cdot \\chi_{inside} \\cdot \\Delta z \\\\ \\text{for compression and tension}

            Q_{b,plugged} = A_{base,full} \\cdot q_{b,plugged,base}

            Q_{b,coring} = A_{base,annulus} \\cdot q_{b,coring,base}

            R_{plugged,compression} = F_{s,outside,compression} + Q_{b,plugged}

            R_{coring,compression} = F_{s,outside,compression} + F_{s,inside,compression} + Q_{b,coring}
            
            R_{plugged,tension} = F_{s,outside,tension} + W_{pile} + W_{\\text{soil plug}}

            R_{coring,tension} = F_{s,outside, tension} + F_{s,inside,tension}
            
        :param circumference: Pile circumference [m]. 
        :param base_area: Pile base area [m2]. Use full end area for plugged conditions, annular area for coring conditions.
        :param plugged: Boolean describing the plugging condition (``plugged=True`` by default for plugged behaviour)
        :param internal_circumference: Internal pile circumference used when ``plugged=False``
        :param compression: Boolean describing whether compression or tension capacity is calculated (``compression=True`` by default for compressive behaviour)
        :param pile_weight: Pile weight in [kN] used for plugged tension capacity
        :param soilplug_weight: Soil plug weight in [kN] used for the plugged tension capacity
        """
        pass
        # TODO