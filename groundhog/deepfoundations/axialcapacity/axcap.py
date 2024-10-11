#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
from logging import warning
import warnings

# 3rd party packages
import numpy as np
import pandas as pd

# Project imports
from groundhog.general.parameter_mapping import SOIL_PARAMETER_MAPPING, reverse_dict
from groundhog.deepfoundations.axialcapacity.skinfriction import SKINFRICTION_METHODS, SKINFRICTION_PARAMETERS
from groundhog.deepfoundations.axialcapacity.endbearing import ENDBEARING_METHODS, ENDBEARING_PARAMETERS
from groundhog.general.soilprofile import CalculationGrid
from groundhog.general.plotting import LogPlot


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
                    elif _param_name == 'z [m]':
                        # 'depth' or 'z [m]' is defined in the grid anyway
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
                    elif _param_name == 'z [m]':
                        # 'depth' or 'z [m]' is defined in the grid anyway
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
        if "Embedded length [m]" in self.output.keys():
            self.output.loc[:, "Embedded length [m]"] = pile_penetration

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

    def calculate_pilecapacity(self, circumference, base_area, internal_circumference=np.nan, annulus_area=np.nan,
        pile_weight=0, soilplug_weight=0):
        """
        Calculates the shaft friction of the pile by summing the unit shaft friction over all pile elements.

        The shaft friction can be calculated for the following modes:
            - Plugged compression (``plugged=True`` and ``compression=True``, default): Outside skin friction and end bearing over the full base area of the pile
            - Plugged tension (``plugged=True`` and ``compression=False``): Outside skin friction in tension + weight of the pile material and/or internal soil plug
            - Coring compression (``plugged=False`` and ``compression=True``): Outside skin friction + inside skin friction + annular end bearing.
            - Coring tension: (``plugged=False`` and ``compression=False``) Outside skin friction in tension + inside skin friction in tension.
   
        The pile circumference and base cross-sectional area need to be known.
        For coring conditions, the argument ``internal_circumference`` needs to be specified.
        For plugged conditions ``base_area`` is the full end area, for coring conditions, ``annulus_area`` is the steel annulus area.
        
        Because pile shapes and sizes may differ, the weight of the pile and/or internal soil plug is not calculated automatically
        but needs to be specified as ``pile_weight`` and ``soilplug_weight``.
        Calculating these components with a separate calculation is straightforward.

        The total capacity for plugged and coring conditions is calculated. A simple plugging criterion is
        also included, which for compression assesses whether the end bearing on the internal pile area is greater than the inside shaft friction.
        If the shaft friction is greater than the internal end bearing, the pile should behave plugged.
        For tension, plugging is expected to occur when the internal shaft friction is greater than the soil plug weight.

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
        :param base_area: Pile base area [m2]. Full end area used for plugged conditions.
        :param internal_circumference: Internal pile circumference used when ``plugged=False``
        :param annulus_area: Pile annular base area [m2]. Use annulus area for coring conditions.
        :param pile_weight: Pile weight in [kN] used for plugged tension capacity (default = 0kN)
        :param soilplug_weight: Soil plug weight in [kN] used for the plugged tension capacity (default = 0kN)

        The columns ``'Fs compression outside [kN]', 'Fs compression inside [kN]', 'Fs tension outside [kN]', 'Fs tension inside [kN]', 'Qb plugged [kN]', 'Qb coring'`` are added to the ``output`` attribute.
        ``'Fs compression outside [kN]'``, ``'Fs compression inside [kN]'``, ``'Fs tension outside [kN]'`` and ``'Fs tension inside [kN]'`` are cumulative sums.
        ``'Qb plugged [kN]'`` and ``'Qb coring'`` are multiplied by the local value.
        """
        self.output["Fs compression outside [kN]"] = \
            (circumference * self.output["dz [m]"] * self.output["Unit skin friction outside compression [kPa]"]).cumsum()
        self.output["Fs tension outside [kN]"] = \
            (circumference * self.output["dz [m]"] * self.output["Unit skin friction outside tension [kPa]"]).cumsum()
        self.output["Fs compression inside [kN]"] = \
            (internal_circumference * self.output["dz [m]"] * self.output["Unit skin friction inside compression [kPa]"]).cumsum()
        self.output["Fs tension inside [kN]"] = \
            (internal_circumference * self.output["dz [m]"] * self.output["Unit skin friction inside tension [kPa]"]).cumsum()
        self.output["Qb plugged [kN]"] = base_area * self.output["Unit end bearing plugged [kPa]"]
        self.output["Qb coring [kN]"] = annulus_area * self.output["Unit end bearing coring [kPa]"]
        self.output["Qb internal [kN]"] = (base_area - annulus_area) * self.output["Unit end bearing plugged [kPa]"]

        _Rs_compression_plugged = self.output["Fs compression outside [kN]"].iloc[-1]
        _Rb_plugged = self.output["Qb plugged [kN]"].iloc[-1]
        _Rs_compression_coring = self.output["Fs compression outside [kN]"].iloc[-1] + \
                    self.output["Fs compression inside [kN]"].iloc[-1]
        _Rb_coring = self.output["Qb coring [kN]"].iloc[-1]
        _Rb_internal = self.output["Qb internal [kN]"].iloc[-1]
        _Rt_compression_plugged = self.output["Fs compression outside [kN]"].iloc[-1] + \
            self.output["Qb plugged [kN]"].iloc[-1]
        _Rt_compression_coring = self.output["Fs compression outside [kN]"].iloc[-1] + \
            self.output["Fs compression inside [kN]"].iloc[-1] + self.output["Qb coring [kN]"].iloc[-1]
            
        if _Rb_internal < self.output["Fs compression inside [kN]"].iloc[-1]:
            _plugged_compression = True
            _Rt_compression = _Rt_compression_plugged
        else:
            _plugged_compression = False
            _Rt_compression = _Rt_compression_coring

        _Rs_tension_plugged = self.output["Fs tension outside [kN]"].iloc[-1]
        _Rs_tension_coring = self.output["Fs tension outside [kN]"].iloc[-1] + \
            self.output["Fs tension inside [kN]"].iloc[-1]
        _Rt_tension_plugged = self.output["Fs tension outside [kN]"].iloc[-1] + pile_weight + soilplug_weight
        _Rt_tension_coring = self.output["Fs tension outside [kN]"].iloc[-1] + \
            self.output["Fs tension inside [kN]"].iloc[-1] + pile_weight
        if self.output["Fs compression inside [kN]"].iloc[-1] > soilplug_weight:
            _plugged_tension = True
            _Rt_tension = _Rt_tension_plugged
        else:
            _plugged_tension = False
            _Rt_tension = _Rt_tension_coring

        self.result = {
            'Rs compression plugged [kN]': _Rs_compression_plugged,
            'Rb plugged [kN]': _Rb_plugged,
            'Rs compression coring [kN]': _Rs_compression_coring,
            'Rb coring [kN]': _Rb_coring,
            'Rb internal [kN]': _Rb_internal,
            'Rt compression plugged [kN]': _Rt_compression_plugged,
            'Rt compression coring [kN]': _Rt_compression_coring,
            'Rt compression [kN]': _Rt_compression,
            'Plugged compression': _plugged_compression,
            'Rs tension plugged [kN]': _Rs_tension_plugged,
            'Rs tension coring [kN]': _Rs_tension_coring,
            'Pile weight [kN]': pile_weight,
            'Soil plug weight [kN]': soilplug_weight,
            'Rt tension plugged [kN]': _Rt_tension_plugged,
            'Rt tension coring [kN]': _Rt_tension_coring,
            'Rt tension [kN]': _Rt_tension,
            'Plugged tension': _plugged_tension
        }

    def calculate_capacity_profile(self, circumference, base_area, internal_circumference=np.nan, annulus_area=np.nan,
        pile_weight_permeter=0, soilplug_weight_permeter=0):
        """
        Calculates compression and tension capacity vs pile penetration.
        Due to the possible dependence of unit skin friction and unit end bearing on pile penetration (e.g. friction fatigue effects),
        The pile capacity profile is calculated for every nodal position (except 0m) and stored in a dataframe.

        :param circumference: Pile circumference [m]. 
        :param base_area: Pile base area [m2]. Full end area used for plugged conditions.
        :param internal_circumference: Internal pile circumference used when ``plugged=False``
        :param annulus_area: Pile annular base area [m2]. Use annulus area for coring conditions.
        :param pile_weight_permeter: Pile weight in [kN/m] used for plugged tension capacity (default = 0kN/m). This value is multiplied by the actual pile penetration to obtain the total pile weight at the considered penetration.
        :param soilplug_weight_permeter: Soil plug weight in [kN/m] used for the plugged tension capacity (default = 0kN/m). This value is multiplied by the actual pile penetration to obtain the total soil plug weight at the considered penetration.
        """
        _capacity_profile = pd.DataFrame()
        for i, _z in enumerate(np.array(self.grid.nodes["z [m]"])[1:]):
            self.set_pilepenetration(pile_penetration=_z)
            self.calculate_unitskinfriction()
            self.calculate_unitendbearing()
            self.calculate_pilecapacity(
                circumference=circumference,
                base_area=base_area,
                internal_circumference=internal_circumference,
                annulus_area=annulus_area,
                pile_weight=pile_weight_permeter * _z,
                soilplug_weight=soilplug_weight_permeter * _z)
            _capacity_profile.loc[i, "Pile penetration [m]"] = _z
            for _key in self.result.keys():
                _capacity_profile.loc[i, _key] = self.result[_key]
        
        self.capacity_profile = _capacity_profile

    def plot_single_penetration(self, return_fig=False, plot_title=None, fillcolordict={'SAND': 'yellow', 'CLAY': 'brown'}, latex_titles=True):
        """
        Plots unit skin friction, unit end bearing, the integration of unit skin friction over the shaft
        and the value of end bearing at the tip
        """
        single_penetration_plot = LogPlot(soilprofile=self.sp, no_panels=4, fillcolordict=fillcolordict)

        z_fs, x_fs = self.output.soilparameter_series('Unit skin friction outside compression [kPa]')
        z_qb, x_qb = self.output.soilparameter_series('Unit end bearing coring [kPa]')

        if latex_titles:
            Fs_comp_out_trace_name = r'$ F_{s,comp,out} $'
            Fs_tens_out_trace_name = r'$ F_{s,tens,out} $'
            Fs_comp_in_trace_name = r'$ F_{s,comp,in} $'
            Fs_tens_in_trace_name = r'$ F_{s,tens,in} $'
            Qb_coring_trace_name = r'$ Q_{b,coring} $'
            Qb_plugged_trace_name = r'$ Q_{b,plugged} $'
        else:
            Fs_comp_out_trace_name = 'Fs,comp,out'
            Fs_tens_out_trace_name = 'Fs,tens,out'
            Fs_comp_in_trace_name = 'Fs,comp,in'
            Fs_tens_in_trace_name = 'Fs,tens,in'
            Qb_coring_trace_name = 'Qb,coring'
            Qb_plugged_trace_name = 'Qb,plugged'

        single_penetration_plot.add_trace(x=x_fs, z=z_fs, showlegend=False, mode='lines',name='fs comp', panel_no=1)
        single_penetration_plot.add_trace(x=x_qb, z=z_qb, showlegend=False, mode='lines',name='qb', panel_no=2)
        single_penetration_plot.add_trace(
            x=self.output["Fs compression outside [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Fs_comp_out_trace_name, panel_no=3, resetaxisrange=False)
        single_penetration_plot.add_trace(
            x=-self.output["Fs tension outside [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Fs_tens_out_trace_name, panel_no=3, resetaxisrange=False)
        single_penetration_plot.add_trace(
            x=self.output["Fs compression inside [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Fs_comp_in_trace_name, panel_no=3, resetaxisrange=False)
        single_penetration_plot.add_trace(
            x=-self.output["Fs tension inside [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Fs_tens_in_trace_name, panel_no=3, resetaxisrange=False)
        single_penetration_plot.add_trace(
            x=self.output["Qb coring [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Qb_coring_trace_name, panel_no=4, resetaxisrange=False)
        single_penetration_plot.add_trace(
            x=self.output["Qb plugged [kN]"],
            z=self.output["z [m]"],
            showlegend=True, mode='lines',name=Qb_plugged_trace_name, panel_no=4, resetaxisrange=False)

        if latex_titles:
            single_penetration_plot.set_xaxis(title=r'$ q_b \ \text{[kPa]} $', panel_no=2, range=(0, x_qb.max()))
            single_penetration_plot.set_xaxis(title=r'$ f_s \ \text{[kPa]} $', panel_no=1, range=(0, x_fs.max()))
            single_penetration_plot.set_xaxis(title=r'$ F_s \ \text{[kN]} $', panel_no=3)
            single_penetration_plot.set_xaxis(title=r'$ Q_b \ \text{[kN]} $', panel_no=4)
            single_penetration_plot.set_zaxis(title=r'$ z \ \text{[m]}$')
        else:
            single_penetration_plot.set_xaxis(title='qb [kPa]', panel_no=2, range=(0, x_qb.max()))
            single_penetration_plot.set_xaxis(title='fs [kPa]', panel_no=1, range=(0, x_fs.max()))
            single_penetration_plot.set_xaxis(title='Fs [kN]', panel_no=3)
            single_penetration_plot.set_xaxis(title='Qb [kN]', panel_no=4)
            single_penetration_plot.set_zaxis(title='z [m]')
        single_penetration_plot.fig['layout'].update(legend=dict(orientation='h', x=0.05, y=-0.1), title=plot_title)

        if return_fig:
            return single_penetration_plot
        else:
            single_penetration_plot.show()

    def plot_all_penetrations(self, return_fig=False, plot_title=None, fillcolordict={'SAND': 'yellow', 'CLAY': 'brown'}, latex_titles=True):
        """
        Plots shaft resistance, tip resistance and total pile resistance for all pile penetrations.
        """
        all_penetrations_plot = LogPlot(soilprofile=self.sp, no_panels=3, fillcolordict=fillcolordict)

        if latex_titles:
            Rs_comp_plugged_trace_name = r'$ R_{s,comp,plugged} $'
            Rs_tens_plugged_trace_name = r'$ R_{s,tens,plugged} $'
            Rs_comp_coring_trace_name = r'$ R_{s,comp,coring} $'
            Rs_tens_coring_trace_name = r'$ R_{s,tens,coring} $'
            Rb_plugged_trace_name = r'$ R_{b,plugged} $'
            Rb_coring_trace_name = r'$ R_{b,coring} $'
            Wpile_trace_name = r'$ W_{pile} $'
            Wsoilplug_trace_name = r'$ W_{soilplug} $'
            Rt_comp_plugged_trace_name = r'$ R_{t,comp,plugged} $'
            Rt_comp_coring_trace_name = r'$ R_{t,comp,coring} $'
            Rt_comp_trace_name = r'$ R_{t,comp} $'
            Rt_tens_plugged_trace_name = r'$ R_{t,tens,plugged} $'
            Rt_tens_coring_trace_name = r'$ R_{t,tens,coring} $'
            Rt_tens_trace_name = r'$ R_{t,tens} $'
        else:
            Rs_comp_trace_name = 'Rs,comp,plugged'
            Rs_tens_plugged_trace_name = 'Rs,tens,plugged'
            Rs_comp_coring_trace_name = 'Rs,comp,coring'
            Rs_tens_coring_trace_name = 'Rs,tens,coring'
            Rb_plugged_trace_name = 'Rb,plugged'
            Rb_coring_trace_name = 'Rb,coring'
            Wpile_trace_name = 'Wpile'
            Wsoilplug_trace_name = 'Wsoilplug'
            Rt_comp_plugged_trace_name = 'Rt,comp,plugged'
            Rt_comp_coring_trace_name = 'Rt,comp,coring'
            Rt_comp_trace_name = 'Rt,comp'
            Rt_tens_plugged_trace_name = 'Rt,tens,plugged'
            Rt_tens_coring_trace_name = 'Rt,tens,coring'
            Rt_tens_trace_name = 'Rt,tens'

        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rs compression plugged [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rs_comp_trace_name, panel_no=1, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Rs tension plugged [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rs_tens_plugged_trace_name, panel_no=1, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rs compression coring [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rs_comp_coring_trace_name, panel_no=1, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Rs tension coring [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rs_tens_coring_trace_name, panel_no=1, resetaxisrange=False)

        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rb plugged [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rb_plugged_trace_name, panel_no=2, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rb coring [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rb_coring_trace_name, panel_no=2, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Pile weight [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Wpile_trace_name, panel_no=2, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Soil plug weight [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Wsoilplug_trace_name, panel_no=2, resetaxisrange=False)

        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rt compression plugged [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rt_comp_plugged_trace_name, panel_no=3, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rt compression coring [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rt_comp_coring_trace_name, panel_no=3, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=self.capacity_profile["Rt compression [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            line=dict(dash='dot'),
            showlegend=True, mode='lines',name=Rt_comp_trace_name, panel_no=3, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Rt tension plugged [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rt_tens_plugged_trace_name, panel_no=3, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Rt tension coring [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            showlegend=True, mode='lines',name=Rt_tens_coring_trace_name, panel_no=3, resetaxisrange=False)
        all_penetrations_plot.add_trace(
            x=-self.capacity_profile["Rt tension [kN]"],
            z=self.capacity_profile["Pile penetration [m]"],
            line=dict(dash='dot'),
            showlegend=True, mode='lines',name=Rt_tens_trace_name, panel_no=3, resetaxisrange=False)

        if latex_titles:
            all_penetrations_plot.set_xaxis(title=r'$ R_s \ \text{[kN]} $', panel_no=1)
            all_penetrations_plot.set_xaxis(title=r'$ R_b \ \text{[kN]} $', panel_no=2)
            all_penetrations_plot.set_xaxis(title=r'$ R_t \ \text{[kN]} $', panel_no=3)
            all_penetrations_plot.set_zaxis(title=r'$ z \ \text{[m]}$')
        else:
            all_penetrations_plot.set_xaxis(title='Rs [kN]', panel_no=1)
            all_penetrations_plot.set_xaxis(title='Rb [kN]', panel_no=2)
            all_penetrations_plot.set_xaxis(title='Rt [kN]', panel_no=3)
            all_penetrations_plot.set_zaxis(title='z [m]')
        all_penetrations_plot.fig['layout'].update(legend=dict(orientation='h', x=0.05, y=-0.1), title=plot_title)
        
        if return_fig:
            return all_penetrations_plot
        else:
            all_penetrations_plot.show()