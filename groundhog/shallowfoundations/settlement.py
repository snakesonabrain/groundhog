#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
from copy import deepcopy

# 3rd party packages
from mimetypes import init
import numpy as np

# Project imports
from groundhog.general.validation import Validator
from groundhog.siteinvestigation.classification.phaserelations import voidratio_bulkunitweight
from groundhog.general.plotting import LogPlot
from groundhog.shallowfoundations.stressdistribution import stresses_stripload, stresses_circle, \
    stresses_rectangle
from groundhog.general.soilprofile import CalculationGrid, profile_from_dataframe


PRIMARYCONSOLIDATIONSETTLEMENT_NC = {
    'initial_height': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'initial_voidratio': {'type': 'float', 'min_value': 0.1, 'max_value': 5.0},
    'initial_effective_stress': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'effective_stress_increase': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'compression_index': {'type': 'float', 'min_value': 0.1, 'max_value': 0.8},
    'e_min': {'type': 'float', 'min_value': 0.1, 'max_value': None},
}

PRIMARYCONSOLIDATIONSETTLEMENT_NC_ERRORRETURN = {
    'delta z [m]': np.nan,
    'delta e [-]': np.nan,
    'e final [-]': np.nan
}


@Validator(PRIMARYCONSOLIDATIONSETTLEMENT_NC, PRIMARYCONSOLIDATIONSETTLEMENT_NC_ERRORRETURN)
def primaryconsolidationsettlement_nc(
        initial_height, initial_voidratio, initial_effective_stress, effective_stress_increase, compression_index, e_min=0.3,
        **kwargs):
    """
    Calculates the primary consolidation settlement for normally consolidated fine grained soil.

    :param initial_height: Initial thickness of the layer (:math:`H_0`) [:math:`m`] - Suggested range: initial_height >= 0.0
    :param initial_voidratio: Initial void ratio of the layer (:math:`e_0`) [:math:`-`] - Suggested range: 0.1 <= initial_voidratio <= 5.0
    :param initial_effective_stress: Initial vertical effective stress in the center of the layer (:math:`\\sigma_{v0}^{\\prime}`) [:math:`kPa`] - Suggested range: initial_effective_stress >= 0.0
    :param effective_stress_increase: Increase in vertical effective stress under the given load (:math:`\\Delta sigma_{v}^{\\prime}`) [:math:`kPa`] - Suggested range: effective_stress_increase >= 0.0
    :param compression_index: Compression index derived from oedometer tests (:math:`C_c`) [:math:`-`] - Suggested range: 0.1 <= compression_index <= 0.8 (derived using logarithm with base 10)
    :param e_min: Minimum void ratio below which no further consolidation occurs (:math:`e_{min}`) [:math:`-`] - Default=0.3

    .. math::
        \\Delta z = \\frac{H_0}{1 + e_0} C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}

        \\Delta e = C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}

    :returns: Dictionary with the following keys:

        - 'delta z [m]': Primary consolidation settlement for normally consolidated soil (:math:`\\Delta z`)  [:math:`m`]
        - 'delta e [-]': Decrease in void ratio for the normally consolidated soil (:math:`\\delta e`)  [:math:`-`]
        - 'e final [-]': Final void ratio after consolidation (:math:` e_{final}`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _delta_e = compression_index * \
        np.log10((initial_effective_stress + effective_stress_increase) / initial_effective_stress)

    if (initial_voidratio - _delta_e) > e_min:
        pass
    else:
        _delta_e = initial_voidratio - e_min    
    
    _delta_z = (initial_height / (1 + initial_voidratio)) * _delta_e
    _e_final = initial_voidratio - _delta_e

    return {
        'delta z [m]': _delta_z,
        'delta e [-]': _delta_e,
        'e final [-]': _e_final
    }


PRIMARYCONSOLIDATIONSETTLEMENT_OC = {
    'initial_height': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'initial_voidratio': {'type': 'float', 'min_value': 0.1, 'max_value': 5.0},
    'initial_effective_stress': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'preconsolidation_pressure': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'effective_stress_increase': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'compression_index': {'type': 'float', 'min_value': 0.1, 'max_value': 0.8},
    'recompression_index': {'type': 'float', 'min_value': 0.015, 'max_value': 0.35},
    'e_min': {'type': 'float', 'min_value': 0.1, 'max_value': None},
}

PRIMARYCONSOLIDATIONSETTLEMENT_OC_ERRORRETURN = {
    'delta z [m]': np.nan,
    'delta e [-]': np.nan,
    'e final [-]': np.nan
}


@Validator(PRIMARYCONSOLIDATIONSETTLEMENT_OC, PRIMARYCONSOLIDATIONSETTLEMENT_OC_ERRORRETURN)
def primaryconsolidationsettlement_oc(
        initial_height, initial_voidratio, initial_effective_stress, preconsolidation_pressure,
        effective_stress_increase, compression_index, recompression_index, e_min=0.3,
        **kwargs):
    """
    Calculates the primary consolidation settlement for an overconsolidated clay. This material is characterised using a compression index and a recompression index which can be derived from oedometer tests.

    The settlement depends on whether the stress increase loads the layer beyond the preconsolidation pressure. If stresses remain below the preconsolidation pressure, the recompression index applies. If stresses go beyond the preconsolidation pressure, the compression index will apply for the increase beyond the preconsolidation pressure.

    Note that a minimum void ratio is set to prevent calculated void ratios from dropping below the minimum.

    :param initial_height: Initial thickness of the layer (:math:`H_0`) [:math:`m`] - Suggested range: initial_height >= 0.0
    :param initial_voidratio: Initial void ratio of the layer (:math:`e_0`) [:math:`-`] - Suggested range: 0.1 <= initial_voidratio <= 5.0
    :param initial_effective_stress: Initial vertical effective stress in the center of the layer (:math:`\\sigma_{v0)^{\\prime}`) [:math:`kPa`] - Suggested range: initial_effective_stress >= 0.0
    :param preconsolidation_pressure: Preconsolidation pressure, maximum vertical stress to which the layer has been subjected (:math:`p_c^{\\prime}`) [:math:`kPa`] - Suggested range: preconsolidation_pressure >= 0.0
    :param effective_stress_increase: Increase in vertical effective stress under the given load (:math:`\\Delta sigma_{v}^{\\prime}`) [:math:`kPa`] - Suggested range: effective_stress_increase >= 0.0
    :param compression_index: Compression index derived from oedometer tests (:math:`C_c`) [:math:`-`] - Suggested range: 0.1 <= compression_index <= 0.8
    :param recompression_index: Recompression index derived from the unloading step in oedometer tests (:math:`C_r`) [:math:`-`] - Suggested range: 0.015 <= recompression_index <= 0.35
    :param e_min: Minimum void ratio below which no further consolidation occurs (:math:`e_{min}`) [:math:`-`] - Default=0.3

    .. math::
        \\Delta z = \\frac{H_0}{1 + e_0} C_r \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} < p_c^{\\prime}

        \\Delta z = \\frac{H_0}{1 + e_0} \\left( C_r \\log_{10} \\frac{p_c^{\\prime}}{\\sigma_{v0}^{\\prime}} + C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{p_c^{\\prime}} \\right); \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} > p_c^{\\prime}

        \\Delta e = C_r \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{\\sigma_{v0}^{\\prime}}; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} < p_c^{\\prime}

        \\Delta e = C_r \\log_{10} \\frac{p_c^{\\prime}}{\\sigma_{v0}^{\\prime}} + C_c \\log_{10} \\frac{\\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime}}{p_c^{\\prime}} ; \\ \\sigma_{v0}^{\\prime} + \\Delta \\sigma_v^{\\prime} > p_c^{\\prime}

    :returns: Dictionary with the following keys:

        - 'delta z [m]': Primary consolidation settlement for the overconsolidated soil (:math:`\\delta z`)  [:math:`m`]
        - 'delta e [-]': Decrease in void ratio for the overconsolidated soil (:math:`\\delta e`)  [:math:`-`]
        - 'e final [-]': Final void ratio after consolidation (:math:` e_{final}`)  [:math:`-`]

    .. figure:: images/primaryconsolidation_settlement.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Cases for calculating the primary consolidation settlement

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    if (initial_effective_stress + effective_stress_increase) < preconsolidation_pressure:
        _delta_e = recompression_index * np.log10((initial_effective_stress + effective_stress_increase) / initial_effective_stress)
        
    else:
        _delta_e = \
            recompression_index * np.log10(preconsolidation_pressure / initial_effective_stress) + \
            compression_index * np.log10(
                (initial_effective_stress + effective_stress_increase) / preconsolidation_pressure)

    if (initial_voidratio - _delta_e) > e_min:
        pass
    else:
        _delta_e = initial_voidratio - e_min    
    
    _delta_z = (initial_height / (1 + initial_voidratio)) * _delta_e
    _e_final = initial_voidratio - _delta_e

    return {
        'delta z [m]': _delta_z,
        'delta e [-]': _delta_e,
        'e final [-]': _e_final
    }

CONSOLIDATIONSETTLEMENT_MV = {
    'initial_height': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'effective_stress_increase': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'compressibility': {'type': 'float', 'min_value': 1e-4, 'max_value': 10}
}

CONSOLIDATIONSETTLEMENT_MV_ERRORRETURN = {
    'delta z [m]': np.nan,
    'delta epsilon [-]': np.nan
}


@Validator(CONSOLIDATIONSETTLEMENT_MV, CONSOLIDATIONSETTLEMENT_MV_ERRORRETURN)
def consolidationsettlement_mv(
        initial_height, effective_stress_increase, compressibility,
        **kwargs):
    """
    Calculates the consolidation settlement using the compressibility :math:`m_v` (inverse of constrained modulus :math:`M`).

    The constrained modulus is stress-dependent and therefore the relation between :math:`M` and :math:`C_c` is also stress-dependent.

    :param initial_height: Initial thickness of the layer (:math:`H_0`) [:math:`m`] - Suggested range: initial_height >= 0.0
    :param effective_stress_increase: Increase in vertical effective stress under the given load (:math:`\\Delta sigma_{v}^{\\prime}`) [:math:`kPa`] - Suggested range: effective_stress_increase >= 0.0
    :param compressibility: Modulus of volumetric compressibility derived from oedometer tests (:math:`m_v`) [:math:`-`] - Suggested range: 1e-4 <= compressibility <= 10 (note that the compressibility is stress-dependent)
    
    .. math::
        \\Delta \\epsilon = m_v \\cdot \\Delta \\sigma_v^{\\prime}

        \\Delta z = \\Delta \\epsilon \\cdot H_0

        m_v = \\frac{C_c}{2.3 \\cdot (1 + e_0) \\cdot \\sigma_{v0}^{\\prime}}

    :returns: Dictionary with the following keys:

        - 'delta z [m]': Consolidation settlement (:math:`\\Delta z`)  [:math:`m`]
        - 'delta epsilon [-]': Change in strain caused by consolidation (:math:`\\delta \\epsilon`)  [:math:`-`]

    Reference - Budhu (2011). Soil mechanics and foundation engineering

    """

    _delta_epsilon = compressibility * effective_stress_increase
    _delta_z = _delta_epsilon * initial_height

    return {
        'delta z [m]': _delta_z,
        'delta epsilon [-]': _delta_epsilon
    }

class SettlementCalculation(object):
    """
    Calculates shallow foundation settlement under a certain distributed load
    """
    
    def __init__(self, soilprofile):
        """
        Initializes the settlement calculation with a certain ``SoilProfile`` object.
        The ``SoilProfile`` object is checked for the presence of the required columns:
        (``'Total unit weight [kN/m3]'``, ``'Cc [-]'``, ``'Cr [-]'``, ``'OCR [-]'``).
        Optionally, a column with the saturation ``S [-]`` can be defined (ranging from 0 to 1).
        If a saturation is defined, it will be taken into account for the calculation of the void ratio.
        """
        self.soilprofile = profile_from_dataframe(deepcopy(soilprofile))
        for _param in ['Total unit weight [kN/m3]', 'Cc [-]', 'Cr [-]', 'mv [1/kPa]','OCR [-]']:
            if not _param in self.soilprofile.numerical_soil_parameters():
                if _param == 'Cc [-]' or _param == 'Cr [-]' or _param == 'OCR [-]':
                    if 'mv [1/kPa]' in self.soilprofile.numerical_soil_parameters():
                        pass
                    else:
                        raise KeyError("'mv [1/kPa]' or 'Cc [-]', 'Cr [-]' and 'OCR [-]' need to be defined")
                if _param == 'mv [1/kPa]':
                    if ('Cc [-]' not in self.soilprofile.numerical_soil_parameters()) or \
                        ('Cr [-]' not in self.soilprofile.numerical_soil_parameters()) or \
                        ('OCR [-]' not in self.soilprofile.numerical_soil_parameters()):
                        raise KeyError("'mv [1/kPa]' or 'Cc [-]', 'Cr [-]' and 'OCR [-]' need to be defined")
                    else:
                        pass
                
        if not 'S [-]' in self.soilprofile.numerical_soil_parameters():
            warnings.warn("Saturation 'S [-]' not defined. Layers above the water table will be assumed dry.")
            self.saturation_defined = False
        else:
            self.saturation_defined = True
                
    def calculate_initial_state(self, waterlevel, specific_gravity=2.65,
                                unitweight_water=10.0, **kwargs):
        """
        Calculates the initial stress distribution and void ratio.
        The water level needs to be set for every calculation (z-axis positive downward).
        """
        self.waterlevel = waterlevel
        self.soilprofile.calculate_overburden(waterlevel=waterlevel)
        self.soilprofile['Depth center [m]'] = 0.5 * (
            self.soilprofile['Depth from [m]'] + self.soilprofile['Depth to [m]'])
        for i, _layer in self.soilprofile.iterrows():
            if _layer['Depth center [m]'] < self.waterlevel:
                if self.saturation_defined:
                    self.soilprofile.loc[i, 'e0 [-]'] = voidratio_bulkunitweight(
                        bulkunitweight=_layer['Total unit weight [kN/m3]'],
                        saturation=_layer['S [-]'],
                        specific_gravity=specific_gravity,
                        unitweight_water=unitweight_water, **kwargs)['e [-]']
                else:
                    self.soilprofile.loc[i, 'e0 [-]'] = voidratio_bulkunitweight(
                        bulkunitweight=_layer['Total unit weight [kN/m3]'],
                        saturation=0,
                        specific_gravity=specific_gravity,
                        unitweight_water=unitweight_water, **kwargs)['e [-]']
            else:
                self.soilprofile.loc[i, 'e0 [-]'] = voidratio_bulkunitweight(
                    bulkunitweight=_layer['Total unit weight [kN/m3]'],
                    saturation=1,
                    specific_gravity=specific_gravity,
                    unitweight_water=unitweight_water)['e [-]']
        if 'mv [1/kPa]' in self.soilprofile.numerical_soil_parameters():
            pass
        else:
            self.soilprofile['pc from [kPa]'] = self.soilprofile['Vertical effective stress from [kPa]'] * \
                self.soilprofile['OCR [-]']
            self.soilprofile['pc to [kPa]'] = self.soilprofile['Vertical effective stress to [kPa]'] * \
                self.soilprofile['OCR [-]']
                                
    def plot_initial_state(self, plot_title="", e0_range=(0, 3), fillcolordict={'SAND': 'yellow', 'CLAY': 'brown'}, latex_titles=True, **kwargs):
        """
        Plots the initial stress vs depth and the initial void ratio vs depth
        """
        if latex_titles:
            e0_trace_title = r'$ e_0 $'
            sigma_v0_eff_trace_title = r'$ \sigma_{vo}^{\prime} $'
            xaxis_title_nc = r'$ \sigma_{vo}, \ \sigma_{vo}^{\prime}, \ \text{[kPa]} $'
            xaxis_title_oc = r'$ \sigma_{vo}, \ \sigma_{vo}^{\prime}, \ \sigma_{vc}^{\prime} \ \text{[kPa]} $'
            sigma_vc_trace_title = r'$ \sigma_{vc}^{\prime} $'
            sigma_v0_trace_title = r'$ \sigma_{vo} $'
            e0_axis_title = r'$ e_0 \ \text{[-]} $'
        else:
            e0_trace_title = 'e0'
            sigma_v0_eff_trace_title = 'sigma,v0,eff'
            xaxis_title_nc = 'sigma,v0, sigma,v0,eff [kPa]'
            xaxis_title_oc = 'sigma,v0, sigma,v0,eff, sigma,vc,eff [kPa]'
            sigma_vc_trace_title = 'sigma,vc,eff'
            sigma_v0_trace_title = 'sigma,v0'
            e0_axis_title = 'e0 [-]'

        self.initial_state_plot = LogPlot(soilprofile=self.soilprofile, no_panels=2, fillcolordict=fillcolordict)
        self.initial_state_plot.add_trace(
            x=self.soilprofile.soilparameter_series('e0 [-]')[1],
            z=self.soilprofile.soilparameter_series('e0 [-]')[0],
            line=dict(color='grey'),
            name=e0_trace_title,
            panel_no=2)
        self.initial_state_plot.add_trace(
            x=self.soilprofile.soilparameter_series('Vertical effective stress [kPa]')[1],
            z=self.soilprofile.soilparameter_series('Vertical effective stress [kPa]')[0],
            name=sigma_v0_eff_trace_title,
            line=dict(color='red'),
            panel_no=1)
        if 'mv [1/kPa]' in self.soilprofile.numerical_soil_parameters():
            _xaxis_title = xaxis_title_nc
            _max_stress = self.soilprofile.soilparameter_series('Vertical total stress [kPa]')[1].max()
        else:
            _xaxis_title = xaxis_title_oc
            _max_stress = max(
                self.soilprofile.soilparameter_series('pc [kPa]')[1].max(),
                self.soilprofile.soilparameter_series('Vertical total stress [kPa]')[1].max())
            self.initial_state_plot.add_trace(
                x=self.soilprofile.soilparameter_series('pc [kPa]')[1],
                z=self.soilprofile.soilparameter_series('pc [kPa]')[0],
                name=sigma_vc_trace_title,
                line=dict(color='violet', dash='dot'),
                panel_no=1)
        self.initial_state_plot.add_trace(
            x=self.soilprofile.soilparameter_series('Vertical total stress [kPa]')[1],
            z=self.soilprofile.soilparameter_series('Vertical total stress [kPa]')[0],
            name=sigma_v0_trace_title,
            line=dict(color='green'),
            panel_no=1)
        self.initial_state_plot.set_xaxis(
            title=_xaxis_title, panel_no=1,
            range=(0, _max_stress))
        self.initial_state_plot.set_xaxis(
            title=e0_axis_title, panel_no=2, range=e0_range)
        self.initial_state_plot.show()
        
    def set_foundation(self, width, shape="strip", length=np.nan, skirt_depth=0):
        """
        Sets the size and shape of the foundation.
        Length only needs to be defined when ``shape='rectangular'``.
        
        TODO: When a skirt depth is defined, the stress increase is transferred to the base of the skirts.
        Compression of the soil inside the skirt is then not taken into account.
        
        :param width: Width of the foundation (diameter for circular foundations) [m]
        :param shape: Shape of the foundation: ``'strip'`` (default), ``'circular'`` or ``'rectangular'``
        :param length: Out-of-plane length (only required for a rectangular foundation)
        :param skirt_depth: Depth of skirts [m]. The skirts are assumed to transfer the load to the base level of the skirts
        """
        self.width = width
        if shape in ['strip', 'circular', 'rectangular']:
            self.shape = shape
        else:
            raise ValueError("Foundation shape must be one of: 'strip', 'circular', 'rectangular'")
        if shape == 'rectangular':
            if np.isnan(length):
                raise ValueError("Length needs to be defined for a rectangular foundation")
        self.length = length
        self.skirt_depth = skirt_depth
        if self.skirt_depth > 0:
            warnings.warn("Functionality for skirted foundations not yet implemented")
            
    def create_grid(self, dz=0.5, custom_nodes=None, **kwargs):
        """
        Creates a grid for calculation
        """
        self.grid = CalculationGrid(soilprofile=self.soilprofile, dz=dz, custom_nodes=custom_nodes, **kwargs)
        
    def calculate_foundation_stress(self, applied_stress, offset=0, poissonsratio=0.3, **kwargs):
        """
        Calculates the vertical stress increase below the foundation.
        By default, the calculation happens below the center of the foundation (``offset=0``).
        The Boussinesq solution for the selected foundation shape is used.
        """
        self.applied_stress = applied_stress
        if self.shape == 'strip':
            self.grid.elements['delta sigma v [kPa]'] = list(map(lambda _z: np.abs(stresses_stripload(
                _z,
                x=self.width * 0.5 + offset,
                width=self.width,
                imposedstress=applied_stress,
                **kwargs)['delta sigma z [kPa]']), self.grid.elements['z [m]']))
            self.grid.nodes['delta sigma v [kPa]'] = list(map(lambda _z: np.abs(stresses_stripload(
                _z,
                x=self.width * 0.5 + offset,
                width=self.width,
                imposedstress=applied_stress,
                **kwargs)['delta sigma z [kPa]']), self.grid.nodes['z [m]']))
        elif self.shape == 'circular':
            if offset != 0:
                warnings.warn('Only stress below the center available for circular footing')
            self.grid.elements['delta sigma v [kPa]'] = list(map(lambda _z: np.abs(stresses_circle(
                _z,
                footing_radius=0.5 * self.width,
                imposedstress=applied_stress,
                poissonsratio=poissonsratio,
                **kwargs)['delta sigma z [kPa]']), self.grid.elements['z [m]']))
            self.grid.nodes['delta sigma v [kPa]'] = list(map(lambda _z: np.abs(stresses_circle(
                _z,
                footing_radius=0.5 * self.width,
                imposedstress=applied_stress,
                poissonsratio=poissonsratio,
                **kwargs)['delta sigma z [kPa]']), self.grid.nodes['z [m]']))
        elif self.shape == 'rectangular':
            if offset == 0:
                self.grid.elements['delta sigma v [kPa]'] = list(map(lambda _z: 4 * np.abs(stresses_rectangle(
                    z=_z,
                    width=0.5 * self.width,
                    imposedstress=applied_stress,
                    length=0.5 * self.length,
                    **kwargs)['delta sigma z [kPa]']), self.grid.elements['z [m]']))
                self.grid.nodes['delta sigma v [kPa]'] = list(map(lambda _z: 4 * np.abs(stresses_rectangle(
                    z=_z,
                    width=0.5 * self.width,
                    imposedstress=applied_stress,
                    length=0.5 * self.length,
                    **kwargs)['delta sigma z [kPa]']), self.grid.nodes['z [m]']))
            else:
                raise ValueError("Calculation for points off-center not implemented yet")
        else:
            raise ValueError("Foundation shape must be one of: 'strip', 'circular', 'rectangular'")
    
    def plot_stress_increase(self, plot_title="", fillcolordict={'SAND': 'yellow', 'CLAY': 'brown'}, latex_titles=True, **kwargs):
        """
        Plots the initial stress vs depth and the stress increase
        """
        if latex_titles:
            delta_sigmav_trace_title = r'$ \Delta \sigma_v $'
            sigma_v0_eff_trace_title = r'$ \sigma_{vo}^{\prime} $'
            sigma_vO_trace_title = r'$ \sigma_{vo} $'
            sigmav_axis_title = r'$ \sigma_{vo}, \ \sigma_{vo}^{\prime} \ \text{[kPa]} $'
            delta_sigma_axis_title = r'$ \Delta \sigma_v \ \text{[kPa]} $'
        else:
            delta_sigmav_trace_title = 'Delta sigma,v'
            sigma_v0_eff_trace_title = 'sigma,v0,eff'
            sigma_vO_trace_title = 'sigma,v0'
            sigmav_axis_title = 'sigma,v0, sigma,v0,eff [kPa]'
            delta_sigma_axis_title = 'Delta sigma,v [kPa]'

        self.stress_increase_plot = LogPlot(soilprofile=self.soilprofile, no_panels=2, fillcolordict=fillcolordict)
        self.stress_increase_plot.add_trace(
            x=self.grid.nodes['delta sigma v [kPa]'],
            z=self.grid.nodes['z [m]'],
            line=dict(color='grey'),
            name=delta_sigmav_trace_title,
            panel_no=2)
        self.stress_increase_plot.add_trace(
            x=self.grid.nodes['Vertical effective stress [kPa]'],
            z=self.grid.nodes['z [m]'],
            name=sigma_v0_eff_trace_title,
            line=dict(color='red'),
            panel_no=1)
        self.stress_increase_plot.add_trace(
            x=self.grid.nodes['Vertical total stress [kPa]'],
            z=self.grid.nodes['z [m]'],
            name=sigma_vO_trace_title,
            line=dict(color='green'),
            panel_no=1)
        self.stress_increase_plot.set_xaxis(
            title=sigmav_axis_title, panel_no=1)
        self.stress_increase_plot.set_xaxis(
            title=delta_sigma_axis_title, panel_no=2, range=(0, self.applied_stress))
        self.stress_increase_plot.fig['layout'].update(legend=dict(x=0.9, y=0.2))
        self.stress_increase_plot.show()
    
    def calculate(self, **kwargs):
        """
        Calculates the consolidation settlement using the specified grid, foundation shape and loading
        """
        for i, row in self.grid.elements.iterrows():
            self.grid.elements.loc[i, "delta z [m]"] = primaryconsolidationsettlement_oc(
                initial_height=row['dz [m]'],
                initial_voidratio=row['e0 [-]'],
                initial_effective_stress=row['Vertical effective stress [kPa]'],
                preconsolidation_pressure=row['pc [kPa]'],
                effective_stress_increase=row['delta sigma v [kPa]'],
                compression_index=row['Cc [-]'],
                recompression_index=row['Cr [-]'],
                **kwargs)['delta z [m]']
        self.settlement = self.grid.elements['delta z [m]'].cumsum().iloc[-1]
        self.grid.nodes['Vertical effective stress final [kPa]'] = \
            self.grid.nodes['Vertical effective stress [kPa]'] + \
            self.grid.nodes['delta sigma v [kPa]']
        self.grid.elements['Vertical effective stress final [kPa]'] = \
            self.grid.elements['Vertical effective stress [kPa]'] + \
            self.grid.elements['delta sigma v [kPa]']
        self.grid.elements['e final [-]'] = \
            self.grid.elements['e0 [-]'] * \
            (1 - (self.grid.elements["delta z [m]"] / self.grid.elements["dz [m]"])) - \
            (self.grid.elements["delta z [m]"] / self.grid.elements["dz [m]"])
        
    def calculate_mv(self, **kwargs):
        """
        Calculates the consolidation settlement using the specified grid, foundation shape and loading.
        Instead of using the compression index and recompression index, the modulus of volumetric compressibility :math:`m_v` is used.
        """
        for i, row in self.grid.elements.iterrows():
            self.grid.elements.loc[i, "delta z [m]"] = consolidationsettlement_mv(
                initial_height=row['dz [m]'],
                effective_stress_increase=row['delta sigma v [kPa]'],
                compressibility=row['mv [1/kPa]'],
                **kwargs)['delta z [m]']
        self.settlement = self.grid.elements['delta z [m]'].cumsum().iloc[-1]
        self.grid.nodes['Vertical effective stress final [kPa]'] = \
            self.grid.nodes['Vertical effective stress [kPa]'] + \
            self.grid.nodes['delta sigma v [kPa]']
        self.grid.elements['Vertical effective stress final [kPa]'] = \
            self.grid.elements['Vertical effective stress [kPa]'] + \
            self.grid.elements['delta sigma v [kPa]']
        
    def plot_result(self, plot_title="", fillcolordict={'SAND': 'yellow', 'CLAY': 'brown'}, latex_titles=True, **kwargs):
        """
        Plots the settlement resulting from the stress increase
        """
        if latex_titles:
            sigmav_fin_trace_title = r'$ \sigma_{v,fin}^{\prime} $'
            sigmav0_trace_title = r'$ \sigma_{v,0}^{\prime} $'
            deltaz_trace_title = r'$ \Delta z $'
            xaxis_1_title = r'$ \sigma_{vo}^{\prime} \ \text{[kPa]} $'
            xaxis_2_title = r'$ \text{Settlement, } \Delta z \ \text{[m]} $'
        else:
            sigmav_fin_trace_title = 'sigma,vfin,eff'
            sigmav0_trace_title = 'sigma,v0,eff'
            deltaz_trace_title = 'Delta z'
            xaxis_1_title = 'sigma,v0,eff'
            xaxis_2_title = 'Settlement [m]'

        self.result_plot = LogPlot(soilprofile=self.soilprofile, no_panels=2, fillcolordict=fillcolordict)
        self.result_plot.add_trace(
            x=self.grid.nodes['Vertical effective stress [kPa]'] + self.grid.nodes['delta sigma v [kPa]'],
            z=self.grid.nodes['z [m]'],
            line=dict(color='grey'),
            name=sigmav_fin_trace_title,
            panel_no=1)
        self.result_plot.add_trace(
            x=self.grid.nodes['Vertical effective stress [kPa]'],
            z=self.grid.nodes['z [m]'],
            line=dict(color='red'),
            name=sigmav0_trace_title,
            panel_no=1)
        self.result_plot.add_trace(
            x=np.flipud(self.grid.elements['delta z [m]']).cumsum(),
            z=np.flipud(self.grid.elements['z [m]']),
            name=deltaz_trace_title,
            line=dict(color='blue'),
            panel_no=2)
        self.result_plot.set_xaxis(
            title=xaxis_1_title, panel_no=1)
        self.result_plot.set_xaxis(
            title=xaxis_2_title, panel_no=2, range=(0, self.settlement))
        self.result_plot.fig['layout'].update(legend=dict(x=0.15, y=0.1))
        self.result_plot.show()
        
        