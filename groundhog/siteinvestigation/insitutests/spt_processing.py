#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

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
    'area ratio [-]': [0.8, ],
    'Cone type': ['U', ],
    'Cone base area [cm2]': [10, ],
    'Cone sleeve_area [cm2]': [150, ],
    'Sleeve cross-sectional area top [cm2]': [np.nan,],
    'Sleeve cross-sectional area bottom [cm2]': [np.nan,]
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

    def map_properties(self, layer_profile, spt_profile=DEFAULT_SPT_PROPERTIES,
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
        :param spt_profile: ``SoilProfile`` object with the spt test properties (default=``DEFAULT_SPT_PROPERTIES``)
        :param initial_vertical_total_stress: Initial vertical total stress at the highest point of the soil profile
        :param vertical_total_stress: Pre-calculated total vertical stress at SPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param vertical_effective_stress: Pre-calculated effective vertical stress at SPT depth nodes (default=None which will lead to calculation of total stress inside the routine)
        :param waterlevel: Waterlevel [m] in the soil (measured from soil surface), default = 0m
        :param extend_cone_profile: Boolean determining whether the cone profile needs to be extended to go to the bottom of the SPT (default = True)
        :param extend_layer_profile: Boolean determining whether the layer profile needs to be extended to the bottom of the SPT (default = True)
        :return: Expands the dataframe `.data` with additional columns for the SPT and soil properties
        """
        super().map_properties(
            layer_profile=layer_profile, initial_vertical_total_stress=initial_vertical_total_stress,
            vertical_total_stress=vertical_total_stress, vertical_effective_stress=vertical_effective_stress,
            waterlevel=waterlevel, extend_layer_profile=extend_layer_profile)

        # TODO: Add mapping for SPT properties