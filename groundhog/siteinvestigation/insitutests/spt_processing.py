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
from groundhog.general.soilprofile import SoilProfile, plot_fence_diagram
from groundhog.general.parameter_mapping import offsets
from groundhog.general.agsconversion import AGSConverter


class SPTProcessing(object):
    """
    The SPTProcessing class implements methods for reading, processing and presentation of Standard Penetration Test (SPT) data.

    `GeoEngineer <https://www.geoengineer.org/education/site-characterization-in-situ-testing-general/standard-penetration-testing-spt/>`_ describes the SPT test as follows:

        Standard Penetration Test (SPT) is a simple and low-cost testing procedure widely used in geotechnical investigation to determine the relative density and angle of shearing resistance of cohesionless soils and also the strength of stiff cohesive soils.

        For this test, a borehole has to be drilled to the desired sampling depth. The split-spoon sampler that is attached to the drill rod is placed at the testing point. A hammer of 63.5 kg (140 lbs) is dropped repeatedly from a height of 76 cm (30 inches) driving the sampler into the ground until reaching a depth of 15 cm (6 inches). The number of the required blows is recorded. This procedure is repeated two more times until a total penetration of 45 cm (18 inches) is achieved. The number of blows required to penetrate the first 15 cm is called “seating drive” and the total number of blows required to penetrate the remaining 30 cm depth is known as the “standard penetration resistance”, or otherwise, the “N-value”. If the N-value exceeds 50 then the test is discontinued and is called a “refusal”. The interpreted results, with several corrections, are used to estimate the geotechnical engineering properties of the soil.

    Check the GeoEngineer website for more useful information and an insightful presentation by Paul W. Mayne on SPT hammer types.

    .. figure:: images/spt_principle.png
        :figwidth: 700.0
        :width: 650.0
        :align: center

        Working principle of the SPT (Mayne, 2016)

    The SPT test is a simple and robust test but it has its drawbacks. For example, measurements are discontinuous, the application of SPT N number in soft clays and silts is not possible, the are energy inefficiency problems, ... The user needs to be aware of these issues at the onset of a SPT processing exercise.
    """

    def __init__(self, title, waterunitweight=10):
        """
        Initialises a SPTProcessing object based on a title. Optionally, a geographical position can be defined. A dictionary for dumping unstructured data (``additionaldata``) is also available.

        An empty dataframe (``.data`) is created for storing the SPT data`

        :param title: Title for the PCPT test
        :param waterunitweight: Unit weight of water used for effective stress calculations (default=10kN/m3 for groundwater)

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

