#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.siteinvestigation.insitutests import pcpt_processing
from groundhog.general.soilprofile import SoilProfile

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Test_PCPTProcessing(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_excel(os.path.join(TESTS_DATA_DIR, 'debeer_example.xlsx'))
        self.data["u2 [MPa]"] = 1e-3 * self.data["u [kPa]"]
        self.pandas_pcpt = pcpt_processing.PCPTProcessing(title="Pandas PCPT")

    def test_pandas_pcpt_creation(self):
        """
        Test PCPT imported as Pandas data
        """
        # Load the PCPT data from a Pandas dataframe
        self.pandas_pcpt.load_pandas(self.data)
        self.assertEqual(self.pandas_pcpt.data["z [m]"].iloc[0], 0)
        self.assertEqual(self.pandas_pcpt.data["z [m]"].iloc[1], 0.02)

    def test_pcpt_mapping(self):
        """
        Test mapping of soil and cone properties to the PCPT grid
        :return:
        """
        # Create layering
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20],
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers)
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "area ratio [-]"], 0.8, 1
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "Water pressure [kPa]"], 42.23, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[210, "Vertical effective stress [kPa]"], 31.51, 2
        )
        self.assertEqual(
            self.pandas_pcpt.data.loc[204, "Total unit weight [kN/m3]"], 17
        )

    def test_pcpt_mapping_nonzero_initial_stress(self):
        """
        Test mapping of soil and cone properties to the PCPT grid
        :return:
        """
        # Create layering
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20],
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers, initial_vertical_total_stress=100)
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "area ratio [-]"], 0.8, 1
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "Water pressure [kPa]"], 42.23, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[210, "Vertical effective stress [kPa]"], 131.51, 2
        )
        self.assertEqual(
            self.pandas_pcpt.data.loc[204, "Total unit weight [kN/m3]"], 17
        )

    def test_pcpt_mapping_errors(self):
        """
        Test mapping of soil and cone properties to the PCPT grid
        :return:
        """
        # Create layering
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 16], # PCPT data exceeds max depth
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.test_pandas_pcpt_creation()
        self.assertRaises(ValueError, self.pandas_pcpt.map_properties, layers)

        layers = SoilProfile({
            "Depth from [m]": [1, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20], # PCPT data exceeds max depth
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.test_pandas_pcpt_creation()
        self.assertRaises(ValueError, self.pandas_pcpt.map_properties, layers)

        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20], # PCPT data exceeds max depth
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers)

        cone_props = pcpt_processing.DEFAULT_CONE_PROPERTIES

        cone_props.loc[0, "Depth from [m]"] = 1
        self.test_pandas_pcpt_creation()
        self.assertRaises(ValueError, self.pandas_pcpt.map_properties, layers, cone_props)

        cone_props.loc[0, "Depth from [m]"] = 0
        cone_props.loc[0, "Depth to [m]"] = 18
        self.test_pandas_pcpt_creation()
        self.assertRaises(ValueError, self.pandas_pcpt.map_properties, layers, cone_props)

        cone_props.loc[0, "Depth to [m]"] = 20
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers)


    def test_pcpt_normalisation(self):
        """
        Test normalisation of the PCPT data
        :return:
        """
        self.test_pcpt_mapping()
        self.pandas_pcpt.normalise_pcpt()
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Qt [-]"], 173.50, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Fr [%]"], 0.4349, 4
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Bq [-]"], -0.00145, 5
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "qnet [MPa]"], 17.2472, 4
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "ft [MPa]"], self.pandas_pcpt.data.loc[600, "fs [MPa]"], 4
        )

    def test_pcpt_normalisation_withsleeve(self):
        """
        Test normalisation of the PCPT data with correction for sleeve
        :return:
        """

        # Create layering
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20],
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        cone_props = SoilProfile({
            'Depth from [m]': [0, ],
            'Depth to [m]': [20, ],
            'area ratio [-]': [0.8, ],
            'Cone type': ['U', ],
            'Cone base area [cm2]': [10, ],
            'Cone sleeve_area [cm2]': [150, ],
            'Sleeve cross-sectional area top [cm2]': [0.8,],
            'Sleeve cross-sectional area bottom [cm2]': [1,]
        })
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers, cone_profile=cone_props)
        self.pandas_pcpt.normalise_pcpt()

        self.assertAlmostEqual(
           self.pandas_pcpt.data.loc[2, 'ft [MPa]'], 0.003 - 0.004 * ((1 - 0.8) / 150), 5
        )

    def test_pcpt_normalisation_nosleevedata(self):
        """
        Test normalisation of the PCPT data with no sleeve dimensions available
        :return:
        """

        # Create layering
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 20],
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        cone_props = SoilProfile({
            'Depth from [m]': [0, ],
            'Depth to [m]': [20, ],
            'area ratio [-]': [0.8, ],
            'Cone type': ['U', ],
            'Cone base area [cm2]': [10, ],
        })
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers, cone_profile=cone_props)
        self.pandas_pcpt.normalise_pcpt()

        self.assertAlmostEqual(
           self.pandas_pcpt.data.loc[2, 'ft [MPa]'], self.pandas_pcpt.data.loc[2, 'fs [MPa]'], 5
        )

    def test_pcpt_correlations(self):
        """
        Test if correlations are correctly applied to PCPT data
        :return:
        """
        self.test_pcpt_normalisation()
        self.pandas_pcpt.apply_correlation('Ic Robertson and Wride (1998)', outkey='Ic [-]', resultkey='Ic [-]')
        self.pandas_pcpt.apply_correlation(
            'Gmax Rix and Stokoe (1991)', outkey='Gmax sand [kPa]', resultkey='Gmax [kPa]',
            apply_for_soiltypes=['SAND', ])
        self.pandas_pcpt.apply_correlation(
            'Gmax Mayne and Rix (1993)', outkey='Gmax clay [kPa]', resultkey='Gmax [kPa]',
            apply_for_soiltypes=['CLAY', ])
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Ic [-]"], 2.993, 2
        )
        self.assertTrue(np.math.isnan(self.pandas_pcpt.data.loc[275, "Gmax sand [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Gmax clay [kPa]"], 45333, 0
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Ic [-]"], 1.505, 3
        )
        self.assertTrue(np.math.isnan(self.pandas_pcpt.data.loc[605, "Gmax clay [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Gmax sand [kPa]"], 107283, 0
        )

