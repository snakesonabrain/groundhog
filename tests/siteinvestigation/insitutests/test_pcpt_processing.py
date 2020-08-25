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
        # Set the cone properties
        self.pandas_pcpt.set_cone_properties(stroke=0.02)
        self.assertEqual(self.pandas_pcpt.coneproperties["z from [m]"].iloc[0], 0)
        # Create layering
        layers = pd.DataFrame({
            "z from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "z to [m]": [3.16, 5.9, 14.86, 15.7, 20],
            "Total unit weight from [kN/m3]": [18, 17, 19.5, 20, 20],
            "Total unit weight to [kN/m3]": [19, 17, 20, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        self.pandas_pcpt.set_layer_properties(layer_data=layers)
        self.assertEqual(self.pandas_pcpt.layerdata["Total unit weight from [kN/m3]"].iloc[2], 19.5)

    def test_pcpt_mapping(self):
        """
        Test mapping of soil and cone properties to the PCPT grid
        :return:
        """
        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties()
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "area ratio [-]"], 0.8, 1
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "Water pressure [kPa]"], 42.23, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[210, "Vertical effective stress [kPa]"], 33.08, 2
        )
        self.assertEqual(
            self.pandas_pcpt.data.loc[204, "Total unit weight [kN/m3]"], 17
        )

    def test_pcpt_normalisation(self):
        """
        Test normalisation of the PCPT data
        :return:
        """
        self.test_pcpt_mapping()
        self.pandas_pcpt.normalise_pcpt()
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Qt [-]"], 172.47, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Fr [%]"], 0.4349, 4
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "Bq [-]"], -0.00145, 5
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[600, "qnet [MPa]"], 17.2466, 4
        )

    def test_pcpt_correlations(self):
        """
        Test if correlations are correctly applied to PCPT data
        :return:
        """
        self.test_pcpt_normalisation()
        self.pandas_pcpt.apply_correlation('Robertson and Wride (1998)', outkey='Ic [-]', resultkey='Ic [-]')
        self.pandas_pcpt.apply_correlation(
            'Rix and Stokoe (1991)', outkey='Gmax sand [kPa]', resultkey='Gmax [kPa]',
            apply_for_soiltypes=['SAND', ])
        self.pandas_pcpt.apply_correlation(
            'Mayne and Rix (1993)', outkey='Gmax clay [kPa]', resultkey='Gmax [kPa]',
            apply_for_soiltypes=['CLAY', ])
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Ic [-]"], 3, 2
        )
        self.assertTrue(np.math.isnan(self.pandas_pcpt.data.loc[275, "Gmax sand [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Gmax clay [kPa]"], 45333, 0
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Ic [-]"], 1.506, 3
        )
        self.assertTrue(np.math.isnan(self.pandas_pcpt.data.loc[605, "Gmax clay [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Gmax sand [kPa]"], 107526, 0
        )

