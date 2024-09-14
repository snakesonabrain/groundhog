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
from groundhog.siteinvestigation.insitutests import spt_processing
from groundhog.general.soilprofile import SoilProfile, read_excel

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Test_SPTProcessing(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example.xlsx'))
        self.pandas_spt = spt_processing.SPTProcessing(title="Pandas SPT")

    def test_pandas_spt_creation(self):
        """
        Test PCPT imported as Pandas data
        """
        # Load the PCPT data from a Pandas dataframe
        self.pandas_spt.load_pandas(self.data, z_key='Depth [ft]', z_multiplier=0.3, N_key='SPT N [-]')
        self.assertEqual(self.pandas_spt.data["z [m]"].iloc[0], 1.5)
        self.assertEqual(self.pandas_spt.data["N [-]"].iloc[0], 7)
        self.assertEqual(self.pandas_spt.data["z [m]"].iloc[2], 4.5)
        self.assertEqual(self.pandas_spt.data["N [-]"].iloc[2], 4)

    def test_excel_spt_creation(self):
        """
        Test PCPT imported as Pandas data
        """
        # Load the PCPT data from an Excel file
        self.excel_spt = spt_processing.SPTProcessing(title="Excel SPT")

        self.excel_spt.load_excel(
            os.path.join(TESTS_DATA_DIR, 'spt_example.xlsx'),
            z_key='Depth [ft]', z_multiplier=0.3, N_key='SPT N [-]')
        self.assertEqual(self.excel_spt.data["z [m]"].iloc[0], 1.5)
        self.assertEqual(self.excel_spt.data["N [-]"].iloc[0], 7)
        self.assertEqual(self.excel_spt.data["z [m]"].iloc[2], 4.5)
        self.assertEqual(self.excel_spt.data["N [-]"].iloc[2], 4)

    def test_spt_mapping(self):
        """
        Test mapping of soil and cone properties to the PCPT grid
        :return:
        """
        # Create layering
        layers = read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example_layering.xlsx'), unit='ft')
        layers.convert_depth_reference(newunit='m', multiplier=0.3048)
        self.test_pandas_spt_creation()
        self.pandas_spt.map_properties(layer_profile=layers)
        self.assertAlmostEqual(
            self.pandas_spt.data.loc[2, "Borehole diameter [mm]"], 100, 1
        )
        self.assertAlmostEqual(
            self.pandas_spt.data.loc[2, "Hydrostatic pressure [kPa]"], 45, 0
        )
        self.assertAlmostEqual(
            self.pandas_spt.data.loc[2, "Vertical effective stress [kPa]"], 35.3, 1
        )
        self.assertEqual(
            self.pandas_spt.data.loc[2, "Total unit weight [kN/m3]"], 18.5
        )

    def test_spt_overburdencorrection(self):
        """
        Test overburden correction for the SPT N values
        :return:
        """
        layers = read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example_layering.xlsx'), unit='ft')
        layers.convert_depth_reference(newunit='m', multiplier=0.3048)
        self.test_pandas_spt_creation()
        self.pandas_spt.map_properties(layer_profile=layers)
        self.pandas_spt.apply_correlation(
            name='Overburden correction Liao and Whitman (1986)',
            outputs={'N1 [-]': 'N1 [-]', 'CN [-]': 'CN [-]'}
        )
        
        self.assertAlmostEqual(self.pandas_spt.data['N1 [-]'].iloc[0], 7, 1)
        self.assertAlmostEqual(self.pandas_spt.data['N1 [-]'].iloc[1], 12.6, 1)
        self.assertAlmostEqual(self.pandas_spt.data['CN [-]'].iloc[0], 1, 1)


    def test_spt_N60correction(self):
        """
        Test N60 calculation for the SPT N values
        :return:
        """
        layers = read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example_layering.xlsx'), unit='ft')
        layers.convert_depth_reference(newunit='m', multiplier=0.3048)
        self.test_pandas_spt_creation()
        self.pandas_spt.map_properties(layer_profile=layers)
        self.pandas_spt.apply_correlation(
            name='N60 correction',
            outputs={'N60 [-]': 'N60 [-]'}
        )
        self.assertAlmostEqual(self.pandas_spt.data['N60 [-]'].iloc[0], 7 * 60 * 1 * 1 * 0.75 / 60, 1)
        self.assertAlmostEqual(self.pandas_spt.data['N60 [-]'].iloc[-1], 53 * 60 * 1 * 1 * 1 / 60, 1)

    def test_spt_N60correction_custom(self):
        """
        Test N60 calculation for the SPT N values
        :return:
        """
        layers = read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example_layering.xlsx'), unit='ft')
        layers.convert_depth_reference(newunit='m', multiplier=0.3048)
        self.test_pandas_spt_creation()

        spt_custom_props = spt_processing.DEFAULT_SPT_PROPERTIES
        spt_custom_props['eta H [%]'].iat[0] = 50

        self.pandas_spt.map_properties(layer_profile=layers, spt_profile=spt_custom_props)
        
        self.pandas_spt.apply_correlation(
            name='N60 correction',
            outputs={'N60 [-]': 'N60 [-]',
             'eta_H [-]': 'eta H [-]',
             'eta_B [-]': 'eta B [-]',
             'eta_S [-]': 'eta S [-]',
             'eta_R [-]': 'eta R [-]'}
        )
        
        self.assertAlmostEqual(self.pandas_spt.data['N60 [-]'].iloc[0], 7 * 50 * 1 * 1 * 0.75 / 60, 1)
        self.assertAlmostEqual(self.pandas_spt.data['N60 [-]'].iloc[-1], 53 * 50 * 1 * 1 * 1 / 60, 1)


    def test_spt_N60correction_withNaNs(self):
        """
        Test N60 calculation for the SPT N values with NaN values in the data
        :return:
        """
        layers = read_excel(os.path.join(TESTS_DATA_DIR, 'spt_example_layering.xlsx'), unit='ft')
        layers.convert_depth_reference(newunit='m', multiplier=0.3048)
        self.test_pandas_spt_creation()
        self.pandas_spt.data['N [-]'].iloc[2] = np.nan
        self.pandas_spt.map_properties(layer_profile=layers)
        self.pandas_spt.apply_correlation(
            name='N60 correction',
            outputs={'N60 [-]': 'N60 [-]'}
        )
        self.assertTrue(np.isnan(self.pandas_spt.data['N60 [-]'].iloc[2]))

