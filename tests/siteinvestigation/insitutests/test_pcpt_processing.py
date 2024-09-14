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
            self.pandas_pcpt.data.loc[206, "Hydrostatic pressure [kPa]"], 42.23, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[210, "Vertical effective stress [kPa]"], 31.51, 2
        )
        self.assertEqual(
            self.pandas_pcpt.data.loc[204, "Total unit weight [kN/m3]"], 17
        )

    def test_pcpt_mapping_extended(self):
        """
        Test automatic extending of soilprofiles for mapping
        :return:
        """
        layers = SoilProfile({
            "Depth from [m]": [0, 3.16, 5.9, 14.86, 15.7],
            "Depth to [m]": [3.16, 5.9, 14.86, 15.7, 18],
            "Total unit weight [kN/m3]": [18, 17, 19.5, 20, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'SAND', 'SAND']
        })
        cone_props = pcpt_processing.DEFAULT_CONE_PROPERTIES
        cone_props['Depth to [m]'] = [18,]

        self.test_pandas_pcpt_creation()
        self.pandas_pcpt.map_properties(layer_profile=layers, cone_profile=cone_props)
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "area ratio [-]"], 0.8, 1
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[206, "Hydrostatic pressure [kPa]"], 42.23, 2
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[210, "Vertical effective stress [kPa]"], 31.51, 2
        )
        self.assertEqual(
            self.pandas_pcpt.data.loc[204, "Total unit weight [kN/m3]"], 17
        )

    def test_excel_output(self):
        """
        Tests whether an output Excel file is correctly written
        :return:
        """
        self.test_pcpt_mapping_extended()
        self.pandas_pcpt.to_excel(output_path=os.path.join(TESTS_DATA_DIR, "output_pcpt.xlsx"))

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
            self.pandas_pcpt.data.loc[206, "Hydrostatic pressure [kPa]"], 42.23, 2
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
        with self.assertRaises(ValueError):
            self.pandas_pcpt.map_properties(layers, extend_layer_profile=False)

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
        with self.assertRaises(ValueError):
            self.pandas_pcpt.map_properties(layers, cone_props, extend_cone_profile=False, extend_layer_profile=False)

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

    def test_pcpt_normalisation_plotting(self):
        """
        Test normalisation of the PCPT data
        :return:
        """
        self.test_pcpt_mapping()
        self.pandas_pcpt.normalise_pcpt()
        self.pandas_pcpt.plot_robertson_chart()
        
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
           self.pandas_pcpt.data.loc[2, 'ft [MPa]'], 0.003 - 0.004 * ((1 - 0.8) / 150), 4
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
        self.pandas_pcpt.apply_correlation(
            'Ic Robertson and Wride (1998)', outputs={'Ic [-]': 'Ic [-]'})
        self.pandas_pcpt.apply_correlation(
            'Gmax Rix and Stokoe (1991)', outputs={'Gmax [kPa]': 'Gmax sand [kPa]'},
            apply_for_soiltypes=['SAND', ])
        self.pandas_pcpt.apply_correlation(
            'Gmax Mayne and Rix (1993)', outputs={'Gmax [kPa]': 'Gmax clay [kPa]'},
            apply_for_soiltypes=['CLAY', ])
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Ic [-]"], 2.993, 2
        )
        self.assertTrue(np.isnan(self.pandas_pcpt.data.loc[275, "Gmax sand [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[275, "Gmax clay [kPa]"], 45333, 0
        )
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Ic [-]"], 1.505, 3
        )
        self.assertTrue(np.isnan(self.pandas_pcpt.data.loc[605, "Gmax clay [kPa]"]))
        self.assertAlmostEqual(
            self.pandas_pcpt.data.loc[605, "Gmax sand [kPa]"], 107283, 0
        )

class Test_pydov_loading(unittest.TestCase):

    def test_pydov_import(self):
        """
        Test CPT import from pydov
        """
        cpt = pcpt_processing.PCPTProcessing('DOV example')
        cpt.load_pydov('GEO-87/143-SVI', z_key='lengte')
        self.assertEqual(cpt.data['sondeernummer'].iloc[1], 'GEO-87/143-SVI')      

class Test_GEFReading(unittest.TestCase):

    def test_longfile_reading(self):
        """
        Test reading the long file from the CUR guide
        """
        filename = os.path.join(TESTS_DATA_DIR, 'gef_file_long.gef')
        self.gef_cpt = pcpt_processing.PCPTProcessing(title="GEF long")

        self.gef_cpt.load_gef(path=filename, inverse_depths=True)
        self.assertEqual(self.gef_cpt.data.loc[1, "qc [MPa]"], 0.382)
        self.assertEqual(self.gef_cpt.data["z [m]"].iloc[-1], 57.64)
        self.assertEqual(self.gef_cpt.easting, 12.345)
        self.assertEqual(self.gef_cpt.elevation, 40.03)
        self.assertEqual(self.gef_cpt.title, 'C2-366')


    def test_shortfile_reading(self):
        """
        Test reading the short file from the CUR guide
        """
        filename = os.path.join(TESTS_DATA_DIR, 'gef_file_short.gef')
        self.gef_cpt = pcpt_processing.PCPTProcessing(title="GEF short")

        self.gef_cpt.load_gef(path=filename, inverse_depths=True)
        self.assertEqual(self.gef_cpt.data.loc[1, "qc [MPa]"], 0.205)
        self.assertEqual(self.gef_cpt.data["z [m]"].iloc[-1], 25.08)
        self.assertEqual(self.gef_cpt.title, 'C2-265')

    def test_realexample_reading(self):
        """
        Test reading a real file from dov.vlaanderen.be
        """
        filename = os.path.join(TESTS_DATA_DIR, 'gef_real_example.gef')
        self.gef_cpt = pcpt_processing.PCPTProcessing(title="GEF real")

        self.gef_cpt.load_gef(path=filename)
        self.assertEqual(self.gef_cpt.data.loc[2, "qc [MPa]"], 1.1)
        self.assertEqual(self.gef_cpt.data["z [m]"].iloc[-1], 7.4)
        self.assertEqual(self.gef_cpt.title, 'GEO-52/1143-S3')

    def test_nodxdy_reading(self):
        """
        Test reading a file without DX and DY (provided thovdl)
        """
        filename = os.path.join(TESTS_DATA_DIR, 'gef_no_dx_dy_example.gef')
        self.gef_cpt = pcpt_processing.PCPTProcessing(title="GEF no DX DY")

        self.gef_cpt.load_gef(path=filename)
        self.assertEqual(self.gef_cpt.data.loc[2, "qc [MPa]"], 0.408)
        self.assertEqual(self.gef_cpt.data["z [m]"].iloc[-1], 16.48)
        self.assertEqual(self.gef_cpt.title, 'CPT000000011611')

class Test_AGSFile_reading(unittest.TestCase):

    def test_loadags(self):
        ags_pcpt = pcpt_processing.PCPTProcessing(title="AGS PCPT")
        ags_pcpt.load_ags(
            os.path.join(TESTS_DATA_DIR, 'N6016_BH_WFS1-2A_AGS4_150909.ags'),
            z_key="Depth [m]",
            qc_key="qc [MN/m2]",
            fs_key="fs [kN/m2]",
            u2_key="u2 [kN/m2]",
            push_key="Test reference or push number",
            fs_multiplier=0.001, u2_multiplier=0.001,
            verbose_keys=True, use_shorthands=True
        )
        self.assertEqual(ags_pcpt.data['z [m]'].iloc[0], 0)
        self.assertEqual(ags_pcpt.data['z [m]'].iloc[1], 10)
        self.assertEqual(ags_pcpt.data['qc [MPa]'].iloc[1], 2.955)