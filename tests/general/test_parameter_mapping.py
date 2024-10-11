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
from groundhog.general import parameter_mapping


class Test_map_depth_properties(unittest.TestCase):

    def setUp(self):
        self.target_df = pd.DataFrame({'z [m]': np.linspace(0, 20, 21),})
        self.layering_df = pd.DataFrame({
            'z from [m]': np.array([0, 5, 10]),
            'z to [m]': np.array([5, 10, 20]),
            'Submerged unit weigth [kN/m3]': np.array([8.5, 7, 10]),
            'Undrained shear strength from [kPa]': np.array([np.nan, 50, np.nan]),
            'Undrained shear strength to [kPa]': np.array([np.nan, 60, np.nan]),
            'Effective friction angle from [deg]': np.array([32.5, np.nan, 40]),
            'Effective friction angle to [deg]': np.array([35, np.nan, 40]),
            'Soil type': np.array(['SAND', 'CLAY', 'SAND'])
        })
        self.target_df_altcol = pd.DataFrame({'Depth [m]': np.linspace(0, 20, 21), })
        self.layering_df_altcols = pd.DataFrame({
            'Depth from [m]': np.array([0, 5, 10]),
            'Depth to [m]': np.array([5, 10, 20]),
            'Submerged unit weigth [kN/m3]': np.array([8.5, 7, 10]),
            'Undrained shear strength from [kPa]': np.array([np.nan, 50, np.nan]),
            'Undrained shear strength to [kPa]': np.array([np.nan, 60, np.nan]),
            'Effective friction angle from [deg]': np.array([32.5, np.nan, 40]),
            'Effective friction angle to [deg]': np.array([35, np.nan, 40]),
            'Soil type': np.array(['SAND', 'CLAY', 'SAND'])
        })


    def test_map_depth_properties(self):
        result = parameter_mapping.map_depth_properties(
            target_df=self.target_df,
            layering_df=self.layering_df)
        self.assertEqual(
            result[result["z [m]"] == 2]["Submerged unit weigth [kN/m3]"].iloc[0], 8.5
        )
        self.assertEqual(
            result[result["z [m]"] == 5]["Soil type"].iloc[0], 'CLAY'
        )
        self.assertEqual(
            result[result["z [m]"] == 5]["Undrained shear strength [kPa]"].iloc[0], 50
        )
        self.assertTrue(
            np.isnan(result[result["z [m]"] == 5]["Effective friction angle [deg]"].iloc[0]))
        result = parameter_mapping.map_depth_properties(
            target_df=self.target_df_altcol,
            layering_df=self.layering_df_altcols,
            target_z_key="Depth [m]",
            layering_zfrom_key="Depth from [m]",
            layering_zto_key="Depth to [m]"
            )
        self.assertEqual(
            result[result["Depth [m]"] == 2]["Submerged unit weigth [kN/m3]"].iloc[0], 8.5
        )
        self.assertEqual(
            result[result["Depth [m]"] == 5]["Soil type"].iloc[0], 'CLAY'
        )
        self.assertEqual(
            result[result["Depth [m]"] == 5]["Undrained shear strength [kPa]"].iloc[0], 50
        )
        self.assertTrue(
            np.isnan(result[result["Depth [m]"] == 5]["Effective friction angle [deg]"].iloc[0])
        )

class Test_offsets(unittest.TestCase):

    def test_offsets_before(self):
        result = parameter_mapping.offsets(
            startpoint=(0, 0),
            endpoint=(1, 0),
            point=(-1, 1)
        )
        self.assertTrue(result['before start'])
        self.assertFalse(result['behind end'])
        self.assertEqual(result['offset to line'], 1)
        self.assertAlmostEqual(result['offset to start projected'], -1, 5)
        self.assertAlmostEqual(result['offset to end projected'], 2, 5)

    def test_offsets_between(self):
        result = parameter_mapping.offsets(
            startpoint=(0, 0),
            endpoint=(0, 2),
            point=(1, 1)
        )
        self.assertFalse(result['before start'])
        self.assertFalse(result['behind end'])
        self.assertAlmostEqual(result['offset to line'], 1, 5)
        self.assertAlmostEqual(result['offset to start projected'], 1, 5)
        self.assertAlmostEqual(result['offset to end projected'], 1, 5)

    def test_offsets_behind(self):
        result = parameter_mapping.offsets(
            startpoint=(2, 2),
            endpoint=(0, 0),
            point=(-1, 0)
        )
        self.assertFalse(result['before start'])
        self.assertTrue(result['behind end'])
        self.assertAlmostEqual(result['offset to line'], np.sqrt(0.5), 5)
        self.assertAlmostEqual(result['offset to start projected'], np.sqrt(8) + np.sqrt(0.5), 5)
        self.assertAlmostEqual(result['offset to end projected'], -np.sqrt(0.5), 5)

    def test_offset_latlon(self):
        distance = parameter_mapping.latlon_distance(lat1=51.215431, lon1=2.928656, lat2=51.315090, lon2=3.130940)
        self.assertAlmostEqual(distance, 17952, 0)