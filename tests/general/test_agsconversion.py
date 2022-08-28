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
from groundhog.general import agsconversion

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_ags4conversion(unittest.TestCase):

    def setUp(self):
        self.agsfile = os.path.join(TESTS_DATA_DIR, "Data_AGS4_Geotech_BH-WFS4-7_WFS IV_151211_Fugro.AGS")
        self.ags = agsconversion.AGSConverter(path = self.agsfile)


    def test_agsconversion(self):
        self.ags.create_dataframes(verbose_keys=True)
        
        self.assertEqual(
            self.ags.data['GEOL']['Depth to the top of stratum [m]'].iloc[0],
            0
        )
        self.assertEqual(
            self.ags.data['GEOL']['Depth to the base of stratum [m]'].iloc[-1],
            51.85
        )

    def test_agsconversion_shorthands(self):
        self.ags.create_dataframes(verbose_keys=True, use_shorthands=True)
        
        self.assertEqual(
            self.ags.data['GEOL']['Depth from [m]'].iloc[0],
            0
        )
        self.assertEqual(
            self.ags.data['GEOL']['Depth to [m]'].iloc[-1],
            51.85
        )

class Test_ags31conversion(unittest.TestCase):

    def setUp(self):
        self.agsfile = os.path.join(TESTS_DATA_DIR, "1SVa.ags")
        self.ags = agsconversion.AGSConverter(path = self.agsfile, agsformat="3.1")

    def test_agsconversion(self):
        self.ags.convert_ags_group('GEOL')

        self.ags.create_dataframes(verbose_keys=True)
        
        self.assertEqual(
            self.ags.data['GEOL']['Depth to the top of stratum [m]'].iloc[0],
            0
        )
        self.assertEqual(
            self.ags.data['GEOL']['Depth to the base of stratum [m]'].iloc[-1],
            5.1
        )
        
    def test_agsconversion_shorthands(self):
        self.ags.create_dataframes(verbose_keys=True, use_shorthands=True)
        
        self.assertEqual(
            self.ags.data['GEOL']['Depth from [m]'].iloc[0],
            0
        )
        self.assertEqual(
            self.ags.data['GEOL']['Depth to [m]'].iloc[-1],
            5.1
        )