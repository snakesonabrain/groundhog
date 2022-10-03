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
from groundhog.deepfoundations.axialcapacity import axcap
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from groundhog.general.soilprofile import SoilProfile

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_axcapcalc(unittest.TestCase):

    def setUp(self):
        self.soilprofile_api = SoilProfile({
            'Depth from [m]': [0, 5, 10],
            'Depth to [m]': [5, 10, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND'],
            'Total unit weight [kN/m3]': [20, 18, 20],
            'Unit skin friction': ['API RP2 GEO Sand', 'API RP2 GEO Clay', 'API RP2 GEO Sand'],
            'Unit end bearing': ['API RP2 GEO Sand', 'API RP2 GEO Clay', 'API RP2 GEO Sand']
        })
        self.soilprofile_api_errors = SoilProfile({
            'Depth from [m]': [0, 5, 10],
            'Depth to [m]': [5, 10, 20],
            'Soil type': ['SAND', 'CLAY', 'SAND'],
            'Total unit weight [kN/m3]': [20, 18, 20],
            'Unit skin friction': ['BPI RP2 GEO Sand', 'API RP2 GEO Clay', 'API RP2 GEO Sand'], # Typo with BPI instead of API
            'Unit end bearing': ['API RP2 GEO Sand', 'BPI RP2 GEO Clay', 'API RP2 GEO Sand']
        })
        
        
    def test_calcinitiation(self):
        self.calc_api = axcap.AxCapCalculation(self.soilprofile_api)
        self.calc_api_errors = axcap.AxCapCalculation(self.soilprofile_api_errors)

    def test_checking(self):
        self.test_calcinitiation()
        self.calc_api.check_methods()
        self.assertRaises(ValueError, self.calc_api.check_methods, True)
        self.calc_api.sp.calculate_overburden()
        self.calc_api.sp['API soil description'] = ['Sand-silt', None, 'Sand']
        self.calc_api.sp['API relative density description'] = ['Medium dense', None, 'Dense']
        self.calc_api.sp['Undrained shear strength from [kPa]'] = [np.nan, 100, np.nan]
        self.calc_api.sp['Undrained shear strength to [kPa]'] = [np.nan, 150, np.nan]
        self.calc_api.check_methods(raise_errors=True)
        self.assertTrue(self.calc_api.checked)
        self.calc_api_errors.check_methods()
        self.assertFalse(self.calc_api_errors.checked)
        self.assertRaises(ValueError, self.calc_api_errors.check_methods, True)

    def test_gridding(self):
        self.test_checking()
        self.calc_api.create_grid()
