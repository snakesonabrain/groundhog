#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import numpy as np
import pandas as pd

# Project imports
from groundhog.standards.eurocode7 import factors

class Test_factors(unittest.TestCase):

    def setUp(self):
        self.factors = factors.Eurocode7_factoring_STR_GEO()

    def test_DA_selection(self):
        self.factors.select_design_approach(design_approach="DA1-1")
        self.assertEqual(self.factors.selected_factors_actions['Variable unfavourable'], 1.5)
        self.assertEqual(self.factors.selected_factors_soil['Effective cohesion'], 1.0)
        self.assertEqual(self.factors.selected_factors_resistance['Bearing'], 1.0)
        self.factors.select_design_approach(design_approach="DA2", foundation_type="Driven pile")
        self.assertEqual(self.factors.selected_factors_actions['Variable unfavourable'], 1.5)
        self.assertEqual(self.factors.selected_factors_soil['Effective cohesion'], 1.0)
        self.assertEqual(self.factors.selected_factors_resistance['Shaft tension'], 1.15)

    def test_NDP_override(self):
        self.factors.override_factors_actions(set='A1', loadtype='Variable unfavourable', value=1.6)
        self.factors.override_factors_soil(set='M1', soilparameter='Effective cohesion', value=1.1)
        self.factors.override_factors_resistance(set='R1', foundationtype="Spread foundation", override_dict={'Bearing': 1.3, 'Sliding': 1.25})
        self.factors.select_design_approach(design_approach="DA1-1")
        self.assertEqual(self.factors.selected_factors_actions['Variable unfavourable'], 1.6)
        self.assertEqual(self.factors.selected_factors_soil['Effective cohesion'], 1.1)
        self.assertEqual(self.factors.selected_factors_resistance['Bearing'], 1.3)

    def test_correlation_factors(self):
        result = self.factors.select_correlation_factors(testtype='Static load test', no_tests=2)
        self.assertEqual(result['ksi_1'], 1.3)
        self.assertEqual(result['ksi_2'], 1.2)
        result = self.factors.select_correlation_factors(testtype='Ground investigation', no_tests=6)
        self.assertEqual(result['ksi_3'], 1.29)
        self.assertEqual(result['ksi_4'], 1.15)
        result = self.factors.select_correlation_factors(testtype='Ground investigation', no_tests=6, interpolate=True)
        self.assertEqual(result['ksi_3'], 1.28)
        self.assertEqual(result['ksi_4'], 1.135)
        result = self.factors.select_correlation_factors(testtype='Dynamic load test', no_tests=12)
        self.assertEqual(result['ksi_5'], 1.45)
        self.assertEqual(result['ksi_6'], 1.3)
        result = self.factors.select_correlation_factors(testtype='Dynamic load test', no_tests=12, interpolate=True)
        self.assertEqual(result['ksi_5'], 1.438)
        self.assertEqual(result['ksi_6'], 1.28)