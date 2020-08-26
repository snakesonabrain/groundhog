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
from groundhog.general import soilprofile as sp


class Test_SoilProfile(unittest.TestCase):

    def setUp(self):
        self.profile = sp.SoilProfile(
            {
                'Depth from [m]': [0, 1, 5, 10],
                'Depth to [m]': [1, 5, 10, 20],
                'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
            }
        )

    def test_wrong_layering(self):
        self.assertRaises(IOError, sp.SoilProfile, ({
            'Depth from [m]': [0, 1, 4, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
        }))

    def test_wrong_depthkey(self):
        self.assertRaises(IOError, sp.SoilProfile, ({
            'z from [m]': [0, 1, 4, 10],
            'z to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
        }))

    def test_wrong_linearvariation(self):
        # Test incomplete linear parameter variation
        self.assertRaises(IOError, sp.SoilProfile, ({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40]
        }))
        # Check that complete parameter variation passes
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [1, 3, 10, 40]
        })

    def test_profile_creation(self):
        self.assertEqual(self.profile.max_depth, 20)
        self.assertEqual(self.profile.min_depth, 0)
        self.assertEqual(self.profile.layer_transitions()[0], 1)
        self.assertEqual(self.profile.layer_transitions(include_top=True)[0], 0)

    def test_soilparameter_retrieval(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [1, 3, 10, 40]
        })
        self.assertIn('qc [MPa]',profile.soil_parameters())
        self.assertIn('Soil type',profile.soil_parameters())
        self.assertIn('qc from [MPa]',profile.soil_parameters(condense_linear=False))
        self.assertIn('qc [MPa]', profile.numerical_soil_parameters())
        self.assertNotIn('Soil type', profile.numerical_soil_parameters())
        self.assertIn('qc from [MPa]', profile.numerical_soil_parameters(condense_linear=False))
        self.assertNotIn('Soil type', profile.numerical_soil_parameters(condense_linear=False))
        self.assertNotIn('qc from [MPa]', profile.string_soil_parameters())
        self.assertIn('Soil type', profile.string_soil_parameters())

    def test_transition_insert(self):
        self.profile.insert_layer_transition(depth=2.5)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 2.5)
        self.profile.insert_layer_transition(depth=2)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 2)

    def test_sign_conversion(self):
        self.profile.convert_depth_sign()
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], -5)
        self.profile.convert_depth_sign()
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 5)

    def test_depth_shift(self):
        self.profile.shift_depths(offset=3)
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 8)
        self.profile.shift_depths(offset=-3)
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 5)
