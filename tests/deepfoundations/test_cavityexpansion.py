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
from groundhog.deepfoundations.boreholestability import cavityexpansion

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_cavityexpansionfunctions(unittest.TestCase):

    def test_stress_elastic_isotropic(self):
        result = cavityexpansion.stress_cylinder_elastic_isotropic(
            radius=1, internal_pressure=0, farfield_pressure=100, borehole_radius=1,
            fail_silently=False
        )
        self.assertEqual(result['radial stress [kPa]'], 0)
        self.assertEqual(result['tangential stress [kPa]'], 200)
        result = cavityexpansion.stress_cylinder_elastic_isotropic(
            radius=1e7, internal_pressure=0, farfield_pressure=100, borehole_radius=1,
            fail_silently=False
        )
        self.assertAlmostEqual(result['radial stress [kPa]'], 100, 3)
        self.assertAlmostEqual(result['tangential stress [kPa]'], 100, 3)
        result = cavityexpansion.stress_cylinder_elastic_isotropic(
            radius=np.linspace(1, 1e7, 100), internal_pressure=0, farfield_pressure=100, borehole_radius=1,
            fail_silently=False
        )
        self.assertEqual(result['radial stress [kPa]'][0], 0)
        self.assertEqual(result['tangential stress [kPa]'][0], 200)
        self.assertAlmostEqual(result['radial stress [kPa]'][-1], 100, 3)
        self.assertAlmostEqual(result['tangential stress [kPa]'][-1], 100, 3)

    def test_expansion_tresca_thicksphere(self):
        result = cavityexpansion.expansion_tresca_thicksphere(
            undrained_shear_strength=10,
            internal_radius=1,
            external_radius=10,
            internal_pressure=150,
            external_pressure=100,
            youngs_modulus=1000,
            poissons_ratio=0.495,
            fail_silently=False)
        self.assertAlmostEqual(result['expanded_radius [m]'], 1.250, 3)

    def test_expansion_cylinder_tresca(self):
        elastic_result = cavityexpansion.expansion_cylinder_tresca(
            insitu_pressure=100,
            borehole_pressure=105,
            diameter=0.4,
            undrained_shear_strength=10,
            shear_modulus=1000)
        self.assertFalse(elastic_result['yielding'])
        self.assertAlmostEqual(elastic_result['elastic wall expansion [m]'], 0.0025, 4)
        self.assertEqual(elastic_result['plastic wall expansion [m]'], 0)
        self.assertEqual(elastic_result['plastic radius [m]'], 0.2)
        plastic_result = cavityexpansion.expansion_cylinder_tresca(
            insitu_pressure=100,
            borehole_pressure=120,
            diameter=0.4,
            undrained_shear_strength=10,
            shear_modulus=1000)
        self.assertTrue(plastic_result['yielding'])
        self.assertAlmostEqual(plastic_result['elastic wall expansion [m]'], 0.01, 3)
        self.assertAlmostEqual(plastic_result['plastic wall expansion [m]'], 0.0018, 4)
        collapsed_result = cavityexpansion.expansion_cylinder_tresca(
            insitu_pressure=100,
            borehole_pressure=250,
            diameter=0.4,
            undrained_shear_strength=10,
            shear_modulus=1000)
        self.assertTrue(collapsed_result['yielding'])
        self.assertTrue(np.isnan(collapsed_result['elastic wall expansion [m]']))