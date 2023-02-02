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
from groundhog.deepfoundations.axialcapacity import skinfriction

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_API_unit_shaft_friction_clay(unittest.TestCase):

    def test_values(self):
        self.assertAlmostEqual(skinfriction.API_unit_shaft_friction_clay(50.0, 100.0)['f_s_comp_out [kPa]'], 35.355, 3)


class Test_API_unit_shaft_friction_sand(unittest.TestCase):

    def test_values(self):
        self.assertEqual(skinfriction.API_unit_shaft_friction_sand_rp2geo("Medium dense", "Sand", 50.0)['f_s_comp_out [kPa]'], 18.5)
        self.assertEqual(skinfriction.API_unit_shaft_friction_sand_rp2geo(
            api_relativedensity="Medium dense",
            api_soildescription="Sand",
            sigma_vo_eff=50.0, fail_silently=False)['f_s_comp_out [kPa]'], 18.5)

    
class Test_almhamre_unitskinfriction_sand(unittest.TestCase):

    def test_almhamre_unitskinfriction_sand(self):
        result = skinfriction.unitskinfriction_sand_almhamre(
            qt=15,
            sigma_vo_eff=100,
            interface_friction_angle=28,
            depth=30,
            embedded_length=30)
        self.assertAlmostEqual(
            result['f_s_initial [kPa]'], 105.098, 3)
        self.assertAlmostEqual(
            result['f_s_res [kPa]'], 21.020, 3)
        self.assertAlmostEqual(
            result['f_s_comp_out [kPa]'], 52.549, 3)
        result = skinfriction.unitskinfriction_sand_almhamre(
            qt=15,
            sigma_vo_eff=100,
            interface_friction_angle=28,
            depth=20,
            embedded_length=30)
        self.assertAlmostEqual(
            result['f_s_initial [kPa]'], 105.098, 3)
        self.assertAlmostEqual(
            result['f_s_res [kPa]'], 21.020, 3)
        self.assertAlmostEqual(
            result['f_s_comp_out [kPa]'], 19.604, 3)


class Test_almhamre_unitskinfriction_clay(unittest.TestCase):

    def test_almhamre_unitskinfriction_clay(self):
        result = skinfriction.unitskinfriction_clay_almhamre(
            depth=30,
            embedded_length=30,
            qt=1.5,
            fs=0.2,
            sigma_vo_eff=100,
            fail_silently=False)
        self.assertAlmostEqual(
            result['f_s_initial [kPa]'], 200, 3)
        self.assertAlmostEqual(
            result['f_s_res [kPa]'], 5.775, 3)
        self.assertAlmostEqual(
            result['f_s_comp_out [kPa]'], 200, 3)
        result = skinfriction.unitskinfriction_clay_almhamre(
            depth=20,
            embedded_length=30,
            qt=1.5,
            fs=0.2,
            sigma_vo_eff=100,
            fail_silently=False)
        self.assertAlmostEqual(
            result['f_s_initial [kPa]'], 200, 3)
        self.assertAlmostEqual(
            result['f_s_res [kPa]'], 5.775, 3)
        self.assertAlmostEqual(
            result['f_s_comp_out [kPa]'], 125.464, 3)