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