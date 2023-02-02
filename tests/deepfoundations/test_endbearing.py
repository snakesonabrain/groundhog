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
from groundhog.deepfoundations.axialcapacity import endbearing

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_API_unit_end_bearing_clay(unittest.TestCase):

    def test_values(self):
        self.assertEqual(endbearing.API_unit_end_bearing_clay(50.0, N_c=9.0)['q_b_plugged [kPa]'], 450.0)
        self.assertEqual(endbearing.API_unit_end_bearing_clay(50.0, N_c=9.0)['q_b_coring [kPa]'], 450.0)


class Test_API_unit_end_bearing_sand_rp2geo(unittest.TestCase):

    def test_values(self):
        result = endbearing.API_unit_end_bearing_sand_rp2geo("Medium dense", "Sand", 100.0)
        self.assertEqual(result['q_b_plugged [kPa]'], 2000.0)
        self.assertEqual(result['q_b_coring [kPa]'], 2000.0)


class Test_unitendbearing_sand_almhamre(unittest.TestCase):

    def test_unitendbearing_sand_almhamre(self):
        result = endbearing.unitendbearing_sand_almhamre(
            qt=15,
            sigma_vo_eff=100)
        self.assertAlmostEqual(
            result['q_b_coring [kPa]'], 6129.157, 3)

class Test_unitendbearing_clay_almhamre(unittest.TestCase):

    def test_unitendbearing_clay_almhamre(self):
        result = endbearing.unitendbearing_clay_almhamre(
            qt=1.5)
        self.assertAlmostEqual(
            result['q_b_coring [kPa]'], 900, 3)