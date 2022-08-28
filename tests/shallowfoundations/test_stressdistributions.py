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
from groundhog.shallowfoundations import stressdistribution

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_stressdistributions(unittest.TestCase):

    def test_stresses_pointload(self):
        result = stressdistribution.stresses_pointload(pointload=100, z=1, r=0, poissonsratio=0.3)
        self.assertAlmostEqual(result['delta sigma z [kPa]'], 47.75, 2)
        self.assertAlmostEqual(result['delta sigma r [kPa]'], -3.18, 2)
        self.assertAlmostEqual(result['delta sigma theta [kPa]'], 3.18, 2)

    def test_stresses_stripload(self):
        result_uniform = stressdistribution.stresses_stripload(
            imposedstress=100, z=1, x=1, width=1)
        self.assertAlmostEqual(result_uniform['delta sigma z [kPa]'], 40.92, 2)
        self.assertAlmostEqual(result_uniform['delta sigma x [kPa]'], 9.08, 2)
        self.assertAlmostEqual(result_uniform['delta tau zx [kPa]'], 15.92, 2)
        result_uniform_center = stressdistribution.stresses_stripload(
            imposedstress=100, z=1e-9, x=0.5, width=1)
        self.assertAlmostEqual(result_uniform_center['delta sigma z [kPa]'], 100, 2)
        result_triangular = stressdistribution.stresses_stripload(
            imposedstress=100, z=1, x=1, width=1, triangular=True)
        self.assertAlmostEqual(result_triangular['delta sigma z [kPa]'], 25, 2)
        self.assertAlmostEqual(result_triangular['delta sigma x [kPa]'], 2.94, 2)
        self.assertAlmostEqual(result_triangular['delta tau zx [kPa]'], 6.83, 2)

    def test_stresses_circle(self):
        result = stressdistribution.stresses_circle(
            imposedstress=100, z=1, footing_radius=1, poissonsratio=0.3)
        self.assertAlmostEqual(result['delta sigma z [kPa]'], 64.64, 2)
        self.assertAlmostEqual(result['delta sigma r [kPa]'], -86.17, 2)

    def test_stresses_rectangle(self):
        result = stressdistribution.stresses_rectangle(
            imposedstress=100, length=1, width=1, z=1)
        self.assertAlmostEqual(result['delta sigma z [kPa]'], 17.52, 2)
        self.assertAlmostEqual(result['delta sigma x [kPa]'], 3.74, 2)
        self.assertAlmostEqual(result['delta sigma y [kPa]'], 3.74, 2)
        self.assertAlmostEqual(result['delta tau zx [kPa]'], 6.66, 2)
