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