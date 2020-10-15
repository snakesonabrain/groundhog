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
from groundhog.shallowfoundations import settlement

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_settlement(unittest.TestCase):

    def test_primaryconsolidationsettlement_nc(self):
        result = settlement.primaryconsolidationsettlement_nc(
            initial_height=1,
            initial_voidratio=1,
            initial_effective_stress=100,
            effective_stress_increase=100,
            compression_index=0.3)
        self.assertAlmostEqual(result['delta z [m]'], 0.045, 3)

    def test_primaryconsolidationsettlement_oc(self):
        result = settlement.primaryconsolidationsettlement_oc(
            initial_height=1,
            initial_voidratio=1,
            initial_effective_stress=100,
            effective_stress_increase=200,
            preconsolidation_pressure=200,
            compression_index=0.3,
            recompression_index=0.03)
        self.assertAlmostEqual(result['delta z [m]'], 0.031, 3)

