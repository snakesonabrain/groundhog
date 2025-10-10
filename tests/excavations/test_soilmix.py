#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import numpy as np
import plotly.express as px

# Project imports
from groundhog.excavations import soilmix

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_bendingstiffness_soilmix(unittest.TestCase):

    def test_method1(self):
        result = soilmix.bendingstiffness_soilmix_method1(
            moment_inertia_reinforcement=1.9429e-5,
            modulus_soilmix=2e6,
            height_soilmix=0.550,
            reinforcement_offset=1.1,
            height_reinforcement=0.2,
            flange_thickness=0.0085,
            connection_thickness=0.0056,
            flange_width=0.1
        )
        self.assertAlmostEqual(result['EI-1 [kNm2]'], 34543.3, 1)
        self.assertAlmostEqual(result['EI-2 [kNm2]'], 13405.9, 1)

    def test_method2(self):
        result = soilmix.bendingstiffness_soilmix_method2(
            bendingstiffness_reinforcement=4.08e3,
            modulus_soilmix=2e6,
            height_soilmix=0.550,
            reinforcement_offset=1.1
        )
        self.assertAlmostEqual(result['EI-eff [kNm2]'], 19331.0, 1)