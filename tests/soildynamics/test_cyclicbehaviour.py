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
from groundhog.soildynamics import cyclicbehaviour

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Test_cyclic_functions(unittest.TestCase):

    def test_cycliccontours_dssclay_andersen(self):
        result = cyclicbehaviour.cycliccontours_dssclay_andersen(
            undrained_shear_strength=10,
            average_shear_stress=5,
            cyclic_shear_stress=3.955
        )
        self.assertAlmostEqual(result['Nf [-]'], 1000, 0)

    def test_cycliccontours_triaxialclay_andersen(self):
        result = cyclicbehaviour.cycliccontours_triaxialclay_andersen(
            undrained_shear_strength=10,
            average_shear_stress=5,
            cyclic_shear_stress=3.1474
        )
        self.assertAlmostEqual(result['Nf [-]'], 978, 0)
        result = cyclicbehaviour.cycliccontours_triaxialclay_andersen(
            undrained_shear_strength=10,
            average_shear_stress=-2.5,
            cyclic_shear_stress=1.215
        )
        self.assertAlmostEqual(result['Nf [-]'], 1124, 0)