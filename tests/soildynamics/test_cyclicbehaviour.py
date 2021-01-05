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

    def test_strainaccumulation_dssclay_andersen(self):
        result = cyclicbehaviour.strainaccumulation_dssclay_andersen(
            undrained_shear_strength=10,
            cyclic_shear_stress=7.5,
            cycle_no=10
        )
        self.assertAlmostEqual(result['cyclic strain [%]'], 1.31, 2)
        result = cyclicbehaviour.strainaccumulation_dssclay_andersen(
            undrained_shear_strength=10,
            cyclic_shear_stress=6,
            cycle_no=100
        )
        self.assertAlmostEqual(result['cyclic strain [%]'], 0.88, 2)

    def test_strainaccumulation_triaxialclay_andersen(self):
        result = cyclicbehaviour.strainaccumulation_triaxialclay_andersen(
            undrained_shear_strength=10,
            cyclic_shear_stress=3,
            cycle_no=10,
            fail_silently=False
        )
        self.assertAlmostEqual(
            result['cyclic strain [%]'], 0.45, 3
        )
        self.assertAlmostEqual(
            result['average strain [%]'], -1.562, 3
        )

    def test_porepressureaccumulation_dssclay_andersen(self):
        result = cyclicbehaviour.porepressureaccumulation_dssclay_andersen(
            undrained_shear_strength=10,
            cyclic_shear_stress=5,
            cycle_no=10,
            fail_silently=False
        )
        self.assertAlmostEqual(
            result['Excess pore pressure ratio [-]'], 0.082, 3)

    def test_porepressureaccumulation_triaxialclay_andersen(self):
        result = cyclicbehaviour.porepressureaccumulation_triaxialclay_andersen(
            undrained_shear_strength=10,
            cyclic_shear_stress=3,
            cycle_no=10,
            fail_silently=False
        )
        self.assertAlmostEqual(
            result['Excess pore pressure ratio [-]'], 0.166, 3)

