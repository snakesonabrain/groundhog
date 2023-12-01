#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import numpy as np

# Project imports
from groundhog.excavations import basic


class Test_earthpressurecoefficients_frictionangle(unittest.TestCase):

    def test_values(self):
        result = basic.earthpressurecoefficients_frictionangle(
            phi_eff=30
        )
        self.assertAlmostEqual(result['Ka [-]'], 1 / 3, 6)
        self.assertAlmostEqual(result['Kp [-]'], 3, 6)
        self.assertAlmostEqual(result['theta_a [radians]'], np.radians(60), 6)
        self.assertAlmostEqual(result['theta_p [radians]'], np.radians(30), 6)

    
class Test_earthpressurecoefficients_poncelet(unittest.TestCase):

    def test_values(self):
        result = basic.earthpressurecoefficients_poncelet(
            phi_eff=30,
            interface_friction_angle=20,
            wall_angle=5,
            top_angle=5,
            fail_silently=False
        )
        self.assertAlmostEqual(result['KaC [-]'], 0.358, 3)
        self.assertAlmostEqual(result['KpC [-]'], 6.605, 3)

class Test_earthpressurecoefficients_rankine(unittest.TestCase):

    def test_values(self):
        result = basic.earthpressurecoefficients_rankine(
            phi_eff=30,
            wall_angle=5,
            top_angle=5,
            fail_silently=False
        )
        self.assertAlmostEqual(result['KaR [-]'], 0.361, 3)
        self.assertAlmostEqual(result['KpR [-]'], 2.997, 3)