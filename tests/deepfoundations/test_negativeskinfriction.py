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
from groundhog.deepfoundations.axialcapacity import negativeskinfriction

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_negativeskinfriction_pilegroup_zeevaertdebeer(unittest.TestCase):

    def setUp(self):
        K0 = 0.8
        DELTA = 25  # Interface friction angle, deg
        OD = 0.4  # Pile outer diameter, m
        INFLUENCE_ZONE = 3 / 0.4  # Multiplier on pile outer diameter
        GAMMA_EFF = 8  # Effective unit weight of soil, kN/m3
        SURCHARGE = 51  # kPa
        MAX_DEPTH = 4.5
        self.depth = np.linspace(0, MAX_DEPTH, 250)
        self.K0 = np.ones(250) * K0
        self.delta = np.ones(250) * DELTA
        self.diameter = OD
        self.diameter_influence = OD * INFLUENCE_ZONE
        self.gamma_eff = np.ones(250) * GAMMA_EFF
        self.p0 = SURCHARGE

    def test_negativeskinfriction_pilegroup_zeevaertdebeer(self):
        result = negativeskinfriction.negativeskinfriction_pilegroup_zeevaertdebeer(
            depths=self.depth,
            effective_unit_weights=self.gamma_eff,
            lateral_earth_pressure_coefficients=self.K0,
            interface_friction_angles=self.delta,
            surcharge=self.p0,
            diameter=self.diameter,
            diameter_influence=self.diameter_influence,
            fail_silently=False
        )
        self.assertAlmostEqual(
            result['negative_skin_friction [kN]'], 145.56, 2
        )
        self.assertAlmostEqual(
            result['negative_skin_friction_group [kN]'], 127.19, 2
        )
