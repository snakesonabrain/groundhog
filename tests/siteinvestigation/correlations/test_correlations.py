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
from groundhog.siteinvestigation.correlations import cohesive, cohesionless


class Test_Correlations(unittest.TestCase):

    def test_voidratio_porosity(self):
        self.assertEqual(
            cohesive.compressionindex_watercontent_koppula(water_content=0.4)['Cc [-]'],
            0.4
        )
        self.assertAlmostEqual(
            cohesive.compressionindex_watercontent_koppula(water_content=0.4)['Cr [-]'],
            0.053, 3
        )

    def test_gmax_sand_hardinblack(self):
        self.assertAlmostEqual(
            cohesionless.gmax_sand_hardinblack(sigma_m0=100, void_ratio=0.3)['Gmax [kPa]'],
            241047, 0
        )

    def test_permeability_d10_hazen(self):
        self.assertAlmostEqual(
            cohesionless.permeability_d10_hazen(0.1)['k [m/s]'],
            1e-4, 5
        )

    def test_frictionangle_plasticityindex(self):
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(50)['Effective friction angle [deg]'],
            25.34,
            2
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(100)['Effective friction angle [deg]'],
            21.4,
            1
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(1000)['Effective friction angle [deg]'],
            1.7,
            1
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(200)['Effective friction angle [deg]'],
            11.5,
            1
        )
