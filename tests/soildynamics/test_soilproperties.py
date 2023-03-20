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
from groundhog.soildynamics import soilproperties


class Test_SoilProperties(unittest.TestCase):

    def test_modulusreduction_plasticity_ishibashi(self):
        result = soilproperties.modulusreduction_plasticity_ishibashi(
            strain=0.01,
            pi=0,
            sigma_m_eff=100
        )
        self.assertAlmostEqual(
            result['G/Gmax [-]'], 0.84, 2
        )
        self.assertAlmostEqual(
            result['dampingratio [pct]'], 3.84, 2
        )
        result = soilproperties.modulusreduction_plasticity_ishibashi(
            strain=0.01,
            pi=100,
            sigma_m_eff=100
        )
        self.assertAlmostEqual(
            result['G/Gmax [-]'], 0.98, 2
        )
        self.assertAlmostEqual(
            result['dampingratio [pct]'], 0.78, 2
        )
        result = soilproperties.modulusreduction_plasticity_ishibashi(
            strain=0.01,
            pi=0,
            sigma_m_eff=400
        )
        self.assertAlmostEqual(
            result['G/Gmax [-]'], 0.98, 2
        )
        self.assertAlmostEqual(
            result['dampingratio [pct]'], 1.61, 2
        )

    def test_gmax_shearwavevelocty(self):
        result = soilproperties.gmax_shearwavevelocity(
            Vs=200,
            gamma=19
        )
        self.assertAlmostEqual(
            result['rho [kg/m3]'], 1937, 0
        )
        self.assertAlmostEqual(
            result['Gmax [kPa]'], 77472, 0
        )

    def test_dampingratio_sandgravel_seed(self):
        result = soilproperties.dampingratio_sandgravel_seed(
            cyclic_shear_strain=0.01
        )
        self.assertAlmostEqual(
            result['D LE [pct]'], 2.48, 2)
        self.assertAlmostEqual(
            result['D BE [pct]'], 5.4, 2)
        self.assertAlmostEqual(
            result['D HE [pct]'], 10.04, 2)

    def test_modulusreduction_darendeli(self):
        result = soilproperties.modulusreduction_darendeli(
            mean_effective_stress=100,
            pi=15,
            ocr=1,
            N=10,
            frequency=1,
            soiltype='all'
        )
        self.assertAlmostEqual(
            np.interp(0.05, result['strains [pct]'], result['G/Gmax [-]']),
            0.5, 1
        )
        self.assertAlmostEqual(
            np.interp(0.05, result['strains [pct]'], result['D [pct]']),
            8.8, 1
        )