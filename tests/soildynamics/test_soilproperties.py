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
            PI=0,
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
            PI=100,
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
            PI=0,
            sigma_m_eff=400
        )
        self.assertAlmostEqual(
            result['G/Gmax [-]'], 0.98, 2
        )
        self.assertAlmostEqual(
            result['dampingratio [pct]'], 1.61, 2
        )

    def test_gmax_shearwavevelocty(self):
        result = soilproperties.gmax_shearwavevelocty(
            Vs=200,
            gamma=19
        )
        self.assertAlmostEqual(
            result['rho [kg/m3]'], 1937, 0
        )
        self.assertAlmostEqual(
            result['Gmax [kPa]'], 77472, 0
        )
