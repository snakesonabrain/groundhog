#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest

# 3rd party packages

# Project imports
from groundhog.consolidation.groundwaterflow import pumpingtests


class Test_pumpingtests(unittest.TestCase):

    def test_hydraulicconductivity_unconfinedaquifer(self):
        result = pumpingtests.hydraulicconductivity_unconfinedaquifer(
            radius_1=15,
            radius_2=30,
            piezometric_height_1=11.5,
            piezometric_height_2=11.7,
            flowrate=10.6e-3
        )
        self.assertAlmostEqual(
            100 * result['hydraulic_conductivity [m/s]'], 0.05, 2)