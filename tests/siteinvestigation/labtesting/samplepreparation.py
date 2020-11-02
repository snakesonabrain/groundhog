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
from groundhog.siteinvestigation.labtesting import samplepreparation


class Test_samplepreparation(unittest.TestCase):

    def test_undercompaction_cohesionless_ladd(self):
        result = samplepreparation.undercompaction_cohesionless_ladd(
            sample_height=0.09, no_layers=5, undercompaction_deepest=8
        )
        self.assertEqual(result['U [-]'][0], 0.08)
        self.assertEqual(result['U [-]'][-1], 0)
        self.assertAlmostEqual(result['h [m]'][0], 0.0194, 4)
        self.assertEqual(result['h [m]'][-1], 0.09, 2)