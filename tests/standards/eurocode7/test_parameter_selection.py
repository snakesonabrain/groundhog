#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import numpy as np
import pandas as pd

# Project imports
from groundhog.standards.eurocode7 import parameter_selection

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Test_parameter_selection(unittest.TestCase):

    def setUp(self):
        self.cpt_data = pd.read_csv(os.path.join(TESTS_DATA_DIR, 'su_hand_penetrometer.csv'))
        self.data = np.array(self.cpt_data['Su [kPa]'])
        self.depths = np.array(self.cpt_data['Depth [m]'])
        self.requested_depths = np.array([10, ])

    def test_constant_value(self):
        result = parameter_selection.constant_value(
            data=self.data,
            mode='Low',
            cov=0.18,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Xk'], 231.52, 2)
        result = parameter_selection.constant_value(
            data=self.data,
            mode='Mean',
            cov=0.18,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Xk'], 320.44, 2)

    def test_linear_trend(self):
        result = parameter_selection.linear_trend(
            data=self.data,
            depths=self.depths,
            requested_depths=self.requested_depths,
            mode='Low',
            fail_silently=False
        )
        self.assertAlmostEqual(result['Xk'][0], 201.51, 2)
        result = parameter_selection.linear_trend(
            data=self.data,
            depths=self.depths,
            requested_depths=self.requested_depths,
            mode='Mean',
            fail_silently=False
        )
        self.assertAlmostEqual(result['Xk'][0], 279.33, 2)