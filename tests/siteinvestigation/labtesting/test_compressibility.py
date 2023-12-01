#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
from unittest.mock import patch
import os

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.siteinvestigation.labtesting import compressibility

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_compressibility(unittest.TestCase):

    @patch('groundhog.siteinvestigation.labtesting.compressibility.selectpoints', return_value=[(0.9677419354838719, 0.009449999999999958), (11.397849462365592, 0.1795500000000002)])
    def test_roottimemethod(self, roottimemethod):
        self.df = pd.read_csv(os.path.join(TESTS_DATA_DIR, 'oedometerdata.csv'))
        result = compressibility.roottimemethod(
            times=np.array(self.df['Load step time [s]']),
            settlements=np.array(-self.df['Load step settlement [mm]']),
            drainagelength=0.009, showfig=False, fail_silently=False)
        
        self.assertAlmostEqual(result['cv [m2/yr]'], 6.298, 3)

    @patch('groundhog.siteinvestigation.labtesting.compressibility.selectpoints', side_effect=[
        [(2.006413839479631, 0.16047619047619138), (2.227559284290414, 0.20142857142857234)],
        [(4.002542459745387, 0.39185714285714357), (4.916222323832045, 0.4420238095238102)],
        [(0.9996201038936956, 0.048829863367755855)]
    ])
    def test_logtimemethod(self, logtimemethod):
        self.df = pd.read_csv(os.path.join(TESTS_DATA_DIR, 'oedometerdata.csv'))
        result = compressibility.logtimemethod(
            times=np.array(self.df['Load step time [s]']),
            settlements=np.array(-self.df['Load step settlement [mm]']),
            drainagelength=0.009, showfig=False, fail_silently=False)
        print(result)
        self.assertAlmostEqual(result['cv [m2/yr]'], 4.757, 3)

    