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
from groundhog.deepfoundations.axialcapacity import piletesting

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_chinkondler(unittest.TestCase):

    def test_example(self):
        loads = [0, 100.0, 200.0, 300.0, 400.0, 501.0, 601.0, 701.0, 801.0, 901.0, 1001.0, 1101.0]
        settlements = [0.0, 0.82, 1.34, 1.82, 2.33, 2.97, 3.59, 4.44, 5.45, 6.59, 8.02, 9.61]
        result = piletesting.piletest_chinkondler(
            loads=loads,
            settlements=settlements,
            no_discard_points=5,
            selected_settlement=42,
            show_fig=False,
            fail_silently=False
        )
        self.assertAlmostEqual(result['slope [1/kN]'], 0.00043958, 5)
        self.assertAlmostEqual(result['Qdisp [kN]'], 1832.3, 1)
        