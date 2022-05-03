# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.consolidation.dissipation import onedimensionalconsolidation


class Test_onedimensionalsolutions(unittest.TestCase):

    def test_fourierseries(self):
        result = onedimensionalconsolidation.pore_pressure_fourier(
            delta_u_0=50,
            depths=np.linspace(0, 1, 50),
            time=1e4,
            cv=100,
            layer_thickness=1
        )
        self.assertAlmostEqual(
            np.interp(0.5, np.linspace(0, 1, 50), result['delta u [kPa]']),
            45.3, 1)