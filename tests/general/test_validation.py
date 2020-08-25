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
from groundhog.general import validation


class Test_check_layer_overlap(unittest.TestCase):

    def setUp(self):
        self.df_no_overlap = pd.DataFrame({
            'z from [m]': np.array([0, 4, 10]),
            'z to [m]': np.array([4, 10, 20])
        })
        self.df_gaps = pd.DataFrame({
            'z from [m]': np.array([0, 5, 10]),
            'z to [m]': np.array([4, 10, 20])
        })
        self.df_overlap = pd.DataFrame({
            'z from [m]': np.array([0, 4, 10]),
            'z to [m]': np.array([5, 10, 20])
        })
        self.df_otherkeys = pd.DataFrame({
            'Depth from [m]': np.array([0, 4, 10]),
            'Depth to [m]': np.array([4, 10, 20])
        })

    def test_check_layer_overlap(self):
        validation.check_layer_overlap(self.df_no_overlap)
        validation.check_layer_overlap(self.df_otherkeys, z_from_key="Depth from [m]", z_to_key="Depth to [m]")
        # An error should be raised
        self.assertRaises(ValueError, validation.check_layer_overlap, df=self.df_gaps)
        self.assertRaises(ValueError, validation.check_layer_overlap, df=self.df_overlap)
        validation.check_layer_overlap(self.df_gaps, raise_error=False)
        validation.check_layer_overlap(self.df_overlap, raise_error=False)
