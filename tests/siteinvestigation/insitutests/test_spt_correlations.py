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
from groundhog.siteinvestigation.insitutests import spt_correlations


class Test_spt_correlations(unittest.TestCase):

    def test_overburdencorrection_spt_liaowhitman(self):
        self.assertEqual(
            spt_correlations.overburdencorrection_spt_liaowhitman(N=50, sigma_vo_eff=100)['CN [-]'],
            1
        )
        self.assertEqual(
            spt_correlations.overburdencorrection_spt_liaowhitman(N=50, sigma_vo_eff=25)['CN [-]'],
            2
        )

    def test_spt_N60_correction(self):
        result = spt_correlations.spt_N60_correction(
            N=20,
            borehole_diameter=70,
            rod_length=5,
            country='United States',
            hammertype='Donut',
            hammerrelease='Rope and pulley'
        )
        self.assertEqual(result['eta_H [pct]'], 45)
        self.assertEqual(result['eta_B [-]'], 1)
        self.assertEqual(result['eta_S [-]'], 1)
        self.assertEqual(result['eta_R [-]'], 0.85)
        result = spt_correlations.spt_N60_correction(
            N=20,
            borehole_diameter=70,
            rod_length=5,
            country='Other',
            hammertype='Donut',
            hammerrelease='Rope and pulley',
            eta_H=50,
            fail_silently=False
        )
        self.assertEqual(result['eta_H [pct]'], 50)
        self.assertEqual(result['eta_B [-]'], 1)
        self.assertEqual(result['eta_S [-]'], 1)
        self.assertEqual(result['eta_R [-]'], 0.85)