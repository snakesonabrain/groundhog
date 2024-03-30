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
        self.assertEqual(result['eta_H [%]'], 45)
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
        self.assertEqual(result['eta_H [%]'], 50)
        self.assertEqual(result['eta_B [-]'], 1)
        self.assertEqual(result['eta_S [-]'], 1)
        self.assertEqual(result['eta_R [-]'], 0.85)

    def test_relativedensity_spt_kulhawymayne(self):
        result = spt_correlations.relativedensity_spt_kulhawymayne(
            N1_60=30,
            d_50=0.2
        )
        self.assertAlmostEqual(result['Dr [pct]'], 83.99, 2)
        result = spt_correlations.relativedensity_spt_kulhawymayne(
            N1_60=6,
            d_50=0.2
        )
        self.assertAlmostEqual(result['Dr [pct]'], 37.56, 2)
        result = spt_correlations.relativedensity_spt_kulhawymayne(
            N1_60=30,
            d_50=0.2,
            ocr=10
        )
        self.assertAlmostEqual(result['Dr [pct]'], 68.27, 2)

    def test_undrainedshearstrength_spt_salgado(self):
        result = spt_correlations.undrainedshearstrength_spt_salgado(
            N_60=30,
            pi=30,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Su [kPa]'], 135, 1)
        result = spt_correlations.undrainedshearstrength_spt_salgado(
            N_60=2,
            pi=60,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Su [kPa]'], 8.6, 1)

    def test_frictionangle_spt_kulhawymayne(self):
        result = spt_correlations.frictionangle_spt_kulhawymayne(
            N=30,
            sigma_vo_eff=100,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Phi [deg]'], 44.2, 1)
        result = spt_correlations.frictionangle_spt_kulhawymayne(
            N=30,
            sigma_vo_eff=300,
            fail_silently=False
        )
        self.assertAlmostEqual(result['Phi [deg]'], 36.5, 1)

    def test_relativedensityclass_spt_terzaghipeck(self):
        result = spt_correlations.relativedensityclass_spt_terzaghipeck(N=25, fail_silently=False)
        self.assertEqual(result['Dr class'], "Medium dense")

    def test_overburdencorrection_spt_ISO(self):
        result = spt_correlations.overburdencorrection_spt_ISO(N=25, sigma_vo_eff=98)
        self.assertEqual(result['N1 [-]'], 25)

    def test_frictionangle_spt_PHT(self):
        result = spt_correlations.frictionangle_spt_PHT(N1_60=25)
        self.assertAlmostEqual(result['Phi [deg]'], 34.3, 1)

    def test_youngsmodulus_spt_AASHTO(self):
        result = spt_correlations.youngsmodulus_spt_AASHTO(N1_60=25, soiltype='Coarse sands')
        self.assertAlmostEqual(result['Es [MPa]'], 25, 1)

    def test_undrainedshearstrengthclass_spt_terzaghipeck(self):
        result = spt_correlations.undrainedshearstrengthclass_spt_terzaghipeck(N=5)
        self.assertEqual(result['Consistency class'], 'Medium')
        self.assertEqual(result['qu min [kPa]'], 50)
        self.assertEqual(result['qu max [kPa]'], 100)