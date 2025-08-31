#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest

# 3rd party packages
import numpy as np

# Project imports
from groundhog.pipelinescables.stability import penetration as pen


class Test_DNV_Undrained_Method1(unittest.TestCase):

    def test_contactwidth(self):
        self.assertEqual(pen.contactwidth(diameter=1, penetration=0)['B [m]'], 0)
        self.assertEqual(pen.contactwidth(diameter=1, penetration=0.5)['B [m]'], 1)
        self.assertEqual(pen.contactwidth(diameter=1, penetration=0.6)['B [m]'], 1)
        
    def test_penetratedarea(self):
        self.assertEqual(pen.penetratedarea(diameter=1, penetration=0)['Abm [m2]'], 0)
        self.assertEqual(
            pen.penetratedarea(diameter=1, penetration=0.5)['Abm [m2]'],
            0.5 * 0.25 * np.pi)
        self.assertEqual(
            pen.penetratedarea(diameter=1, penetration=1)['Abm [m2]'],
            0.5 * 0.25 * np.pi + 0.5)
        
    def test_penetration_undrained_method1(self):
        result = pen.embedment_undrained_method1(
            diameter=1,
            undrained_shear_strength=5,
            k_su=0,
            gamma_eff=4,
            penetration=0.5)
        
        self.assertAlmostEqual(result['Qv0 [kN/m]'], 5.14 * 5)
        result = pen.embedment_undrained_method1(
            diameter=1,
            undrained_shear_strength=5,
            k_su=1,
            gamma_eff=4,
            penetration=0.5)
        self.assertAlmostEqual(result['Qv0 [kN/m]'], 28.17, 2)