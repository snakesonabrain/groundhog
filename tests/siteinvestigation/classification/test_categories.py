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
from groundhog.siteinvestigation.classification import categories


class Test_Categories(unittest.TestCase):


    def test_relativedensity_categories(self):
        self.assertEqual(
            categories.relativedensity_categories(
                relative_density=0.3)['Relative density'], "Loose")
        self.assertEqual(
            categories.relativedensity_categories(
                relative_density=0.9)['Relative density'], "Very dense")
        self.assertEqual(
            categories.relativedensity_categories(
                relative_density=0.85)['Relative density'], "Very dense")

    def test_su_categories(self):
        self.assertEqual(
            categories.su_categories(
                undrained_shear_strength=50
            )['strength class'], 'Medium'
        )
        self.assertEqual(
            categories.su_categories(
                undrained_shear_strength=60, standard='ASTM D-2488'
            )['strength class'], 'Stiff'
        )

    def test_uscs_categories(self):
        self.assertEqual(
            categories.uscs_categories('OL')['Soil type'],
            "Organic clays organic silt-clays of low plasticity"
        )

    def test_samplequality_voidratio_lunne(self):
        self.assertEqual(
            categories.samplequality_voidratio_lunne(voidratio=1.0, voidratio_change=0.1, ocr=1.5)['Quality category'],
            'Poor'
        )
        self.assertEqual(
            categories.samplequality_voidratio_lunne(voidratio=1.0, voidratio_change=0.11, ocr=3)['Quality category'],
            'Very poor'
        )