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
from groundhog.deepfoundations.axialcapacity import koppejan

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_pilecalckoppejan(unittest.TestCase):

    def setUp(self):
        self.cpt_data = pd.read_excel(os.path.join(TESTS_DATA_DIR, 'cpt_koppejan.xlsx'))
        self.layering = pd.DataFrame({
            'Depth from [m]': [0, 3, 7, 14, 18.2, 21.3],
            'Depth to [m]': [3, 7, 14, 18.2, 21.3, 22],
            'Total unit weight [kN/m3]': [17, 20, 18, 20, 20, 20]
        })

    def test_mapping(self):
        """
        Tests the mapping of soil parameters to the grid
        """
        self.koppejan = koppejan.KoppejanCalculation(
            depth=self.cpt_data['z [m]'],
            qc=self.cpt_data['qc [MPa]'],
            diameter=0.4,
            penetration=16.5
        )
        self.koppejan.set_layer_properties(
            layer_data=self.layering)
        self.assertEqual(
            np.interp(
                5,
                self.koppejan.data['z [m]'],
                self.koppejan.data['Total unit weight [kN/m3]']), 20
        )
        self.assertEqual(
            np.interp(
                21.5,
                self.koppejan.data['z [m]'],
                self.koppejan.data['qclim [MPa]']), 12
        )

    def test_shaft_capacity(self):
        """
        Test the shaft capacity calculation according to Koppejan
        """
        self.test_mapping()
        self.koppejan.calculate_side_friction(alpha_s=0.01)
        self.assertAlmostEqual(self.koppejan.Frs, 1132, 0)

    def test_base_capacity(self):
        """
        Test the base capacity calculation according to Koppejan
        """
        self.test_mapping()
        self.koppejan.calculate_base_resistance(alpha_p=1)
        self.assertAlmostEqual(self.koppejan.qcII, 16.13, 2)
        self.assertAlmostEqual(self.koppejan.qcIII, 8.02, 2)
        self.assertAlmostEqual(self.koppejan.qcI, 12.51, 2)
        self.assertAlmostEqual(self.koppejan.Frb, 1404, 0)