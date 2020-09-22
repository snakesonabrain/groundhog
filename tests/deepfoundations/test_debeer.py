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
from groundhog.deepfoundations.axialcapacity import debeer
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from groundhog.general.soilprofile import SoilProfile

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_pilecalcdebeer(unittest.TestCase):

    def setUp(self):
        self.cpt_data = pd.read_excel(os.path.join(TESTS_DATA_DIR, 'debeer_example.xlsx'))

    def test_pilecalcbeer(self):
        calc = debeer.DeBeerCalculation(
            depth=self.cpt_data['z [m]'],
            qc=self.cpt_data['qc [MPa]'],
            diameter_pile=0.4,
            diameter_cone=0.0357)
        calc.resample_data()
        profile = SoilProfile({
            'Depth from [m]': [0, 3, 6, 15],
            'Depth to [m]': [3, 6, 15, 20],
            'Soil type': ['Sand', 'Clay', 'Sand', 'Loam (silt)']
        })
        calc.set_soil_layers(soilprofile=profile)
        calc.calculate_base_resistance()
        calc.correct_shaft_qc(cone_type='U')
        calc.calculate_average_qc()
        calc.calculate_unit_shaft_friction()
        calc.set_shaft_base_factors(
            alpha_b_tertiary_clay=1.0,
            alpha_b_other=1.0,
            alpha_s_tertiary_clay=0.6,
            alpha_s_other=0.6)
        calc.calculate_pile_resistance(
            pile_penetration=16, base_area=0.25*np.pi*(0.4 ** 2), circumference=np.pi * 0.4)
        self.assertAlmostEqual(calc.Rs, 1313.1, 1)
        self.assertAlmostEqual(calc.Rb, 2800.4, 1)
        self.assertAlmostEqual(calc.Rc, 4113.5, 1)


class Test_shaftcalcdebeer(unittest.TestCase):

    def setUp(self):
        self.cpt = PCPTProcessing(title="Shaft test")
        self.cpt.load_excel(os.path.join(TESTS_DATA_DIR, "cpt_lecture3.xlsx"), sheet_name="CPT1")
        self.calc = debeer.DeBeerCalculation(
            depth=self.cpt.data["z [m]"],
            qc=self.cpt.data["qc [MPa]"],
            diameter_pile=0.4)
        self.calc.resample_data(spacing=0.2)
        profile = SoilProfile({
            'Depth from [m]': [0, 3, 7, 14, 18],
            'Depth to [m]': [3, 7, 14, 18, 24],
            'Soil type': ['Clay', 'Sand', 'Clay', 'Sand', 'Clayey sand / loam (silt)']
        })
        self.calc.set_soil_layers(soilprofile=profile)

    def test_baseresistance(self):
        self.calc.calculate_base_resistance()
        self.assertAlmostEqual(
            np.interp(15, self.calc.depth_qb, self.calc.qb),
            5.26,
            2
        )

    def test_shaftresistance(self):
        self.calc.correct_shaft_qc()
        self.calc.calculate_average_qc()
        self.assertAlmostEqual(
            self.calc.layering['qc avg [MPa]'].iloc[0],
            0.85,
            2
        )