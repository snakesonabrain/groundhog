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
from groundhog.shallowfoundations import settlement
from groundhog.general.soilprofile import SoilProfile


TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_settlement(unittest.TestCase):

    def test_primaryconsolidationsettlement_nc(self):
        result = settlement.primaryconsolidationsettlement_nc(
            initial_height=1,
            initial_voidratio=1,
            initial_effective_stress=100,
            effective_stress_increase=100,
            compression_index=0.3)
        self.assertAlmostEqual(result['delta z [m]'], 0.045, 3)

    def test_primaryconsolidationsettlement_oc(self):
        result = settlement.primaryconsolidationsettlement_oc(
            initial_height=1,
            initial_voidratio=1,
            initial_effective_stress=100,
            effective_stress_increase=200,
            preconsolidation_pressure=200,
            compression_index=0.3,
            recompression_index=0.03)
        self.assertAlmostEqual(result['delta z [m]'], 0.031, 3)


    def test_settlement_calculation(self):
        sp = SoilProfile({
            'Depth from [m]': [0, 4.2],
            'Depth to [m]': [4.2, 20],
            'Soil type': ['Clay', 'Clay'],
            'Total unit weight [kN/m3]': [15, 17],
            'Cc [-]': [0.7, 0.45],
            'Cr [-]': [0.07, 0.045],
            'OCR [-]': [1, 4]
        })
        calc = settlement.SettlementCalculation(sp)
        calc.calculate_initial_state(waterlevel=0.8)
        calc.create_grid()
        calc.set_foundation(shape='rectangular', length=8, width=5)
        calc.calculate_foundation_stress(applied_stress=100)
        calc.calculate()
        self.assertAlmostEqual(
            calc.settlement,
            0.741,
            3)

    def test_settlement_calculation_budhu(self):
        sp = SoilProfile({
            'Depth from [m]': [0, 3, 10.4, 12.4],
            'Depth to [m]': [3, 10.4, 12.4, 15],
            'Soil type': ['Sand', 'Sand', 'Clay', 'Sand'],
            'Total unit weight [kN/m3]': [19.3, 19.5, 17.5, 19.5],
            'Cc [-]': [0, 0, 0.3, 0],
            'Cr [-]': [0, 0, 0.03, 0],
            'OCR [-]': [1, 1, 1, 1]
        })
        calc = settlement.SettlementCalculation(sp)
        calc.calculate_initial_state(waterlevel=3, specific_gravity=2.7, unitweight_water=9.8)
        calc.create_grid()
        calc.set_foundation(shape='rectangular', length=8, width=5)
        calc.calculate_foundation_stress(applied_stress=1100)
        calc.calculate(compression_index__min=0, recompression_index__min=0)
        self.assertAlmostEqual(
            calc.settlement,
            0.086,
            3)
        
    def test_settlement_calculation_mv(self):
        sp = SoilProfile({
            'Depth from [m]': [0, 4.2],
            'Depth to [m]': [4.2, 20],
            'Soil type': ['Clay', 'Clay'],
            'Total unit weight [kN/m3]': [15, 17],
            'mv [1/kPa]': [1e-3, 2e-3]
        })
        calc = settlement.SettlementCalculation(sp)
        calc.calculate_initial_state(waterlevel=0.8)
        calc.create_grid()
        calc.set_foundation(shape='rectangular', length=8, width=5)
        calc.calculate_foundation_stress(applied_stress=100)
        calc.calculate_mv()
        self.assertAlmostEqual(calc.settlement, 0.864, 3)