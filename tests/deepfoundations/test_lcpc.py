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
from groundhog.deepfoundations.axialcapacity import lcpc
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from groundhog.general.soilprofile import SoilProfile

TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_pilecalclcpc(unittest.TestCase):

    def setUp(self):
        self.cpt_data = pd.read_excel(os.path.join(TESTS_DATA_DIR, 'debeer_example.xlsx'))

    def test_pilecalc_lcpc(self):
        calc = lcpc.LCPCAxcapCalculation(
            depth=self.cpt_data['z [m]'],
            qc=self.cpt_data['qc [MPa]'],
            diameter_pile=0.4,
            group_base='II', group_shaft='IA')
        profile = SoilProfile({
            'Depth from [m]': [0, 3, 6, 15, 20],
            'Depth to [m]': [3, 6, 15, 20, 25],
            "Total unit weight [kN/m3]": [16, 19, 17.5, 20, 21],
            'Soil type': ['Clay', 'Sand', 'Silt', 'Gravel', 'Chalk'],
            "Ignore shaft friction": [True, False, False, False, False]
        })
        calc.set_soil_layers(soilprofile=profile)
        calc.qca_calculation()

        calc.calculate_base_resistance()
        calc.calculate_shaft_resistance()
        
        result = calc.get_axialpileresistance(17)

        self.assertAlmostEqual(result["Rs [kN]"], 783, 0)
        self.assertAlmostEqual(result["Rb [kN]"], 2166, 0)
        self.assertAlmostEqual(result["Rc [kN]"], 2948, 0)
