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
from groundhog.siteinvestigation.classification import phaserelations


class Test_PhaseRelations(unittest.TestCase):

    def test_voidratio_porosity(self):
        self.assertAlmostEqual(
            phaserelations.voidratio_porosity(porosity=0.4)['voidratio [-]'],
            0.667, 3
        )

    def test_porosity_voidratio(self):
        self.assertAlmostEqual(
            phaserelations.porosity_voidratio(voidratio=0.667)['porosity [-]'],
            0.4, 3
        )

    def test_saturation(self):
        self.assertAlmostEqual(
            phaserelations.saturation_watercontent(
                water_content=0.1,
                voidratio=0.7)['saturation [-]'],
            0.379, 3
        )

    def test_bulkunitweight(self):
        self.assertAlmostEqual(
            phaserelations.bulkunitweight(
                saturation=0.8,
                voidratio=0.4,
            )['bulk unit weight [kN/m3]'], 21.214, 3
        )
        self.assertAlmostEqual(
            phaserelations.bulkunitweight(
                saturation=0.8,
                voidratio=0.4,
            )['effective unit weight [kN/m3]'], 11.214, 3
        )

    def test_dryunitweight_watercontent(self):
        self.assertAlmostEqual(
            phaserelations.dryunitweight_watercontent(
                watercontent=0.9, bulkunitweight=18
            )['dry unit weight [kN/m3]'],
            9.474, 3
        )

    def test_bulkunitweight_dryunitweight(self):
        self.assertAlmostEqual(
            phaserelations.bulkunitweight_dryunitweight(
                dryunitweight=9.474,
                watercontent=0.9
            )['bulk unit weight [kN/m3]'], 18, 2
        )
        self.assertAlmostEqual(
            phaserelations.bulkunitweight_dryunitweight(
                dryunitweight=9.474,
                watercontent=0.9
            )['effective unit weight [kN/m3]'], 8, 2
        )

    def test_relative_density(self):
        self.assertAlmostEqual(
            phaserelations.relative_density(
                void_ratio=1.2,
                e_min=0.8,
                e_max=2
            )['Dr [-]'], 0.3333, 4
        )

    def test_voidratio_bulkunitweight(self):
        self.assertAlmostEqual(
            phaserelations.voidratio_bulkunitweight(
                bulkunitweight=18
            )['e [-]'], 1.0625, 4
        )
        self.assertAlmostEqual(
            phaserelations.voidratio_bulkunitweight(
                bulkunitweight=18
            )['w [-]'], 0.401, 3
        )

    def test_unitweight_watercontent_saturated(self):
        self.assertAlmostEqual(
            phaserelations.unitweight_watercontent_saturated(water_content=0.4)['gamma [kN/m3]'],
            18.01, 2
        )

    def test_density_unitweight(self):
        self.assertAlmostEqual(
            phaserelations.density_unitweight(
                gamma=20, g=10)['Density [kg/m3]'],
                2000, 1)
        
    def test_unitweight_density(self):
        self.assertAlmostEqual(
            phaserelations.unitweight_density(
                density=2000, g=10)['Unit weight [kN/m3]'],
                20, 1)
        
    def test_watercontent_voidratio(self):
        self.assertAlmostEqual(
            phaserelations.watercontent_voidratio(
                voidratio=1)['Water content [-]'],
                0.377, 3)
        
    def test_voidratio_watercontent(self):
        self.assertAlmostEqual(
            phaserelations.voidratio_watercontent(
                water_content=0.377)['Void ratio [-]'],
                1, 1)