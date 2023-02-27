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
from groundhog.siteinvestigation.correlations import cohesive, cohesionless, general


class Test_Correlations(unittest.TestCase):

    def test_voidratio_porosity(self):
        self.assertEqual(
            cohesive.compressionindex_watercontent_koppula(water_content=0.4)['Cc [-]'],
            0.4
        )
        self.assertAlmostEqual(
            cohesive.compressionindex_watercontent_koppula(water_content=0.4)['Cr [-]'],
            0.053, 3
        )

    def test_gmax_sand_hardinblack(self):
        self.assertAlmostEqual(
            cohesionless.gmax_sand_hardinblack(sigma_m0=100, void_ratio=0.3)['Gmax [kPa]'],
            241047, 0
        )

    def test_permeability_d10_hazen(self):
        self.assertAlmostEqual(
            cohesionless.permeability_d10_hazen(0.1)['k [m/s]'],
            1e-4, 5
        )

    def test_frictionangle_plasticityindex(self):
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(50)['Effective friction angle [deg]'],
            25.34,
            2
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(100)['Effective friction angle [deg]'],
            21.4,
            1
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(1000)['Effective friction angle [deg]'],
            1.7,
            1
        )
        self.assertAlmostEqual(
            cohesive.frictionangle_plasticityindex(200)['Effective friction angle [deg]'],
            11.5,
            1
        )

    def test_cv_liquidlimit_usnavy(self):
        self.assertAlmostEqual(
            cohesive.cv_liquidlimit_usnavy(liquid_limit=60)['cv [m2/yr]'],
            3.2,
            1
        )
        self.assertAlmostEqual(
            cohesive.cv_liquidlimit_usnavy(liquid_limit=80, trend='Remoulded')['cv [m2/yr]'],
            0.5,
            1
        )
        self.assertAlmostEqual(
            cohesive.cv_liquidlimit_usnavy(liquid_limit=80, trend='OC')['cv [m2/yr]'],
            3.2,
            1
        )

    def test_hssmall_parameters_sand(self):
        result = cohesionless.hssmall_parameters_sand(
            relative_density=50
        )
        self.assertAlmostEqual(
            result['E50_ref [kPa]'], 30000, 0
        )
        self.assertAlmostEqual(
            result['phi_eff [deg]'], 34.3, 1
        )
        self.assertAlmostEqual(
            result['gamma_07 [-]'], 0.000150, 3
        )

    def test_gmax_plasticityocr_andersen(self):
        result = cohesive.gmax_plasticityocr_andersen(pi=50, ocr=1, sigma_vo_eff=100)
        self.assertAlmostEqual(
            result['Gmax [kPa]'], 17151, 0
        )
        result = cohesive.gmax_plasticityocr_andersen(pi=100, ocr=1, sigma_vo_eff=100)
        self.assertAlmostEqual(
            result['Gmax [kPa]'], 10282, 0
        )
        result = cohesive.gmax_plasticityocr_andersen(pi=50, ocr=10, sigma_vo_eff=100)
        self.assertAlmostEqual(
            result['Gmax [kPa]'], 54236, 0
        )

    def test_acousticimpedance_bulkunitweight_chen(self):
        result = general.acousticimpedance_bulkunitweight_chen(
            bulkunitweight=18.5)
        self.assertAlmostEqual(result['I [(m/s).(g/cm3)]'], 3023, 0)
        result = general.acousticimpedance_bulkunitweight_chen(
            bulkunitweight=13.0)
        self.assertAlmostEqual(result['I [(m/s).(g/cm3)]'], 1891, 0)
        result = general.acousticimpedance_bulkunitweight_chen(
            bulkunitweight=21.0)
        self.assertAlmostEqual(result['I [(m/s).(g/cm3)]'], 3868, 0)

    def test_shearwavevelocity_compressionindex_cha(self):
        result = general.shearwavevelocity_compressionindex_cha(
            Cc=0.05, sigma_eff_particle_motion=100, sigma_eff_wave_propagation=100)
        self.assertAlmostEqual(result['alpha [-]'], 89, 0)
        self.assertAlmostEqual(result['beta [-]'], 0.21, 2)
        self.assertAlmostEqual(result['Vs [m/s]'], 233, 0)

    def test_stress_dilatancy_bolton(self):
        result_triaxialstrain = cohesionless.stress_dilatancy_bolton(
            relative_density=0.6, p_eff=200)
        self.assertAlmostEqual(result_triaxialstrain['Ir [-]'], 1.82, 2)
        self.assertAlmostEqual(result_triaxialstrain['Dilation angle [deg]'], 5.46, 2)
        result_triaxialstrain_highstress = cohesionless.stress_dilatancy_bolton(
            relative_density=0.6, p_eff=400)
        self.assertAlmostEqual(result_triaxialstrain_highstress['Ir [-]'], 1.41, 2)
        self.assertAlmostEqual(result_triaxialstrain_highstress['Dilation angle [deg]'], 4.22, 2)
        result_planestrain = cohesionless.stress_dilatancy_bolton(
            relative_density=0.6, p_eff=200, stress_condition='plane strain')
        self.assertAlmostEqual(result_planestrain['Ir [-]'], 1.82, 2)
        self.assertAlmostEqual(result_planestrain['Dilation angle [deg]'], 11.38, 2)

    def test_k0_frictionangle_mesri(self):
        result = general.k0_frictionangle_mesri(
            phi_cs=32)
        self.assertAlmostEqual(result['K0 [-]'], 0.47, 2)
        result_ocr = general.k0_frictionangle_mesri(
            phi_cs=32, ocr=10)
        self.assertAlmostEqual(result_ocr['K0 [-]'], 1.59, 2)
        
    def test_k0_plasticity_kenney(self):
        result = cohesive.k0_plasticity_kenney(
            pi=30, fail_silently=False)
        self.assertAlmostEqual(result['K0 [-]'], 0.53, 2)
        result_ocr = cohesive.k0_plasticity_kenney(
            pi=30, ocr=10)
        self.assertAlmostEqual(result_ocr['K0 [-]'], 1.41, 2)
        
    