#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import numpy as np

# Project imports
from groundhog.siteinvestigation.insitutests import pcpt_correlations


TESTS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class Test_behaviourindex_pcpt_robertsonwride(unittest.TestCase):

    def test_behaviourindex_pcpt_robertsonwride(self):
        result = pcpt_correlations.behaviourindex_pcpt_robertsonwride(
            qt=40,
            fs=0.1,
            sigma_vo=150,
            sigma_vo_eff=50
        )
        self.assertAlmostEqual(result['Ic [-]'], 1.005, 3)
        result = pcpt_correlations.behaviourindex_pcpt_robertsonwride(
            qt=1,
            fs=0.1,
            sigma_vo=150,
            sigma_vo_eff=50
        )
        self.assertAlmostEqual(result['Ic [-]'], 3.203, 3)


class Test_gmax_sand_rixstokoe(unittest.TestCase):

    def test_gmax_sand_rixstokoe(self):
        result = pcpt_correlations.gmax_sand_rixstokoe(
            qc=15,
            sigma_vo_eff=100
        )
        self.assertAlmostEqual(result['Gmax [kPa]'], 101689.2, 1)


class Test_gmax_clay_maynerix(unittest.TestCase):

    def test_gmax_clay_maynerix(self):
        result = pcpt_correlations.gmax_clay_maynerix(
            qc=1
        )
        self.assertAlmostEqual(result['Gmax [kPa]'], 28121.9, 1)


class Test_relativedensity_ncsand_baldi(unittest.TestCase):

    def test_relativedensity_ncsand_baldi(self):
        result = pcpt_correlations.relativedensity_ncsand_baldi(
            qc=20,
            sigma_vo_eff=200
        )
        self.assertAlmostEqual(result['Dr [-]'], 0.8, 1)


class Test_relativedensity_ocsand_baldi(unittest.TestCase):

    def test_relativedensity_ocsand_baldi(self):
        result = pcpt_correlations.relativedensity_ocsand_baldi(
            qc=20,
            sigma_vo_eff=200,
            k0=1
        )
        self.assertAlmostEqual(result['Dr [-]'], 0.686, 3)


class Test_coneresistance_ocsand_baldi(unittest.TestCase):

    def test_coneresistance_ocsand_baldi(self):
        result = pcpt_correlations.coneresistance_ocsand_baldi(
            dr=0.686,
            sigma_vo_eff=200,
            k0=1
        )
        self.assertAlmostEqual(result['qc [MPa]'], 20, 1)


class Test_relativedensity_sand_jamiolkowski(unittest.TestCase):

    def test_relativedensity_sand_jamiolkowski(self):
        result = pcpt_correlations.relativedensity_sand_jamiolkowski(
            qc=20,
            sigma_vo_eff=200,
            k0=1
        )
        self.assertAlmostEqual(result['Dr dry [-]'], 0.60, 2)
        self.assertAlmostEqual(result['Dr sat [-]'], 0.68, 2)


class Test_friction_angle_kleven(unittest.TestCase):
    def test_values(self):
        self.assertEqual(
            pcpt_correlations.frictionangle_overburden_kleven(
                10.0, 100.0, Ko=1.0, max_friction_angle=50.0)['phi [deg]'], 47.497)
        self.assertEqual(
            pcpt_correlations.frictionangle_overburden_kleven(
                10.0, 100.0, Ko=1.0, )['phi [deg]'], 45.0)

    def test_ranges(self):
        self.assertRaises(
            ValueError,
            pcpt_correlations.frictionangle_overburden_kleven, 1.0, 100.0, Ko=0.6, fail_silently=False)


class Test_ocr_cpt_lunne(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.ocr_cpt_lunne(
                Qt=10)
        self.assertAlmostEqual(result['OCR_Qt_LE [-]'], 2.28, 2)
        self.assertAlmostEqual(result['OCR_Qt_BE [-]'], 3.01, 2)
        self.assertAlmostEqual(result['OCR_Qt_HE [-]'], 4.68, 2)
        self.assertTrue(np.math.isnan(result['OCR_Bq_LE [-]']))

        result = pcpt_correlations.ocr_cpt_lunne(
            Qt=10, Bq=0.6)
        self.assertAlmostEqual(result['OCR_Qt_LE [-]'], 2.28, 2)
        self.assertAlmostEqual(result['OCR_Qt_BE [-]'], 3.01, 2)
        self.assertAlmostEqual(result['OCR_Qt_HE [-]'], 4.68, 2)
        self.assertAlmostEqual(result['OCR_Bq_LE [-]'], 1.41, 2)
        self.assertAlmostEqual(result['OCR_Bq_BE [-]'], 2.07, 2)
        self.assertAlmostEqual(result['OCR_Bq_HE [-]'], 3.09, 2)


class Test_sensitivity_frictionratio_lunne(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.sensitivity_frictionratio_lunne(Rf=1)
        self.assertAlmostEqual(result['St LE [-]'], 5.71, 2)
        self.assertAlmostEqual(result['St BE [-]'], 7.47, 2)
        self.assertAlmostEqual(result['St HE [-]'], 9.63, 2)


class Test_unitweight_mayne(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.unitweight_mayne(ft=0.25, sigma_vo_eff=100)
        self.assertAlmostEqual(result['gamma [kN/m3]'], 1.95 * 10.25 * (2.5 ** 0.06) * (1 ** 0.06), 2)


class Test_vs_ic_robertsoncabal(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.vs_ic_robertsoncabal(qt=10, ic=2.3, sigma_vo=200)
        self.assertAlmostEqual(result['alpha_vs [-]'], 881.05, 2)
        self.assertAlmostEqual(result['Vs [m/s]'], 293.84, 2)


class Test_k0_sand_mayne(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.k0_sand_mayne(qt=10, sigma_vo_eff=100, ocr=1)
        self.assertAlmostEqual(result['K0 CPT [-]'], 0.53, 2)


class Test_gmax_cpt_puechen(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.gmax_cpt_puechen(
            qc=15,
            sigma_vo_eff=100,
            Bq=0
        )
        self.assertAlmostEqual(result['Gmax [kPa]'], 101689.2, 1)
        result = pcpt_correlations.gmax_cpt_puechen(
            qc=15,
            sigma_vo_eff=100,
            Bq=-0.2
        )
        self.assertAlmostEqual(result['Gmax [kPa]'], 101689.2, 1)


class Test_behaviourindex_pcpt_nonnormalised(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.behaviourindex_pcpt_nonnormalised(
            qc=10,
            Rf=1
        )
        self.assertAlmostEqual(result['Isbt [-]'], 1.9, 1)
        self.assertEqual(result['Isbt class number [-]'], 6)


class Test_drainedsecantmodulus_sand_bellotti(unittest.TestCase):

    def test_values(self):
        result = pcpt_correlations.drainedsecantmodulus_sand_bellotti(
            qc=10,
            sigma_vo_eff=100,
            K0=1,
            sandtype='NC'
        )
        self.assertAlmostEqual(result['Es_qc [-]'], 2.3, 1)
        result = pcpt_correlations.drainedsecantmodulus_sand_bellotti(
            qc=10,
            sigma_vo_eff=100,
            K0=1,
            sandtype='Aged NC'
        )
        self.assertAlmostEqual(result['Es_qc [-]'], 4.4, 1)
        result = pcpt_correlations.drainedsecantmodulus_sand_bellotti(
            qc=10,
            sigma_vo_eff=100,
            K0=1,
            sandtype='OC'
        )
        self.assertAlmostEqual(result['Es_qc [-]'], 8.7, 1)