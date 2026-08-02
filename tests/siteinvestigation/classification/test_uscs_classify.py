#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only

__author__ = 'Heston Norcott'

# Native Python packages
import unittest

# 3rd party packages
import numpy as np

# Project imports
from groundhog.siteinvestigation.classification import uscs_classify


class Test_USCSClassify(unittest.TestCase):

    def test_fine_grained_lean_clay(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=35.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'CL')
        self.assertEqual(result['Group name'], 'Lean clay')
        self.assertTrue(result['Fine grained'])
        self.assertAlmostEqual(result['Plasticity index'], 15.0)

    def test_fine_grained_fat_clay(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=70.0, plastic_limit=30.0)
        self.assertEqual(result['Group symbol'], 'CH')
        self.assertEqual(result['Group name'], 'Fat clay')
        self.assertTrue(result['High plasticity'])

    def test_fine_grained_silt(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=30.0, plastic_limit=25.0)
        self.assertEqual(result['Group symbol'], 'ML')
        self.assertEqual(result['Group name'], 'Silt')

    def test_fine_grained_elastic_silt(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=50.0, plastic_limit=35.0)
        self.assertEqual(result['Group symbol'], 'MH')
        self.assertEqual(result['Group name'], 'Elastic silt')

    def test_fine_grained_borderline_cl_ml(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=25.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'CL-ML')
        self.assertEqual(result['Group name'], 'Silty clay')

    def test_fine_grained_organic(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=35.0, plastic_limit=20.0, organic=True)
        self.assertEqual(result['Group symbol'], 'OL')
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=70.0, plastic_limit=30.0, organic=True)
        self.assertEqual(result['Group symbol'], 'OH')

    def test_above_uline_warning(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=60.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=35.0, plastic_limit=1.0)
        self.assertIsNotNone(result['Data quality warning'])
        self.assertIn('U-line', result['Data quality warning'])

    def test_coarse_sand_clean_poorly_graded(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=3.0,
            percent_passing_4=80.0,
            d60=0.5, d30=0.3, d10=0.15,
            liquid_limit=35.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'SP')
        self.assertEqual(result['Group name'], 'Poorly graded sand')
        self.assertFalse(result['Fine grained'])
        self.assertFalse(result['Gravel governs'])

    def test_coarse_gravel_clean_well_graded(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=4.0,
            percent_passing_4=35.0,
            d60=4.0, d30=2.0, d10=1.0,
            liquid_limit=35.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'GW')
        self.assertEqual(result['Group name'], 'Well-graded gravel')
        self.assertTrue(result['Gravel governs'])
        self.assertTrue(result['Well graded'])

    def test_coarse_sand_dual_symbol(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=12.0,
            percent_passing_4=70.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=35.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'SW-SC')
        self.assertEqual(result['Group name'], 'Well-graded sand with clay')
        self.assertFalse(result['Gravel governs'])

    def test_coarse_sand_fines_govern_silty(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=25.0,
            percent_passing_4=85.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=25.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'SM')
        self.assertEqual(result['Group name'], 'Silty sand')

    def test_coarse_gravel_fines_govern_clayey(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=30.0,
            percent_passing_4=40.0,
            d60=4.0, d30=2.0, d10=1.0,
            liquid_limit=40.0, plastic_limit=20.0)
        self.assertEqual(result['Group symbol'], 'GC')
        self.assertEqual(result['Group name'], 'Clayey gravel')

    def test_all_fines(self):
        result = uscs_classify.uscs_classify(
            percent_passing_200=100.0,
            percent_passing_4=100.0,
            d60=2.0, d30=0.5, d10=0.1,
            liquid_limit=35.0, plastic_limit=20.0)
        self.assertTrue(result['Fine grained'])

    def test_validation_out_of_range(self):
        with self.assertWarns(UserWarning):
            result = uscs_classify.uscs_classify(
                percent_passing_200=101.0,
                percent_passing_4=70.0,
                d60=2.0, d30=0.5, d10=0.1,
                liquid_limit=35.0, plastic_limit=20.0)
        self.assertIsNone(result['Group symbol'])


if __name__ == "__main__":
    unittest.main()
