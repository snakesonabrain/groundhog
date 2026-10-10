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
            liquid_limit=25.0, plastic_limit=24.0)
        # PI = 1 < 4 -> ML fines -> SM (PI = 5 would be CL-ML fines -> SC-SM)
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


def _classify(**kwargs):
    """Classifies with a default sample (clean gravel, sand-free) with selected inputs overridden"""
    params = dict(
        percent_passing_200=2.0, percent_passing_4=40.0,
        d60=4.0, d30=2.0, d10=1.0,
        liquid_limit=40.0, plastic_limit=20.0)
    params.update(kwargs)
    return uscs_classify.uscs_classify(**params)


def _fine(liquid_limit, plastic_limit, **kwargs):
    """Classifies a fine-grained soil (70 % fines) with given Atterberg limits"""
    return _classify(
        percent_passing_200=70.0, percent_passing_4=100.0,
        liquid_limit=liquid_limit, plastic_limit=plastic_limit, **kwargs)


class Test_USCSClassifyBoundaries(unittest.TestCase):
    """Boundary tests: every limit of ASTM D2487-17 is tested on the limit and just beside it"""

    # --- Coarse / fine split (ASTM D2487 Section 1.1) ---

    def test_fines_exactly_50(self):
        # "50 % or more passes the No. 200 sieve" is fine-grained
        result = _classify(percent_passing_200=50.0, percent_passing_4=100.0, liquid_limit=35.0, plastic_limit=20.0)
        self.assertTrue(result['Fine grained'])
        self.assertEqual(result['Group symbol'], 'CL')

    def test_fines_just_below_50(self):
        # More than 50 % retained on the No. 200 sieve -> coarse-grained; > 12 % fines -> SC (PI 15 above A-line)
        result = _classify(percent_passing_200=49.9, percent_passing_4=100.0, liquid_limit=35.0, plastic_limit=20.0)
        self.assertFalse(result['Fine grained'])
        self.assertEqual(result['Group symbol'], 'SC')

    # --- Gravel / sand split on the coarse fraction (Table 1) ---

    def test_gravel_sand_exactly_half_of_coarse_is_sand(self):
        # 98 % coarse, 49 % retained on No. 4 = exactly 50 % of the coarse fraction: gravel needs MORE than 50 %
        result = _classify(percent_passing_200=2.0, percent_passing_4=51.0)
        self.assertFalse(result['Gravel governs'])
        self.assertEqual(result['Group symbol'][0], 'S')

    def test_gravel_sand_just_over_half_of_coarse_is_gravel(self):
        # 50 % of the total retained on No. 4 = 51 % of the 98 % coarse fraction -> gravel
        result = _classify(percent_passing_200=2.0, percent_passing_4=50.0)
        self.assertTrue(result['Gravel governs'])
        self.assertEqual(result['Group symbol'][0], 'G')

    def test_gravel_sand_split_uses_coarse_fraction_not_total(self):
        # 40 % of the total retained on No. 4 is only 44 % of a 90 % coarse fraction (not > 50 %) -> sand
        result = _classify(percent_passing_200=10.0, percent_passing_4=60.0, d60=0.6, d30=0.25, d10=0.1)
        self.assertEqual(result['Group symbol'][0], 'S')

    # --- Fines content bands: < 5 clean, 5-12 dual, > 12 fines govern (Table 1) ---

    def test_fines_just_below_5_is_clean(self):
        # Less than 5 % fines -> single grading symbol
        self.assertEqual(_classify(percent_passing_200=4.99)['Group symbol'], 'GW')

    def test_fines_exactly_5_is_dual(self):
        # 5 to 12 % fines are borderline -> dual symbol; PI 20 is above the A-line (14.6) -> GC
        self.assertEqual(_classify(percent_passing_200=5.0)['Group symbol'], 'GW-GC')

    def test_fines_exactly_12_is_dual(self):
        # 12 % is still inside "5 to 12 %" -> dual symbol
        self.assertEqual(_classify(percent_passing_200=12.0)['Group symbol'], 'GW-GC')

    def test_fines_just_above_12_fines_govern(self):
        # More than 12 % fines -> grading is dropped, clay fines -> GC
        self.assertEqual(_classify(percent_passing_200=12.01)['Group symbol'], 'GC')

    def test_all_dual_symbols(self):
        silty = dict(liquid_limit=30.0, plastic_limit=28.0)    # PI 2 < 4 -> ML fines
        clayey = dict(liquid_limit=40.0, plastic_limit=20.0)   # PI 20 above A-line -> CL fines
        poor_gravel = dict(d60=4.0, d30=1.0, d10=1.0)          # Cc = 0.25 < 1 -> poorly graded
        # nothing retained on the No. 4 sieve -> sand; Cu 5 < 6 -> poorly graded
        poor_sand = dict(d60=0.5, d30=0.2, d10=0.1, percent_passing_4=100.0)
        well_sand = dict(d60=0.6, d30=0.25, d10=0.08, percent_passing_4=100.0)  # Cu 7.5, Cc 1.3 -> well graded
        cases = [
            (dict(**silty), 'GW-GM'), (dict(**clayey), 'GW-GC'),
            (dict(**silty, **poor_gravel), 'GP-GM'), (dict(**clayey, **poor_gravel), 'GP-GC'),
            (dict(**silty, **well_sand), 'SW-SM'), (dict(**clayey, **well_sand), 'SW-SC'),
            (dict(**silty, **poor_sand), 'SP-SM'), (dict(**clayey, **poor_sand), 'SP-SC'),
        ]
        for kwargs, expected in cases:
            kwargs = dict(kwargs)
            result = _classify(percent_passing_200=8.0, **kwargs)
            self.assertEqual(result['Group symbol'], expected, msg=str(kwargs))

    def test_dual_symbol_cl_ml_fines_above_12(self):
        # Table 1 note: fines in the CL-ML zone give the dual symbols GC-GM / SC-SM when > 12 % fines
        gravel = _classify(percent_passing_200=20.0, liquid_limit=25.0, plastic_limit=20.0)
        self.assertEqual(gravel['Group symbol'], 'GC-GM')
        sand = _classify(percent_passing_200=20.0, percent_passing_4=100.0, liquid_limit=25.0, plastic_limit=20.0)
        self.assertEqual(sand['Group symbol'], 'SC-SM')

    # --- Gradation criteria (Table 1) ---

    def test_gravel_cu_exactly_4(self):
        # GW needs Cu >= 4: D60 / D10 = 4 exactly, Cc = 1 -> GW
        self.assertEqual(_classify(d60=4.0, d30=2.0, d10=1.0)['Group symbol'], 'GW')

    def test_gravel_cu_just_below_4(self):
        # Cu = 3.9 < 4 -> not well graded (Cc = 1.0 here) -> GP
        self.assertEqual(_classify(d60=3.9, d30=1.97, d10=1.0)['Group symbol'], 'GP')

    def test_sand_cu_exactly_6_floating_point(self):
        # SW needs Cu >= 6; 0.6 / 0.1 = 5.999999999999999 in floating point but is exactly 6 -> SW (Cc = 1.04)
        result = _classify(percent_passing_4=100.0, d60=0.6, d30=0.25, d10=0.1)
        self.assertEqual(result['Group symbol'], 'SW')

    def test_sand_cu_4_is_poorly_graded(self):
        # Cu = 4 satisfies the gravel criterion (>= 4) but not the sand criterion (>= 6) -> SP
        result = _classify(percent_passing_4=100.0, d60=0.4, d30=0.2, d10=0.1)
        self.assertEqual(result['Group symbol'], 'SP')

    def test_sand_cu_just_below_6(self):
        # Cu = 5.9 < 6 -> SP (Cc = 1.1)
        result = _classify(percent_passing_4=100.0, d60=0.59, d30=0.253, d10=0.1)
        self.assertEqual(result['Group symbol'], 'SP')

    def test_gravel_cu_5_is_well_graded(self):
        # Cu = 5 >= 4 for gravel (but < 6 as for sand), Cc = 1.28 -> GW
        self.assertEqual(_classify(d60=5.0, d30=2.5, d10=1.0)['Group symbol'], 'GW')

    def test_cc_exactly_1(self):
        # Cc = 2 x 2 / (4 x 1) = 1 is inside 1 <= Cc <= 3 -> GW
        result = _classify(d60=4.0, d30=2.0, d10=1.0)
        self.assertAlmostEqual(result['Coefficient of curvature C_c'], 1.0)
        self.assertEqual(result['Group symbol'], 'GW')

    def test_cc_just_below_1(self):
        # Cc = 0.99 < 1 -> GP
        self.assertEqual(_classify(d60=4.0, d30=1.99, d10=1.0)['Group symbol'], 'GP')

    def test_cc_exactly_3_floating_point(self):
        # Cc = d30^2 / (d60 d10) = 12 / 4 = 3 (sqrt(12) is inexact in floating point) is inside 1 <= Cc <= 3 -> GW
        result = _classify(d60=4.0, d30=np.sqrt(12.0), d10=1.0)
        self.assertEqual(result['Group symbol'], 'GW')

    def test_cc_just_above_3(self):
        # Cc = 3.01 > 3 -> GP
        self.assertEqual(_classify(d60=4.0, d30=np.sqrt(12.04), d10=1.0)['Group symbol'], 'GP')

    # --- Plasticity chart: A-line (Fig. 4) ---

    def test_on_aline_is_clay(self):
        # LL 40: A-line PI = 0.73 x 20 = 14.6. "On or above the A-line" is clay -> CL
        result = _fine(40.0, 25.4)
        self.assertEqual(result['Group symbol'], 'CL')
        self.assertTrue(result['Above A-line'])

    def test_just_below_aline_is_silt(self):
        # PI 14.5 < 14.6 -> below A-line -> ML
        self.assertEqual(_fine(40.0, 25.5)['Group symbol'], 'ML')

    def test_on_aline_high_plasticity(self):
        # LL 60: A-line PI = 29.2; on the line -> CH
        self.assertEqual(_fine(60.0, 30.8)['Group symbol'], 'CH')

    def test_below_aline_high_plasticity(self):
        # LL 60, PI 29 < 29.2 -> MH
        self.assertEqual(_fine(60.0, 31.0)['Group symbol'], 'MH')

    def test_aline_horizontal_part_below_ll_25_5(self):
        # For LL < 25.5 the A-line is horizontal at PI = 4 (Fig. 4): LL 22, PI 4 is on the line, PI 3 below it
        self.assertTrue(_fine(22.0, 18.0)['Above A-line'])
        self.assertFalse(_fine(22.0, 19.0)['Above A-line'])

    # --- Plasticity chart: CL-ML hatched zone (Fig. 4) ---

    def test_cl_ml_pi_4(self):
        # LL 25: A-line = 4 (horizontal part); PI = 4 on the line, in the 4-7 hatched zone -> CL-ML
        self.assertEqual(_fine(25.0, 21.0)['Group symbol'], 'CL-ML')

    def test_cl_ml_pi_just_below_4(self):
        # PI 3.9 < 4 -> ML
        self.assertEqual(_fine(25.0, 21.1)['Group symbol'], 'ML')

    def test_cl_ml_pi_7(self):
        # PI = 7 is the top of the hatched zone (CL needs PI > 7) -> CL-ML
        self.assertEqual(_fine(25.0, 18.0)['Group symbol'], 'CL-ML')

    def test_cl_ml_pi_just_above_7(self):
        # PI 7.1 > 7 and above the A-line -> CL
        self.assertEqual(_fine(25.0, 17.9)['Group symbol'], 'CL')

    def test_pi_in_zone_but_below_aline(self):
        # LL 35: A-line PI = 10.95; PI 5 is below the A-line, so not CL-ML -> ML
        self.assertEqual(_fine(35.0, 30.0)['Group symbol'], 'ML')

    def test_cl_ml_on_aline_floating_point(self):
        # LL 28: A-line PI = 5.84; PI 5.84 on the line and within 4-7 -> CL-ML
        self.assertEqual(_fine(28.0, 22.16)['Group symbol'], 'CL-ML')

    # --- LL = 50 boundary (Table 1) ---

    def test_ll_just_below_50_is_low_plasticity(self):
        # LL < 50 is low plasticity: PI 25 above A-line (21.8) -> CL
        self.assertEqual(_fine(49.9, 24.9)['Group symbol'], 'CL')

    def test_ll_exactly_50_is_high_plasticity(self):
        # "LL 50 or more" is high plasticity: PI 25 above the A-line (21.9) -> CH
        self.assertEqual(_fine(50.0, 25.0)['Group symbol'], 'CH')

    def test_ll_50_on_aline(self):
        # LL 50: A-line PI = 21.9; on the line -> CH
        self.assertEqual(_fine(50.0, 28.1)['Group symbol'], 'CH')

    def test_ll_50_silt(self):
        # LL 50 and PI 10 below the A-line -> MH
        self.assertEqual(_fine(50.0, 40.0)['Group symbol'], 'MH')

    def test_ll_just_below_50_silt(self):
        # LL 49.9, PI 10 below the A-line -> ML
        self.assertEqual(_fine(49.9, 39.9)['Group symbol'], 'ML')

    def test_pi_zero_non_plastic(self):
        # PI = 0 (LL = PL) is below the A-line -> ML
        self.assertEqual(_fine(30.0, 30.0)['Group symbol'], 'ML')

    # --- Organic soils (Table 1) ---

    def test_organic_ll_boundary(self):
        # Organic fines: LL < 50 -> OL, LL >= 50 -> OH
        self.assertEqual(_fine(49.9, 30.0, organic=True)['Group symbol'], 'OL')
        self.assertEqual(_fine(50.0, 30.0, organic=True)['Group symbol'], 'OH')

    def test_organic_clay_or_silt_name(self):
        # Organic clay plots on/above the A-line with PI >= 4; organic silt plots below it
        self.assertEqual(_fine(40.0, 20.0, organic=True)['Group name'], 'Organic clay')
        self.assertEqual(_fine(40.0, 30.0, organic=True)['Group name'], 'Organic silt')

    # --- U-line (Fig. 4) ---

    def test_on_uline_no_warning(self):
        # LL 40: U-line PI = 0.9 x 32 = 28.8; a sample on the U-line is the upper limit and still valid
        self.assertIsNone(_fine(40.0, 11.2)['Data quality warning'])

    def test_above_uline_warning(self):
        # PI 29 > 28.8 plots above the U-line, which natural soils do not
        self.assertIn('U-line', _fine(40.0, 11.0)['Data quality warning'])

    # --- Invalid inputs ---

    def test_negative_percentage(self):
        with self.assertWarns(UserWarning):
            result = _classify(percent_passing_200=-1.0)
        self.assertIsNone(result['Group symbol'])

    def test_percentage_above_100(self):
        with self.assertWarns(UserWarning):
            result = _classify(percent_passing_4=100.1)
        self.assertIsNone(result['Group symbol'])

    def test_negative_limits_and_diameters(self):
        for key in ('liquid_limit', 'plastic_limit', 'd60', 'd30', 'd10'):
            with self.assertWarns(UserWarning):
                result = _classify(**{key: -1.0})
            self.assertIsNone(result['Group symbol'], msg=key)

    def test_missing_diameters_for_clean_soil(self):
        # Cu / Cc cannot be computed -> no silent "poorly graded"; None and NaN are both rejected
        for key in ('d60', 'd30', 'd10'):
            for missing in (None, np.nan, 0.0):
                with self.assertWarns(UserWarning):
                    result = _classify(**{key: missing})
                self.assertIsNone(result['Group symbol'], msg="%s=%s" % (key, missing))

    def test_missing_diameters_for_dual_symbol_soil(self):
        # 5-12 % fines also need grading
        with self.assertWarns(UserWarning):
            result = _classify(percent_passing_200=8.0, d10=np.nan)
        self.assertIsNone(result['Group symbol'])

    def test_diameters_not_needed_above_12_percent_fines(self):
        # With > 12 % fines the grading is not used, so missing diameters are acceptable
        result = _classify(percent_passing_200=20.0, d60=np.nan, d30=np.nan, d10=np.nan)
        self.assertEqual(result['Group symbol'], 'GC')

    def test_diameters_not_needed_for_fine_grained(self):
        result = _fine(35.0, 20.0, d60=np.nan, d30=np.nan, d10=np.nan)
        self.assertEqual(result['Group symbol'], 'CL')

    def test_diameters_not_in_order(self):
        # D10 <= D30 <= D60 must hold for a grain size curve
        with self.assertWarns(UserWarning):
            result = _classify(d60=1.0, d30=2.0, d10=4.0)
        self.assertIsNone(result['Group symbol'])

    def test_missing_limits_for_fine_grained(self):
        with self.assertWarns(UserWarning):
            result = _fine(np.nan, 20.0)
        self.assertIsNone(result['Group symbol'])

    def test_plastic_limit_above_liquid_limit(self):
        with self.assertWarns(UserWarning):
            result = _fine(30.0, 35.0)
        self.assertIsNone(result['Group symbol'])

    def test_passing_4_below_passing_200(self):
        # Cannot pass the No. 4 sieve less than the No. 200 sieve
        with self.assertWarns(UserWarning):
            result = _classify(percent_passing_200=30.0, percent_passing_4=20.0)
        self.assertIsNone(result['Group symbol'])

    def test_clean_soil_does_not_need_limits(self):
        # Plasticity is not used below 5 % fines
        result = _classify(liquid_limit=np.nan, plastic_limit=np.nan)
        self.assertEqual(result['Group symbol'], 'GW')


if __name__ == "__main__":
    unittest.main()
