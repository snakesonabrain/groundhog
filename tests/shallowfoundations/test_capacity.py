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
from groundhog.shallowfoundations.capacity import ShallowFoundationCapacityUndrained, \
    ShallowFoundationCapacityDrained, failuremechanism_prandtl


class Test_UndrainedCapacity(unittest.TestCase):

    def setUp(self):
        self.rectangle_analysis = ShallowFoundationCapacityUndrained(title="Undrained test rectangle")
        self.circle_analysis = ShallowFoundationCapacityUndrained(title="Undrained test circle")
        self.rectangle_analysis.set_geometry(length=5, width=5)
        self.circle_analysis.set_geometry(option='circle', diameter=5)
        self.rectangle_analysis.set_soilparameters_undrained(unit_weight=17, su_base=10)
        self.circle_analysis.set_soilparameters_undrained(unit_weight=17, su_base=10)

    def test_rectangle_noneccentric(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0, eccentricity_length=0)
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            1.18 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            1.18 * 5.14 * 10 * 5 * 5, 3
        )

    def test_circle_noneccentric(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0)
        self.circle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            1.18 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            1.18 * 5.14 * 10 * 0.25 * np.pi * 25, 3
        )

    def test_rectangle_eccentric(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0.5, eccentricity_length=0.5)
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            1.18 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            1.18 * 5.14 * 10 * 4 * 4, 3
        )

    def test_circle_eccentric(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0.5)
        self.circle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            1.14696938 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            1.14696938 * 5.14 * 10 * 3.460747347564257 * 4.2385325651113686, 3
        )

    def test_rectangle_deep(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0, eccentricity_length=0)
        self.rectangle_analysis.depth = 1
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            1.23921 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            (1.23921 * 5.14 * 10 + 17) * 5 * 5, 1
        )

    def test_circle_deep(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0)
        self.circle_analysis.depth = 1
        self.circle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            1.2465872851440176 * 5.14 * 10, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            (1.2465872851440176 * 5.14 * 10 + 17) * 0.25 * np.pi * 25 , 3
        )

    def test_sliding_rectangle(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=1)
        self.rectangle_analysis.calculate_sliding_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_base_only,
            10 * 5 * 5, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_full,
            10 * 5 * 5, 3
        )

    def test_sliding_circle(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0.5)
        self.circle_analysis.calculate_sliding_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.sliding_base_only,
            10 * 0.25 * np.pi * 5 ** 2, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.sliding_full,
            10 * 0.25 * np.pi * 5 ** 2, 3
        )

    def test_sliding_rectangle_deep(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=1)
        self.rectangle_analysis.su_above_base = 10
        self.rectangle_analysis.depth = 1
        self.rectangle_analysis.calculate_sliding_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_base_only,
            10 * 5 * 5, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_full,
            10 * 5 * 5 + 4 * 5 * 10, 3
        )

    def test_sliding_circle_deep(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0.5)
        self.circle_analysis.su_above_base = 10
        self.circle_analysis.depth = 1
        self.circle_analysis.calculate_sliding_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.sliding_base_only,
            10 * 0.25 * np.pi * 5 ** 2, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.sliding_full,
            10 * 0.25 * np.pi * 5 ** 2 + 4 * 5 * 10, 3
        )

    def test_envelope_rectangle(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=0)
        self.rectangle_analysis.calculate_sliding_capacity()
        self.rectangle_analysis.calculate_bearing_capacity()
        self.rectangle_analysis.calculate_envelope()
        self.assertAlmostEqual(
            np.array(self.rectangle_analysis.envelope_V_unfactored).max(),
            self.rectangle_analysis.ultimate_capacity,
            3
        )
        self.assertAlmostEqual(
            np.array(self.rectangle_analysis.envelope_H_unfactored).max(),
            self.rectangle_analysis.sliding_full,
            3
        )
        self.assertAlmostEqual(
            np.array(self.rectangle_analysis.envelope_V_factored).max(),
            self.rectangle_analysis.ultimate_capacity / 2,
            3
        )
        self.assertAlmostEqual(
            np.array(self.rectangle_analysis.envelope_H_factored).max(),
            self.rectangle_analysis.sliding_full / 1.5,
            3
        )

    def test_envelope_rectangle_embedded(self):
        self.rectangle_analysis.depth = 1
        self.rectangle_analysis.skirted = True
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=0)
        self.rectangle_analysis.calculate_sliding_capacity()
        self.rectangle_analysis.calculate_bearing_capacity()
        self.rectangle_analysis.calculate_envelope()


    def test_envelope_circle(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0)
        self.circle_analysis.calculate_sliding_capacity()
        self.circle_analysis.calculate_bearing_capacity()
        self.circle_analysis.calculate_envelope()
        self.assertAlmostEqual(
            np.array(self.circle_analysis.envelope_V_unfactored).max(),
            self.circle_analysis.ultimate_capacity,
            3
        )
        self.assertAlmostEqual(
            np.array(self.circle_analysis.envelope_H_unfactored).max(),
            self.circle_analysis.sliding_full,
            3
        )
        self.assertAlmostEqual(
            np.array(self.circle_analysis.envelope_V_factored).max(),
            self.circle_analysis.ultimate_capacity / 2,
            3
        )
        self.assertAlmostEqual(
            np.array(self.circle_analysis.envelope_H_factored).max(),
            self.circle_analysis.sliding_full / 1.5,
            3
        )


class Test_DrainedCapacity(unittest.TestCase):

    def setUp(self):
        self.rectangle_analysis = ShallowFoundationCapacityDrained(title="Drained test rectangle")
        self.circle_analysis = ShallowFoundationCapacityDrained(title="Drained test circle")
        self.rectangle_analysis.set_geometry(length=5, width=5)
        self.circle_analysis.set_geometry(option='circle', diameter=5)
        self.rectangle_analysis.set_soilparameters_drained(
            effective_unit_weight=9, friction_angle=38, effective_stress_base=0
        )
        self.circle_analysis.set_soilparameters_drained(
            effective_unit_weight=9, friction_angle=38, effective_stress_base=0
        )

    def test_rectangle_noneccentric(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0, eccentricity_length=0)
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            0.5 * 5 * 9 * 56.17434205174994 * 0.6, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            0.5 * 5 * 9 * 56.17434205174994 * 0.6 * 5 * 5, 3
        )

    def test_circle_noneccentric(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0)
        self.circle_analysis.calculate_bearing_capacity()

        # Effective area from effective length and effective width needs to
        # equal area of circle for non-eccentric case
        self.assertAlmostEqual(
            self.circle_analysis.effective_width * self.circle_analysis.effective_length,
            0.25 * np.pi * 5 ** 2, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            0.5 * 4.43113462726379 * 9 * 56.17434205174994 * 0.6, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            0.5 * 4.43113462726379 * 9 * 56.17434205174994 * 0.6 * 0.25 * np.pi * 25, 3
        )


    def test_rectangle_eccentric(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0.5, eccentricity_length=0.5)
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            0.5 * 4 * 9 * 56.17434205174994 * 0.6, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            0.5 * 4 * 9 * 56.17434205174994 * 0.6 * 4 * 4, 3
        )

    def test_circle_eccentric(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0.5)
        self.circle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.capacity['s_gamma [-]'],
            1 - 0.4 * (self.circle_analysis.effective_width / self.circle_analysis.effective_length),
            3
        )
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            0.5 * 3.460747347564257 * 9 * 56.17434205174994 * 0.6734013676289096, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            0.5 * 3.460747347564257 * 9 * 56.17434205174994 * 0.6734013676289096
            * 3.460747347564257 * 4.2385325651113686, 3
        )

    def test_rectangle_deep(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_width=0, eccentricity_length=0)
        self.rectangle_analysis.depth = 1
        self.rectangle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.rectangle_analysis.capacity['d_gamma [-]'],
            1, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.net_bearing_pressure,
            0.5 * 5 * 9 * 56.17434205174994 * 0.6, 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            0.5 * 5 * 9 * 56.17434205174994 * 0.6 * 5 * 5, 3
        )

    def test_circle_deep(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=0)
        self.circle_analysis.depth = 1
        self.circle_analysis.calculate_bearing_capacity()
        self.assertAlmostEqual(
            self.circle_analysis.net_bearing_pressure,
            0.5 * 4.43113462726379 * 9 * 56.17434205174994 * 0.6, 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.ultimate_capacity,
            0.5 * 4.43113462726379 * 9 * 56.17434205174994 * 0.6 * 0.25 * np.pi * 25, 3
        )

    def test_sliding_rectangle(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=1)
        self.rectangle_analysis.calculate_sliding_capacity(vertical_load=100)
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_base_only,
            100 * np.tan(np.radians(38 - 5)), 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_full,
            100 * np.tan(np.radians(38 - 5)), 3
        )

    def test_sliding_circle(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=1)
        self.circle_analysis.calculate_sliding_capacity(vertical_load=100)
        self.assertAlmostEqual(
            self.circle_analysis.sliding_base_only,
            100 * np.tan(np.radians(38 - 5)), 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.sliding_full,
            100 * np.tan(np.radians(38 - 5)), 3
        )

    def test_sliding_rectangle_deep(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=1)
        self.rectangle_analysis.depth = 1
        self.rectangle_analysis.calculate_sliding_capacity(vertical_load=100)
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_base_only,
            100 * np.tan(np.radians(38 - 5)), 3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.sliding_full,
            100 * np.tan(np.radians(38 - 5)) + 0.5 * 3.097319104870605 * 9 * 1 * 5, 3
        )

    def test_sliding_circle_deep(self):
        self.circle_analysis.set_eccentricity(eccentricity_width=1)
        self.circle_analysis.depth = 1
        self.circle_analysis.calculate_sliding_capacity(vertical_load=100)
        print(self.circle_analysis.sliding)
        self.assertAlmostEqual(
            self.circle_analysis.sliding_base_only,
            100 * np.tan(np.radians(38 - 5)), 3
        )
        self.assertAlmostEqual(
            self.circle_analysis.sliding_full,
            100 * np.tan(np.radians(38 - 5)) + 0.5 * 3.097319104870605 * 9 * 1 * 5, 3
        )

    def test_envelope_rectangle(self):
        self.rectangle_analysis.set_eccentricity(eccentricity_length=0, eccentricity_width=0)
        self.rectangle_analysis.calculate_bearing_capacity()
        self.rectangle_analysis.calculate_sliding_capacity(vertical_load=100)
        self.rectangle_analysis.calculate_envelope()
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity,
            self.rectangle_analysis.envelope_V_unfactored.max(),
            3
        )
        self.assertAlmostEqual(
            self.rectangle_analysis.ultimate_capacity / 2,
            self.rectangle_analysis.envelope_V_factored.max(),
            3
        )

    def test_failuremechanism_prandtl(self):
        result = failuremechanism_prandtl(
            friction_angle=0, width=5)

        self.assertAlmostEqual(result['X [m]'].max(), 1.5 * 5, 4)