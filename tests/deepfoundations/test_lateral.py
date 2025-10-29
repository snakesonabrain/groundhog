__author__ = "Wouter Karreman and Ping Li"

import unittest

# Project imports
from groundhog.deepfoundations.lateralresponse import lateral


class Test_lateral(unittest.TestCase):

    def test_reinforcement_inertia(self):
        result = lateral.reinforced_circularsection_inertia(diameter=0.42, n_bars=3, offset=0.5, rebar_diameter=0.016, fail_silently=False)
        self.assertAlmostEqual(result["Start angle [deg]"], 30, 1)
        self.assertAlmostEqual(result["Rebar inertia [m4]"], 7.54078e-5, 8)
        result_1 = lateral.reinforced_circularsection_inertia(diameter=0.42, n_bars=2, offset=0.5, rebar_diameter=0.016, maximum_resistance=True)
        result_2 = lateral.reinforced_circularsection_inertia(diameter=0.42, n_bars=2, offset=0.5, rebar_diameter=0.016, maximum_resistance=False)
        self.assertAlmostEqual(result_2["Start angle [deg]"], 0, 1)
        self.assertGreaterEqual(result_1["Rebar inertia [m4]"], result_2["Rebar inertia [m4]"])
        