__author__ = "Wouter Karreman and Ping Li"

import unittest

# Project imports
from groundhog.deepfoundations.lateralresponse import lateral


class Test_lateral(unittest.TestCase):

    def test_reinforcement_inertia(self):
        result = lateral.reinforced_circularsection_inertia(diameter=0.42, modulus_ratio=7, n_bars=3, offset=0.5, rebar_diameter=0.016, fail_silently=False)
        self.assertAlmostEqual(result["Start angle [deg]"], 30, 1)
        self.assertAlmostEqual(result["Rebar inertia [m4]"], 0.00053, 5)
        result_1 = lateral.reinforced_circularsection_inertia(diameter=0.42, modulus_ratio=7, n_bars=2, offset=0.5, rebar_diameter=0.016, maximum_resistance=True)
        result_2 = lateral.reinforced_circularsection_inertia(diameter=0.42, modulus_ratio=7, n_bars=2, offset=0.5, rebar_diameter=0.016, maximum_resistance=False)
        self.assertAlmostEqual(result_2["Start angle [deg]"], 0, 1)
        self.assertGreaterEqual(result_1["Rebar inertia [m4]"], result_2["Rebar inertia [m4]"])

    def test_pilegroupeffect_reesevanimpe(self):
        result = lateral.pilegroupeffect_reesevanimpe(
            pile_x=[0, 1.58, 0, 1.58, 0, 1.58, 0, 1.58],
            pile_y=[0, 0, 1.08, 1.08, 2.16, 2.16, 3.18, 3.18],
            pile_diameters=[0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
            load_x=2, load_y=2, show_fig=False
        )
        self.assertAlmostEqual(result['efficiencies'][0], 0.404, 3)
        self.assertAlmostEqual(result['efficiencies'][-1], 0.707, 3)
        