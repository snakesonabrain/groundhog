#!/usr/bin/env python
# -*- coding: utf-8 -*-
from groundhog.shallowfoundations.settlement import primaryconsolidationsettlement_nc

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.general import soilprofile as sp
from groundhog.siteinvestigation.classification.phaserelations import voidratio_bulkunitweight
from groundhog.soildynamics.liquefaction import liquefactionprobability_moss


class Test_SoilProfile(unittest.TestCase):

    def setUp(self):
        self.profile = sp.SoilProfile(
            {
                'Depth from [m]': [0, 1, 5, 10],
                'Depth to [m]': [1, 5, 10, 20],
                'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
            }
        )
        self.profile.title = "Test"

    def test_adjustlayertransition(self):
        self.profile.adjust_layertransition(currentdepth=5, newdepth=6)
        self.assertEqual(
            self.profile.loc[1, "Depth to [m]"], 6
        )
        self.profile.adjust_layertransition(currentdepth=20, newdepth=21)
        self.assertEqual(
            self.profile.loc[3, "Depth to [m]"], 21
        )
        self.profile.adjust_layertransition(currentdepth=0, newdepth=-1)
        self.assertEqual(
            self.profile.loc[0, "Depth from [m]"], -1
        )


    def test_layerthickness(self):
        self.profile.calculate_layerthickness()
        self.assertEqual(
            self.profile.loc[1, "Layer thickness [m]"], 4
        )

    def test_layercenter(self):
        self.profile.calculate_center()
        self.assertEqual(
            self.profile.loc[1, "Depth center [m]"], 3
        )

    def test_wrong_layering(self):
        self.assertRaises(IOError, sp.SoilProfile, ({
            'Depth from [m]': [0, 1, 4, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
        }))

    def test_wrong_depthkey(self):
        self.assertRaises(IOError, sp.SoilProfile, ({
            'z from [m]': [0, 1, 4, 10],
            'z to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND']
        }))

    def test_wrong_linearvariation(self):
        # Test incomplete linear parameter variation
        self.assertRaises(IOError, sp.SoilProfile, ({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40]
        }))
        # Check that complete parameter variation passes
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [1, 3, 10, 40]
        })

    def test_calculate_parameter_center(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [1, 3, 10, 40]
        })
        profile.calculate_parameter_center('qc [MPa]')
        self.assertEqual(profile.loc[2, 'qc center [MPa]'], 10)

    def test_profile_creation(self):
        self.assertEqual(self.profile.max_depth, 20)
        self.assertEqual(self.profile.min_depth, 0)
        self.assertEqual(self.profile.layer_transitions()[0], 1)
        self.assertEqual(self.profile.layer_transitions(include_top=True)[0], 0)

    def test_profile_depthreference_modification(self):
        self.profile.convert_depth_reference(newname='Depth', newunit='ft', multiplier=1/0.3048)
        self.assertAlmostEqual(self.profile.max_depth, 20 / 0.3048, 3)
        self.assertEqual(self.profile.min_depth, 0)
        self.assertAlmostEqual(self.profile.layer_transitions()[0], 1 / 0.3048, 3)
        self.profile.convert_depth_reference(newname='Depth', newunit='m', multiplier=0.3048)
        self.assertAlmostEqual(self.profile.max_depth, 20, 3)
        self.assertEqual(self.profile.min_depth, 0)
        self.assertAlmostEqual(self.profile.layer_transitions()[0], 1, 3)

    def test_soilparameter_retrieval(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [1, 3, 10, 40]
        })
        self.assertIn('qc [MPa]',profile.soil_parameters())
        self.assertIn('Soil type',profile.soil_parameters())
        self.assertIn('qc from [MPa]',profile.soil_parameters(condense_linear=False))
        self.assertIn('qc [MPa]', profile.numerical_soil_parameters())
        self.assertNotIn('Soil type', profile.numerical_soil_parameters())
        self.assertIn('qc from [MPa]', profile.numerical_soil_parameters(condense_linear=False))
        self.assertNotIn('Soil type', profile.numerical_soil_parameters(condense_linear=False))
        self.assertNotIn('qc from [MPa]', profile.string_soil_parameters())
        self.assertIn('Soil type', profile.string_soil_parameters())
        self.assertTrue(profile.check_linear_variation('qc [MPa]'))
        self.assertFalse(profile.check_linear_variation('Dr [%]'))

    def test_transition_insert(self):
        self.profile.insert_layer_transition(depth=2.5)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 2.5)
        self.profile.insert_layer_transition(depth=2)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 2)

    def test_sign_conversion(self):
        self.profile.convert_depth_sign()
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], -5)
        self.profile.convert_depth_sign()
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 5)

    def test_depth_shift(self):
        self.profile.shift_depths(offset=3)
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 8)
        self.profile.shift_depths(offset=-3)
        self.assertEqual(self.profile.loc[1, 'Depth to [m]'], 5)

    def test_soilparameter_series(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 3, 20, 50]
        })
        self.assertEqual(
            profile.soilparameter_series("qc [MPa]")[1][2], 3)
        self.assertEqual(
            profile.soilparameter_series("Dr [%]")[1][3], 60)

    def test_parameter_mapping(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 3, 20, 50]
        })
        mapped_df = profile.map_soilprofile(np.linspace(0, 20, 21))
        self.assertEqual(mapped_df.loc[3, 'Soil type'], 'SILT')
        self.assertRaises(ValueError, profile.map_soilprofile, (np.linspace(-1, 20, 22),))
        self.assertEqual(mapped_df.loc[15, 'qc [MPa]'], 45)

    def test_profile_selection_constant(self):
        self.profile.selection_soilparameter(
            parameter="Su [kPa]",
            depths=[6, 7, 8, 9],
            values=[10, 11, 12, 13]
        )
        self.assertEqual(self.profile.loc[2, "Su [kPa]"], 11.5)
        self.profile.selection_soilparameter(
            parameter="Dr [%]",
            depths=[6, 7, 8, 9],
            values=[10, 11, 12, 13],
            rule="min"
        )
        self.assertEqual(self.profile.loc[2, "Dr [%]"], 10)
        self.profile.selection_soilparameter(
            parameter="gamma [kN/m3]",
            depths=[6, 7, 8, 9],
            values=[10, 11, 12, 13],
            rule="max"
        )
        self.assertEqual(self.profile.loc[2, "gamma [kN/m3]"], 13)

    def test_profile_selection_linear(self):
        self.profile.selection_soilparameter(
            parameter="Su [kPa]",
            depths=[6, 7, 8, 9],
            values=[10, 11, 12, 13],
            linearvariation=True
        )
        print(self.profile)
        self.assertAlmostEqual(self.profile.loc[2, "Su from [kPa]"], 9, 4)
        self.assertAlmostEqual(self.profile.loc[2, "Su to [kPa]"], 14, 4)
        # TODO: Check linear variation with min and max rule

    def test_merge_layers_top(self):
        self.profile.merge_layers(layer_ids=(1, 2))
        self.assertEqual(self.profile.loc[1, "Depth from [m]"], 1)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 10)
        self.assertEqual(self.profile.loc[1, "Soil type"], "SILT")

    def test_merge_layers_bottom(self):
        self.profile.merge_layers(layer_ids=(1, 2), keep='bottom')
        self.assertEqual(self.profile.loc[1, "Depth from [m]"], 1)
        self.assertEqual(self.profile.loc[1, "Depth to [m]"], 10)
        self.assertEqual(self.profile.loc[1, "Soil type"], "CLAY")

    def test_remove_parameter(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 3, 20, 50]
        })
        profile.remove_parameter('qc [MPa]')
        def test_func():
            return profile.loc[0, "qc from [MPa]"]
        self.assertRaises(KeyError, test_func)
        profile.remove_parameter('Dr [%]')
        def test_func_2():
            return profile.loc[0, "Dr [%]"]

        self.assertRaises(KeyError, test_func_2)

    def test_cut_profile(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 4, 20, 50]
        })
        pf = profile.cut_profile(top_depth=1.5, bottom_depth=19)
        self.assertEqual(pf.min_depth, 1.5)
        self.assertEqual(pf.max_depth, 19)
        self.assertEqual(pf.loc[0, "qc from [MPa]"], 3.125)
        self.assertEqual(pf["qc to [MPa]"].iloc[-1], 49)

    def test_depth_integration(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Unit weight [kN/m3]': [9, 8, 7, 10],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 4, 20, 50]
        })
        profile.depth_integration(
            parameter='Unit weight [kN/m3]',
            outputparameter='Total vertical stress [kPa]'
        )
        self.assertEqual(profile['Total vertical stress from [kPa]'].iloc[0], 0)
        self.assertEqual(profile['Total vertical stress to [kPa]'].iloc[0], 9)
        self.assertEqual(profile['Total vertical stress to [kPa]'].iloc[1], 41)

    def test_convert_to_constant(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Total unit weight from [kN/m3]': [19, 18, 17, 20],
            'Total unit weight to [kN/m3]': [19, 18, 19, 20],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 4, 20, 50]
        })
        profile.convert_to_constant("qc [MPa]")
        profile.convert_to_constant("Total unit weight [kN/m3]", rule='min')
        self.assertEqual(profile['qc [MPa]'].iloc[0], 1.5)
        self.assertEqual(profile['Total unit weight [kN/m3]'].iloc[2], 17)

    def test_calculate_overburden(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Total unit weight [kN/m3]': [19, 18, 17, 20],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, 3, 10, 40],
            'qc to [MPa]': [2, 4, 20, 50]
        })
        profile.calculate_overburden(waterlevel=4)
        self.assertEqual(profile['Vertical total stress to [kPa]'].iloc[-1], 376)
        self.assertEqual(profile['Vertical effective stress to [kPa]'].iloc[-1], 216)
        self.assertEqual(profile['Hydrostatic pressure to [kPa]'].iloc[-1], 160)

    def test_applyfunction(self):
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Total unit weight [kN/m3]': [19, 18, 17, 20],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, np.nan, 10, 40],
            'qc to [MPa]': [2, np.nan, 10, 50]
        })
        profile.applyfunction(
            function=voidratio_bulkunitweight,
            outputkey="Void ratio [-]",
            resultkey="e [-]",
            parametermapping={
                'bulkunitweight': 'Total unit weight [kN/m3]'
            }
        )
        self.assertAlmostEqual(
            profile.loc[1, "Void ratio [-]"], 1.0625, 4
        )
        profile.applyfunction(
            function=liquefactionprobability_moss,
            outputkey='Liquefaction probability [pct]',
            resultkey='Pl [pct]',
            parametermapping={
                'qc': 'qc [MPa]'
            },
            sigma_vo_eff=100,
            Rf=0.4,
            CSR=0.2,
            CSR_star=0.2
        )
        self.assertAlmostEqual(
            profile.loc[2, 'Liquefaction probability from [pct]'], 27, 0
        )
        self.assertAlmostEqual(
            profile.loc[2, 'Liquefaction probability to [pct]'], 27, 0
        )

    def test_profile_from_dataframe(self):
        """
        Check whether a SoilProfile is correctly created from a dataframe, even when the index does not start at 0
        :return:
        """
        df = pd.DataFrame({
            'Depth from [m]': [-2, 0, 2, 4],
            'Depth to [m]': [0, 2, 4, 6],
            'Soil type': ['SAND', 'CLAY', 'SAND', 'CLAY']
        })
        sp.profile_from_dataframe(df[1:])

    def test_parameter_at_depth(self):
        """
        Check whether the routine for selection of parameters at a given depth works
        :return:
        """
        profile = sp.SoilProfile({
            'Depth from [m]': [0, 1, 5, 10],
            'Depth to [m]': [1, 5, 10, 20],
            'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
            'Relative density': ['Loose', 'Medium dense', None, 'Dense'],
            'Total unit weight [kN/m3]': [19, 18, 17, 20],
            'Dr [%]': [40, 60, np.nan, 80],
            'qc from [MPa]': [1, np.nan, 10, 40],
            'qc to [MPa]': [2, np.nan, 10, 50]
        })
        # String parameter
        self.assertEqual(
            profile.parameter_at_depth(depth=0.5, parameter='Soil type'), 'SAND')
        self.assertEqual(
            profile.parameter_at_depth(depth=3, parameter='Soil type'), 'SILT')
        self.assertEqual(
            profile.parameter_at_depth(depth=5, parameter='Soil type'), 'SILT')
        self.assertEqual(
            profile.parameter_at_depth(depth=5, parameter='Soil type', shallowest=False), 'CLAY')
        # Errors when out of bounds
        self.assertRaises(
            ValueError, profile.parameter_at_depth, -1, 'Soil type')
        self.assertRaises(
            ValueError, profile.parameter_at_depth, 100, 'Soil type')
        # Errors when parameter not available
        self.assertRaises(
            ValueError, profile.parameter_at_depth, 12, 'Color')
        # Test Numerical soil parameter with constant value
        self.assertEqual(
            profile.parameter_at_depth(depth=3, parameter='Total unit weight [kN/m3]'), 18)
        self.assertEqual(
            profile.parameter_at_depth(depth=13, parameter='Dr [%]'), 80)
        # Test Numerical soil parameter with linear variation
        self.assertEqual(
            profile.parameter_at_depth(depth=0.5, parameter='qc [MPa]'), 1.5)
        self.assertEqual(
            profile.parameter_at_depth(depth=7, parameter='qc [MPa]'), 10)
        self.assertEqual(
            profile.parameter_at_depth(depth=15, parameter='qc [MPa]'), 45)
        
    def test_dummy_profile(self):
        _sp = sp.create_blank_soilprofile(max_depth=40)
        self.assertEqual(_sp.max_depth, 40)
        self.assertEqual(_sp['Soil type'].iat[0], 'Unknown')
        self.assertEqual(_sp['Total unit weight [kN/m3]'].iat[0], 20)

        _sp = sp.create_blank_soilprofile(max_depth=40, soiltype='CLAY', bulkunitweight=18)
        self.assertEqual(_sp.max_depth, 40)
        self.assertEqual(_sp['Soil type'].iat[0], 'CLAY')
        self.assertEqual(_sp['Total unit weight [kN/m3]'].iat[0], 18)


class Test_CalculationGrid(unittest.TestCase):

    def setUp(self):
        self.profile = sp.SoilProfile(
            {
                'Depth from [m]': [0, 1, 5, 10],
                'Depth to [m]': [1, 5, 10, 20],
                'Soil type': ['SAND', 'SILT', 'CLAY', 'SAND'],
                'Gmax from [kPa]': [10e3, 50e3, 75e3, 90e3],
                'Gmax to [kPa]': [20e3, 60e3, 85e3, 100e3],
                'Su from [kPa]': [np.nan, np.nan, 100, np.nan],
                'Su to [kPa]': [np.nan, np.nan, 150, np.nan]
            }
        )
        self.profile.title = "Test"

    def test_gridcreation(self):
        grid = sp.CalculationGrid(self.profile, dz=1)
        self.assertEqual(grid.elements.loc[9, "Su to [kPa]"], 150)
        self.assertEqual(grid.elements.loc[5, "Su from [kPa]"], 100)