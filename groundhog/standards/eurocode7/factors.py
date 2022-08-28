#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages


# 3rd party packages
from json import load
from multiprocessing.sharedctypes import Value
import numpy as np

# Project imports


class Eurocode7_factoring_STR_GEO(object):

    def __init__(self):
        """
        This class sets the necessary factors for Eurocode 7 factoring of effects of actions and resistances for the STR and GEO limit states.

        The initialisation sets four sets dictionaries for factors, each with sub-dictionaries for the case under consideration

            - Effects of actions: Factor sets A1 and A2
                - Permanent unfavourable
                - Permanent favourable
                - Variable unfavourable
                - Variable favourable
            - Soil parameters: Factor sets M1 and M2
                - Angle of shearing resistance
                - Effective cohesion
                - Undrained shear strength
                - Unconfined strength
                - Weight density
            - Resistances: Factor sets R1, R2, R3 and R4 for different types of foundations and retaining structures
                - Spread foundation
                    - Bearing
                    - Sliding
                - Driven pile
                    - Base
                    - Shaft
                    - Total compression
                    - Shaft tension
                - Bored pile
                    - Base
                    - Shaft
                    - Total compression
                    - Shaft tension
                - CFA pile
                    - Base
                    - Shaft
                    - Total compression
                    - Shaft tension
                - Prestressed anchorage
                    - Temporary
                    - Permanent
                - Retaining structure
                    - Bearing capacity
                    - Sliding resistance
                    - Earth resistance
                - Slopes
                    - Earth resistance
            - Correlation factors
                - :math:`\\xi_1,\\xi_2` for pile design based on results from static pile load tests (SLT)
                - :math:`\\xi_3,\\xi_4` for pile design based on results from geotechnical investigations
                - :math:`\\xi_5,\\xi_6` for pile design based on results from dynamic pile load tests (DPLT)

        By default, the factors from EN 1997-1 Appendix A are adopted. If the user wants to override these using Nationally Determined Parameters,
        they need to use the overriding methods.
        """
        self.factors_actions = {
            'A1': {
                'Permanent unfavourable': 1.35,
                'Permanent favourable': 1.0,
                'Variable unfavourable': 1.5,
                'Variable favourable': 0
            },
            'A2': {
                'Permanent unfavourable': 1.0,
                'Permanent favourable': 1.0,
                'Variable unfavourable': 1.3,
                'Variable favourable': 0
            }
        }

        self.factors_soil = {
            'M1': {
                'Angle of shearing resistance': 1.0,
                'Effective cohesion': 1.0,
                'Undrained shear strength': 1.0,
                'Unconfined strength': 1.0,
                'Weight density': 1.0
            },
            'M2': {
                'Angle of shearing resistance': 1.25,
                'Effective cohesion': 1.25,
                'Undrained shear strength': 1.4,
                'Unconfined strength': 1.4,
                'Weight density': 1.0
            }
        }

        self.factors_resistance = {
            'R1': {
                'Spread foundation': {
                    'Bearing': 1.0,
                    'Sliding': 1.0
                },
                'Driven pile': {
                    'Base': 1.0,
                    'Shaft': 1.0,
                    'Total compression': 1.0,
                    'Shaft tension': 1.25
                },
                'Bored pile': {
                    'Base': 1.25,
                    'Shaft': 1.0,
                    'Total compression': 1.15,
                    'Shaft tension': 1.25
                },
                'CFA pile': {
                    'Base': 1.1,
                    'Shaft': 1.0,
                    'Total compression': 1.1,
                    'Shaft tension': 1.25
                },
                'Prestressed anchorage': {
                    'Temporary': 1.1,
                    'Permanent': 1.1
                },
                'Retaining structure': {
                    'Bearing capacity': 1.0,
                    'Sliding resistance': 1.0,
                    'Earth resistance': 1.0
                },
                'Slopes': {
                    'Earth resistance': 1.0
                }
            },
            'R2': {
                'Spread foundation': {
                    'Bearing': 1.4,
                    'Sliding': 1.1
                },
                'Driven pile': {
                    'Base': 1.1,
                    'Shaft': 1.1,
                    'Total compression': 1.1,
                    'Shaft tension': 1.15
                },
                'Bored pile': {
                    'Base': 1.1,
                    'Shaft': 1.1,
                    'Total compression': 1.1,
                    'Shaft tension': 1.15
                },
                'CFA pile': {
                    'Base': 1.1,
                    'Shaft': 1.1,
                    'Total compression': 1.1,
                    'Shaft tension': 1.15
                },
                'Prestressed anchorage': {
                    'Temporary': 1.1,
                    'Permanent': 1.1
                },
                'Retaining structure': {
                    'Bearing capacity': 1.4,
                    'Sliding resistance': 1.1,
                    'Earth resistance': 1.4
                },
                'Slopes': {
                    'Earth resistance': 1.1
                }
            },
            'R3': {
                'Spread foundation': {
                    'Bearing': 1.0,
                    'Sliding': 1.0
                },
                'Driven pile': {
                    'Base': 1.0,
                    'Shaft': 1.0,
                    'Total compression': 1.0,
                    'Shaft tension': 1.1
                },
                'Bored pile': {
                    'Base': 1.0,
                    'Shaft': 1.0,
                    'Total compression': 1.0,
                    'Shaft tension': 1.1
                },
                'CFA pile': {
                    'Base': 1.0,
                    'Shaft': 1.0,
                    'Total compression': 1.0,
                    'Shaft tension': 1.1
                },
                'Prestressed anchorage': {
                    'Temporary': 1.0,
                    'Permanent': 1.0
                },
                'Retaining structure': {
                    'Bearing capacity': 1.0,
                    'Sliding resistance': 1.0,
                    'Earth resistance': 1.0
                },
                'Slopes': {
                    'Earth resistance': 1.0
                }
            },
            'R4': {
                'Spread foundation': {
                    'Bearing': np.nan,
                    'Sliding': np.nan
                },
                'Driven pile': {
                    'Base': 1.3,
                    'Shaft': 1.3,
                    'Total compression': 1.3,
                    'Shaft tension': 1.6
                },
                'Bored pile': {
                    'Base': 1.6,
                    'Shaft': 1.3,
                    'Total compression': 1.5,
                    'Shaft tension': 1.6
                },
                'CFA pile': {
                    'Base': 1.45,
                    'Shaft': 1.3,
                    'Total compression': 1.4,
                    'Shaft tension': 1.6
                },
                'Prestressed anchorage': {
                    'Temporary': 1.1,
                    'Permanent': 1.1
                },
                'Retaining structure': {
                    'Bearing capacity': np.nan,
                    'Sliding resistance': np.nan,
                    'Earth resistance': np.nan
                },
                'Slopes': {
                    'Earth resistance': np.nan
                }
            }
        }
        self.correlation_factors = {
            'ksi_1': {
                '1': 1.4,
                '2': 1.3,
                '3': 1.2,
                '4': 1.1,
                '>=5': 1 
            },
            'ksi_2': {
                '1': 1.4,
                '2': 1.2,
                '3': 1.05,
                '4': 1.0,
                '>=5': 1.0 
            },
            'ksi_3': {
                '1': 1.4,
                '2': 1.35,
                '3': 1.33,
                '4': 1.31,
                '5': 1.29,
                '7': 1.27,
                '10': 1.25
            },
            'ksi_4': {
                '1': 1.4,
                '2': 1.27,
                '3': 1.23,
                '4': 1.20,
                '5': 1.15,
                '7': 1.12,
                '10': 1.08
            },
            'ksi_5': {
                '>=2': 1.6,
                '>=5': 1.5,
                '>=10': 1.45,
                '>=15': 1.42,
                '>=20': 1.40
            },
            'ksi_6': {
                '>=2': 1.5,
                '>=5': 1.35,
                '>=10': 1.3,
                '>=15': 1.25,
                '>=20': 1.25
            }
        }

    def select_design_approach(self, design_approach="DA1-1", foundation_type="Spread foundation"):
        """
        The user selects the design approach (``"DA1-1"``, ``"DA1-1"``, ``"DA2"``, ``"DA3-1"``, ``"DA3-2"``) based on the design problem and national requirements.
        Note that design approach 1 is split in Combination 1 (``"DA1-1"``) and Combination 2 (``"DA1-2"``) as the two combinations need to be checked.
        A separate object needs to be created for each set.
        Design approach 3 is also split into two cases (``"DA3-1"`` and ``"DA3-2"``) depending on whether A1 or A2 is used.
        
        The following combinations are stored in the ``.selected_factors_actions``, ``.selected_factors_soil`` and ``.selected_factors_resistance`` attributes:

            - DA1-1: A1 + M1 + R1
            - DA1-2: A2 + M2 + R1
            - DA2: A1 + M1 + R2
            - DA3-1: A1 + M2 + R3
            - DA3-2: A2 + M2 + R3

        For the resistance factors, the user must also specify a foundation type from the following list:
            - Spread foundation
            - Driven pile
            - Bored pile
            - CFA pile
            - Prestressed anchorage
            - Retaining structure
            - Slopes
        """
        if design_approach not in ['DA1-1', 'DA1-2', 'DA2', 'DA3-1', 'DA3-2']:
            raise ValueError("Design approach should be one of 'DA1-1', 'DA1-2', 'DA2', 'DA3-1', 'DA3-2'")
        else:
            pass

        if foundation_type not in ['Spread foundation', 'Driven pile', 'Bored pile', 'CFA pile', 'Prestressed anchorage',
                                   'Retaining structure', 'Slopes']:
            raise ValueError("Foundation type should be one of 'Spread foundation', 'Driven pile', 'Bored pile', 'CFA pile', 'Prestressed anchorage', 'Retaining structure', 'Slopes'")
        else:
            pass          

        if design_approach == "DA1-1":
            self.selected_factors_actions = self.factors_actions['A1']
            self.selected_factors_soil = self.factors_soil['M1']
            self.selected_factors_resistance = self.factors_resistance['R1'][foundation_type]
        elif design_approach == "DA1-2":
            self.selected_factors_actions = self.factors_actions['A2']
            self.selected_factors_soil = self.factors_soil['M2']
            self.selected_factors_resistance = self.factors_resistance['R1'][foundation_type]
        elif design_approach == "DA2":
            self.selected_factors_actions = self.factors_actions['A1']
            self.selected_factors_soil = self.factors_soil['M1']
            self.selected_factors_resistance = self.factors_resistance['R2'][foundation_type]
        elif design_approach == "DA3-1":
            self.selected_factors_actions = self.factors_actions['A1']
            self.selected_factors_soil = self.factors_soil['M2']
            self.selected_factors_resistance = self.factors_resistance['R3'][foundation_type]
        elif design_approach == "DA3-2":
            self.selected_factors_actions = self.factors_actions['A2']
            self.selected_factors_soil = self.factors_soil['M2']
            self.selected_factors_resistance = self.factors_resistance['R3'][foundation_type]
        else:
            raise ValueError("Design approach not recognised")

    def override_factors_actions(self, set, loadtype=None, value=None, override_dict=None):
        """
        Overrides the factor on effect of actions.
        An individual factor can be overridden by specifying its set, loadtype and value.
        Alternatively, an entire dictionary can be specified to override one of the sets, completely replacing the defaults
        """
        if set not in ['A1', 'A2']:
            raise ValueError("Set should be one of 'A1', 'A2'")
        else:
            pass  

        if override_dict is None:
            if loadtype is None or value is None or loadtype not in ['Permanent unfavourable', 'Permanent favourable', 'Variable unfavourable', 'Variable favourable']:
                raise ValueError("Load type and value need to be specified and load type needs to be one of 'Permanent unfavourable', 'Permanent favourable', 'Variable unfavourable', 'Variable favourable'")
            else:
                self.factors_actions[set][loadtype] = value
        else:
            self.factors_actions[set] = override_dict

    def override_factors_soil(self, set, soilparameter=None, value=None, override_dict=None):
        """
        Overrides the factor on soil properties.
        An individual factor can be overridden by specifying its set, soilparameter type and value.
        Alternatively, an entire dictionary can be specified to override one of the sets, completely replacing the defaults
        """
        if set not in ['M1', 'M2']:
            raise ValueError("Set should be one of 'M1', 'M2'")
        else:
            pass  

        if override_dict is None:
            if soilparameter is None or value is None or soilparameter not in ['Angle of shearing resistance', 'Effective cohesion', 'Undrained shear strength', 'Unconfined strength', 'Weight density']:
                raise ValueError("Soil parameter type and value need to be specified and soil parameter type needs to be one of 'Angle of shearing resistance', 'Effective cohesion', 'Undrained shear strength', 'Unconfined strength', 'Weight density'")
            else:
                self.factors_soil[set][soilparameter] = value
        else:
            self.factors_soil[set] = override_dict

    def override_factors_resistance(self, set, foundationtype, resistancecomponent=None, value=None, override_dict=None):
        """
        Overrides the factor on resistances.
        An individual factor can be overridden by specifying its set, foundation type, resistance component and value.
        Alternatively, an entire dictionary can be specified to override one of the sets for a given foundation type, completely replacing the defaults
        """
        if set not in ['R1', 'R2', 'R3', 'R4']:
            raise ValueError("Set should be one of 'R1', 'R2', 'R3', 'R4'")
        else:
            pass  

        if foundationtype not in ['Spread foundation', 'Driven pile', 'Bored pile', 'CFA pile', 'Prestressed anchorage',
                                   'Retaining structure', 'Slopes']:
            raise ValueError("Foundation type should be one of 'Spread foundation', 'Driven pile', 'Bored pile', 'CFA pile', 'Prestressed anchorage', 'Retaining structure', 'Slopes'")
        else:
            pass  

        if override_dict is None:
            if resistancecomponent is None or value is None:
                raise ValueError("Resistance component and value need to be specified")
            else:
                self.factors_resistance[set][foundationtype][resistancecomponent] = value
        else:
            self.factors_resistance[set][foundationtype] = override_dict

    def select_correlation_factors(self, testtype, no_tests, interpolate=False):
        """
        Selects the correlation factors for a given testtype ('Static load test', 'Ground investigation', 'Dynamic load test')
        If ``interpolate`` is set to True, interpolation between the two nearest categories is considered.
        A dictionary with the selected correlation factors is returned.
        """
        if testtype not in ['Static load test', 'Ground investigation', 'Dynamic load test']:
            raise ValueError("Test type should be one of 'Static load test', 'Ground investigation', 'Dynamic load test'")
        else:
            pass
        if testtype == 'Static load test':
            factor_name_mean = 'ksi_1'
            factor_name_min = 'ksi_2'
            if no_tests == 1:
                factor_mean = self.correlation_factors[factor_name_mean]['1']
                factor_min = self.correlation_factors[factor_name_min]['1']
            elif no_tests == 2:
                factor_mean = self.correlation_factors[factor_name_mean]['2']
                factor_min = self.correlation_factors[factor_name_min]['2']
            elif no_tests == 3:
                factor_mean = self.correlation_factors[factor_name_mean]['3']
                factor_min = self.correlation_factors[factor_name_min]['3']
            elif no_tests == 4:
                factor_mean = self.correlation_factors[factor_name_mean]['4']
                factor_min = self.correlation_factors[factor_name_min]['4']
            elif no_tests >= 5:
                factor_mean = self.correlation_factors[factor_name_mean]['>=5']
                factor_min = self.correlation_factors[factor_name_min]['>=5']
            else:
                raise ValueError('Invalid number of tests')
        elif testtype == 'Ground investigation':
            factor_name_mean = 'ksi_3'
            factor_name_min = 'ksi_4'
            if no_tests == 1:
                factor_mean = self.correlation_factors[factor_name_mean]['1']
                factor_min = self.correlation_factors[factor_name_min]['1']
            elif no_tests == 2:
                factor_mean = self.correlation_factors[factor_name_mean]['2']
                factor_min = self.correlation_factors[factor_name_min]['2']
            elif no_tests == 3:
                factor_mean = self.correlation_factors[factor_name_mean]['3']
                factor_min = self.correlation_factors[factor_name_min]['3']
            elif no_tests == 4:
                factor_mean = self.correlation_factors[factor_name_mean]['4']
                factor_min = self.correlation_factors[factor_name_min]['4']
            elif 5 <= no_tests < 7:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [5, 7],
                        [self.correlation_factors[factor_name_mean]['5'], self.correlation_factors[factor_name_mean]['7']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [5, 7],
                        [self.correlation_factors[factor_name_min]['5'], self.correlation_factors[factor_name_min]['7']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['5']
                    factor_min = self.correlation_factors[factor_name_min]['5']
            elif 7 <= no_tests < 10:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [7, 10],
                        [self.correlation_factors[factor_name_mean]['7'], self.correlation_factors[factor_name_mean]['10']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [7, 10],
                        [self.correlation_factors[factor_name_min]['7'], self.correlation_factors[factor_name_min]['10']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['7']
                    factor_min = self.correlation_factors[factor_name_min]['7']
            elif no_tests >= 10:
                factor_mean = self.correlation_factors[factor_name_mean]['10']
                factor_min = self.correlation_factors[factor_name_min]['10']
            else:
                raise ValueError('Invalid number of tests')
        elif testtype == 'Dynamic load test':
            factor_name_mean = 'ksi_5'
            factor_name_min = 'ksi_6'
            if 2 <= no_tests < 5:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [2, 5],
                        [self.correlation_factors[factor_name_mean]['>=2'], self.correlation_factors[factor_name_mean]['>=5']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [2, 5],
                        [self.correlation_factors[factor_name_min]['>=2'], self.correlation_factors[factor_name_min]['>=5']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['>=2']
                    factor_min = self.correlation_factors[factor_name_min]['>=2']
            elif 5 <= no_tests < 10:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [5, 10],
                        [self.correlation_factors[factor_name_mean]['>=5'], self.correlation_factors[factor_name_mean]['>=10']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [5, 10],
                        [self.correlation_factors[factor_name_min]['>=5'], self.correlation_factors[factor_name_min]['>=10']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['>=5']
                    factor_min = self.correlation_factors[factor_name_min]['>=5']
            elif 10 <= no_tests < 15:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [10, 15],
                        [self.correlation_factors[factor_name_mean]['>=10'], self.correlation_factors[factor_name_mean]['>=15']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [10, 15],
                        [self.correlation_factors[factor_name_min]['>=10'], self.correlation_factors[factor_name_min]['>=15']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['>=10']
                    factor_min = self.correlation_factors[factor_name_min]['>=10']
            elif 15 <= no_tests < 20:
                if interpolate:
                    factor_mean = np.interp(
                        no_tests,
                        [15, 20],
                        [self.correlation_factors[factor_name_mean]['>=15'], self.correlation_factors[factor_name_mean]['>=20']]

                    )
                    factor_min = np.interp(
                        no_tests,
                        [15, 20],
                        [self.correlation_factors[factor_name_min]['>=15'], self.correlation_factors[factor_name_min]['>=20']]

                    )
                else:
                    factor_mean = self.correlation_factors[factor_name_mean]['>=15']
                    factor_min = self.correlation_factors[factor_name_min]['>=15']
            elif no_tests >= 20:
                factor_mean = self.correlation_factors[factor_name_mean]['>=20']
                factor_min = self.correlation_factors[factor_name_min]['>=20']
            else:
                raise ValueError('Invalid number of tests')

        return {
            factor_name_mean: factor_mean,
            factor_name_min: factor_min
        }