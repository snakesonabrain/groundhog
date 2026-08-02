#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only

__author__ = 'Heston Norcott'

# Native Python packages
import os

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

USCS_CLASSIFY_PARAMETERS = {
    'percent_passing_200': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'percent_passing_4': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'd60': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'd30': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'd10': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'liquid_limit': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'plastic_limit': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'organic': {'type': 'boolean'},
}

USCS_CLASSIFY_ERRORRETURN = {
    'Group symbol': None,
    'Group name': None,
    'Classification path': None,
    'Fine grained': None,
    'Gravel governs': None,
    '% coarse fraction': None,
    '% retained on No. 4 (of total)': None,
    '% gravel of coarse fraction': None,
    'Coefficient of uniformity C_u': np.nan,
    'Coefficient of curvature C_c': np.nan,
    'Well graded': None,
    'Plasticity index': np.nan,
    'A-line plasticity index': np.nan,
    'U-line plasticity index': np.nan,
    'Above A-line': None,
    'High plasticity': None,
    'Fines classification': None,
    'Data quality warning': None,
}

USCS_GROUP_NAMES = {
    "GW": "Well-graded gravel",
    "GP": "Poorly graded gravel",
    "GM": "Silty gravel",
    "GC": "Clayey gravel",
    "SW": "Well-graded sand",
    "SP": "Poorly graded sand",
    "SM": "Silty sand",
    "SC": "Clayey sand",
    "ML": "Silt",
    "CL": "Lean clay",
    "OL": "Organic clay or organic silt (low plasticity)",
    "MH": "Elastic silt",
    "CH": "Fat clay",
    "OH": "Organic clay or organic silt (high plasticity)",
    "CL-ML": "Silty clay",
}


@Validator(USCS_CLASSIFY_PARAMETERS, USCS_CLASSIFY_ERRORRETURN)
def uscs_classify(
        percent_passing_200,
        percent_passing_4,
        d60,
        d30,
        d10,
        liquid_limit,
        plastic_limit,
        organic=False, ** kwargs):
    """
    Classifies a soil according to the Unified Soil Classification System (USCS, ASTM D2487)
    from grain size and plasticity data.

    The split between coarse-grained and fine-grained soil is made on the percentage passing
    the No. 200 sieve (50 %). For coarse-grained soils, the split between gravel and sand is
    made on the percentage retained on the No. 4 sieve, expressed as a fraction of the coarse
    fraction (not of the total sample). The plasticity chart (A-line and U-line) is used to
    distinguish silts from clays and, for fine-grained soils, to derive the group symbol.

    :param percent_passing_200: Percentage of the sample passing the No. 200 sieve (0.075 mm) [:math:`\\%`] - Suggested range: 0.0 <= percent_passing_200 <= 100.0
    :param percent_passing_4: Percentage of the sample passing the No. 4 sieve (4.75 mm) [:math:`\\%`] - Suggested range: 0.0 <= percent_passing_4 <= 100.0
    :param d60: Particle diameter at 60% passing on the grain size distribution curve (:math:`D_{60}`) [:math:`mm`] - Suggested range: 0.0 <= d60
    :param d30: Particle diameter at 30% passing on the grain size distribution curve (:math:`D_{30}`) [:math:`mm`] - Suggested range: 0.0 <= d30
    :param d10: Particle diameter at 10% passing on the grain size distribution curve (:math:`D_{10}`) [:math:`mm`] - Suggested range: 0.0 <= d10
    :param liquid_limit: Liquid limit of the minus-40 fraction (:math:`LL`) [:math:`\\%`] - Suggested range: 0.0 <= liquid_limit <= 100.0
    :param plastic_limit: Plastic limit of the minus-40 fraction (:math:`PL`) [:math:`\\%`] - Suggested range: 0.0 <= plastic_limit <= 100.0
    :param organic: Whether the soil is marked organic (odour, colour, or LL ratio < 0.75) [boolean] (optional, default= False)

    .. math::
        PI = LL - PL

        A-line: PI = 0.73 (LL - 20)

        U-line: PI = 0.9 (LL - 8)

        C_u = \\frac{D_{60}}{D_{10}}

        C_c = \\frac{D_{30}^2}{D_{60} D_{10}}

    :returns: Dictionary with the following keys:

        - 'Group symbol': Two-letter USCS group symbol, with dual symbols where applicable (e.g. ``SW-SM``)
        - 'Group name': Descriptive name of the group
        - 'Classification path': Ordered list of the decision steps taken, for traceability of the classification
        - 'Fine grained': True if the soil passes 50% or more on the No. 200 sieve
        - 'Gravel governs': True if more than half of the coarse fraction is retained on the No. 4 sieve
        - '% coarse fraction': Percentage of the sample retained on the No. 200 sieve [:math:`\\%`]
        - '% retained on No. 4 (of total)': Percentage of the total sample retained on the No. 4 sieve [:math:`\\%`]
        - '% gravel of coarse fraction': Percentage of the coarse fraction retained on the No. 4 sieve [:math:`\\%`]
        - 'Coefficient of uniformity C_u': Coefficient of uniformity (coarse soils) [:math:`-`]
        - 'Coefficient of curvature C_c': Coefficient of curvature (coarse soils) [:math:`-`]
        - 'Well graded': True if the gradation criteria for the group are met (coarse soils)
        - 'Plasticity index': Plasticity index [:math:`\\%`]
        - 'A-line plasticity index': Plasticity index on the A-line for the given liquid limit [:math:`\\%`]
        - 'U-line plasticity index': Plasticity index on the U-line for the given liquid limit [:math:`\\%`]
        - 'Above A-line': True if the sample plots above the A-line
        - 'High plasticity': True if the liquid limit is 50 or more
        - 'Fines classification': 'clayey' or 'silty' (coarse soils with 5% or more fines)
        - 'Data quality warning': Warning if the sample plots above the U-line (suggests re-checking lab data)

    Reference - ASTM D2487

    """

    plastic_index = liquid_limit - plastic_limit
    aline_pi = 0.73 * (liquid_limit - 20)
    uline_pi = 0.9 * (liquid_limit - 8)
    above_aline = plastic_index >= aline_pi
    high_plasticity = liquid_limit >= 50
    classification_path = []

    if percent_passing_200 >= 50:
        # FINE-GRAINED
        classification_path.append(
            "#200: %.1f%% passing >= 50%% -> fine-grained" % percent_passing_200)
        classification_path.append(
            "PI: LL %.1f - PL %.1f = %.1f" % (
                liquid_limit, plastic_limit, plastic_index))
        classification_path.append(
            "A-line: PI = 0.73(LL - 20) = %.1f, sample %s it" % (
                aline_pi, "above" if above_aline else "below"))

        if organic:
            group_symbol = "OH" if high_plasticity else "OL"
            classification_path.append(
                "organic: marked organic -> O prefix, %s plasticity" % (
                    "high" if high_plasticity else "low"))
        elif not high_plasticity and 4 <= plastic_index <= 7 and above_aline:
            group_symbol = "CL-ML"
            classification_path.append(
                "dual: PI between 4 and 7 and above A-line -> borderline CL-ML")
        elif above_aline and plastic_index >= 7:
            group_symbol = "CH" if high_plasticity else "CL"
            classification_path.append(
                "above A-line and PI >= 7 -> %s" % (
                    "fat clay (CH)" if high_plasticity else "lean clay (CL)"))
        else:
            group_symbol = "MH" if high_plasticity else "ML"
            classification_path.append(
                "below A-line or PI < 7 -> %s" % (
                    "elastic silt (MH)" if high_plasticity else "silt (ML)"))

        data_quality_warning = None
        if plastic_index > uline_pi and liquid_limit > 0:
            data_quality_warning = (
                "Sample plots ABOVE the U-line. Real soils do not - "
                "re-check the lab results.")

        return {
            'Group symbol': group_symbol,
            'Group name': USCS_GROUP_NAMES[group_symbol],
            'Classification path': classification_path,
            'Fine grained': True,
            'Gravel governs': None,
            '% coarse fraction': 100.0 - percent_passing_200,
            '% retained on No. 4 (of total)': 100.0 - percent_passing_4,
            '% gravel of coarse fraction': None,
            'Coefficient of uniformity C_u': np.nan,
            'Coefficient of curvature C_c': np.nan,
            'Well graded': None,
            'Plasticity index': plastic_index,
            'A-line plasticity index': aline_pi,
            'U-line plasticity index': uline_pi,
            'Above A-line': above_aline,
            'High plasticity': high_plasticity,
            'Fines classification': None,
            'Data quality warning': data_quality_warning,
        }

    # COARSE-GRAINED
    coarse_fraction = 100.0 - percent_passing_200
    retained_4_total = 100.0 - percent_passing_4
    percent_gravel = (
        (retained_4_total / coarse_fraction) * 100.0 if coarse_fraction > 0 else 0.0)
    gravel_governs = percent_gravel > 50
    first_letter = "G" if gravel_governs else "S"

    if d10 > 0:
        cu = d60 / d10
    else:
        cu = np.nan
    if d60 > 0 and d10 > 0:
        cc = (d30 ** 2) / (d60 * d10)
    else:
        cc = np.nan

    cu_min = 4.0 if gravel_governs else 6.0
    well_graded = (cu >= cu_min) and (1.0 <= cc <= 3.0)
    fines_are_clay = above_aline and plastic_index >= 7
    fines_classification = "clayey" if fines_are_clay else "silty"

    classification_path.append(
        "#200: %.1f%% passing < 50%% -> coarse-grained" % percent_passing_200)
    classification_path.append(
        "#4: %.1f%% of total retained = %.1f%% of coarse fraction -> %s "
        "(split is on the coarse fraction, not the total)" % (
            retained_4_total, percent_gravel,
            "GRAVEL (G)" if gravel_governs else "SAND (S)"))

    if percent_passing_200 < 5:
        classification_path.append(
            "fines: %.1f%% < 5%% -> clean, grade it" % percent_passing_200)
        classification_path.append(
            "Cu/Cc: Cu = %.1f (needs >= %.0f), Cc = %.2f (needs 1-3) -> %s" % (
                cu, cu_min, cc,
                "well graded" if well_graded else "poorly graded"))
        group_symbol = first_letter + ("W" if well_graded else "P")
        group_name = (
            "Well-graded " if well_graded else "Poorly graded ") + (
            "gravel" if gravel_governs else "sand")
    elif percent_passing_200 <= 12:
        classification_path.append(
            "fines: %.1f%% in the 5-12%% band -> DUAL SYMBOL" % percent_passing_200)
        classification_path.append(
            "Cu/Cc: Cu = %.1f, Cc = %.2f -> %s" % (
                cu, cc, "well graded" if well_graded else "poorly graded"))
        classification_path.append(
            "fines type: PI %.1f vs A-line %.1f -> %s (C/M)" % (
                plastic_index, aline_pi,
                "clayey" if fines_are_clay else "silty"))
        group_symbol = (
            first_letter + ("W" if well_graded else "P") + "-"
            + first_letter + ("C" if fines_are_clay else "M"))
        group_name = (
            ("Well-graded " if well_graded else "Poorly graded ")
            + ("gravel" if gravel_governs else "sand")
            + (" with clay" if fines_are_clay else " with silt"))
    else:
        classification_path.append(
            "fines: %.1f%% > 12%% -> fines govern the symbol" % percent_passing_200)
        classification_path.append(
            "fines type: PI %.1f vs A-line %.1f -> %s (C/M)" % (
                plastic_index, aline_pi,
                "clayey" if fines_are_clay else "silty"))
        group_symbol = first_letter + ("C" if fines_are_clay else "M")
        group_name = (
            ("Clayey " if fines_are_clay else "Silty ")
            + ("gravel" if gravel_governs else "sand"))

    return {
        'Group symbol': group_symbol,
        'Group name': group_name,
        'Classification path': classification_path,
        'Fine grained': False,
        'Gravel governs': gravel_governs,
        '% coarse fraction': coarse_fraction,
        '% retained on No. 4 (of total)': retained_4_total,
        '% gravel of coarse fraction': percent_gravel,
        'Coefficient of uniformity C_u': cu,
        'Coefficient of curvature C_c': cc,
        'Well graded': well_graded,
        'Plasticity index': plastic_index,
        'A-line plasticity index': aline_pi,
        'U-line plasticity index': uline_pi,
        'Above A-line': above_aline,
        'High plasticity': high_plasticity,
        'Fines classification': fines_classification,
        'Data quality warning': None,
    }
