#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only

__author__ = 'Heston Norcott'

# Native Python packages
import warnings

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
    'organic': {'type': 'bool'},
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
    "GC-GM": "Silty clayey gravel",
    "SC-SM": "Silty clayey sand",
}

# Tolerance for comparisons against exact ASTM D2487 limits, so that values lying on a limit
# (e.g. Cu = 6 from D60 / D10 = 0.6 / 0.1, or PI on the A-line) are not misclassified by floating point round-off
TOLERANCE = 1e-9


def _fines_symbol(liquid_limit, plastic_index):
    """
    Plasticity chart classification of the fines (ASTM D2487-17, Section 1.1 / Fig. 4, Table 1).
    Returns a tuple of the symbol, the A-line PI and whether the sample plots on or above the A-line.
    """
    # ASTM D2487 Fig. 4: the A-line is horizontal at PI = 4 up to LL = 25.5 and PI = 0.73 (LL - 20) beyond
    if liquid_limit < 25.5:
        aline_pi = 4.0
    else:
        aline_pi = 0.73 * (liquid_limit - 20.0)
    # "on or above the A-line" (Table 1)
    above_aline = plastic_index >= aline_pi - TOLERANCE
    high_plasticity = liquid_limit >= 50  # Table 1: LL < 50 low, LL >= 50 high plasticity
    if above_aline and plastic_index > 7 + TOLERANCE:
        symbol = "CH" if high_plasticity else "CL"
    elif (not high_plasticity) and above_aline and plastic_index >= 4 - TOLERANCE:
        # Fig. 4 hatched zone: 4 <= PI <= 7, on or above the A-line, LL < 50
        symbol = "CL-ML"
    else:
        symbol = "MH" if high_plasticity else "ML"
    return symbol, aline_pi, above_aline


def _invalid(message):
    """Warns and returns the default error output, like the Validator does for out-of-range inputs"""
    warnings.warn(message)
    result = dict(USCS_CLASSIFY_ERRORRETURN)
    result['Data quality warning'] = message
    return result



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
    the No. 200 sieve (50 %, ASTM D2487 Section 1.1). For coarse-grained soils, the split between gravel and sand is
    made on the percentage retained on the No. 4 sieve, expressed as a fraction of the coarse
    fraction (not of the total sample). Coarse-grained soils with less than 5 % fines are graded with
    :math:`C_u` and :math:`C_c` (GW: :math:`C_u \\geq 4`, SW: :math:`C_u \\geq 6`, both with :math:`1 \\leq C_c \\leq 3`),
    soils with 5 to 12 % fines get a dual symbol (e.g. GW-GM, SP-SC) and soils with more than 12 % fines are
    classified on the plasticity of the fines (GM, GC, SM, SC; GC-GM and SC-SM if the fines plot in the CL-ML zone).
    The plasticity chart (A-line and U-line) is used to distinguish silts from clays and, for fine-grained soils,
    to derive the group symbol. A soil marked as organic is classified as OL or OH.

    The grain size diameters are only needed for coarse-grained soils with 12 % fines or less and the Atterberg limits
    only for soils with 5 % fines or more. Use ``np.nan`` for data which is not available. If the data
    needed for the classification are missing or inconsistent (e.g. :math:`D_{10} \\leq 0`, :math:`PL > LL`),
    a warning is issued and the error output (``None`` for the group symbol) is returned.

    :param percent_passing_200: Percentage of the sample passing the No. 200 sieve (0.075 mm) [:math:`\\%`] - Suggested range: 0.0 <= percent_passing_200 <= 100.0
    :param percent_passing_4: Percentage of the sample passing the No. 4 sieve (4.75 mm) [:math:`\\%`] - Suggested range: 0.0 <= percent_passing_4 <= 100.0
    :param d60: Particle diameter at 60% passing on the grain size distribution curve (:math:`D_{60}`) [:math:`mm`] - Suggested range: 0.0 <= d60
    :param d30: Particle diameter at 30% passing on the grain size distribution curve (:math:`D_{30}`) [:math:`mm`] - Suggested range: 0.0 <= d30
    :param d10: Particle diameter at 10% passing on the grain size distribution curve (:math:`D_{10}`) [:math:`mm`] - Suggested range: 0.0 <= d10
    :param liquid_limit: Liquid limit of the minus-40 fraction (:math:`LL`) [:math:`\\%`] - Suggested range: 0.0 <= liquid_limit <= 100.0
    :param plastic_limit: Plastic limit of the minus-40 fraction (:math:`PL`) [:math:`\\%`] - Suggested range: 0.0 <= plastic_limit <= 100.0
    :param organic: Whether the soil is marked organic (odour, colour, or oven-dried LL / undried LL < 0.75, ASTM D2487 Section 1.1) [boolean] (optional, default= False)

    .. math::
        PI = LL - PL

        \\text{A-line: } PI = 0.73 (LL - 20) \\ (\\text{horizontal at } PI = 4 \\text{ for } LL < 25.5)

        \\text{U-line: } PI = 0.9 (LL - 8)

        C_u = \\frac{D_{60}}{D_{10}}

        C_c = \\frac{D_{30}^2}{D_{60} D_{10}}

    :returns: Dictionary with the following keys:

        - 'Group symbol': USCS group symbol, with dual symbols where applicable (e.g. ``SW-SM``)
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
        - 'Above A-line': True if the sample plots on or above the A-line
        - 'High plasticity': True if the liquid limit is 50 or more
        - 'Fines classification': 'clayey', 'silty' or 'silty-clayey' (CL-ML fines) for coarse soils with 5% or more fines
        - 'Data quality warning': Warning if the sample plots above the U-line (suggests re-checking lab data) or if the data are insufficient for the classification

    Example:
        >>> result = uscs_classify(
        ...     percent_passing_200=8.0, percent_passing_4=80.0,
        ...     d60=0.5, d30=0.25, d10=0.1,
        ...     liquid_limit=30.0, plastic_limit=15.0)
        >>> print(result['Group symbol'], '-', result['Group name'])
        SP-SC - Poorly graded sand with clay

    Reference - ASTM D2487-17, Standard Practice for Classification of Soils for Engineering Purposes (Unified Soil Classification System)

    """

    # Input consistency (all NaN checks are explicit as comparisons with NaN are always False)
    if percent_passing_4 < percent_passing_200:
        return _invalid(
            "percent_passing_4 (%s) cannot be smaller than percent_passing_200 (%s)" % (
                percent_passing_4, percent_passing_200))

    fine_grained = percent_passing_200 >= 50  # ASTM D2487 Section 1.1: 50 % or more passes the No. 200 sieve
    needs_grading = (not fine_grained) and percent_passing_200 <= 12
    needs_plasticity = fine_grained or percent_passing_200 >= 5

    if needs_grading:
        if not (d10 > 0 and d30 > 0 and d60 > 0):
            return _invalid(
                "D10, D30 and D60 must be positive to grade a coarse-grained soil with %.1f%% fines "
                "(D60 = %s, D30 = %s, D10 = %s)" % (percent_passing_200, d60, d30, d10))
        if not (d10 <= d30 <= d60):
            return _invalid(
                "Grain sizes must satisfy D10 <= D30 <= D60 (D60 = %s, D30 = %s, D10 = %s)" % (d60, d30, d10))
    if needs_plasticity:
        if np.isnan(liquid_limit) or np.isnan(plastic_limit):
            return _invalid(
                "Liquid limit and plastic limit are needed for a soil with %.1f%% fines" % percent_passing_200)
        if plastic_limit > liquid_limit:
            return _invalid(
                "plastic_limit (%s) cannot be greater than liquid_limit (%s)" % (plastic_limit, liquid_limit))

    if needs_plasticity:
        plastic_index = liquid_limit - plastic_limit
        fines_symbol, aline_pi, above_aline = _fines_symbol(liquid_limit, plastic_index)
        # ASTM D2487 Fig. 4: U-line is the upper limit of natural soils
        uline_pi = 0.9 * (liquid_limit - 8)
        high_plasticity = liquid_limit >= 50
    else:
        plastic_index = liquid_limit - plastic_limit if not (np.isnan(liquid_limit) or np.isnan(plastic_limit)) else np.nan
        aline_pi = np.nan
        uline_pi = np.nan
        above_aline = None
        high_plasticity = None
        fines_symbol = None
    classification_path = []

    if fine_grained:
        # ASTM D2487 Section 1.1 / Fig. 4 / Table 1: fine-grained soils
        classification_path.append(
            "#200: %.1f%% passing >= 50%% -> fine-grained" % percent_passing_200)
        classification_path.append(
            "PI: LL %.1f - PL %.1f = %.1f" % (
                liquid_limit, plastic_limit, plastic_index))
        classification_path.append(
            "A-line: PI = %.1f, sample %s it" % (
                aline_pi, "on or above" if above_aline else "below"))

        if organic:
            # ASTM D2487 Section 1.1 / Table 1: organic fines are OL (LL < 50) or OH (LL >= 50)
            group_symbol = "OH" if high_plasticity else "OL"
            organic_clay = above_aline and plastic_index >= 4 - TOLERANCE
            group_name = "Organic clay" if organic_clay else "Organic silt"
            classification_path.append(
                "organic: marked organic -> O prefix, %s plasticity" % (
                    "high" if high_plasticity else "low"))
        else:
            group_symbol = fines_symbol
            group_name = USCS_GROUP_NAMES[group_symbol]
            classification_path.append("plasticity chart (Fig. 4) -> %s" % group_symbol)

        data_quality_warning = None
        if plastic_index > uline_pi and liquid_limit > 0:
            data_quality_warning = (
                "Sample plots ABOVE the U-line. Real soils do not - "
                "re-check the lab results.")

        return {
            'Group symbol': group_symbol,
            'Group name': group_name,
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

    # COARSE-GRAINED (ASTM D2487 Section 1.1: more than 50 % retained on the No. 200 sieve)
    coarse_fraction = 100.0 - percent_passing_200
    retained_4_total = 100.0 - percent_passing_4
    percent_gravel = (retained_4_total / coarse_fraction) * 100.0
    # Gravel if MORE than 50 % of the coarse fraction is retained on the No. 4 sieve; otherwise sand
    gravel_governs = percent_gravel > 50 + TOLERANCE
    first_letter = "G" if gravel_governs else "S"

    # Grading is only reported (and needed) for soils with 12 % fines or less
    cu = d60 / d10 if (d10 > 0 and d60 > 0) else np.nan
    cc = (d30 ** 2) / (d60 * d10) if (d10 > 0 and d60 > 0) else np.nan
    # ASTM D2487 Table 1 / Section 1.1: GW Cu >= 4, SW Cu >= 6, both 1 <= Cc <= 3
    cu_min = 4.0 if gravel_governs else 6.0
    well_graded = bool((cu >= cu_min - TOLERANCE) and (1.0 - TOLERANCE <= cc <= 3.0 + TOLERANCE))
    if needs_plasticity:
        if fines_symbol in ("CL", "CH"):
            fines_classification = "clayey"
        elif fines_symbol == "CL-ML":
            fines_classification = "silty-clayey"
        else:
            fines_classification = "silty"
    else:
        fines_classification = None

    classification_path.append(
        "#200: %.1f%% passing < 50%% -> coarse-grained" % percent_passing_200)
    classification_path.append(
        "#4: %.1f%% of total retained = %.1f%% of coarse fraction -> %s "
        "(split is on the coarse fraction, not the total)" % (
            retained_4_total, percent_gravel,
            "GRAVEL (G)" if gravel_governs else "SAND (S)"))

    material = "gravel" if gravel_governs else "sand"
    if percent_passing_200 < 5:
        # ASTM D2487 Table 1: less than 5 % fines -> GW, GP, SW, SP
        classification_path.append(
            "fines: %.1f%% < 5%% -> clean, grade it" % percent_passing_200)
        classification_path.append(
            "Cu/Cc: Cu = %.1f (needs >= %.0f), Cc = %.2f (needs 1-3) -> %s" % (
                cu, cu_min, cc,
                "well graded" if well_graded else "poorly graded"))
        group_symbol = first_letter + ("W" if well_graded else "P")
        group_name = ("Well-graded " if well_graded else "Poorly graded ") + material
    elif percent_passing_200 <= 12:
        # ASTM D2487 Table 1: 5 to 12 % fines -> dual symbols (GW-GM, GW-GC, GP-GM, GP-GC, SW-SM, ...)
        classification_path.append(
            "fines: %.1f%% in the 5-12%% band -> DUAL SYMBOL" % percent_passing_200)
        classification_path.append(
            "Cu/Cc: Cu = %.1f, Cc = %.2f -> %s" % (
                cu, cc, "well graded" if well_graded else "poorly graded"))
        # Fines of CL-ML type have no dual symbol in D2487 for the 5-12 % band; they are treated as silty (M)
        clayey = fines_symbol in ("CL", "CH")
        classification_path.append(
            "fines type: PI %.1f vs A-line %.1f -> %s (C/M)" % (
                plastic_index, aline_pi, "clayey" if clayey else "silty"))
        group_symbol = (
            first_letter + ("W" if well_graded else "P") + "-"
            + first_letter + ("C" if clayey else "M"))
        group_name = (
            ("Well-graded " if well_graded else "Poorly graded ")
            + material + (" with clay" if clayey else " with silt"))
    else:
        # ASTM D2487 Table 1: more than 12 % fines -> GM, GC, SM, SC (GC-GM / SC-SM for CL-ML fines)
        classification_path.append(
            "fines: %.1f%% > 12%% -> fines govern the symbol" % percent_passing_200)
        classification_path.append(
            "fines type: PI %.1f vs A-line %.1f -> %s" % (
                plastic_index, aline_pi, fines_symbol))
        if fines_symbol == "CL-ML":
            group_symbol = first_letter + "C-" + first_letter + "M"
            group_name = "Silty clayey " + material
        else:
            clayey = fines_symbol in ("CL", "CH")
            group_symbol = first_letter + ("C" if clayey else "M")
            group_name = ("Clayey " if clayey else "Silty ") + material

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
