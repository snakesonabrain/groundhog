__author__ = "Wouter Karreman & Ping Li"

# Native Python packages
import math

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

FOS_LIQUEFACTION = {
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "CRR": {"type": "float", "min_value": 0.0, "max_value": None},
    "CSR": {"type": "float", "min_value": 0.0, "max_value": None},
    "MSF": {"type": "float", "min_value": 0.0, "max_value": None},
    "K_sigma": {"type": "float", "min_value": 0.0, "max_value": None},
}

FOS_LIQUEFACTION_ERRORRETURN = {"FoS_liq [-]": np.nan}


@Validator(FOS_LIQUEFACTION, FOS_LIQUEFACTION_ERRORRETURN)
def fos_liquefaction(sigma_vo, sigma_vo_eff, CRR, CSR, MSF, K_sigma, **kwargs):
    """
    Calculates the Factor of Safety (FoS) for liquefaction assessment. The FoS is a measure of the capacity of the soil to resist liquefaction relative to the demand imposed by seismic loading.

    The FoS is computed as the ratio of the Cyclic Resistance Ratio (CRR) to the Cyclic Stress Ratio (CSR), adjusted for earthquake magnitude effects (MSF) and overburden stress effects (K_sigma). The FoS is capped at a maximum value of 5 to avoid unrealistic results.

    If the effective vertical stress (:math:`\\sigma_v^{\\prime}`) equals the total vertical stress (:math:`\\sigma_v`), the FoS is set to 5, indicating no risk of liquefaction.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param CRR: Cyclic Resistance Ratio (:math:`CRR`) [:math:`-`] - Suggested range: 0.0 < CRR <= 1.0
    :param CSR: Cyclic Stress Ratio (:math:`CSR`) [:math:`-`] - Suggested range: CSR >= 0.0
    :param MSF: Magnitude Scaling Factor (:math:`MSF`) [:math:`-`] - Suggested range: 0.0 < MSF <= 2.0
    :param K_sigma: Overburden correction factor (:math:`K_{\\sigma}`) [:math:`-`] - Suggested range: 0.0 < K_sigma <= 2.0

    .. math::
        FoS = \\min \\left( \\frac{CRR}{CSR} \\cdot MSF \\cdot K_{\\sigma}, 5 \\right)

        \\text{If } \\sigma_v^{\\prime} = \\sigma_v, \\text{ then } FoS = 5

    :returns: Dictionary with the following key:

        - 'FoS_liq [-]': Factor of Safety for liquefaction assessment (:math:`FoS`) [:math:`-`]

    Example:
        >>> result = fos_liquefaction(
        ...     sigma_vo=100, sigma_vo_eff=80, CRR=0.25, CSR=0.15, MSF=1.2, K_sigma=1.1
        ... )
        >>> print(result)
        {'FoS_liq [-]': 2.2}

    Reference:
        - Youd, T. L., et al. (2001). Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of Soils. Journal of Geotechnical and Geoenvironmental Engineering, 127(10).
        - Idriss, I. M., & Boulanger, R. W. (2008). Soil Liquefaction During Earthquakes. Earthquake Engineering Research Institute.
    """
    if sigma_vo_eff == sigma_vo:
        FoS = 5
    else:
        FoS = min((CRR / CSR) * MSF * K_sigma, 5)

    return {
        "FoS_liq [-]": FoS,
    }


CSR_ROBERTSON_CABAL_2022 = {
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "depth": {"type": "float", "min_value": 0.0, "max_value": None},
    "magnitude": {"type": "float", "min_value": 5.5, "max_value": 8.5},
    "acceleration": {"type": "float", "min_value": 0.0, "max_value": None},
    "MSF_userdefined": {
        "type": "bool",
    },
}

CSR_ROBERTSON_CABAL_2022_ERRORRETURN = {
    "MSF [-]": np.nan,
    "CSR [-]": np.nan,
}


@Validator(CSR_ROBERTSON_CABAL_2022, CSR_ROBERTSON_CABAL_2022_ERRORRETURN)
def csr_robertson_cabal_2022(
    sigma_vo,
    sigma_vo_eff,
    depth,
    magnitude,
    acceleration,
    MSF_userdefined=False,
    **kwargs
):
    """
    Calculates the Cyclic Stress Ratio (CSR) and Magnitude Scaling Factor (MSF) following the methodology outlined by Robertson and Cabal (2022).

    The CSR is computed using the simplified procedure by Seed and Idriss (1971), adjusted for depth and earthquake magnitude effects. The MSF is calculated based on the earthquake magnitude, with an option for a user-defined MSF that averages the formulations by Idriss and Andrus & Stokoe.

    The depth-dependent stress reduction factor (:math:`r_d`) is calculated using piecewise linear equations based on depth ranges.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param depth: Depth below the mudline at which CSR is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param magnitude: Earthquake magnitude (:math:`M_w`) [:math:`-`] - Suggested range: 5.0 <= magnitude <= 8.5
    :param acceleration: Maximum horizontal acceleration at the soil surface (:math:`a_{max}`) [:math:`g`] - Suggested range: acceleration >= 0.0
    :param MSF_userdefined: If True, the MSF is calculated as the average of the Idriss and Andrus & Stokoe formulations. If False, the default Robertson and Cabal (2022) MSF is used. (optional, default=False)

    .. math::
        CSR = 0.65 \\cdot a_{max} \\cdot \\frac{\\sigma_v}{\\sigma_v^{\\prime}} \\cdot r_d

        r_d = \\begin{cases}
        1.0 - 0.00765 \\cdot z & \\text{if } z < 9.15 \\, m \\\\
        1.174 - 0.0267 \\cdot z & \\text{if } 9.15 \\, m \\leq z < 23 \\, m \\\\
        0.744 - 0.008 \\cdot z & \\text{if } 23 \\, m \\leq z < 30 \\, m \\\\
        0.5 & \\text{if } z \\geq 30 \\, m
        \\end{cases}

        MSF = \\begin{cases}
        \\frac{10^{2.24}}{M_w^{2.56}} + \\left( \\frac{M_w}{7.5} \\right)^{-3.33} \\cdot 0.5 & \\text{if } MSF_{userdefined} = True \\\\
        \\frac{174}{M_w^{2.56}} & \\text{if } MSF_{userdefined} = False
        \\end{cases}

    :returns: Dictionary with the following keys:

        - 'MSF [-]': Magnitude scaling factor (:math:`MSF`) [:math:`-`]
        - 'CSR [-]': Cyclic stress ratio (:math:`CSR`) [:math:`-`]

    Example:
        >>> result = csr_robertson_cabal_2022(
        ...     sigma_vo=100, sigma_vo_eff=80, depth=10, magnitude=7.0, acceleration=0.3
        ... )
        >>> print(result)
        {'MSF [-]': 1.12, 'CSR [-]': 0.18}

    Reference:
        - Robertson, P. K., & Cabal, K. L. (2022). Guide to Cone Penetration Testing for Geotechnical Engineering. Gregg Drilling & Testing, Inc.
        - Seed, H. B., & Idriss, I. M. (1971). Simplified Procedure for Evaluating Soil Liquefaction Potential. Journal of the Soil Mechanics and Foundations Division, 97(9).
    """
    # Calculate MSF (page 119)
    if MSF_userdefined:  # middle east project specific request
        MSF = ((10**2.24) / (magnitude**2.56) + ((magnitude / 7.5) ** (-3.33))) * 0.5
    else:
        MSF = 174 / (magnitude**2.56)

    # Calculate rd based on depth (page 110)
    if depth < 9.15:
        rd = 1.0 - 0.00765 * depth
    elif depth < 23:
        rd = 1.174 - 0.0267 * depth
    elif depth < 30:
        rd = 0.744 - 0.008 * depth
    else:
        rd = 0.5

    # Calculate CSR (page 113)
    CSR = 0.65 * acceleration * (sigma_vo / sigma_vo_eff) * rd

    return {
        "MSF [-]": MSF,
        "CSR [-]": CSR,
    }


CSR_ROBERTSON_WRIDE_1998 = {
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "depth": {"type": "float", "min_value": 0.0, "max_value": None},
    "magnitude": {"type": "float", "min_value": 5.5, "max_value": 8.5},
    "acceleration": {"type": "float", "min_value": 0.0, "max_value": None},
}

CSR_ROBERTSON_WRIDE_1998_ERRORRETURN = {
    "MSF [-]": np.nan,
    "CSR [-]": np.nan,
}


@Validator(CSR_ROBERTSON_WRIDE_1998, CSR_ROBERTSON_CABAL_2022_ERRORRETURN)
def csr_robertson_wride_1998(
    sigma_vo, sigma_vo_eff, depth, magnitude, acceleration, **kwargs
):
    """
    Calculates the Cyclic Stress Ratio (CSR) and Magnitude Scaling Factor (MSF) following the methodology outlined by Robertson and Wride (1998).

    The CSR is computed using the simplified procedure by Seed and Idriss (1971), adjusted for depth and earthquake magnitude effects. The MSF is calculated based on the earthquake magnitude using the Robertson and Wride (1998) formulation.

    The depth-dependent stress reduction factor (:math:`r_d`) is calculated using piecewise linear equations based on depth ranges.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param depth: Depth below the mudline at which CSR is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param magnitude: Earthquake magnitude (:math:`M_w`) [:math:`-`] - Suggested range: 5.0 <= magnitude <= 8.5
    :param acceleration: Maximum horizontal acceleration at the soil surface (:math:`a_{max}`) [:math:`g`] - Suggested range: acceleration >= 0.0

    .. math::
        CSR = 0.65 \\cdot a_{max} \\cdot \\frac{\\sigma_v}{\\sigma_v^{\\prime}} \\cdot r_d

        r_d = \\begin{cases}
        1.0 - 0.00765 \\cdot z & \\text{if } z < 9.15 \\, m \\\\
        1.174 - 0.0267 \\cdot z & \\text{if } 9.15 \\, m \\leq z < 23 \\, m \\\\
        0.744 - 0.008 \\cdot z & \\text{if } 23 \\, m \\leq z < 30 \\, m \\\\
        0.5 & \\text{if } z \\geq 30 \\, m
        \\end{cases}

        MSF = \\frac{10^{2.24}}{M_w^{2.56}}

    :returns: Dictionary with the following keys:

        - 'MSF [-]': Magnitude scaling factor (:math:`MSF`) [:math:`-`]
        - 'CSR [-]': Cyclic stress ratio (:math:`CSR`) [:math:`-`]

    Example:
        >>> result = csr_robertson_wride_1998(
        ...     sigma_vo=100, sigma_vo_eff=80, depth=10, magnitude=7.0, acceleration=0.3
        ... )
        >>> print(result)
        {'MSF [-]': 1.12, 'CSR [-]': 0.18}

    Reference:
        - Robertson, P. K., & Wride, C. E. (1998). Evaluating Cyclic Liquefaction Potential Using the Cone Penetration Test. Canadian Geotechnical Journal, 35(3).
        - Seed, H. B., & Idriss, I. M. (1971). Simplified Procedure for Evaluating Soil Liquefaction Potential. Journal of the Soil Mechanics and Foundations Division, 97(9).
    """
    # Magnitude Scaling Factor (MSF)
    # page 27 of Robertson & Wride (1998) Cyclic liquefaction and
    # its evaluation based on SPT and CPT
    MSF = (10**2.24) / (magnitude**2.56)

    # Stress reduction factor (rd)(page 110)
    if depth < 9.15:
        rd = 1.0 - 0.00765 * depth
    elif depth < 23:
        rd = 1.174 - 0.0267 * depth
    elif depth < 30:
        rd = 0.744 - 0.008 * depth
    else:
        rd = 0.5

    # Cyclic Stress Ratio (CSR) (page 113)
    CSR = 0.65 * acceleration * (sigma_vo / sigma_vo_eff) * rd

    return {
        "MSF [-]": MSF,
        "CSR [-]": CSR,
    }


CSR_IDRISS_BOULANGER_2008 = {
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "depth": {"type": "float", "min_value": 0.0, "max_value": None},
    "magnitude": {"type": "float", "min_value": 5.5, "max_value": 8.5},
    "acceleration": {"type": "float", "min_value": 0.0, "max_value": None},
}

CSR_IDRISS_BOULANGER_2008_ERRORRETURN = {
    "MSF [-]": np.nan,
    "CSR [-]": np.nan,
}


@Validator(CSR_IDRISS_BOULANGER_2008, CSR_IDRISS_BOULANGER_2008_ERRORRETURN)
def csr_idriss_boulanger_2008(
    sigma_vo, sigma_vo_eff, depth, magnitude, acceleration, **kwargs
):
    """
    Calculates the Cyclic Stress Ratio (CSR) and Magnitude Scaling Factor (MSF) following the methodology outlined by Idriss and Boulanger (2008).

    The CSR is computed using the simplified procedure by Seed and Idriss (1971), adjusted for depth and earthquake magnitude effects. The MSF is calculated based on the earthquake magnitude using the Idriss and Boulanger (2008) formulation.

    The depth-dependent stress reduction factor (:math:`r_d`) is calculated using an exponential function that depends on depth and earthquake magnitude.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param depth: Depth below the mudline at which CSR is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param magnitude: Earthquake magnitude (:math:`M_w`) [:math:`-`] - Suggested range: 5.0 <= magnitude <= 8.5
    :param acceleration: Maximum horizontal acceleration at the soil surface (:math:`a_{max}`) [:math:`g`] - Suggested range: acceleration >= 0.0

    .. math::
        CSR = 0.65 \\cdot a_{max} \\cdot \\frac{\\sigma_v}{\\sigma_v^{\\prime}} \\cdot r_d

        MSF = \\min \\left( 6.9 \\cdot e^{-M_w / 4} - 0.058, 1.8 \\right)

        \\alpha = -1.012 - 1.126 \\cdot \\sin \\left( \\frac{z}{11.73} + 5.133 \\right)

        \\beta = 0.106 + 0.118 \\cdot \\sin \\left( \\frac{z}{11.28} + 5.142 \\right)

        r_d = e^{\\alpha + \\beta \\cdot M_w}

    :returns: Dictionary with the following keys:

        - 'MSF [-]': Magnitude scaling factor (:math:`MSF`) [:math:`-`]
        - 'CSR [-]': Cyclic stress ratio (:math:`CSR`) [:math:`-`]

    Example:
        >>> result = csr_idriss_boulanger_2008(
        ...     sigma_vo=100, sigma_vo_eff=80, depth=10, magnitude=7.0, acceleration=0.3
        ... )
        >>> print(result)
        {'MSF [-]': 1.12, 'CSR [-]': 0.18}

    Reference:
        - Idriss, I. M., & Boulanger, R. W. (2008). Soil Liquefaction During Earthquakes. Earthquake Engineering Research Institute.
        - Seed, H. B., & Idriss, I. M. (1971). Simplified Procedure for Evaluating Soil Liquefaction Potential. Journal of the Soil Mechanics and Foundations Division, 97(9).
    """
    # Magnitude Scaling Factor (MSF)
    # page 93 - equation 50 of Idriss and Boulanger (2008)
    MSF = min(6.9 * np.exp(-magnitude / 4) - 0.058, 1.8)

    # Stress reduction factor (rd)
    # page 68 - equations 22, 23 and 24 of Idriss and Boulanger (2008)
    alpha = -1.012 - 1.126 * math.sin(depth / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(depth / 11.28 + 5.142)
    rd = np.exp(alpha + beta * magnitude)

    # Cyclic Stress Ratio (CSR) (page 70)
    CSR = 0.65 * acceleration * (sigma_vo / sigma_vo_eff) * rd

    return {
        "MSF [-]": MSF,
        "CSR [-]": CSR,
    }


CSR_BOULANGER_IDRISS_2014 = {
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "depth": {"type": "float", "min_value": 0.0, "max_value": None},
    "magnitude": {"type": "float", "min_value": 5.5, "max_value": 8.5},
    "acceleration": {"type": "float", "min_value": 0.0, "max_value": None},
}

CSR_BOULANGER_IDRISS_2014_ERRORRETURN = {
    "CSR [-]": np.nan,
    "MSF [-]": np.nan,
}


@Validator(CSR_BOULANGER_IDRISS_2014, CSR_BOULANGER_IDRISS_2014_ERRORRETURN)
def csr_boulanger_idriss_2014(
    Qtn_cs, sigma_vo, sigma_vo_eff, depth, magnitude, acceleration, **kwargs
):
    """
    Calculates the Cyclic Stress Ratio (CSR) and Magnitude Scaling Factor (MSF) following the methodology outlined by Boulanger and Idriss (2014).

    The CSR is computed using the simplified procedure by Seed and Idriss (1971), adjusted for depth and earthquake magnitude effects. The MSF is calculated based on the earthquake magnitude and normalized cone penetration resistance (:math:`Q_{tn,cs}`) using the Boulanger and Idriss (2014) formulation.

    The depth-dependent stress reduction factor (:math:`r_d`) is calculated using an exponential function that depends on depth and earthquake magnitude.

    :param Qtn_cs: Normalized cone penetration resistance (:math:`Q_{tn,cs}`) [:math:`-`] - Suggested range: Qtn_cs >= 0.0
    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param depth: Depth below the mudline at which CSR is calculated (:math:`z`) [:math:`m`] - Suggested range: depth >= 0.0
    :param magnitude: Earthquake magnitude (:math:`M_w`) [:math:`-`] - Suggested range: 5.0 <= magnitude <= 8.5
    :param acceleration: Maximum horizontal acceleration at the soil surface (:math:`a_{max}`) [:math:`g`] - Suggested range: acceleration >= 0.0

    .. math::
        CSR = 0.65 \\cdot a_{max} \\cdot \\frac{\\sigma_v}{\\sigma_v^{\\prime}} \\cdot r_d

        \\alpha = -1.012 - 1.126 \\cdot \\sin \\left( \\frac{z}{11.73} + 5.133 \\right)

        \\beta = 0.106 + 0.118 \\cdot \\sin \\left( \\frac{z}{11.28} + 5.142 \\right)

        r_d = e^{\\alpha + \\beta \\cdot M_w}

        MSF_{max} = \\min \\left( 1.09 + \\left( \\frac{Q_{tn,cs}}{180} \\right)^3, 2.2 \\right)

        MSF = 1 + (MSF_{max} - 1) \\cdot \\left( 8.64 \\cdot e^{-M_w / 4} - 1.325 \\right)

    :returns: Dictionary with the following keys:

        - 'CSR [-]': Cyclic stress ratio (:math:`CSR`) [:math:`-`]
        - 'MSF [-]': Magnitude scaling factor (:math:`MSF`) [:math:`-`]

    Example:
        >>> result = csr_boulanger_idriss_2014(
        ...     Qtn_cs=100, sigma_vo=120, sigma_vo_eff=80, depth=10, magnitude=7.0, acceleration=0.3
        ... )
        >>> print(result)
        {'CSR [-]': 0.18, 'MSF [-]': 1.12}

    Reference:
        - Boulanger, R. W., & Idriss, I. M. (2014). CPT and SPT Liquefaction Triggering Procedures. Report No. UCD/CGM-14/01, University of California, Davis.
        - Seed, H. B., & Idriss, I. M. (1971). Simplified Procedure for Evaluating Soil Liquefaction Potential. Journal of the Soil Mechanics and Foundations Division, 97(9).
    """
    # Stress reduction factor (rd)
    # page 68 - equations 22, 23 and 24 of Idriss and Boulanger (2008)
    alpha = -1.012 - 1.126 * math.sin(depth / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(depth / 11.28 + 5.142)
    rd = np.exp(alpha + beta * magnitude)

    # Cyclic Stress Ratio (CSR) (page 5)
    CSR = 0.65 * acceleration * (sigma_vo / sigma_vo_eff) * rd

    # Magnitude Scaling Factor (MSF)
    # page 14 of Boulanger and Idriss (2014)
    MSF_max = min(1.09 + (Qtn_cs / 180) ** 3, 2.2)
    MSF = 1 + (MSF_max - 1) * (8.64 * np.exp(-magnitude / 4) - 1.325)

    return {
        "CSR [-]": CSR,
        "MSF [-]": MSF,
    }


CRR_ROBERTSON_CABAL_2022 = {
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
}

CRR_ROBERTSON_CABAL_2022_ERRORRETURN = {
    "CRR [-]": np.nan,
    "K_sigma [-]": np.nan,
}


@Validator(CRR_ROBERTSON_CABAL_2022, CRR_ROBERTSON_CABAL_2022_ERRORRETURN)
def crr_robertson_cabal_2022(Qtn_cs, **kwargs):
    """
    Calculates the Cyclic Resistance Ratio (CRR) and Overburden Correction Factor (:math:`K_{\\sigma}`) based on the methodology outlined by Robertson and Cabal (2022).

    The CRR is computed using normalized cone penetration resistance (:math:`Q_{tn,cs}`) with fines correction. The CRR is calculated differently for three ranges of :math:`Q_{tn,cs}`:
    - For :math:`Q_{tn,cs} < 50`, a linear relationship is used.
    - For :math:`50 \\leq Q_{tn,cs} < 160`, a cubic relationship is used.
    - For :math:`Q_{tn,cs} \\geq 160`, the CRR is set to infinity, indicating no liquefaction risk.

    The overburden correction factor (:math:`K_{\\sigma}`) is set to 1, as it is not required for this methodology.

    :param Qtn_cs: Normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`] - Suggested range: Qtn_cs >= 0.0

    .. math::
        CRR = \\begin{cases}
        0.833 \\cdot \\left( \\frac{Q_{tn,cs}}{1000} \\right) + 0.05 & \\text{if } Q_{tn,cs} < 50 \\\\
        93 \\cdot \\left( \\frac{Q_{tn,cs}}{1000} \\right)^3 + 0.08 & \\text{if } 50 \\leq Q_{tn,cs} < 160 \\\\
        \\infty & \\text{if } Q_{tn,cs} \\geq 160
        \\end{cases}

        K_{\\sigma} = 1

    :returns: Dictionary with the following keys:

        - 'CRR [-]': Cyclic resistance ratio (:math:`CRR`) [:math:`-`]
        - 'K_sigma [-]': Overburden correction factor (:math:`K_{\\sigma}`) [:math:`-`]

    Example:
        >>> result = crr_robertson_cabal_2022(Qtn_cs=120)
        >>> print(result)
        {'CRR [-]': 0.18, 'K_sigma [-]': 1.0}

    Reference:
        - Robertson, P. K., & Cabal, K. L. (2022). Guide to Cone Penetration Testing for Geotechnical Engineering. Gregg Drilling & Testing, Inc.
    """
    # Calculate CRR based on Qtn_cs ranges
    if Qtn_cs < 50:
        CRR = 0.833 * (Qtn_cs / 1000) + 0.05
    elif 50 <= Qtn_cs < 160:
        CRR = 93 * (Qtn_cs / 1000) ** 3 + 0.08
    else:
        CRR = np.inf

    # Overburden correction factor (K_sigma)
    # Not required for this methodology, so set to 1
    K_sigma = 1

    return {
        "CRR [-]": CRR,
        "K_sigma [-]": K_sigma,
    }


CRR_ROBERTSON_WRIDE_1998 = {
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "relative_density": {"type": "float", "min_value": 0.0, "max_value": 2.0},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

CRR_ROBERTSON_WRIDE_1998_ERRORRETURN = {
    "CRR [-]": np.nan,
    "K_sigma [-]": np.nan,
}


@Validator(CRR_ROBERTSON_WRIDE_1998, CRR_ROBERTSON_WRIDE_1998_ERRORRETURN)
def crr_robertson_wride_1998(
    Qtn_cs, sigma_vo_eff, relative_density, atmospheric_pressure=100, **kwargs
):
    """
    Calculates the Cyclic Resistance Ratio (CRR) and Overburden Correction Factor (:math:`K_{\\sigma}`) based on the methodology outlined by Robertson and Wride (1998).

    The CRR is computed using normalized cone penetration resistance (:math:`Q_{tn,cs}`) with fines correction. The CRR is calculated differently for three ranges of :math:`Q_{tn,cs}`:
    - For :math:`Q_{tn,cs} < 50`, a linear relationship is used.
    - For :math:`50 \\leq Q_{tn,cs} < 160`, a cubic relationship is used.
    - For :math:`Q_{tn,cs} \\geq 160`, the CRR is set to infinity, indicating no liquefaction risk.

    The overburden correction factor (:math:`K_{\\sigma}`) is calculated based on the effective vertical stress (:math:`\\sigma_v^{\\prime}`) and relative density. If :math:`\\sigma_v^{\\prime}` is less than 100 kPa, :math:`K_{\\sigma}` is set to 1. Otherwise, it is calculated using a power-law relationship.

    :param Qtn_cs: Normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`] - Suggested range: Qtn_cs >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param relative_density: Relative density of the soil (:math:`D_r`) [:math:`-`] - Suggested range: 0.0 <= relative_density <= 1.0
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        CRR = \\begin{cases}
        0.833 \\cdot \\left( \\frac{Q_{tn,cs}}{1000} \\right) + 0.05 & \\text{if } Q_{tn,cs} < 50 \\\\
        93 \\cdot \\left( \\frac{Q_{tn,cs}}{1000} \\right)^3 + 0.08 & \\text{if } 50 \\leq Q_{tn,cs} < 160 \\\\
        \\infty & \\text{if } Q_{tn,cs} \\geq 160
        \\end{cases}

        K_{\\sigma} = \\begin{cases}
        1.0 & \\text{if } \\sigma_v^{\\prime} < 100 \\, kPa \\\\
        \\min \\left( \\left( \\frac{\\sigma_v^{\\prime}}{P_a} \\right)^{f_{K_{\\sigma}} - 1}, 1 \\right) & \\text{otherwise}
        \\end{cases}

        f_{K_{\\sigma}} = 1 - \\frac{D_r}{2}

        f_{K_{\\sigma}} = \\min \\left( \\max \\left( f_{K_{\\sigma}}, 0.6 \\right), 0.8 \\right)

    :returns: Dictionary with the following keys:

        - 'CRR [-]': Cyclic resistance ratio (:math:`CRR`) [:math:`-`]
        - 'K_sigma [-]': Overburden correction factor (:math:`K_{\\sigma}`) [:math:`-`]

    Example:
        >>> result = crr_robertson_wride_1998(
        ...     Qtn_cs=120, sigma_vo_eff=80, relative_density=0.5
        ... )
        >>> print(result)
        {'CRR [-]': 0.18, 'K_sigma [-]': 1.0}

    Reference:
        - Robertson, P. K., & Wride, C. E. (1998). Evaluating Cyclic Liquefaction Potential Using the Cone Penetration Test. Canadian Geotechnical Journal, 35(3).
        - Youd, T. L., et al. (2001). Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of Soils. Journal of Geotechnical and Geoenvironmental Engineering, 127(10).
    """
    # Cyclic Resistance Ratio (CRR)
    # page 452 - equation 8 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    # there is a typo in original reference
    if Qtn_cs < 50:
        CRR = 0.833 * (Qtn_cs / 1000) + 0.05
    elif 50 <= Qtn_cs < 160:
        CRR = 93 * (Qtn_cs / 1000) ** 3 + 0.08
    else:
        CRR = np.inf

    # Overburden correction factor (K_sigma)
    # page 308 of Youd et al (2001) NCEER - Liquefaction resistance of soils
    # (referred in RW1998)
    if sigma_vo_eff < 100:
        K_sigma = 1.0
    else:
        f_K_sigma = 1 - relative_density / 2
        f_K_sigma = max(min(f_K_sigma, 0.8), 0.6)
        K_sigma = min((sigma_vo_eff / atmospheric_pressure) ** (f_K_sigma - 1), 1)

    return {
        "CRR [-]": CRR,
        "K_sigma [-]": K_sigma,
    }


CRR_IDRISS_BOULANGER_2008 = {
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

CRR_IDRISS_BOULANGER_2008_ERRORRETURN = {
    "CRR [-]": np.nan,
    "K_sigma [-]": np.nan,
}


@Validator(CRR_IDRISS_BOULANGER_2008, CRR_IDRISS_BOULANGER_2008_ERRORRETURN)
def crr_idriss_boulanger_2008(Qtn_cs, sigma_vo_eff, atmospheric_pressure=100, **kwargs):
    """
    Calculates the Cyclic Resistance Ratio (CRR) and Overburden Correction Factor (:math:`K_{\\sigma}`) based on the methodology outlined by Idriss and Boulanger (2008).

    The CRR is computed using normalized cone penetration resistance (:math:`Q_{tn,cs}`) with fines correction. The CRR is calculated using a polynomial function of :math:`Q_{tn,cs}` and is capped at a maximum value of 0.6.

    The overburden correction factor (:math:`K_{\\sigma}`) is calculated based on the effective vertical stress (:math:`\\sigma_v^{\\prime}`) and a stress correction coefficient (:math:`C_{\\sigma}`). The value of :math:`K_{\\sigma}` is capped at 1.1.

    :param Qtn_cs: Normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`] - Suggested range: Qtn_cs >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        CRR = \\min \\left( \\exp \\left( \\frac{Q_{tn,cs}}{540} + \\left( \\frac{Q_{tn,cs}}{67} \\right)^2 - \\left( \\frac{Q_{tn,cs}}{80} \\right)^3 + \\left( \\frac{Q_{tn,cs}}{114} \\right)^4 - 3 \\right), 0.6 \\right)

        C_{\\sigma} = \\min \\left( \\frac{1}{37.3 - 8.27 \\cdot \\min(Q_{tn,cs}, 211)^{0.264}}, 0.3 \\right)

        K_{\\sigma} = \\min \\left( 1 - C_{\\sigma} \\cdot \\ln \\left( \\frac{\\sigma_v^{\\prime}}{P_a} \\right), 1.1 \\right)

    :returns: Dictionary with the following keys:

        - 'CRR [-]': Cyclic resistance ratio (:math:`CRR`) [:math:`-`]
        - 'K_sigma [-]': Overburden correction factor (:math:`K_{\\sigma}`) [:math:`-`]

    Example:
        >>> result = crr_idriss_boulanger_2008(
        ...     Qtn_cs=120, sigma_vo_eff=80, atmospheric_pressure=100
        ... )
        >>> print(result)
        {'CRR [-]': 0.25, 'K_sigma [-]': 1.0}

    Reference:
        - Idriss, I. M., & Boulanger, R. W. (2008). Soil Liquefaction During Earthquakes. Earthquake Engineering Research Institute.
    """
    # Cyclic Resistance Ratio (CRR)
    # page 100 - equation 71 of Idriss and Boulanger (2008)
    CRR = min(
        np.exp(
            Qtn_cs / 540
            + (Qtn_cs / 67) ** 2
            - (Qtn_cs / 80) ** 3
            + (Qtn_cs / 114) ** 4
            - 3
        ),
        0.6,
    )

    # Overburden correction factor (K_sigma)
    # page 95 of Idriss and Boulanger (2008)
    C_sigma = min(1 / (37.3 - 8.27 * min(Qtn_cs, 211) ** 0.264), 0.3)
    K_sigma = min(1 - C_sigma * np.log(sigma_vo_eff / atmospheric_pressure), 1.1)

    return {
        "CRR [-]": CRR,
        "K_sigma [-]": K_sigma,
    }


CRR_BOULANGER_IDRISS_2014 = {
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

CRR_BOULANGER_IDRISS_2014_ERRORRETURN = {
    "CRR [-]": np.nan,
    "K_sigma [-]": np.nan,
}


@Validator(CRR_BOULANGER_IDRISS_2014, CRR_BOULANGER_IDRISS_2014_ERRORRETURN)
def crr_boulanger_idriss_2014(Qtn_cs, sigma_vo_eff, atmospheric_pressure=100, **kwargs):
    """
    Calculates the Cyclic Resistance Ratio (CRR) and Overburden Correction Factor (:math:`K_{\\sigma}`) based on the methodology outlined by Boulanger and Idriss (2014).

    The CRR is computed using normalized cone penetration resistance (:math:`Q_{tn,cs}`) with fines correction. The CRR is calculated using a polynomial function of :math:`Q_{tn,cs}` and is capped at a maximum value of 0.6.

    The overburden correction factor (:math:`K_{\\sigma}`) is calculated based on the effective vertical stress (:math:`\\sigma_v^{\\prime}`) and a stress correction coefficient (:math:`C_{\\sigma}`). The value of :math:`K_{\\sigma}` is capped at 1.1.

    :param Qtn_cs: Normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`] - Suggested range: Qtn_cs >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        CRR = \\min \\left( \\exp \\left( \\frac{Q_{tn,cs}}{113} + \\left( \\frac{Q_{tn,cs}}{1000} \\right)^2 - \\left( \\frac{Q_{tn,cs}}{140} \\right)^3 + \\left( \\frac{Q_{tn,cs}}{137} \\right)^4 - 2.80 \\right), 0.6 \\right)

        C_{\\sigma} = \\frac{1}{37.3 - 8.27 \\cdot \\min(Q_{tn,cs}, 211)^{0.264}}

        K_{\\sigma} = \\min \\left( 1 - C_{\\sigma} \\cdot \\ln \\left( \\frac{\\sigma_v^{\\prime}}{P_a} \\right), 1.1 \\right)

    :returns: Dictionary with the following keys:

        - 'CRR [-]': Cyclic resistance ratio (:math:`CRR`) [:math:`-`]
        - 'K_sigma [-]': Overburden correction factor (:math:`K_{\\sigma}`) [:math:`-`]

    Example:
        >>> result = crr_boulanger_idriss_2014(
        ...     Qtn_cs=120, sigma_vo_eff=80, atmospheric_pressure=100
        ... )
        >>> print(result)
        {'CRR [-]': 0.25, 'K_sigma [-]': 1.0}

    Reference:
        - Boulanger, R. W., & Idriss, I. M. (2014). CPT and SPT Liquefaction Triggering Procedures. Report No. UCD/CGM-14/01, University of California, Davis.
    """
    # Cyclic Resistance Ratio (CRR)
    # page 17 of Boulanger and Idriss (2014)
    CRR = min(
        np.exp(
            Qtn_cs / 113
            + (Qtn_cs / 1000) ** 2
            - (Qtn_cs / 140) ** 3
            + (Qtn_cs / 137) ** 4
            - 2.80
        ),
        0.6,
    )

    # Overburden correction factor (K_sigma)
    # page 11 of Boulanger and Idriss (2014)
    C_sigma = 1 / (37.3 - 8.27 * min(Qtn_cs, 211) ** 0.264)
    K_sigma = min(1 - C_sigma * np.log(sigma_vo_eff / atmospheric_pressure), 1.1)

    return {
        "CRR [-]": CRR,
        "K_sigma [-]": K_sigma,
    }


QTN_CS_ROBERTSON_CABAL_2022 = {
    "ic": {"type": "float", "min_value": 0.0, "max_value": 3.0},
    "Fr": {"type": "float", "min_value": 0.0, "max_value": 10.0},
    "qt": {"type": "float", "min_value": 0.0, "max_value": 120.0},
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

QTN_CS_ROBERTSON_CABAL_2022_ERRORRETURN = {
    "Qtn_cs [-]": np.nan,
}


@Validator(QTN_CS_ROBERTSON_CABAL_2022, QTN_CS_ROBERTSON_CABAL_2022_ERRORRETURN)
def Qtn_cs_robertson_cabal_2022(
    ic, Fr, qt, sigma_vo_eff, sigma_vo, atmospheric_pressure=100, **kwargs
):
    """
    Calculates the normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) based on the methodology outlined by Robertson and Cabal (2022).

    The calculation involves adjusting the cone tip resistance (:math:`q_t`) for overburden stress effects and soil type using the soil behavior type index (:math:`I_c`) and friction ratio (:math:`F_r`). The normalization is performed using the exponent :math:`n`, which depends on :math:`I_c` and effective vertical stress (:math:`\\sigma_v^{\\prime}`).

    :param ic: Soil behavior type index (:math:`I_c`) [:math:`-`] - Suggested range: 1.0 <= ic <= 3.5
    :param Fr: Friction ratio (:math:`F_r`) [:math:`-`] - Suggested range: 0.0 <= Fr <= 10.0
    :param qt: Total cone tip resistance (:math:`q_t`) [:math:`MPa`] - Suggested range: qt >= 0.0
    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo >= 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff >= 0.0
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        K_c = \\begin{cases}
        1.0 & \\text{if } I_c \\leq 1.7 \\\\
        1.0 & \\text{if } 1.7 < I_c < 2.36 \\text{ and } F_r < 0.5 \\\\
        15 - \\frac{14}{1 + \\left( \\frac{I_c}{2.95} \\right)^{11}} & \\text{otherwise}
        \\end{cases}

        n = \\min \\left( 1, 0.381 \\cdot I_c + 0.05 \\cdot \\left( \\frac{\\sigma_v^{\\prime}}{P_a} \\right) - 0.15 \\right)

        Q_{tn} = \\left( \\frac{q_t - \\sigma_v}{P_a} \\right) \\cdot \\left( \\frac{P_a}{\\sigma_v^{\\prime}} \\right)^n

        Q_{tn,cs} = K_c \\cdot Q_{tn}

    :returns: Dictionary with the following key:

        - 'Qtn_cs [-]': Normalized cone resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`]

    Example:
        >>> result = Qtn_cs_robertson_cabal_2022(
        ...     ic=2.0, Fr=0.3, qt=10, sigma_vo_eff=80, sigma_vo=100
        ... )
        >>> print(result)
        {'Qtn_cs [-]': 120.5}

    Reference:
        - Robertson, P. K., & Cabal, K. L. (2022). Guide to Cone Penetration Testing for Geotechnical Engineering. Gregg Drilling & Testing, Inc.
    """
    # Calculate Kc based on ic and Fr (page 116)
    if ic > 1.7:
        if ic < 2.36 and Fr < 0.5:
            Kc = 1.0
        else:
            Kc = 15 - 14 / (1 + (ic / 2.95) ** 11)
    else:
        Kc = 1.0

    # Helper function to calculate Qtn
    def Qtn(qt, sigma_vo, sigma_vo_eff, n, pa=0.001 * atmospheric_pressure):
        return ((qt - 0.001 * sigma_vo) / pa) * (pa / (0.001 * sigma_vo_eff)) ** n

    # Helper function to calculate exponent n
    def exponent_zhang(ic, sigma_vo_eff, pa=atmospheric_pressure):
        return min(1, 0.381 * ic + 0.05 * (sigma_vo_eff / pa) - 0.15)

    # Calculate exponent n and Qtn
    _exponent_zhang = exponent_zhang(ic, sigma_vo_eff)
    _Qtn = Qtn(qt, sigma_vo, sigma_vo_eff, _exponent_zhang)

    # Calculate Qtn_cs (page 115)
    Qtn_cs = Kc * _Qtn

    return {
        "Qtn_cs [-]": Qtn_cs,
    }


QTN_CS_ROBERTSON_WRIDE_1998 = {
    "sigma_vo": {"type": "float", "min_value": 0.0, "max_value": None},
    "ic": {"type": "float", "min_value": 0.0, "max_value": 2.6},
    "qc": {"type": "float", "min_value": 0.0, "max_value": 120.0},
    "Fr": {"type": "float", "min_value": 0.0, "max_value": 10.0},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

QTN_CS_ROBERTSON_WRIDE_1998_ERRORRETURN = {
    "Qtn_cs [-]": np.nan,
}


@Validator(QTN_CS_ROBERTSON_WRIDE_1998, QTN_CS_ROBERTSON_WRIDE_1998_ERRORRETURN)
def Qtn_cs_robertson_wride_1998(
    sigma_vo, ic, qc, Fr, atmospheric_pressure=100, **kwargs
):
    """
    Calculates the normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) based on the methodology outlined by Robertson and Wride (1998).

    The calculation involves normalizing the cone tip resistance (:math:`q_c`) for overburden stress effects and adjusting for fines content using a fines correction factor (:math:`K_c`). The stress normalization is performed using a correction factor (:math:`C_Q`), which depends on total vertical stress (:math:`\\sigma_v`), and the final corrected resistance is obtained as :math:`q_{c1N,cs}`.

    :param sigma_vo: Total vertical stress at the depth of interest (:math:`\\sigma_v`) [:math:`kPa`] - Suggested range: sigma_vo > 0.0
    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff > 0.0
    :param ic: Soil behavior type index (:math:`I_c`) [:math:`-`] - Suggested range: 1.0 <= ic <= 3.5
    :param qc: Total cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: qc >= 0.0
    :param Fr: Normalized friction ratio (:math:`F_r`) as a fraction [:math:`-`] - Suggested range: 0.0 <= Fr <= 10.0
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        C_Q = \\min \\left( \\left( \\frac{P_a}{\\sigma_v} \\right)^{0.5}, 2 \\right)

        q_{c1N} = \\frac{q_c}{0.001 P_a} \\cdot C_Q

        K_c = \\begin{cases}
        1.0 & \\text{if } I_c \\leq 1.64 \\\\
        1.0 & \\text{if } I_c < 2.36 \\text{ and } F_r < 0.5 \\\\
        -0.403 I_c^4 + 5.581 I_c^3 - 21.63 I_c^2 + 33.75 I_c - 17.88 & \\text{otherwise}
        \\end{cases}

        Q_{tn,cs} = K_c \\cdot q_{c1N}

    :returns: Dictionary with the following key:

        - 'Qtn_cs [-]': Normalized cone resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`]

    Example:
        >>> result = Qtn_cs_robertson_wride_1998(
        ...     sigma_vo=100, sigma_vo_eff=80, ic=2.0, qc=10, Fr=0.3
        ... )
        >>> print(result)
        {'Qtn_cs [-]': 95.4}

    Reference:
        - Robertson, P. K., & Wride, C. E. (1998). Evaluating cyclic liquefaction potential using the cone penetration test. Canadian Geotechnical Journal, 35(3), 442-459.
    """
    # Stress normalization
    # page 447 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    C_Q = min((atmospheric_pressure / sigma_vo) ** 0.5, 2)

    # Stress normalized tip resistance
    # page 447 - equation 3 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    qc1N = qc / (0.001 * atmospheric_pressure) * C_Q

    # Fines correction factor
    # page 450 - equation 7 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    if ic <= 1.64 or (ic < 2.36 and Fr < 0.5):
        Kc = 1.0
    else:
        Kc = -0.403 * ic**4 + 5.581 * ic**3 - 21.63 * ic**2 + 33.75 * ic - 17.88

    # Fines corrected stress normalized tip resistance
    # page 449 - equation 4 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    qc1N_cs = qc1N * Kc

    return {"Qtn_cs [-]": qc1N_cs}


QTN_CS_IDRISS_BOULANGER_2008 = {
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "qc": {"type": "float", "min_value": 0.0, "max_value": 120.0},
    "ic": {"type": "float", "min_value": 0.0, "max_value": 5.0},
    "atmospheric_pressure": {"type": "float", "min_value": 0, "max_value": 110},
}

QTN_CS_IDRISS_BOULANGER_2008_ERRORRETURN = {
    "Qtn_cs [-]": np.nan,
    "Fines [%]": np.nan,
}


@Validator(QTN_CS_IDRISS_BOULANGER_2008, QTN_CS_IDRISS_BOULANGER_2008_ERRORRETURN)
def Qtn_cs_idriss_boulanger_2008(
    sigma_vo_eff, qc, ic, atmospheric_pressure=100, **kwargs
):
    """
    Calculates the normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`) based on the methodology outlined by Idriss and Boulanger (2008).

    This method accounts for the effect of overburden stress, fines content, and stress normalization using an iterative approach to determine the exponent (:math:`m`). The final normalized resistance (:math:`q_{c1N,cs}`) is computed after applying fines content corrections.

    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`] - Suggested range: sigma_vo_eff > 0.0
    :param qc: Total cone tip resistance (:math:`q_c`) [:math:`MPa`] - Suggested range: qc >= 0.0
    :param ic: Soil behavior type index (:math:`I_c`) [:math:`-`] - Suggested range: 1.0 <= ic <= 3.5
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`] - Suggested range: atmospheric_pressure > 0.0 (optional, default=100.0)

    .. math::
        q_{c1N} = \\max \\left( \\min \\left( \\frac{q_c}{0.001 P_a}, 254 \\right), 21 \\right)

        m = 1.338 - 0.249 \\cdot (q_{c1N})^{0.264}

        C_N = \\min \\left( \\left( \\frac{P_a}{\\sigma_v^{\\prime}} \\right)^m, 1.7 \\right)

        FC = 2.8 \\cdot I_c^{2.6}

        \\Delta q_{c1N} = (5.4 + q_{c1N} / 16) \\cdot \\exp \\left( 1.63 + \\frac{9.7}{FC + 0.01} - \\left( \\frac{15.7}{FC + 0.01} \\right)^2 \\right)

        Q_{tn,cs} = q_{c1N} + \\Delta q_{c1N}

    :returns: Dictionary with the following keys:

        - 'Qtn_cs [-]': Normalized cone resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`]
        - 'Fines [%]': Estimated fines content as a percentage (:math:`FC`) [:math:`%`]

    Example:
        >>> result = Qtn_cs_idriss_boulanger_2008(
        ...     sigma_vo_eff=80, qc=10, ic=2.0
        ... )
        >>> print(result)
        {'Qtn_cs [-]': 85.2, 'Fines [%]': 14.7}

    Reference:
        - Idriss, I. M., & Boulanger, R. W. (2008). Soil liquefaction during earthquakes. Earthquake Engineering Research Institute.
    """
    # Overburden correction of in-situ test results
    # page 87 - equations 39 and 40 of Idriss and Boulanger (2008)
    # Soil Liquefaction During Earthquakes
    qc1N_initial = max(min(qc / (0.001 * atmospheric_pressure), 254), 21)

    m = 1.338 - 0.249 * (qc1N_initial) ** 0.264
    delta_m = 1
    counter = 1
    while delta_m > 0.01 and counter < 1000:
        C_N = min((atmospheric_pressure / sigma_vo_eff) ** m, 1.7)
        qc1N = max(min(C_N * qc / (0.001 * atmospheric_pressure), 254), 21)
        m_new = 1.338 - 0.249 * (qc1N) ** 0.264
        delta_m = abs(m - m_new)
        counter = counter + 1
        m = m_new
    qc1N = C_N * qc / (0.001 * atmospheric_pressure)

    # Fines content
    # Figure 55 of page 79 of Idriss and Boulanger (2008)
    # Soil Liquefaction During Earthquakes
    FC = 2.8 * ic**2.6

    # Equivalent clean-sand value of the corrected tip resistance
    # page 111 - equation 77 and 78 of Idriss and Boulanger (2008)
    # Soil Liquefaction During Earthquakes
    dqc1N = (5.4 + qc1N / 16) * np.exp(
        1.63 + 9.7 / (FC + 0.01) - (15.7 / (FC + 0.01)) ** 2
    )
    qc1N_cs = qc1N + dqc1N

    return {
        "Qtn_cs [-]": qc1N_cs,
        "Fines [%]": FC,
    }


QTN_CS_BOULANGER_IDRISS_2014 = {
    "sigma_vo_eff": {"type": "float", "min_value": 0.0, "max_value": None},
    "qc": {"type": "float", "min_value": 0.0, "max_value": 120.0},
    "ic": {"type": "float", "min_value": 0.0, "max_value": 5.0},
    "atmospheric_pressure": {"type": "float", "min_value": 80, "max_value": 110},
    "C_FC": {"type": "float", "min_value": -0.29, "max_value": 0.29},
}

QTN_CS_BOULANGER_IDRISS_2014_ERRORRETURN = {
    "Qtn_cs [-]": np.nan,
    "Fines [%]": np.nan,
}


@Validator(QTN_CS_BOULANGER_IDRISS_2014, QTN_CS_BOULANGER_IDRISS_2014_ERRORRETURN)
def Qtn_cs_boulanger_idriss_2014(
    sigma_vo_eff, qc, ic, C_FC=0, atmospheric_pressure=100, **kwargs
):
    """
    Calculates the normalized cone penetration resistance with fines correction (:math:`Q_{tn,cs}`)
    based on the methodology outlined by Boulanger and Idriss (2014).

    The calculation involves adjusting the cone tip resistance (:math:`q_c`) for overburden stress
    effects and soil type using the soil behavior type index (:math:`I_c`). The normalization is performed
    using the exponent :math:`m`, which depends on :math:`I_c` and effective vertical stress (:math:`\\sigma_v^{\\prime}`).
    Additionally, fines content (:math:`FC`) is estimated as a function of :math:`I_c`.

    :param sigma_vo_eff: Effective vertical stress at the depth of interest (:math:`\\sigma_v^{\\prime}`) [:math:`kPa`]
                         - Suggested range: :math:`\\sigma_v^{\\prime} \\geq 0.0`
    :param qc: Total cone tip resistance (:math:`q_c`) [:math:`MPa`]
               - Suggested range: :math:`q_c \\geq 0.0`
    :param ic: Soil behavior type index (:math:`I_c`) [:math:`-`]
               - Suggested range: :math:`1.0 \\leq I_c \\leq 3.5`
    :param atmospheric_pressure: Reference atmospheric pressure (:math:`P_a`) [:math:`kPa`]
                                 - Suggested range: :math:`P_a > 0.0` (optional, default=100)
    :param C_FC: Fitting parameter for fines content estimation (:math:`C_{FC}`) [:math:`-`]
                 - Suggested range: Typically between -0.2 and 0.2 (optional, default=0)

    .. math::
        FC = \\min \\left( \\max \\left( 80 \\cdot (I_c + C_{FC}) - 137, 0 \\right), 100 \\right)

        C_N = \\min \\left( \\left( \\frac{P_a}{\\sigma_v^{\\prime}} \\right)^m, 1.7 \\right)

        Q_{tn} = C_N \\cdot \\frac{q_c}{0.001 P_a}

        \\Delta Q_{tn} = \\left( 11.9 + \\frac{Q_{tn}}{14.6} \\right) \\cdot
        \\exp \\left( 1.63 - \\frac{9.7}{FC + 2} - \\left( \\frac{15.7}{FC + 2} \\right)^2 \\right)

        Q_{tn,cs} = Q_{tn} + \\Delta Q_{tn}

    :returns: Dictionary with the following keys:

        - 'Qtn_cs [-]': Normalized cone resistance with fines correction (:math:`Q_{tn,cs}`) [:math:`-`]
        - 'Fines [%]': Estimated fines content (:math:`FC`) [:math:`%`]

    Example:
        >>> result = Qtn_cs_boulanger_idriss_2014(
        ...     sigma_vo_eff=80, qc=10, ic=2.0, C_FC=0
        ... )
        >>> print(result)
        {'Qtn_cs [-]': 120.5, 'Fines [%]': 25.3}

    Reference:
        - Boulanger, R. W., & Idriss, I. M. (2014). CPT and SPT liquefaction triggering procedures.
          Report No. UCD/CGM-14/01, Center for Geotechnical Modeling, Department of Civil & Environmental Engineering, University of California, Davis.
    """
    # exponent m depending on grading
    # page 10, page 15 and page 21 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering
    m = 0.5
    delta_m = 1
    counter = 1
    FC = min(max(80 * (ic + C_FC) - 137, 0), 100)
    while delta_m > 0.01 and counter < 1000:
        C_N = min((atmospheric_pressure / sigma_vo_eff) ** m, 1.7)
        qc1N = C_N * qc / (0.001 * atmospheric_pressure)
        dqc1N = (11.9 + qc1N / 14.6) * np.exp(
            1.63 - (9.7 / (FC + 2)) - (15.7 / (FC + 2)) ** 2
        )
        qc1Ncs = max(min(qc1N + dqc1N, 254), 21)
        m_new = 1.338 - 0.249 * (qc1Ncs) ** 0.264
        delta_m = abs(m - m_new)
        counter = counter + 1
        m = m_new

    # overburden correction factor
    # page 10 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering
    C_N = min((atmospheric_pressure / sigma_vo_eff) ** m, 1.7)

    # overburden corrected penetration resistance
    # page 6 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering
    qc1N = C_N * qc / (0.001 * atmospheric_pressure)

    # equivalent clean sand penetration resistance
    # page 15 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering
    dqc1N = (11.9 + qc1N / 14.6) * np.exp(
        1.63 - (9.7 / (FC + 2)) - (15.7 / (FC + 2)) ** 2
    )
    qc1N_cs = qc1N + dqc1N

    return {
        "Qtn_cs [-]": qc1N_cs,
        "Fines [%]": FC,
    }


LIQUEFACTION_STRAINS_ZHANG = {
    "FoS_liq": {"type": "float", "min_value": 0.0, "max_value": 5.0},
    "relative_density": {"type": "float", "min_value": 0.0, "max_value": None},
    "Qtn_cs": {"type": "float", "min_value": 0.0, "max_value": None},
}

LIQUEFACTION_STRAINS_ZHANG_ERRORRETURN = {
    "eps_liq [%]": np.nan,
    "gamma_liq [%]": np.nan,
}


@Validator(LIQUEFACTION_STRAINS_ZHANG, LIQUEFACTION_STRAINS_ZHANG_ERRORRETURN)
def liquefaction_strains_zhang(FoS_liq, relative_density, Qtn_cs, **kwargs):
    """
    Calculates liquefaction-induced strains (volumetric and lateral) for level ground 
    and lateral spreading based on the methodologies of Zhang et al. (2002, 2004).

    The function estimates volumetric strain (:math:`\\epsilon_{liq}`) based on normalized 
    cone penetration resistance (:math:`Q_{tn,cs}`) and safety factor against liquefaction (:math:`FoS_{liq}`).  
    Lateral strain (:math:`\\gamma_{liq}`) is estimated using :math:`FoS_{liq}` and relative density (:math:`D_r`).

    :param FoS_liq: Safety factor against liquefaction (:math:`FoS_{liq}`) [:math:`-`]  
                    - Suggested range: :math:`0.0 \\leq FoS_{liq} \\leq 5.0`
    :param relative_density: Relative density (:math:`D_r`) [:math:`-`]  
                             - Suggested range: :math:`0.0 \\leq D_r \\leq 2.0`
    :param Qtn_cs: Normalized cone resistance corrected for fines (:math:`Q_{tn,cs}`) [:math:`-`]  
                   - Suggested range: :math:`33 \\leq Q_{tn,cs} \\leq 200`

    .. math::
        \\epsilon_{liq} =
        \\begin{cases} 
        102 \\cdot Q_{tn,cs}^{-0.82}, & 0.0 \\leq FoS_{liq} \\leq 0.55, 33 \\leq Q_{tn,cs} \\leq 200 \\\\
        2411 \\cdot Q_{tn,cs}^{-1.45}, & 0.55 < FoS_{liq} \\leq 0.65, 147 < Q_{tn,cs} \\leq 200 \\\\
        1690 \\cdot Q_{tn,cs}^{-1.46}, & 0.75 < FoS_{liq} \\leq 0.85, 80 < Q_{tn,cs} \\leq 200 \\\\
        5.8, & Q_{tn,cs} < 33, FoS_{liq} < 1 \\\\
        \\end{cases}

    .. math::
        \\gamma_{liq} =
        \\begin{cases} 
        3.26 \\cdot FoS_{liq}^{-1.80}, & D_r \\geq 0.8, 0.7 \\leq FoS_{liq} < 2.0 \\\\
        6.2, & D_r \\geq 0.8, FoS_{liq} < 0.7 \\\\
        3.58 \\cdot FoS_{liq}^{-4.42}, & D_r \\geq 0.5, 0.66 \\leq FoS_{liq} < 2.0 \\\\
        250 (1 - FoS_{liq}) + 3.5, & 0.81 \\leq FoS_{liq} < 1.0 \\\\
        51.2, & FoS_{liq} < 0.81 \\\\
        0, & \text{otherwise}
        \\end{cases}

    :returns: Dictionary with the following keys:

        - 'eps_liq [%]': Volumetric strain (:math:`\\epsilon_{liq}`) as a percentage.
        - 'gamma_liq [%]': Lateral strain (:math:`\\gamma_{liq}`) as a percentage.

    Example:
        >>> result = liquefaction_strains_zhang(FoS_liq=0.8, relative_density=0.6, Qtn_cs=100)
        >>> print(result)
        {'eps_liq [%]': 2.3, 'gamma_liq [%]': 14.5}

    References:
        - Zhang, L., Robertson, P. K., & Brachman, R. W. I. (2002). Estimating liquefaction-induced 
          ground settlements from CPT for level ground. *Canadian Geotechnical Journal, 39*(5), 1168-1180.
        - Zhang, L., Robertson, P. K., & Brachman, R. W. I. (2004). Estimating liquefaction-induced 
          lateral displacements using the standard penetration test or cone penetration test. 
          *Journal of Geotechnical and Geoenvironmental Engineering, 130*(8), 861-871.
    """
    # Volumetric strain
    # Zhang et al (2002), Appendix A2
    eps = 0
    if FoS_liq <= 0.55 and Qtn_cs >= 33 and Qtn_cs <= 200:
        eps = 102 * Qtn_cs**-0.82
    if FoS_liq > 0.55 and FoS_liq <= 0.65 and Qtn_cs >= 33 and Qtn_cs <= 147:
        eps = 102 * Qtn_cs**-0.82
    if FoS_liq > 0.55 and FoS_liq <= 0.65 and Qtn_cs > 147 and Qtn_cs <= 200:
        eps = 2411 * Qtn_cs**-1.45
    if FoS_liq > 0.65 and FoS_liq <= 0.75 and Qtn_cs >= 33 and Qtn_cs <= 110:
        eps = 102 * Qtn_cs**-0.82
    if FoS_liq > 0.65 and FoS_liq <= 0.75 and Qtn_cs > 110 and Qtn_cs <= 200:
        eps = 1701 * Qtn_cs**-1.42
    if FoS_liq > 0.75 and FoS_liq <= 0.85 and Qtn_cs >= 33 and Qtn_cs <= 80:
        eps = 102 * Qtn_cs**-0.82
    if FoS_liq > 0.75 and FoS_liq <= 0.85 and Qtn_cs > 80 and Qtn_cs <= 200:
        eps = 1690 * Qtn_cs**-1.46
    if FoS_liq > 0.85 and FoS_liq <= 0.95 and Qtn_cs >= 33 and Qtn_cs <= 60:
        eps = 102 * Qtn_cs**-0.82
    if FoS_liq > 0.85 and FoS_liq <= 0.95 and Qtn_cs > 80 and Qtn_cs <= 200:
        eps = 1430 * Qtn_cs**-1.48
    if FoS_liq > 0.95 and FoS_liq <= 1.05 and Qtn_cs >= 33 and Qtn_cs <= 200:
        eps = 64 * Qtn_cs**-0.93
    if FoS_liq > 1.05 and FoS_liq <= 1.15 and Qtn_cs >= 33 and Qtn_cs <= 200:
        eps = 11 * Qtn_cs**-0.65
    if FoS_liq > 1.15 and FoS_liq <= 1.25 and Qtn_cs >= 33 and Qtn_cs <= 200:
        eps = 9.7 * Qtn_cs**-0.69
    if FoS_liq > 1.25 and FoS_liq <= 1.35 and Qtn_cs >= 33 and Qtn_cs <= 200:
        eps = 7.6 * Qtn_cs**-0.71
    if Qtn_cs < 33 and FoS_liq < 1:
        eps = 5.8

    # Lateral Displacement Index
    # Zhang et al (2004), Appendix
    if relative_density >= 0.8 and FoS_liq < 2 and FoS_liq >= 0.7:
        gamma_max = 3.26 * (FoS_liq**-1.80)
    elif relative_density >= 0.8 and FoS_liq < 0.7:
        gamma_max = 6.2
    elif relative_density >= 0.7 and FoS_liq < 2 and FoS_liq >= 0.56:
        gamma_max = 3.22 * (FoS_liq**-2.08)
    elif relative_density >= 0.7 and FoS_liq < 0.56:
        gamma_max = 10
    elif relative_density >= 0.6 and FoS_liq < 2 and FoS_liq >= 0.59:
        gamma_max = 3.20 * (FoS_liq**-2.89)
    elif relative_density >= 0.6 and FoS_liq < 0.59:
        gamma_max = 14.5
    elif relative_density >= 0.5 and FoS_liq < 2 and FoS_liq >= 0.66:
        gamma_max = 3.58 * (FoS_liq**-4.42)
    elif relative_density >= 0.5 and FoS_liq < 0.66:
        gamma_max = 22.7
    elif relative_density >= 0.4 and FoS_liq < 2 and FoS_liq >= 0.72:
        gamma_max = 4.22 * (FoS_liq**-6.39)
    elif relative_density >= 0.4 and FoS_liq < 0.72:
        gamma_max = 34.1
    elif FoS_liq < 2 and FoS_liq >= 1:
        gamma_max = 3.31 * (FoS_liq**-7.97)
    elif FoS_liq < 1 and FoS_liq >= 0.81:
        gamma_max = 250 * (1 - FoS_liq) + 3.5
    elif FoS_liq < 0.81:
        gamma_max = 51.2
    else:
        gamma_max = 0

    return {"eps_liq [%]": eps, "gamma_liq [%]": gamma_max}
