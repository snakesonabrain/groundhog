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
    Calculates the Factor of Safety (FoS) for liquefaction assessment

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        CRR [-]: Cyclic resistance ratio
        CSR [-]: cyclic stress ratio
        MSF [-]: Magnitude scaling factor
        K_sigma [-]: Overburden correction factor

    Returns:
        FoS_liq [-]: Factor of Safety (FoS) for liquefaction assessment
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
    Calculates CSR following Robertson and Cabal (2022)

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        depth [m]: Depth below mudline
        magnitude [-]: Earthquake magnitude
        acceleration [-]: Maximum horizontal acceleration at the soil surface
        MSF_usderdefined [-]: MSF defined by user with averaging Idriss and Andrus and Stokoe

    Returns:
        MSF [-]: Magnitude scaling factor
        CSR [-]: Cyclic stress ratio
    """
    # page 119
    if MSF_userdefined:
        MSF = ((10**2.24) / (magnitude**2.56) + ((magnitude / 7.5) ** (-3.33))) * 0.5
    else:
        MSF = 174 / (magnitude**2.56)

    # page 110
    if depth < 9.15:
        rd = 1.0 - 0.00765 * depth
    elif depth < 23:
        rd = 1.174 - 0.0267 * depth
    elif depth < 30:
        rd = 0.744 - 0.008 * depth
    else:
        rd = 0.5

    # page 113
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
    Calculates CSR following Robertson and Wride (1998)

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        depth [m]: Depth below mudline
        magnitude [-]: Earthquake magnitude
        acceleration [-]: Maximum horizontal acceleration at the soil surface

    Returns:
        MSF [-]: Magnitude scaling factor
        CSR [-]: Cyclic stress ratio
    """

    # Magnitude Scaling factor
    # page 27 of Robertson & Wride (1998) Cyclic liquefaction and
    # its evaluation based on SPT and CPT
    MSF = (10**2.24) / (magnitude**2.56)

    # page 110
    if depth < 9.15:
        rd = 1.0 - 0.00765 * depth
    elif depth < 23:
        rd = 1.174 - 0.0267 * depth
    elif depth < 30:
        rd = 0.744 - 0.008 * depth
    else:
        rd = 0.5

    # page 113
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
    Calculates CSR following Idriss and Boulanger (2008)

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        depth [m]: Depth below mudline
        magnitude [-]: Earthquake magnitude
        acceleration [-]: Maximum horizontal acceleration at the soil surface

    Returns:
        MSF [-]: Magnitude scaling factor
        CSR [-]: Cyclic stress ratio
    """

    # Magnitude Scaling Factor
    # page 93 - equation 50 of Idriss and Boulanger (2008)
    # Soil Liquefaction during Earthquakes
    MSF = min(6.9 * np.exp(-magnitude / 4) - 0.058, 1.8)

    # Stress reduction coefficient
    # page 68 - equations 22, 23 and 24 of Idriss and Boulanger (2008)
    # Soil Liquefaction during Earthquakes
    alpha = -1.012 - 1.126 * math.sin(depth / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(depth / 11.28 + 5.142)
    rd = np.exp(alpha + beta * magnitude)

    # page 70
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
        Calculates CSR following Idriss and Boulanger (2014)

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        depth [m]: Depth below mudline
        magnitude [-]: Earthquake magnitude
        acceleration [-]: Maximum horizontal acceleration at the soil surface

    Returns:
        CSR [-]: Cyclic stress ratio
    """
    # Stress reduction coefficient
    # page 68 - equations 22, 23 and 24 of Idriss and Boulanger (2008)
    # Soil Liquefaction during Earthquakes
    alpha = -1.012 - 1.126 * math.sin(depth / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(depth / 11.28 + 5.142)
    rd = np.exp(alpha + beta * magnitude)

    # page 5
    CSR = 0.65 * acceleration * (sigma_vo / sigma_vo_eff) * rd

    # magnitude scaling factor
    # page 14 of Boulanger and Idriss (2014) CPT and
    # SPT liquefaction triggering
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
    Calculates the CRR based on the technique Robertson and Cabal (2022).
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
    using a magnitude scaling factor factor (MSF).

    Args:
        Qtn_cs [-]: normalised cone resistance with fines corrrection

    returns:
        CRR [-] : Cyclic resistance ratio (:math:`CRR`)
        K_sigma [-]: Overburden correction factor

    Reference - Robertson and Cabal (2022) CPT guide 7th edition

    """

    # page 113
    if Qtn_cs < 50:
        CRR = 0.833 * (Qtn_cs / 1000) + 0.05
    if Qtn_cs >= 50 and Qtn_cs < 160:
        CRR = 93 * (Qtn_cs / 1000) ** 3 + 0.08
    if Qtn_cs >= 160:
        CRR = np.inf

    # FoS liquefaction calculation standard input.
    # Since this parameter is not required for this methodology, it is assigned as 1
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
    Calculates the CRR based on the technique Robertson & Wride (1998).
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
        using a magnitude scaling factor factor (MSF).

    Args:
        Qtn_cs [-]: normalised cone resistance with fines corrrection
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        relative_density [-]: Relative density as a number between 0 and 1
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        CRR [-] : Cyclic resistance ratio (:math:`CRR`)
        K_sigma [-]: Overburden correction factor

    Reference - Robertson & Wride (1998)
        Evaluating cyclic liquefaction potential using cpt.
    """

    # Cyclic Resistance Ratio
    # page 452 - equation 8 of Robertson and Wride (1998)
    # Evaluating cyclic liquefaction potential usig the cone penetration test
    # there is a typo in original reference
    if Qtn_cs < 50:
        CRR = 0.833 * (Qtn_cs / 1000) + 0.05
    if Qtn_cs >= 50 and Qtn_cs < 160:
        CRR = 93 * (Qtn_cs / 1000) ** 3 + 0.08
    if Qtn_cs >= 160:
        CRR = np.inf

    # Overburden stress correction factor
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
    Calculates the CRR based on the technique Idriss and Boulanger (2008).
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
        using a magnitude scaling factor factor (MSF).

    Args:
        Qtn_cs [-]: normalised cone resistance with fines corrrection
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        CRR [-] : Cyclic resistance ratio (:math:`CRR`)
        K_sigma [-]: Overburden correction factor

    Reference - Idriss and Boulanger (2008)
    Soil liquefaction during earthquakes

    """

    # Cyclic Resistance Ratio
    # page 100 - equation 71 of Idriss and Boulanger (2008)
    #  Soil Liquefaction During Earthquakes
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
    # Overburden correction factor,
    # page 95 of Idriss and Boulanger (2008)
    # Soil Liquefaction During Earthquakes
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
    Calculates the CRR based on Boulanger and Idriss (2014)
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
    using a magnitude scaling factor factor (MSF).

    Args:
        Qtn_cs [-]: normalised cone resistance with fines corrrection
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        CRR [-] : Cyclic resistance ratio (:math:`CRR`)
        K_sigma [-]: Overburden correction factor

    Reference - Boulanger and Idriss (2014) CPT and SPT liquefaction triggering

    """
    # exponent m depending on grading
    # page 10, page 15 and page 21 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering

    # cyclic resistance ratio
    # page 17 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering

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

    # Overburden correction factor
    # page 11 of Boulanger and Idriss (2014)
    # CPT and SPT liquefaction triggering
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
    calculates the Qtn_cs based on Robertson & Cabal (2022)
    Args:
        ic [-]: soil type index
        Fr [-]: friction ratio
        qt [MPa]: total cone tip resistance
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        Qtn_cs [-]: normalised cone resistance with fines corrrection

    Reference - Robertson and Cabal (2022) CPT guide 7th edition

    """

    # page 116
    if ic > 1.7:
        if ic < 2.36 and Fr < 0.5:
            Kc = 1.0
        else:
            Kc = 15 - 14 / (1 + (ic / 2.95) ** 11)
    else:
        Kc = 1.0

    # recalculate the _Qtn in case qc (qt) is updated with SCF.

    def Qtn(
        qt,
        sigma_vo,
        sigma_vo_eff,
        n,
        pa=0.001 * atmospheric_pressure,
    ):
        return ((qt - 0.001 * sigma_vo) / pa) * ((pa / (0.001 * sigma_vo_eff)) ** n)

    def exponent_zhang(ic, sigma_vo_eff, pa=atmospheric_pressure):
        return min(1, 0.381 * ic + 0.05 * (sigma_vo_eff / pa) - 0.15)

    _exponent_zhang = exponent_zhang(ic, sigma_vo_eff)
    _Qtn = Qtn(qt, sigma_vo, sigma_vo_eff, _exponent_zhang)

    # page 115
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
    Calculates the Qtn_cs [-] based on the technique Robertson & Wride (1998).
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
        using a magnitude scaling factor factor (MSF).

    Args:
        sigma_vo [kPa]: Total vertical stress at the depth
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        qc [MPa]: Cone tip resistance
        Fr [MPa]: Normalised friction ratio as fraction
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        Qtn_cs [-]: normalised cone resistance with fines corrrection

    Reference - Robertson & Wride (1998)
        Evaluating cyclic liquefaction potential using the cpt.

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
    Calculates the Qtn_cs based on the technique Idriss and Boulanger (2008).
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
    using a magnitude scaling factor factor (MSF).

    Args:
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        qc [MPa]: Cone tip resistance
        ic [-]: Soil behaviour type index class number
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.

    returns:
        Qtn_cs [-]: normalised cone resistance with fines corrrection
        Fines [%]: Fines content

    Reference - Idriss and Boulanger (2008)
    Soil liquefaction during earthquakes

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
    Calculates the Qtn_cs based on Boulanger and Idriss (2014)
    Earthquake events are scaled to an equivalent event with magnitude of 7.5
        using a magnitude scaling factor factor (MSF).

    Args:
        sigma_vo_eff [kPa]: Effective vertical stress at the depth
        qc [MPa]: Cone tip resistance
        ic [-]: Soil behaviour type index class number
        atmospheric_pressure [kPa]: Reference atmospheric pressure.
            Default = 100.
        C_FC [-]: Fitting parameter for relation between fines content and ic

    returns:
        Qtn_cs [-]: normalised cone resistance with fines corrrection
        Fines [%]: Fines content

    Reference - Boulanger and Idriss (2014) CPT and SPT liquefaction triggering

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
    Calculates the liquefaction induced strains (gamma) for level ground and
        for lateral spreading based on the techniques by
        Zhang et al (2002) and Zhange et al (2004)
    Input requires factors of safety, relative density
        and normalized tip resistance.

    The parameter Qtn_cs is in principle the same to qc1N_cs and
        the latter can be used as input as well.

    :param FoS_liq: Safety factor against liquefaciton
    - Suggested range: 0.0 <= FoS <= 5.0
    :param relative_density: Relative density as a number between 0 and 1.5
    - Suggested range: 0.0 <= relative_density <= 2.0
    :param Qtn_cs': Normalized cone resistance corrected for fines

    :returns: Dictionary with the following keys:
    'volumetric strain [-]': volumetric strain as a fraction
    'lateral strain [-]': lateral strain as a fraction

    References:
    Zhang, Robertson and Brachmann (2002)
        Estimating liquefaction-induced ground settlements
        from CPT for level ground
    Zhang, Robertson and Brachmann (2004)
        Estimating liquefaction-induced lateral displacements
        using the standard penetration test or cone penetration test

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
