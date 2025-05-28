__author__ = "Wouter Karreman"

import unittest

import numpy as np

from groundhog.soildynamics.cptliquefaction import (
    Qtn_cs_boulanger_idriss_2014,
    Qtn_cs_idriss_boulanger_2008,
    Qtn_cs_robertson_cabal_2022,
    Qtn_cs_robertson_wride_1998,
    crr_boulanger_idriss_2014,
    crr_idriss_boulanger_2008,
    crr_robertson_cabal_2022,
    crr_robertson_wride_1998,
    csr_boulanger_idriss_2014,
    csr_idriss_boulanger_2008,
    csr_robertson_cabal_2022,
    csr_robertson_wride_1998,
    fos_liquefaction,
    liquefaction_strains_zhang,
)


class TestCptLiquefaction(unittest.TestCase):
    """ """

    def setUp(self):
        pass

    def test_fos_liquefaction(self):
        # check if calculation is okay for normal case
        sigma_vo = 130
        sigma_vo_eff = 90
        CRR = 0.15
        CSR = 0.16
        MSF = 1.5
        K_sigma = 1.0

        out = fos_liquefaction(sigma_vo, sigma_vo_eff, CRR, CSR, MSF, K_sigma)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 1.41, 2)

        # check if calculation is okay for above water case
        sigma_vo = 130
        sigma_vo_eff = 130
        CRR = 0.15
        CSR = 0.16
        MSF = 1.5
        K_sigma = 1.0

        out = fos_liquefaction(sigma_vo, sigma_vo_eff, CRR, CSR, MSF, K_sigma)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 5, 0)

        # check if calculation is okay for high CRR case
        sigma_vo = 130
        sigma_vo_eff = 130
        CRR = 0.5
        CSR = 0.16
        MSF = 1.5
        K_sigma = 1.0

        out = fos_liquefaction(sigma_vo, sigma_vo_eff, CRR, CSR, MSF, K_sigma)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 5, 0)

        # check if calculation is okay for error
        sigma_vo = np.nan
        sigma_vo_eff = np.nan
        CRR = 0.5
        CSR = 0
        MSF = np.nan
        K_sigma = 1.0

        out = fos_liquefaction(sigma_vo, sigma_vo_eff, CRR, CSR, MSF, K_sigma)
        assert np.isnan(out.get("FoS_liq [-]"))

    def test_csr_robertson_cabal_2022(self):
        # check if calculation is okay for depth < 9.15
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_cabal_2022(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.178, 3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.772, 3)

        # check if calculation is okay for depth > 9.15 and < 23 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 20
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_cabal_2022(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.120, 3)

        # check if calculation is okay for depth > 23 and < 30 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 25
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_cabal_2022(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.102, 3)

        # check if calculation is okay for depth > 30 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 31
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_cabal_2022(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.094, 3)

        # check if calculation is okay for error
        sigma_vo = np.nan
        sigma_vo_eff = 90
        depth = 31
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_cabal_2022(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        assert np.isnan(out.get("CSR [-]"))

    def test_csr_robertson_wride_1998(self):
        # check if calculation is okay for depth < 9.15
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_wride_1998(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.178, 3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.770, 3)

        # check if calculation is okay for depth > 9.15 and < 23 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 20
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_wride_1998(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.120, 3)

        # check if calculation is okay for depth > 23 and < 30 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 25
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_wride_1998(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.102, 3)

        # check if calculation is okay for depth > 30 m
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 31
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_wride_1998(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.094, 3)

        # check if calculation is okay for error (nan for sigma_vo)
        sigma_vo = np.nan
        sigma_vo_eff = 90
        depth = 31
        magnitude = 6.0
        acceleration = 0.20

        out = csr_robertson_wride_1998(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        assert np.isnan(out.get("CSR [-]"))

    def test_csr_idriss_boulanger_2008(self):
        # check if calculation is okay
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 6.0
        acceleration = 0.20

        out = csr_idriss_boulanger_2008(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.164, 3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.482, 3)

        # check if calculation with error (low magnitude)
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 5.0
        acceleration = 0.20

        out = csr_idriss_boulanger_2008(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        assert np.isnan(out.get("CSR [-]"))

    def test_csr_boulanger_idriss_2014(self):
        # check if calculation is okay
        Qtn_cs = 90
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 6.0
        acceleration = 0.20

        out = csr_boulanger_idriss_2014(
            Qtn_cs, sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("CSR [-]"), 0.164, 3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.130, 3)

        # check if calculation is okay with high Qtn_cs
        Qtn_cs = 400
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 6.0
        acceleration = 0.20

        out = csr_boulanger_idriss_2014(
            Qtn_cs, sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        self.assertAlmostEqual(out.get("MSF [-]"), 1.723, 3)

        # check if calculation with error (low magnitude)
        sigma_vo = 130
        sigma_vo_eff = 90
        depth = 7
        magnitude = 5.0
        acceleration = 0.20

        out = csr_boulanger_idriss_2014(
            sigma_vo, sigma_vo_eff, depth, magnitude, acceleration
        )
        assert np.isnan(out.get("CSR [-]"))

    def test_crr_robertson_cabal_2022(self):
        # check if calculation is okay for Qtn_cs < 50
        Qtn_cs = 45

        out = crr_robertson_cabal_2022(Qtn_cs)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, 4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1, 0)

        # check if calculation is okay for Qtn_cs >50 and <160
        Qtn_cs = 120

        out = crr_robertson_cabal_2022(Qtn_cs)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.2407, 4)

        # check if calculation is okay for Qtn_cs > 160
        Qtn_cs = 180

        out = crr_robertson_cabal_2022(Qtn_cs)
        assert np.isinf(out.get("CRR [-]"))

        # check if calculation is okay for error (negative Qtn_cs)
        Qtn_cs = -5

        out = crr_robertson_cabal_2022(Qtn_cs)
        assert np.isnan(out.get("CRR [-]"))

    def test_crr_robertson_wride_1998(self):
        # check if calculation is okay for Qtn_cs < 50 and sigma_vo_eff < 100
        Qtn_cs = 45
        sigma_vo_eff = 90
        relative_density = 0.6

        out = crr_robertson_wride_1998(Qtn_cs, sigma_vo_eff, relative_density)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, 4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1, 0)

        # check if calculation is okay for Qtn_cs >50 and <160 and sigma_vo_eff < 100
        Qtn_cs = 120
        sigma_vo_eff = 90
        relative_density = 0.6

        out = crr_robertson_wride_1998(Qtn_cs, sigma_vo_eff, relative_density)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.2407, 4)

        # check if calculation is okay for Qtn_cs >160 and sigma_vo_eff < 100
        Qtn_cs = 180
        sigma_vo_eff = 90
        relative_density = 0.6

        out = crr_robertson_wride_1998(Qtn_cs, sigma_vo_eff, relative_density)
        assert np.isinf(out.get("CRR [-]"))

        # check if calculation is okay for sigma_vo_eff > 100
        Qtn_cs = 45
        sigma_vo_eff = 150
        relative_density = 0.6

        out = crr_robertson_wride_1998(Qtn_cs, sigma_vo_eff, relative_density)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, 4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 0.8855, 4)

        # check if calculation is okay for error (negative Qtn_cs)
        Qtn_cs = -5
        sigma_vo_eff = 150
        relative_density = 0.6

        out = crr_robertson_wride_1998(Qtn_cs, sigma_vo_eff, relative_density)
        assert np.isnan(out.get("CRR [-]"))

    def test_crr_idriss_boulanger_2008(self):
        # check if calculation is okay
        Qtn_cs = 45
        sigma_vo_eff = 90

        out = crr_idriss_boulanger_2008(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("CRR [-]"), 0.0729, 4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.0072, 4)

        # check if calculation is okay for low pressure
        Qtn_cs = 45
        sigma_vo_eff = 10

        out = crr_idriss_boulanger_2008(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.1000, 4)

        # check if calculation is okay for high Qtn_cs
        Qtn_cs = 500
        sigma_vo_eff = 90

        out = crr_idriss_boulanger_2008(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("CRR [-]"), 0.6000, 4)

        # check if calculation is okay for error (negative Qtn_cs)
        Qtn_cs = -5
        sigma_vo_eff = 150

        out = crr_idriss_boulanger_2008(Qtn_cs, sigma_vo_eff)
        assert np.isnan(out.get("CRR [-]"))

    def test_crr_boulanger_idriss_2014(self):
        # check if calculation is okay
        Qtn_cs = 45
        sigma_vo_eff = 90

        out = crr_boulanger_idriss_2014(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("CRR [-]"), 0.0888, 4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.0072, 4)

        # check if calculation is okay for low pressure
        Qtn_cs = 45
        sigma_vo_eff = 10

        out = crr_boulanger_idriss_2014(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.1000, 4)

        # check if calculation is okay for high Qtn_cs
        Qtn_cs = 500
        sigma_vo_eff = 90

        out = crr_boulanger_idriss_2014(Qtn_cs, sigma_vo_eff)

        self.assertAlmostEqual(out.get("CRR [-]"), 0.6000, 4)

        # check if calculation is okay for error (negative Qtn_cs)
        Qtn_cs = -5
        sigma_vo_eff = 150

        out = crr_boulanger_idriss_2014(Qtn_cs, sigma_vo_eff)
        assert np.isnan(out.get("CRR [-]"))

    def test_Qtn_cs_robertson_cabal_2022(self):
        # check if calculation is okay for low ic
        ic = 1.4
        Fr = 2
        qt = 20
        sigma_vo_eff = 100
        sigma_vo = 130

        out = Qtn_cs_robertson_cabal_2022(ic, Fr, qt, sigma_vo_eff, sigma_vo)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 198.7, 2)

        # check if calculation is okay for ic > 1.7 but < 3.0
        ic = 2.1
        Fr = 2
        qt = 20
        sigma_vo_eff = 100
        sigma_vo = 130

        out = Qtn_cs_robertson_cabal_2022(ic, Fr, qt, sigma_vo_eff, sigma_vo)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 263.3377, 4)

        # check if calculation is okay for ic < 2.36 and Fr < 0.5
        ic = 2.1
        Fr = 0.4
        qt = 20
        sigma_vo_eff = 100
        sigma_vo = 130

        out = Qtn_cs_robertson_cabal_2022(ic, Fr, qt, sigma_vo_eff, sigma_vo)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 198.7, 2)

        # check if calculation gives error if outside the range
        ic = 3.5
        Fr = 2
        qt = 20
        sigma_vo_eff = 100
        sigma_vo = 130

        out = Qtn_cs_robertson_cabal_2022(ic, Fr, qt, sigma_vo_eff, sigma_vo)
        assert np.isnan(out.get("Qtn_cs [-]"))

    def test_Qtn_cs_robertson_wride_1998(self):
        # check if calculation is okay for clean sand
        sigma_vo = 130
        ic = 1.4
        qc = 7
        Fr = 0.5

        out = Qtn_cs_robertson_wride_1998(sigma_vo, ic, qc, Fr)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 61.3941, 4)

        # check if calculation is okay for ic > 1.64
        sigma_vo = 130
        ic = 2.1
        qc = 7
        Fr = 2

        out = Qtn_cs_robertson_wride_1998(sigma_vo, ic, qc, Fr)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 89.3134, 4)

        # check if calculation gives error if outside the range
        sigma_vo = 130
        ic = 4.0
        qc = 7
        Fr = 2

        out = Qtn_cs_robertson_wride_1998(sigma_vo, ic, qc, Fr)
        assert np.isnan(out.get("Qtn_cs [-]"))

    def test_Qtn_cs_idriss_boulanger_2008(self):
        # check if calculation is okay
        sigma_vo_eff = 90
        ic = 1.4
        qc = 7

        out = Qtn_cs_idriss_boulanger_2008(sigma_vo_eff, qc, ic)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 75.1971, 4)
        self.assertAlmostEqual(out.get("Fines [%]"), 6.7157, 4)

        # check if calculation gives error outside range
        sigma_vo_eff = 90
        ic = 6
        qc = 7

        out = Qtn_cs_idriss_boulanger_2008(sigma_vo_eff, qc, ic)
        assert np.isnan(out.get("Qtn_cs [-]"))

    def test_Qtn_cs_boulanger_idriss_2014(self):
        # check if calculation is okay for no fines
        sigma_vo_eff = 90
        ic = 1.4
        qc = 7
        C_FC = 0

        out = Qtn_cs_boulanger_idriss_2014(sigma_vo_eff, qc, ic, C_FC)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 74.2666, 4)
        self.assertAlmostEqual(out.get("Fines [%]"), 0, 0)

        # check if calculation is okay for higher fines
        sigma_vo_eff = 90
        ic = 1.8
        qc = 7
        C_FC = 0

        out = Qtn_cs_boulanger_idriss_2014(sigma_vo_eff, qc, ic, C_FC)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 75.6441, 4)
        self.assertAlmostEqual(out.get("Fines [%]"), 7.0, 1)

        # check if calculation is okay for higher fines and C_FC
        sigma_vo_eff = 90
        ic = 1.8
        qc = 7
        C_FC = 0.29

        out = Qtn_cs_boulanger_idriss_2014(sigma_vo_eff, qc, ic, C_FC)

        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 123.7922, 4)
        self.assertAlmostEqual(out.get("Fines [%]"), 30.2, 1)

        # check if calculation gives error outside range
        sigma_vo_eff = 90
        ic = 5.5
        qc = 7
        C_FC = 0

        out = Qtn_cs_boulanger_idriss_2014(sigma_vo_eff, qc, ic, C_FC)
        assert np.isnan(out.get("Qtn_cs [-]"))

    def test_liquefaction_strains_zhang(self):
        # check if calculation is okay for high FoS_liq
        FoS_liq = 5.0
        relative_density = 0.9
        Qtn_cs = 100

        out = liquefaction_strains_zhang(FoS_liq, relative_density, Qtn_cs)

        self.assertAlmostEqual(out.get("eps_liq [%]"), 0, 0)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 0, 0)

        # check if calculation is okay for low FoS_liq
        FoS_liq = 0.5
        relative_density = 0.3
        Qtn_cs = 35

        out = liquefaction_strains_zhang(FoS_liq, relative_density, Qtn_cs)

        self.assertAlmostEqual(out.get("eps_liq [%]"), 5.53, 2)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 51.2, 1)

        # check if calculation is okay for low intermediate FoS_liq
        FoS_liq = 0.87
        relative_density = 0.4
        Qtn_cs = 120

        out = liquefaction_strains_zhang(FoS_liq, relative_density, Qtn_cs)

        self.assertAlmostEqual(out.get("eps_liq [%]"), 1.20, 2)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 10.3, 1)

        # check if calculation gives error outside range
        FoS_liq = 0.87
        relative_density = -0.4
        Qtn_cs = 120

        out = liquefaction_strains_zhang(FoS_liq, relative_density, Qtn_cs)
        assert np.isnan(out.get("eps_liq [%]"))
