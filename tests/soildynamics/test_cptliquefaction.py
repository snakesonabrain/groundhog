__author__ = "Wouter Karreman and Ping Li"

import unittest

import numpy as np

from vo_geotools.seismic.cptliquefaction import (
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
        # Normal case
        out = fos_liquefaction(130, 90, 0.15, 0.16, 1.5, 1.0)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 1.41, places=2)

        # Above water case
        out = fos_liquefaction(130, 130, 0.15, 0.16, 1.5, 1.0)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 5, places=0)

        # High CRR case
        out = fos_liquefaction(130, 130, 0.5, 0.16, 1.5, 1.0)
        self.assertAlmostEqual(out.get("FoS_liq [-]"), 5, places=0)

        # Error case (nan)
        out = fos_liquefaction(np.nan, np.nan, 0.5, 0, np.nan, 1.0)
        self.assertTrue(np.isnan(out.get("FoS_liq [-]")))

    def test_csr_robertson_cabal_2022(self):
        # depth < 9.15
        out = csr_robertson_cabal_2022(130, 90, 7, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.178, places=3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.772, places=3)

        # 9.15 < depth < 23
        out = csr_robertson_cabal_2022(130, 90, 20, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.120, places=3)

        # 23 < depth < 30
        out = csr_robertson_cabal_2022(130, 90, 25, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.102, places=3)

        # depth > 30
        out = csr_robertson_cabal_2022(130, 90, 31, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.094, places=3)

        # Error case (nan)
        out = csr_robertson_cabal_2022(np.nan, 90, 31, 6.0, 0.20)
        self.assertTrue(np.isnan(out.get("CSR [-]")))

    def test_csr_robertson_wride_1998(self):
        # depth < 9.15
        out = csr_robertson_wride_1998(130, 90, 7, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.178, places=3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.770, places=3)

        # 9.15 < depth < 23
        out = csr_robertson_wride_1998(130, 90, 20, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.120, places=3)

        # 23 < depth < 30
        out = csr_robertson_wride_1998(130, 90, 25, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.102, places=3)

        # depth > 30
        out = csr_robertson_wride_1998(130, 90, 31, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.094, places=3)

        # Error case (nan)
        out = csr_robertson_wride_1998(np.nan, 90, 31, 6.0, 0.20)
        self.assertTrue(np.isnan(out.get("CSR [-]")))

    def test_csr_idriss_boulanger_2008(self):
        # Normal case
        out = csr_idriss_boulanger_2008(130, 90, 7, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.164, places=3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.482, places=3)

        # Error case (low magnitude)
        out = csr_idriss_boulanger_2008(130, 90, 7, 5.0, 0.20)
        self.assertTrue(np.isnan(out.get("CSR [-]")))

    def test_csr_boulanger_idriss_2014(self):
        # Normal case
        out = csr_boulanger_idriss_2014(90, 130, 90, 7, 6.0, 0.20)
        self.assertAlmostEqual(out.get("CSR [-]"), 0.164, places=3)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.130, places=3)

        # High Qtn_cs
        out = csr_boulanger_idriss_2014(400, 130, 90, 7, 6.0, 0.20)
        self.assertAlmostEqual(out.get("MSF [-]"), 1.723, places=3)

        # Error case (low magnitude)
        out = csr_boulanger_idriss_2014(130, 90, 7, 5.0, 0.20)
        self.assertTrue(np.isnan(out.get("CSR [-]")))

    def test_crr_robertson_cabal_2022(self):
        # Qtn_cs < 50
        out = crr_robertson_cabal_2022(45)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, places=4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1, places=0)

        # 50 < Qtn_cs < 160
        out = crr_robertson_cabal_2022(120)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.2407, places=4)

        # Qtn_cs > 160
        out = crr_robertson_cabal_2022(180)
        self.assertTrue(np.isinf(out.get("CRR [-]")))

        # Error case (negative Qtn_cs)
        out = crr_robertson_cabal_2022(-5)
        self.assertTrue(np.isnan(out.get("CRR [-]")))

    def test_crr_robertson_wride_1998(self):
        # Qtn_cs < 50, sigma_vo_eff < 100
        out = crr_robertson_wride_1998(45, 90, 0.6)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, places=4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1, places=0)

        # 50 < Qtn_cs < 160, sigma_vo_eff < 100
        out = crr_robertson_wride_1998(120, 90, 0.6)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.2407, places=4)

        # Qtn_cs > 160, sigma_vo_eff < 100
        out = crr_robertson_wride_1998(180, 90, 0.6)
        self.assertTrue(np.isinf(out.get("CRR [-]")))

        # sigma_vo_eff > 100
        out = crr_robertson_wride_1998(45, 150, 0.6)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0875, places=4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 0.8855, places=4)

        # Error case (negative Qtn_cs)
        out = crr_robertson_wride_1998(-5, 150, 0.6)
        self.assertTrue(np.isnan(out.get("CRR [-]")))

    def test_crr_idriss_boulanger_2008(self):
        # Normal case
        out = crr_idriss_boulanger_2008(45, 90)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0729, places=4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.0072, places=4)

        # Low pressure
        out = crr_idriss_boulanger_2008(45, 10)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.1000, places=4)

        # High Qtn_cs
        out = crr_idriss_boulanger_2008(500, 90)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.6000, places=4)

        # Error case (negative Qtn_cs)
        out = crr_idriss_boulanger_2008(-5, 150)
        self.assertTrue(np.isnan(out.get("CRR [-]")))

    def test_crr_boulanger_idriss_2014(self):
        # Normal case
        out = crr_boulanger_idriss_2014(45, 90)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.0888, places=4)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.0072, places=4)

        # Low pressure
        out = crr_boulanger_idriss_2014(45, 10)
        self.assertAlmostEqual(out.get("K_sigma [-]"), 1.1000, places=4)

        # High Qtn_cs
        out = crr_boulanger_idriss_2014(500, 90)
        self.assertAlmostEqual(out.get("CRR [-]"), 0.6000, places=4)

        # Error case (negative Qtn_cs)
        out = crr_boulanger_idriss_2014(-5, 150)
        self.assertTrue(np.isnan(out.get("CRR [-]")))

    def test_Qtn_cs_robertson_cabal_2022(self):
        # Low ic
        out = Qtn_cs_robertson_cabal_2022(1.4, 2, 20, 100, 130)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 198.7, places=2)

        # 1.7 < ic < 3.0
        out = Qtn_cs_robertson_cabal_2022(2.1, 2, 20, 100, 130)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 263.3377, places=4)

        # ic < 2.36 and Fr < 0.5
        out = Qtn_cs_robertson_cabal_2022(2.1, 0.4, 20, 100, 130)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 198.7, places=2)

        # Error case (ic out of range)
        out = Qtn_cs_robertson_cabal_2022(3.5, 2, 20, 100, 130)
        self.assertTrue(np.isnan(out.get("Qtn_cs [-]")))

    def test_Qtn_cs_robertson_wride_1998(self):
        # Clean sand
        out = Qtn_cs_robertson_wride_1998(130, 1.4, 7, 0.5)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 61.3941, places=4)

        # ic > 1.64
        out = Qtn_cs_robertson_wride_1998(130, 2.1, 7, 2)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 89.3134, places=4)

        # Error case (ic out of range)
        out = Qtn_cs_robertson_wride_1998(130, 4.0, 7, 2)
        self.assertTrue(np.isnan(out.get("Qtn_cs [-]")))

    def test_Qtn_cs_idriss_boulanger_2008(self):
        # Normal case
        out = Qtn_cs_idriss_boulanger_2008(90, 7, 1.4)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 75.1971, places=4)
        self.assertAlmostEqual(out.get("Fines [%]"), 6.7157, places=4)

        # Error case (ic out of range)
        out = Qtn_cs_idriss_boulanger_2008(90, 7, 6)
        self.assertTrue(np.isnan(out.get("Qtn_cs [-]")))

    def test_Qtn_cs_boulanger_idriss_2014(self):
        # No fines
        out = Qtn_cs_boulanger_idriss_2014(90, 7, 1.4, 0)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 74.2666, places=4)
        self.assertAlmostEqual(out.get("Fines [%]"), 0, places=0)

        # Higher fines
        out = Qtn_cs_boulanger_idriss_2014(90, 7, 1.8, 0)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 75.6441, places=4)
        self.assertAlmostEqual(out.get("Fines [%]"), 7.0, places=1)

        # Higher fines and C_FC
        out = Qtn_cs_boulanger_idriss_2014(90, 7, 1.8, 0.29)
        self.assertAlmostEqual(out.get("Qtn_cs [-]"), 123.7922, places=4)
        self.assertAlmostEqual(out.get("Fines [%]"), 30.2, places=1)

        # Error case (ic out of range)
        out = Qtn_cs_boulanger_idriss_2014(90, 7, 5.5, 0)
        self.assertTrue(np.isnan(out.get("Qtn_cs [-]")))

    def test_liquefaction_strains_zhang(self):
        # High FoS_liq
        out = liquefaction_strains_zhang(5.0, 0.9, 100)
        self.assertAlmostEqual(out.get("eps_liq [%]"), 0, places=0)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 0, places=0)

        # Low FoS_liq
        out = liquefaction_strains_zhang(0.5, 0.3, 35)
        self.assertAlmostEqual(out.get("eps_liq [%]"), 5.53, places=2)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 51.2, places=1)

        # Intermediate FoS_liq
        out = liquefaction_strains_zhang(0.87, 0.4, 120)
        self.assertAlmostEqual(out.get("eps_liq [%]"), 1.20, places=2)
        self.assertAlmostEqual(out.get("gamma_liq [%]"), 10.3, places=1)

        # Error case (negative relative_density)
        out = liquefaction_strains_zhang(0.87, -0.4, 120)
        self.assertTrue(np.isnan(out.get("eps_liq [%]")))
