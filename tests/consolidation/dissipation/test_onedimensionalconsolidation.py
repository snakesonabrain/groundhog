# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.consolidation.dissipation import onedimensionalconsolidation


class Test_onedimensionalsolutions(unittest.TestCase):

    def test_fourierseries(self):
        result = onedimensionalconsolidation.pore_pressure_fourier(
            delta_u_0=50,
            depths=np.linspace(0, 1, 50),
            time=1e4,
            cv=100,
            layer_thickness=1
        )
        self.assertAlmostEqual(
            np.interp(0.5, np.linspace(0, 1, 50), result['delta u [kPa]']),
            45.3, 1)

    def test_consolidation_degree(self):
        result = onedimensionalconsolidation.consolidation_degree(
            time=1e4,
            cv=100,
            drainage_length=0.5
        )
        self.assertAlmostEqual(
            result['U [pct]'],
            40.2,
            1
        )
        result_triangular = onedimensionalconsolidation.consolidation_degree(
            time=1e4,
            cv=100,
            drainage_length=0.5,
            distribution="triangular"
        )
        self.assertAlmostEqual(
            result_triangular['U [pct]'],
            24.8,
            1
        )

    def test_fd_doubledrainage(self):
        calc = onedimensionalconsolidation.ConsolidationCalculation(
            height=1, total_time=1e4, no_nodes=21)
        calc.set_cv(cv=np.array([100, 100]), cv_depths=np.array([0, 1]), uniform=False)
        calc.set_top_boundary(freedrainage=True)
        calc.set_bottom_boundary(freedrainage=True)
        calc.set_initial(u0=[50, 50], u0_depths=[0, 1])
        calc.set_output_times(output_times=[1, 10, 100, 1000, 10000])
        calc.calculate()
        self.assertAlmostEqual(
            np.interp(0.5, calc.z, calc.u_steps[-1]),
            45.23,
            2
        )

    def test_fd_doubledrainage_cv_as_list(self):
        calc = onedimensionalconsolidation.ConsolidationCalculation(
            height=1, total_time=1e4, no_nodes=21)
        calc.set_cv(cv=[100, 100], cv_depths=np.array([0, 1]), uniform=False)
        calc.set_top_boundary(freedrainage=True)
        calc.set_bottom_boundary(freedrainage=True)
        calc.set_initial(u0=[50, 50], u0_depths=[0, 1])
        calc.set_output_times(output_times=[1, 10, 100, 1000, 10000])
        calc.calculate()
        self.assertAlmostEqual(
            np.interp(0.5, calc.z, calc.u_steps[-1]),
            45.23,
            2
        )

    def test_fd_topdrainage(self):
        calc = onedimensionalconsolidation.ConsolidationCalculation(
            height=1, total_time=1e4, no_nodes=21)
        calc.set_cv(cv=np.array([100, 100]), cv_depths=np.array([0, 1]), uniform=False)
        calc.set_top_boundary(freedrainage=True)
        calc.set_bottom_boundary(freedrainage=False)
        calc.set_initial(u0=[0, 50], u0_depths=[0, 1])
        calc.set_output_times(output_times=[0, 2e3, 4e3, 6e3, 8e3, 1e4])
        calc.calculate()
        self.assertAlmostEqual(
            np.interp(1, calc.z, calc.u_steps[-1]),
            39.98,
            2
        )

    def test_fd_bottomdrainage(self):
        calc = onedimensionalconsolidation.ConsolidationCalculation(
            height=1, total_time=1e4, no_nodes=21)
        calc.set_cv(cv=np.array([100, 100]), cv_depths=np.array([0, 1]), uniform=False)
        calc.set_top_boundary(freedrainage=False)
        calc.set_bottom_boundary(freedrainage=True)
        calc.set_initial(u0=[50, 0], u0_depths=[0, 1])
        calc.set_output_times(output_times=[0, 2e3, 4e3, 6e3, 8e3, 1e4])
        calc.calculate()
        self.assertAlmostEqual(
            np.interp(0, calc.z, calc.u_steps[-1]),
            39.98,
            2
        )