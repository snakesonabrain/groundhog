#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

UNDERCOMPACTION_COHESIONLESS_LADD = {
    'sample_height': {'type': 'float', 'min_value': 0.0, 'max_value': 1.0},
    'no_layers': {'type': 'int', 'min_value': 1.0, 'max_value': 10.0},
    'undercompaction_deepest': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'undercompaction_shallowest': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
}

UNDERCOMPACTION_COHESIONLESS_LADD_ERRORRETURN = {
    'U [-]': None,
    'h [m]': None,
}


@Validator(UNDERCOMPACTION_COHESIONLESS_LADD, UNDERCOMPACTION_COHESIONLESS_LADD_ERRORRETURN)
def undercompaction_cohesionless_ladd(
        sample_height, no_layers, undercompaction_deepest, undercompaction_shallowest=0,
        **kwargs):
    """
    When soil sample have to be reconstituted to a specific relative density,  the sample is generally prepared in several layers of equal mass and tamping or vibration is used to obtain the desired volume in the sample mould.

    If each layer is however compacted to the desired relative density, a non-uniform density profile will be obtained as the lower layers will still be compacted to a certain degree by the tamping or vibration on the upper layers.

    To address this shortcoming, the undercompaction method is used (Ladd, 1978) which compacts the deeper layers to a lesser degree than the higher layers. The degree of undercompaction of the deepest layer is chosen based on experience (e.g. from density profiling using core loggers) and the undercompaction degree is then chosen to vary linearly from the bottom to the top of the sample.

    This function calculates the undercompaction degrees for each layer (using a linear variation) and the height to the top of each layer for the given undercompaction degrees.

    :param sample_height: Total height of the sample (:math:`H_0`) [:math:`m`] - Suggested range: 0.0 <= sample_height <= 1.0
    :param no_layers: Number of layers for the sample (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= no_layers <= 10.0
    :param undercompaction_deepest: Chosen undercompaction degree of the deepest layer (:math:`U_1`) [:math:`pct`] - Suggested range: 0.0 <= undercompaction_deepest <= 10.0
    :param undercompaction_shallowest: Chosen undercompaction degree of the shallowest layer (:math:`U_N`) [:math:`pct`] (default=0pct) - Suggested range: 0.0 <= undercompaction_deepest <= 10.0

    .. math::
        U_i = U_1 - \\left[ \\frac{U_1 - U_N}{N - 1} \\cdot (i - 1) \\right]

        h_i = \\frac{H_0}{N} \\left[ (i - 1) + (1 + U_i) \\right]

    :returns: Dictionary with the following keys:

        - 'U [-]': Undercompaction degrees of each layer starting from the deepest layer (:math:`U`)  [:math:`-`]
        - 'h [m]': Height to the top of each layer (:math:`h`)  [:math:`m`]

    .. figure:: images/undercompaction_cohesionless_ladd_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Density profiles of triaxial silt samples using different undercompaction degrees

    Reference - R. Ladd, "Preparing Test Specimens Using Undercompaction," Geotechnical Testing Journal 1, no. 1 (1978): 16-23. https://doi.org/10.1520/GTJ10364J

    """

    # Convert to dimensionless
    undercompaction_deepest = 0.01 * undercompaction_deepest
    undercompaction_shallowest = 0.01 * undercompaction_shallowest

    _layer_no = np.linspace(1, no_layers, no_layers)

    _U = []
    for i, _n in enumerate(_layer_no):
        Ui = undercompaction_deepest - (
                ((undercompaction_deepest - undercompaction_shallowest) /
                 (no_layers - 1)) * i)  # Note: Python uses zero-indexing
        _U.append(Ui)

    _h = []
    for i, _u in enumerate(_U):
        hi = (sample_height / no_layers) * (i + (1 + _u))  # Note: Python uses zero-indexing
        _h.append(hi)

    return {
        'U [-]': _U,
        'h [m]': _h,
    }