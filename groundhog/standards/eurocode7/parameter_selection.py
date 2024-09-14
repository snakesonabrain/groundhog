#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np
from scipy.stats import t

# Project imports
from groundhog.general.validation import Validator

CONSTANT_VALUE = {
    'data': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'mode': {'type': 'string', 'options': ('Low', 'Mean'), 'regex': None},
    'cov': {'type': 'float', 'min_value': 0, 'max_value': 10.0},
    'confidence': {'type': 'float', 'min_value': 0.1, 'max_value': 0.99999},
}

CONSTANT_VALUE_ERRORRETURN = {
    'n': np.nan,
    't_nminus1': np.nan,
    'kn': np.nan,
    'Xk': np.nan,
}


@Validator(CONSTANT_VALUE, CONSTANT_VALUE_ERRORRETURN)
def constant_value(data, mode='Low', cov=np.nan, confidence=0.95, **kwargs):
    """
    Selects the characteristic value from a set of measurements using Eurocode 7 rules.
    For a local low value, the 5% fractile is taken. For a mean value, a 95% confidence value or the mean is taken.
    The selection process assumes that the parameter under consideration is stationary and is normally distributed.
    For lognormally distributed parameters a transformation to the logarithm is required.

    :param data: List or numpy array with the measurements. The number of measurements is derived from the length.
    :param mode: Determines whether a local low value ``"Low"`` or mean vaue ``"Mean"`` needs to be taken
    :param cov: Coefficient of variation (:math:`V_x = \\sigma / \\mu`) (given as the ratio of standard deviation to the mean, not in percent). If CoV is unknown, leave blank.
    :param confidence: Confidence level used for calculations (default = 95%)

    .. math::
        X_k = X_{mean} \\cdot \\left( 1 - k_n \\cdot V_x \\right)

        V_x \\text{unknown}: k_{n,mean} = t_{n-1}^{0.95} \\sqrt{ \\frac{1}{n} }, \\ k_{n,low} = t_{n-1}^{0.95} \\sqrt{ \\frac{1}{n} + 1}

        V_x \\text{known}: k_{n,mean} = 1.64 \\sqrt{ \\frac{1}{n} }, \\ k_{n,low} = 1.64 \\sqrt{ \\frac{1}{n} + 1}

    :returns: Dictionary with the following keys:
        - 'n': Number of datapoints
        - 't_nminus1': Student-t factor
        - 'kn': kn value
        - 'Xk': Characteristic value of the parameter under consideration
    """
    n = data.__len__()
    r = n - 1

    t_nminus1 = t.ppf(confidence, r)

    if mode == 'Low':
        if np.isnan(cov):
            # COV unknown
            kn = t_nminus1 * np.sqrt((1 / n) + 1)
            cov = np.array(data).std() / np.array(data).mean()
        else:
            # COV known
            kn = 1.64 * np.sqrt((1 / n) + 1)
    elif mode == "Mean":
        if np.isnan(cov):
            # COV unknown
            kn = t_nminus1 * np.sqrt(1 / n)
            cov = np.array(data).std() / np.array(data).mean()
        else:
            # COV known
            kn = 1.64 * np.sqrt(1 / n)
    else:
        raise ValueError("Mode %s is not defined, use 'Low' or 'Mean'" % mode)

    Xk = np.array(data).mean() * (1 - kn * cov)

    return {
        'n': n,
        't_nminus1': t_nminus1,
        'kn': kn,
        'Xk': Xk,
    }

LINEAR_TREND = {
    'data': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'depths': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'requested_depths': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'mode': {'type': 'string', 'options': ('Low', 'Mean'), 'regex': None},
    'confidence': {'type': 'float', 'min_value': 0.1, 'max_value': 0.99999},
}

LINEAR_TREND_ERRORRETURN = {
    'n': np.nan,
    't_nminus2': np.nan,
    's1': np.nan,
    's2': np.nan,
    'b': np.nan,
    'Xk': None
}


@Validator(LINEAR_TREND, LINEAR_TREND_ERRORRETURN)
def linear_trend(data, depths, requested_depths, mode='Low', confidence=0.95, **kwargs):
    """
    Selects the characteristic value from a set of measurements using Eurocode 7 rules.
    A linear trend is assumed in the data.
    For a local low value, the 5% fractile is taken. For a mean value, a 95% confidence value or the mean is taken.
    The selection process assumes that the parameter under consideration is stationary and, when de-trended, is normally distributed.
    For lognormally distributed parameters a transformation to the logarithm is required.

    :param data: List or numpy array with the measurements. The number of measurements is derived from the length.
    :param depths: List or numpy array with the depths. The number of depths needs to be identical to the number of measurements.
    :param requested_depths: List or numpy array with the depths where the characteristic value is requested.
    :param mode: Determines whether a local low value ``"Low"`` or mean vaue ``"Mean"`` needs to be taken
    :param confidence: Confidence level used for calculations (default = 95%)

    .. math::
        x^{*} = \\bar{x} + b ( z - \\bar{z} )

        \\bar{x} = \\frac{1}{n} \\left( x_1 + x_2 + ... + x_n \\right)

        \\bar{z} = \\frac{1}{n} \\left( z_1 + z_2 + ... + z_n \\right)

        b = \\frac{\sum_{i=1}^n (x_i - \\bar{x}) (z_i - \\bar{z})}{\sum_{i=1}^n (z_i - \\bar{z})^2}

        \\text{Mean value}

        s_1 = \\sqrt{ \\frac{1}{n-2} \\left( \\frac{1}{n} + \\frac{(z- \\bar{z})^2}{\\sum_{i=1}^{n} (z_i - \\bar{z})^2} \\right) \\sum_{i=1}^n \\left[ (x_i - \\bar{x}) - b (z_i - \\bar{z}) \\right]^2 }

        X_k = \\left[ \\bar{x} + b (z - \\bar{z}) \\right] - t_{n-2}^{0.95} s_1

        \\text{Local low value}

        s_2 = \\sqrt{ \\frac{1}{n-2} \\left(1+ \\frac{1}{n} + \\frac{(z- \\bar{z})^2}{\\sum_{i=1}^{n} (z_i - \\bar{z})^2} \\right) \\sum_{i=1}^n \\left[ (x_i - \\bar{x}) - b (z_i - \\bar{z}) \\right]^2 }

        X_k = \\left[ \\bar{x} + b (z - \\bar{z}) \\right] - t_{n-2}^{0.95} s_2


    :returns: Dictionary with the following keys:
        - 'n': Number of datapoints
        - 't_nminus1': Student-t factor
        - 'kn': kn value
        - 'Xk': Characteristic values (Numpy array) of the parameter under consideration at the requested depths
    """
    n = data.__len__()
    r = n - 2

    t_nminus2 = t.ppf(confidence, r)

    data_mean = np.array(data).mean()
    depth_mean = np.array(depths).mean()

    b_array_nominator = np.cumsum((np.array(data) - data_mean) * (np.array(depths) - depth_mean))[-1]
    b_array_denominator = np.cumsum((np.array(depths) - depth_mean) ** 2)[-1]
    b = b_array_nominator / b_array_denominator

    final_part = np.cumsum(
        ((np.array(data) - data_mean) - b * (np.array(depths) - depth_mean)) ** 2)[-1]

    if mode == 'Low':
        s1 = np.nan
        s2 = np.sqrt(
            (1 / (n - 2)) *
            (1 + (1 / n) + (((np.array(requested_depths) - depth_mean) ** 2) / b_array_denominator)) *
            final_part
        )
        Xk = (data_mean + b * (np.array(requested_depths) - depth_mean)) - t_nminus2 * s2
    elif mode == "Mean":
        s1 = np.sqrt(
            (1 / (n - 2)) *
            ((1 / n) + (((np.array(requested_depths) - depth_mean) ** 2) / b_array_denominator)) *
            final_part
        )
        s2 = np.nan
        Xk = (data_mean + b * (np.array(requested_depths) - depth_mean)) - t_nminus2 * s1
    else:
        raise ValueError("Mode %s is not defined, use 'Low' or 'Mean'" % mode)

    return {
        'n': n,
        't_nminus2': t_nminus2,
        's1': s1,
        's2': s2,
        'b': b,
        'Xk': Xk
    }