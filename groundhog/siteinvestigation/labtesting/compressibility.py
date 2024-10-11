#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings

# 3rd party packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root

# Project imports
from groundhog.general.validation import Validator

def selectpoints(nopoints, timeout=60):
    return plt.ginput(nopoints, timeout=timeout)

ROOTTIMEMETHOD = {
    'times': {'type': 'list', 'elementtype': 'float', 'order': 'ascending', 'unique': True, 'empty_allowed': False},
    'settlements': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'drainagelength': {'type': 'float', 'min_value': 0, 'max_value': None},
    'initialguess_override': {'type': 'float', 'min_value': 0.0, 'max_value': None}
}

ROOTTIMEMETHOD_ERRORRETURN = {
    't90 [s]': np.nan,
    'cv [m2/s]': np.nan,
    'cv [m2/yr]': np.nan,
    'A': None,
    'B': None,
    'plot': None
}

@Validator(ROOTTIMEMETHOD, ROOTTIMEMETHOD_ERRORRETURN)
def roottimemethod(times, settlements, drainagelength, initialguess_override=np.nan, xrange=(0, 100), showfig=True, **kwargs):
    """
    Calculates the root-time construction for determining the coefficient of consolidation for an oedometer test
    (or any other soil mechanical test involving consolidation).

    The following procedure is applied:

    #. Plot the displacement gage readings versus square root of times.
    #. Draw the best straight line through the initial part of the curve intersecting the ordinate (displacement reading) at :math:`O` and the abscissa (:math:`\\sqrt{\\text{time}}`) at  :math:`A`.
    #. Note the time at point :math:`A`; let us say it is :math:`\sqrt{t_A}`.
    #. Locate a point :math:`B`, :math:`1.15 \\sqrt{t_A}`, on the abscissa.
    #. Join :math:`OB`.
    #. The intersection of the line :math:`OB` with the curve, point :math:`C`, gives the displacement gage reading and the time for 90% consolidation (:math:`t_{90}`). You should note that the value read off the abscissa is :math:`\\sqrt{t_{90}}`. Now when :math:`U` = 90%,  :math:`T_v` = 0.848 and from one-dimensional consolidation equation, we obtain:
   
    .. math::
        c_v = \\frac{0.848 H_{dr}^2}{t_{90}}

    Because the construction relies heavily on the laboratory data and the judgement of the user, a semi-automated procedure is followed in which the user selects
    the origin :math:`O` and point :math:`A` in an interactive matplotlib plot. Any notebook needs to ensure that matplotlib plots are generated with the ``qt`` backend using the following magic command:

    .. code-block:: python

        %matplotlib qt

    The following input parameters are expected:

    :param times: Array with time values in seconds, increasing from 0s at the start of the test
    :param settlements: Array with settlement values, increasing from 0 at the origin. The units are not important as only the time for 90% consolidation is determined.
    :param drainagelength: Drainage length for the consolidation (:math:`H_{dr}`) [m] - Suggested range: drainagelength > 0
    :param initialguess_override: Override for the initial guess for :math:`\\sqrt{t_{90}}`, default=np.nan

    .. figure:: images/root_time.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Root-time construction

    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.
    """
    # Coerce the times and settlements to Numpy arrays
    times = np.array(times)
    settlements = np.array(settlements)

    settlement_abcis = 1.1 * settlements.max()

    # Close all open plots
    plt.close('all')

    # Generate the plot
    if showfig:
        plt.figure(1, figsize=(12,6))
        plt.subplot(111)
        plt.plot(np.sqrt(times), settlements, label='Root-time data')
        plt.xlabel('$ \sqrt{t} $', size=15)
        plt.ylabel('$ \Delta z $ [mm]', size=15)
        plt.xlim(xrange)
        plt.ylim((settlement_abcis, 0))
        plt.grid()
        plt.show()

    print("First step: Select the position of the origin O")
    print("Second step: Select a point on the initial straight portion OA")
    # Get the input from the user
    xy = selectpoints(2)

    # Interpolate the selection
    pointO = (xy[0][0], np.interp(xy[0][0], np.sqrt(times), settlements))
    pointD = (xy[1][0], np.interp(xy[1][0], np.sqrt(times), settlements))

    # Create the interpolation function
    roottime_interpolation_func = interp1d([pointO[1], pointD[1]], [pointO[0], pointD[0]], fill_value='extrapolate')

    # Calculate the position of the point A
    pointA = (roottime_interpolation_func(settlement_abcis), settlement_abcis)

    # Calculate the position of the point B
    pointB = (1.15 * pointA[0], settlement_abcis)

    # Find the intersection of OB and the data
    OB_func = interp1d([pointO[0], pointB[0]], [pointO[1], pointB[1]], fill_value='extrapolate')
    data_func = interp1d(np.sqrt(times), settlements)

    def intersection(x):
        return OB_func(x) - data_func(x)
    
    if np.isnan(initialguess_override):
        initial_guess = pointA[0]
    else:
        initial_guess = initialguess_override

    sqrt_t90 = root(intersection, initial_guess).x[0]

    # Determine cv from the calculation
    cv_roottime = (0.848 * (drainagelength) ** 2) / (sqrt_t90 ** 2) # m2/s
    cv_roottime_m2year = cv_roottime * 3600 * 24 * 365 # m2/yr
    
    # Generate the resulting plot
    if showfig:
        plt.plot([pointO[0], pointA[0]], [pointO[1], pointA[1]], label='OA', ls='--')
        plt.plot([pointO[0], pointB[0]], [pointO[1], pointB[1]], label='OB', ls='-.')
        plt.scatter([sqrt_t90,], [OB_func(sqrt_t90),], label='C')
        plt.gca().figure.canvas.draw()

        resultfig = plt.gcf()
    else:
        resultfig = None

    return {
        't90 [s]': sqrt_t90 ** 2,
        'cv [m2/s]': cv_roottime,
        'cv [m2/yr]': cv_roottime_m2year,
        'A': pointA,
        'B': pointB,
        'plot': resultfig
    }


LOGTIMEMETHOD = {
    'times': {'type': 'list', 'elementtype': 'float', 'order': 'ascending', 'unique': True, 'empty_allowed': False},
    'settlements': {'type': 'list', 'elementtype': 'float', 'order': None, 'unique': False, 'empty_allowed': False},
    'drainagelength': {'type': 'float', 'min_value': 0, 'max_value': None},
    'initialguess_override': {'type': 'float', 'min_value': 0.0, 'max_value': None}
}

LOGTIMEMETHOD_ERRORRETURN = {
    't100 [s]': np.nan,
    't50 [s]': np.nan,
    'cv [m2/s]': np.nan,
    'cv [m2/yr]': np.nan,
    'A': None,
    'B': None,
    'C': None,
    'D': None,
    'E': None,
    'plot': None
}

@Validator(LOGTIMEMETHOD, LOGTIMEMETHOD_ERRORRETURN)
def logtimemethod(times, settlements, drainagelength, initialguess_override=np.nan, ignore_warnings=True, showfig=True, **kwargs):
    """
    Calculates the log-time construction for determining the coefficient of consolidation for an oedometer test
    (or any other soil mechanical test involving consolidation).

    The following steps need to be performed:

   #. Project the straight portions of the primary consolidation and secondary compression to intersect at :math:`A`. The ordinate of A, :math:`d_{100}`, is the displacement gage reading for 100% primary consolidation.
   #. Correct the initial portion of the curve to make it a parabola. Select a time :math:`t_1`, point :math:`B`, near the head of the initial portion of the curve (:math:`U < 60%`) and then another time :math:`t_2`, point :math:`C`, such that :math:`t_2` = 4 :math:`t_1`.
   #. Calculate the difference in displacement reading, :math:`\Delta d = d_2 - d_1`, between :math:`t_2` and :math:`t_1`. Plot a point :math:`D` at a vertical distance :math:`\Delta d` from :math:`B`. The ordinate of point :math:`D` is the corrected initial displacement gage reading, :math:`d_o`, at the beginning of primary consolidation.
   #. Calculate the ordinate for 50% consolidation as :math:`d_{50} = (d_{100} + d_o)/2`. Draw a horizontal line through this point to intersect the curve at :math:`E`. The abscissa of point :math:`E` is the time for 50% consolidation, :math:`t_{50}`.
   #. You will recall that the time factor for 50% consolidation is 0.197, and from the one-dimensional consolidation equation we obtain:

    .. math::
        c_v = \\frac{0.197 H_{dr}^2}{t_{50}}

    Because the construction relies heavily on the laboratory data and the judgement of the user, a semi-automated procedure is followed in which the user first selects
    two points on primary consolidation part, then two points on the secondary consolidation part and finally a point B close to the origin of the curve.
    Any notebook needs to ensure that matplotlib plots are generated with the ``qt`` backend using the following magic command:

    .. code-block:: python

        %matplotlib qt

    The following input parameters are expected:

    :param times: Array with time values in seconds, increasing from 0s at the start of the test
    :param settlements: Array with settlement values, increasing from 0 at the origin. The units are not important as only the time for 90% consolidation is determined.
    :param drainagelength: Drainage length for the consolidation (:math:`H_{dr}`) [m] - Suggested range: drainagelength > 0
    :param initialguess_override: Override for the initial guess for :math:`\\sqrt{t_{100}}`, default=np.nan

    .. figure:: images/log_time.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Log-time construction

    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.
    """
    # Coerce the times and settlements to Numpy arrays
    times = np.array(times)
    settlements = np.array(settlements)

    settlement_abcis = 1.1 * settlements.max()

    if ignore_warnings:
        warnings.filterwarnings('ignore')

    # Close all open plots
    plt.close('all')

    # Generate the plot
    if showfig:
        plt.figure(1, figsize=(12,6))
        plt.subplot(111)
        plt.plot(np.log10(times), settlements, label='Root-time data')
        plt.xlabel('$ \log_{10}(t) $', size=15)
        plt.ylabel('$ \Delta z $ [mm]', size=15)
        plt.ylim([settlement_abcis, 0])
        plt.grid()
        plt.show()

    # Get the input from the user for the primary consolidation part
    print("First step: Select two points on the straight portions of the primary consolidation part of the curve (in ascending order).")
    xy_primary = selectpoints(2)
    
    # Interpolate the selection
    x1_primary = xy_primary[0][0]
    y1_primary = np.interp(x1_primary, np.log10(times), settlements)
    x2_primary = xy_primary[1][0]
    y2_primary = np.interp(x2_primary, np.log10(times), settlements)

    # Create the interpolation function for primary settlement
    primary_func = interp1d([x1_primary, x2_primary], [y1_primary, y2_primary], fill_value='extrapolate')

    # Get the input from the user for the secondary consolidation part
    print("Second step: Select two points on the straight portions of the secondary consolidation part of the curve (in ascending order).")
    xy_secondary = selectpoints(2)
    
    # Interpolate the selection
    x1_secondary = xy_secondary[0][0]
    y1_secondary = np.interp(x1_secondary, np.log10(times), settlements)
    x2_secondary = xy_secondary[1][0]
    y2_secondary = np.interp(x2_secondary, np.log10(times), settlements)

    # Create the interpolation function for secondary settlement
    secondary_func = interp1d([x1_secondary, x2_secondary], [y1_secondary, y2_secondary], fill_value='extrapolate')

    # Determine the intersection of the primary and secondary part
    def intersection_primary_secondary(x):
        return primary_func(x) - secondary_func(x)

    if np.isnan(initialguess_override):
        initial_guess = 1.5 * x1_primary
    else:
        initial_guess = initialguess_override

    # Calculate the position of the point A
    log_t100 = root(intersection_primary_secondary, initial_guess).x[0]
    xA = log_t100
    yA = primary_func(xA)
    
    # Get the input of the user for the point B
    print("Third step: Select the point B close to the head of the curve (U < 60%).")
    xy_B = selectpoints(1)
    
    # Interpolate to have B perfectly on the curve
    xB = xy_B[0][0]
    yB = np.interp(xB, np.log10(times), settlements)

    # Calculate the position of C
    xC = np.log10(10 ** xB * 4)
    yC = np.interp(xC, np.log10(times), settlements)

    # Calculate the difference in settlement between B and C
    Delta_d = np.abs(yC - yB)

    # Calculate the settlement at point D
    yD = yB - Delta_d

    # Calculate the position of the point E which determines t50
    yE = 0.5 * (yD + yA)
    xE = np.interp(yE, settlements, np.log10(times))

    # t50 is calculated from the position E
    t50 = 10 ** xE

    # Determine cv from the calculation
    cv_logtime = (0.197 * (drainagelength) ** 2) / (t50) # m2/s
    cv_logtime_m2year = cv_logtime * 3600 * 24 * 365 # m2/yr
    
    # Generate the resulting plot
    if showfig:
        plt.plot([x1_primary, x1_secondary], primary_func([x1_primary, x1_secondary]), label='Primary', ls='--')
        plt.plot([x2_primary, x2_secondary], secondary_func([x2_primary, x2_secondary]), label='Secondary', ls='--')
        plt.scatter([xA,], [yA,],)
        plt.scatter([xB, xC, xB], [yB, yC, yD],)
        plt.scatter([xE,], [yE,],)
        plt.gca().figure.canvas.draw()

        resultfig = plt.gcf()
    else:
        resultfig = None

    if ignore_warnings:
        warnings.filterwarnings('default')

    return {
        't100 [s]': 10 ** log_t100,
        't50 [s]': t50,
        'cv [m2/s]': cv_logtime,
        'cv [m2/yr]': cv_logtime_m2year,
        'A': (xA, yA),
        'B': (xB, yB),
        'C': (xC, yC),
        'D': (xB, yD),
        'E': (xE, yE),
        'plot': resultfig
    }