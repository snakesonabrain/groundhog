
# -*- coding: utf-8 -*-




__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings

# 3rd party packages
import numpy as np
from plotly import subplots
import plotly.graph_objs as go

# Project imports
from groundhog.general.validation import Validator
from groundhog.general.plotting import GROUNDHOG_PLOTTING_CONFIG

CYCLICCONTOURS_DSSCLAY_ANDERSEN = {
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'average_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
}

CYCLICCONTOURS_DSSCLAY_ANDERSEN_ERRORRETURN = {
    'Nf [-]': np.nan,
    'tau_cy / Su_DSS [-]': None,
    'Nf interpolated [-]': None
}

@Validator(CYCLICCONTOURS_DSSCLAY_ANDERSEN, CYCLICCONTOURS_DSSCLAY_ANDERSEN_ERRORRETURN)
def cycliccontours_dssclay_andersen(
        undrained_shear_strength, average_shear_stress, cyclic_shear_stress,
        **kwargs):

    """
    Calculates the number of cycles to failure for a cyclic DSS test with  a given combination of average and cyclic shear stress for a sample with a given undrained shear strength. Cyclic failure is defined as reaching a cyclic or average shear strain of 15%. Contours for N=10, N=100 and N=1000 are defined. Logarithmic interpolation is used between the contours. Three points are extracted from the digitised graphs and a logarithmic relation is fitted. If the number of cycles to failure is lower than 10 or greater than 1000, extrapolation is used but a warning is raised to the user.

    The data used for this interaction diagram originates from cyclic tests on normally consolidated Drammen clay, a marine clay with a plasticity index of 27%. Average cyclic stresses were applied for 1 to 2hrs before starting the cyclic shearing and the cyclic loading period was 10s.

    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a DSS test (:math:`S_u^{DSS}`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param average_shear_stress: Magnitude of the applied average shear stress (:math:`\\tau_a`) [:math:`kPa`] - Suggested range: 0.0 <= average_shear_stress <= 100.0
    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0

    :returns: Dictionary with the following keys:

        - 'Nf [-]': Number of cycles to failure (:math:`N_f`)  [:math:`-`]

    .. figure:: images/cycliccontours_dssclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Contour diagram for DSS tests on normally consolidated Drammen clay

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    N_10 = (
       (0, 0.9235412474849094),
       (0.028865979381443252, 0.9215291750503017),
       (0.06597938144329896, 0.9175050301810864),
       (0.09072164948453604, 0.9154929577464788),
       (0.11958762886597935, 0.9134808853118712),
       (0.1422680412371134, 0.9094567404426559),
       (0.1711340206185567, 0.9054325955734406),
       (0.19793814432989693, 0.9014084507042253),
       (0.22474226804123715, 0.8973843058350099),
       (0.2474226804123711, 0.8933601609657946),
       (0.27010309278350514, 0.8873239436619718),
       (0.29896907216494845, 0.8832997987927564),
       (0.32577319587628867, 0.8752515090543259),
       (0.3525773195876288, 0.8712273641851106),
       (0.37319587628865974, 0.8631790744466801),
       (0.40412371134020614, 0.8511066398390341),
       (0.41855670103092785, 0.8370221327967806),
       (0.43298969072164945, 0.8229376257545271),
       (0.4536082474226804, 0.8048289738430583),
       (0.47628865979381446, 0.7826961770623742),
       (0.5092783505154639, 0.7505030181086518),
       (0.5381443298969071, 0.7203219315895372),
       (0.560824742268041, 0.6941649899396378),
       (0.5855670103092783, 0.663983903420523),
       (0.6082474226804122, 0.6378269617706237),
       (0.6350515463917525, 0.6036217303822937),
       (0.6659793814432988, 0.5633802816901409),
       (0.7175257731958762, 0.49295774647887314),
       (0.7525773195876289, 0.44064386317907445),
       (0.7711340206185566, 0.41046277665995967),
       (0.7999999999999998, 0.3661971830985915),
       (0.8371134020618556, 0.3078470824949697),
       (0.8680412371134019, 0.25553319919517103),
       (0.8907216494845358, 0.21529175050301808),
       (0.9113402061855669, 0.17907444668008043),
       (0.9546391752577319, 0.09255533199195187),
       (0.9773195876288658, 0.05432595573440646),
       (0.9958762886597936, 0.016096579476861272),
       (1, 0)
    )

    _tau_avg_N_10 = list(map(lambda _x: _x[0], N_10))
    _tau_cyc_N_10 = list(map(lambda _x: _x[1], N_10))

    N_100 = (
        (0.0020618556701030855, 0.69215291750503),
        (0.030927835051546393, 0.69215291750503),
        (0.05567010309278353, 0.69215291750503),
        (0.10927835051546392, 0.6901408450704225),
        (0.15876288659793808, 0.6861167002012072),
        (0.2144329896907216, 0.6800804828973842),
        (0.2639175257731959, 0.6760563380281689),
        (0.3030927835051546, 0.6680080482897384),
        (0.34432989690721644, 0.6378269617706237),
        (0.37525773195876283, 0.6136820925553319),
        (0.40412371134020614, 0.5915492957746478),
        (0.4371134020618556, 0.5674044265593561),
        (0.46391752577319584, 0.545271629778672),
        (0.5010309278350515, 0.5150905432595573),
        (0.5278350515463917, 0.49295774647887314),
        (0.5567010309278351, 0.46881287726358145),
        (0.6041237113402063, 0.4285714285714286),
        (0.6309278350515461, 0.40643863179074446),
        (0.6948453608247422, 0.34406438631790737),
        (0.7237113402061854, 0.3138832997987928),
        (0.7567010309278348, 0.2816901408450705),
        (0.7835051546391751, 0.2515090543259556),
        (0.8144329896907214, 0.22132796780684105),
        (0.847422680412371, 0.18712273641851107),
        (0.8886597938144329, 0.1428571428571428),
        (0.9113402061855669, 0.11267605633802824),
        (0.938144329896907, 0.07847082494969815),
        (0.9628865979381442, 0.046277665995975825),
        (0.9896907216494846, 0.014084507042253502),
        (1, -0.00201207243460777)
    )

    _tau_avg_N_100 = list(map(lambda _x: _x[0], N_100))
    _tau_cyc_N_100 = list(map(lambda _x: _x[1], N_100))

    N_1000 = (
        (0, 0.5472837022132796),
        (0.032989690721649534, 0.5432595573440643),
        (0.0742268041237113, 0.5412474849094567),
        (0.11340206185567009, 0.5372233400402414),
        (0.15051546391752574, 0.5352112676056338),
        (0.2020618556701031, 0.5311871227364184),
        (0.245360824742268, 0.5251509054325955),
        (0.290721649484536, 0.5050301810865191),
        (0.334020618556701, 0.4849094567404426),
        (0.37319587628865974, 0.4668008048289738),
        (0.4268041237113402, 0.4386317907444668),
        (0.4680412371134021, 0.414486921529175),
        (0.5154639175257731, 0.386317907444668),
        (0.5525773195876289, 0.36418511066398385),
        (0.5979381443298968, 0.3319919517102615),
        (0.6329896907216495, 0.3078470824949697),
        (0.6680412371134019, 0.2816901408450705),
        (0.6989690721649482, 0.25955734406438635),
        (0.736082474226804, 0.22937625754527158),
        (0.7752577319587628, 0.19718309859154926),
        (0.8144329896907214, 0.1670020120724347),
        (0.847422680412371, 0.13480885311871238),
        (0.8845360824742268, 0.10060362173038229),
        (0.9237113402061854, 0.06438631790744465),
        (0.9546391752577319, 0.03621730382293764),
        (0.981443298969072, 0.010060362173038184),
        (1, 0)
    )

    _tau_avg_N_1000 = list(map(lambda _x: _x[0], N_1000))
    _tau_cyc_N_1000 = list(map(lambda _x: _x[1], N_1000))


    _tau_a_norm = average_shear_stress / undrained_shear_strength
    _tau_cy_norm = cyclic_shear_stress / undrained_shear_strength

    if _tau_a_norm < 0 or _tau_a_norm > 1:
        raise ValueError("Normalised average shear stress should be between 0 and 1")
    if _tau_cy_norm < 0 or _tau_cy_norm > 1:
        raise ValueError("Normalised cyclic shear stress should be between 0 and 1")

    _tau_cy_interp = [
        np.interp(_tau_a_norm, _tau_avg_N_10, _tau_cyc_N_10),
        np.interp(_tau_a_norm, _tau_avg_N_100, _tau_cyc_N_100),
        np.interp(_tau_a_norm, _tau_avg_N_1000, _tau_cyc_N_1000),
    ]

    fit_coeff = np.polyfit(
        x=_tau_cy_interp,
        y=[1, 2, 3],
        deg=1)
    fit_func = np.poly1d(fit_coeff)

    Nf_tau_cy_norm_0 = fit_func(0)
    Nf_tau_cy_norm_1 = fit_func(1)

    _Nf = 10 ** (np.interp(
        _tau_cy_norm,
        [0, _tau_cy_interp[2], _tau_cy_interp[1], _tau_cy_interp[0], 1],
        [Nf_tau_cy_norm_0, 3, 2, 1, Nf_tau_cy_norm_1]))

    if _Nf < 10 or _Nf > 1000:
        warnings.warn(
            "Interpolated number of cycles to failure is greater outside the bounds of the published diagram")

    return {
        'Nf [-]': _Nf,
        'tau_cy / Su_DSS [-]': [0, _tau_cy_interp[2], _tau_cy_interp[1], _tau_cy_interp[0], 1],
        'Nf interpolated [-]': [Nf_tau_cy_norm_0, 3, 2, 1, Nf_tau_cy_norm_1],
    }

def plotcycliccontours_dssclay_andersen():
    """
    Returns a Plotly figure with the cyclic contours for DSS tests on normally consolidated Drammen clay
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)

    N_10 = (
        (0, 0.9235412474849094),
        (0.028865979381443252, 0.9215291750503017),
        (0.06597938144329896, 0.9175050301810864),
        (0.09072164948453604, 0.9154929577464788),
        (0.11958762886597935, 0.9134808853118712),
        (0.1422680412371134, 0.9094567404426559),
        (0.1711340206185567, 0.9054325955734406),
        (0.19793814432989693, 0.9014084507042253),
        (0.22474226804123715, 0.8973843058350099),
        (0.2474226804123711, 0.8933601609657946),
        (0.27010309278350514, 0.8873239436619718),
        (0.29896907216494845, 0.8832997987927564),
        (0.32577319587628867, 0.8752515090543259),
        (0.3525773195876288, 0.8712273641851106),
        (0.37319587628865974, 0.8631790744466801),
        (0.40412371134020614, 0.8511066398390341),
        (0.41855670103092785, 0.8370221327967806),
        (0.43298969072164945, 0.8229376257545271),
        (0.4536082474226804, 0.8048289738430583),
        (0.47628865979381446, 0.7826961770623742),
        (0.5092783505154639, 0.7505030181086518),
        (0.5381443298969071, 0.7203219315895372),
        (0.560824742268041, 0.6941649899396378),
        (0.5855670103092783, 0.663983903420523),
        (0.6082474226804122, 0.6378269617706237),
        (0.6350515463917525, 0.6036217303822937),
        (0.6659793814432988, 0.5633802816901409),
        (0.7175257731958762, 0.49295774647887314),
        (0.7525773195876289, 0.44064386317907445),
        (0.7711340206185566, 0.41046277665995967),
        (0.7999999999999998, 0.3661971830985915),
        (0.8371134020618556, 0.3078470824949697),
        (0.8680412371134019, 0.25553319919517103),
        (0.8907216494845358, 0.21529175050301808),
        (0.9113402061855669, 0.17907444668008043),
        (0.9546391752577319, 0.09255533199195187),
        (0.9773195876288658, 0.05432595573440646),
        (0.9958762886597936, 0.016096579476861272),
        (1, 0)
    )

    _tau_avg_N_10 = list(map(lambda _x: _x[0], N_10))
    _tau_cyc_N_10 = list(map(lambda _x: _x[1], N_10))

    N_100 = (
        (0.0020618556701030855, 0.69215291750503),
        (0.030927835051546393, 0.69215291750503),
        (0.05567010309278353, 0.69215291750503),
        (0.10927835051546392, 0.6901408450704225),
        (0.15876288659793808, 0.6861167002012072),
        (0.2144329896907216, 0.6800804828973842),
        (0.2639175257731959, 0.6760563380281689),
        (0.3030927835051546, 0.6680080482897384),
        (0.34432989690721644, 0.6378269617706237),
        (0.37525773195876283, 0.6136820925553319),
        (0.40412371134020614, 0.5915492957746478),
        (0.4371134020618556, 0.5674044265593561),
        (0.46391752577319584, 0.545271629778672),
        (0.5010309278350515, 0.5150905432595573),
        (0.5278350515463917, 0.49295774647887314),
        (0.5567010309278351, 0.46881287726358145),
        (0.6041237113402063, 0.4285714285714286),
        (0.6309278350515461, 0.40643863179074446),
        (0.6948453608247422, 0.34406438631790737),
        (0.7237113402061854, 0.3138832997987928),
        (0.7567010309278348, 0.2816901408450705),
        (0.7835051546391751, 0.2515090543259556),
        (0.8144329896907214, 0.22132796780684105),
        (0.847422680412371, 0.18712273641851107),
        (0.8886597938144329, 0.1428571428571428),
        (0.9113402061855669, 0.11267605633802824),
        (0.938144329896907, 0.07847082494969815),
        (0.9628865979381442, 0.046277665995975825),
        (0.9896907216494846, 0.014084507042253502),
        (1, -0.00201207243460777)
    )

    _tau_avg_N_100 = list(map(lambda _x: _x[0], N_100))
    _tau_cyc_N_100 = list(map(lambda _x: _x[1], N_100))

    N_1000 = (
        (0, 0.5472837022132796),
        (0.032989690721649534, 0.5432595573440643),
        (0.0742268041237113, 0.5412474849094567),
        (0.11340206185567009, 0.5372233400402414),
        (0.15051546391752574, 0.5352112676056338),
        (0.2020618556701031, 0.5311871227364184),
        (0.245360824742268, 0.5251509054325955),
        (0.290721649484536, 0.5050301810865191),
        (0.334020618556701, 0.4849094567404426),
        (0.37319587628865974, 0.4668008048289738),
        (0.4268041237113402, 0.4386317907444668),
        (0.4680412371134021, 0.414486921529175),
        (0.5154639175257731, 0.386317907444668),
        (0.5525773195876289, 0.36418511066398385),
        (0.5979381443298968, 0.3319919517102615),
        (0.6329896907216495, 0.3078470824949697),
        (0.6680412371134019, 0.2816901408450705),
        (0.6989690721649482, 0.25955734406438635),
        (0.736082474226804, 0.22937625754527158),
        (0.7752577319587628, 0.19718309859154926),
        (0.8144329896907214, 0.1670020120724347),
        (0.847422680412371, 0.13480885311871238),
        (0.8845360824742268, 0.10060362173038229),
        (0.9237113402061854, 0.06438631790744465),
        (0.9546391752577319, 0.03621730382293764),
        (0.981443298969072, 0.010060362173038184),
        (1, 0)
    )

    _tau_avg_N_1000 = list(map(lambda _x: _x[0], N_1000))
    _tau_cyc_N_1000 = list(map(lambda _x: _x[1], N_1000))


    trace_N10 = go.Scatter(
        x=_tau_avg_N_10,
        y=_tau_cyc_N_10,
        showlegend=True, mode='lines', name=r'$ N_f = 10 $')
    fig.append_trace(trace_N10, 1, 1)

    trace_N100 = go.Scatter(
        x=_tau_avg_N_100,
        y=_tau_cyc_N_100,
        showlegend=True, mode='lines', name=r'$ N_f = 100 $')
    fig.append_trace(trace_N100, 1, 1)

    trace_N1000 = go.Scatter(
        x=_tau_avg_N_1000,
        y=_tau_cyc_N_1000,
        showlegend=True, mode='lines', name=r'$ N_f = 1000 $')
    fig.append_trace(trace_N1000, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ \tau_{avg} / S_u^{DSS} $', range=(0, 1), dtick=0.25)
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{DSS} $', range=(0, 1), dtick=0.25)
    fig['layout'].update(height=500, width=500,
                         title=r'$ \text{Contour diagram for cyclic DSS tests om Drammen clay} $',
                         hovermode='closest')

    return fig


CYCLICCONTOURS_TRIAXIALCLAY_ANDERSEN = {
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'average_shear_stress': {'type': 'float', 'min_value': -50.0, 'max_value': 100.0},
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
}

CYCLICCONTOURS_TRIAXIALCLAY_ANDERSEN_ERRORRETURN = {
    'Nf [-]': np.nan,
    'tau_cy / Su_C [-]': None,
    'Nf interpolated [-]': None,
}


@Validator(CYCLICCONTOURS_TRIAXIALCLAY_ANDERSEN, CYCLICCONTOURS_TRIAXIALCLAY_ANDERSEN_ERRORRETURN)
def cycliccontours_triaxialclay_andersen(
        undrained_shear_strength, average_shear_stress, cyclic_shear_stress,
        **kwargs):
    """
    Calculates the number of cycles to failure for cyclic triaxial test with a given combination of average and cyclic shear stress for a sample with a given undrained shear strength. Cyclic failure is defined as reaching a cyclic or average shear strain of 15%. Contours for N=10, N=100 and N=1000 are defined. Logarithmic interpolation is used between the contours. Three points are extracted from the digitised graphs and a logarithmic relation is fitted. If the number of cycles to failure is lower than 10 or greater than 1000, extrapolation is used but a warning is raised to the user.

    The data used for this interaction diagram originates from cyclic tests on normally consolidated Drammen clay, a marine clay with a plasticity index of 27%. Average cyclic stresses were applied for 1 to 2hrs before starting the cyclic shearing and the cyclic loading period was 10s.

    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a triaxial compression test (:math:`S_u^C`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param average_shear_stress: Magnitude of the applied average shear stress (:math:`\\tau_a`) [:math:`kPa`] - Suggested range: -50.0 <= average_shear_stress <= 100.0
    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0

    :returns: Dictionary with the following keys:

        - 'Nf [-]': Number of cycles to failure (:math:`N_f`)  [:math:`-`]
        - 'tau_cy / Su_C [-]': Ratios of cyclic shear stress to the compressive undrained shear strength used in interpolation [:math:`-`]
        - 'Nf interpolated [-]': Number of cycles to failure used in the interpolation [:math:`-`]

    .. figure:: images/cycliccontours_triaxialclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Contour diagram for cyclic triaxial tests on normally consolidated Drammen clay

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    _Nf_10_tau_avg = np.array([-0.5       , -0.48814229, -0.46640316, -0.4486166 , -0.42490119,
       -0.40316206, -0.37821453, -0.35898525, -0.33975597, -0.32052668,
       -0.3012974 , -0.28206811, -0.26283883, -0.24360955, -0.22438026,
       -0.20515098, -0.18592169, -0.16669241, -0.14746312, -0.12823384,
       -0.10900456, -0.08977527, -0.07054599, -0.05306482, -0.01581028,
        0.02173913,  0.06324111,  0.11857708,  0.16205534,  0.21146245,
        0.24505929,  0.27272727,  0.31027668,  0.35573123,  0.38932806,
        0.41368235,  0.43291163,  0.45214092,  0.4713702 ,  0.49059948,
        0.50982877,  0.52905805,  0.54828734,  0.56751662,  0.58674591,
        0.60597519,  0.62520447,  0.64443376,  0.66366304,  0.68289233,
        0.70212161,  0.72135089,  0.74058018,  0.75980946,  0.77903875,
        0.79826803,  0.8157492 ,  0.83148225,  0.8472153 ,  0.86207429,
        0.87605923,  0.89004416,  0.90315503,  0.91539185,  0.92675461,
        0.93724331,  0.94773201,  0.95734665,  0.96503837,  0.97113735,
        0.97757486,  0.9843176 ,  0.98764734,  0.99407115,  0.99604743,
        1.        ])
    _Nf_10_tau_cyc = np.array([0.        , 0.00996016, 0.03386454, 0.0498008 , 0.07370518,
       0.0936255 , 0.11870426, 0.13696546, 0.15546694, 0.1742888 ,
       0.19263009, 0.21169223, 0.23059417, 0.24877528, 0.26815779,
       0.28657918, 0.30516076, 0.32390252, 0.34232391, 0.36122585,
       0.37988752, 0.39870938, 0.41753123, 0.4338702 , 0.44621514,
       0.4561753 , 0.47011952, 0.48406375, 0.49800797, 0.51394422,
       0.52589641, 0.53386454, 0.54183267, 0.55577689, 0.56772908,
       0.57363248, 0.56914728, 0.56522272, 0.56121807, 0.55673286,
       0.55144672, 0.54616059, 0.53991333, 0.53286515, 0.52485585,
       0.51676646, 0.50883725, 0.5002673 , 0.49153716, 0.48232647,
       0.47239494, 0.46158238, 0.45012908, 0.43675355, 0.42193635,
       0.40543719, 0.38717599, 0.3686745 , 0.34850887, 0.32902847,
       0.30920545, 0.2880609 , 0.26685342, 0.24696747, 0.22697664,
       0.20774097, 0.18777112, 0.16780126, 0.14789014, 0.1285468 ,
       0.10776584, 0.08511096, 0.07235711, 0.04780876, 0.02589641,
       0.])

    _Nf_100_tau_avg = np.array([-0.5       , -0.49802372, -0.46442688, -0.42885375, -0.39328063,
       -0.36758893, -0.3432522 , -0.32402292, -0.30514325, -0.28556435,
       -0.26633506, -0.24710578, -0.22787649, -0.20864721, -0.18941793,
       -0.17018864, -0.15095936, -0.14047066, -0.02859119, -0.01111002,
        0.02766798,  0.06521739,  0.10869565,  0.15612648,  0.20750988,
        0.26679842,  0.31422925,  0.35770751,  0.40118577,  0.42417105,
        0.44340033,  0.46262962,  0.4818589 ,  0.50108819,  0.52031747,
        0.53954675,  0.55877604,  0.57800532,  0.59723461,  0.61646389,
        0.63569317,  0.65492246,  0.67415174,  0.69338103,  0.71261031,
        0.73183959,  0.75087464,  0.77029816,  0.78952745,  0.80875673,
        0.82711196,  0.84459312,  0.86207429,  0.87955546,  0.89616257,
        0.91189562,  0.92762867,  0.94336172,  0.95909477,  0.97482782,
        0.98968681,  1.        ])
    _Nf_100_tau_cyc = np.array([0.        , 0.00199203, 0.02589641, 0.05179283, 0.07569721,
       0.09163347, 0.10813198, 0.12110705, 0.13379378, 0.14705718,
       0.15995215, 0.17284712, 0.18598237, 0.19911763, 0.2120126 ,
       0.22498766, 0.23796273, 0.24445026, 0.32037842, 0.33216099,
       0.34661355, 0.35657371, 0.3685259 , 0.37848606, 0.39442231,
       0.40836653, 0.42430279, 0.43426295, 0.44621514, 0.44476285,
       0.43763457, 0.43034611, 0.42337802, 0.41608956, 0.40888119,
       0.40103207, 0.39326305, 0.38509357, 0.37604306, 0.3664319 ,
       0.35690083, 0.34688921, 0.33623684, 0.32486363, 0.31228903,
       0.29867322, 0.2840963 , 0.26751705, 0.25005677, 0.23163538,
       0.21273344, 0.19370334, 0.17502565, 0.15564315, 0.13676968,
       0.11807241, 0.0987878 , 0.07832849, 0.05777128, 0.03711619,
       0.01956915, 0.        ])

    _Nf_1000_tau_avg = np.array([-0.5       , -0.49604743, -0.46640316, -0.43478261, -0.41501976,
       -0.40118577, -0.38870323, -0.36947395, -0.35024467, -0.33101538,
       -0.3117861 , -0.29255681, -0.27332753, -0.25409825, -0.23486896,
       -0.21563968, -0.19641039, -0.17718111, -0.15795182, -0.13872254,
       -0.11949326, -0.10026397, -0.08890121, -0.06126482, -0.03162055,
       -0.00592885,  0.01511173,  0.02647449,  0.06126482,  0.09881423,
        0.14229249,  0.18577075,  0.23913043,  0.27470356,  0.31620553,
        0.35375494,  0.39130435,  0.42094862,  0.42591917,  0.44514845,
        0.46437773,  0.48360702,  0.5028363 ,  0.52206559,  0.54129487,
        0.56052415,  0.57975344,  0.59898272,  0.61821201,  0.63744129,
        0.65667057,  0.67589986,  0.69512914,  0.71435843,  0.73358771,
        0.752817  ,  0.77204628,  0.79127556,  0.81050485,  0.82973413,
        0.84896342,  0.8681927 ,  0.88742198,  0.90665127,  0.92588055,
        0.94510984,  0.96346506,  0.98418972,  1.])
    _Nf_1000_tau_cyc = np.array([ 0.        , -0.00199203,  0.01195219,  0.02988048,  0.03984064,
        0.04581673,  0.05310809,  0.06295953,  0.07305125,  0.08298278,
        0.09283422,  0.10260556,  0.11269728,  0.12254872,  0.13248025,
        0.14241178,  0.15226322,  0.16219475,  0.17212629,  0.18245828,
        0.19190926,  0.20184079,  0.20612576,  0.22111554,  0.23705179,
        0.24900398,  0.26199063,  0.26779737,  0.27689243,  0.28486056,
        0.29282869,  0.30278884,  0.31075697,  0.32270916,  0.32868526,
        0.33864542,  0.34462151,  0.35258964,  0.35209525,  0.34224381,
        0.33199191,  0.32206038,  0.31228903,  0.3023575 ,  0.29234588,
        0.28225416,  0.27216244,  0.26191054,  0.25149845,  0.24092617,
        0.23083445,  0.21986171,  0.20888897,  0.19767595,  0.18606247,
        0.17388833,  0.16139382,  0.14841876,  0.13528351,  0.12198807,
        0.10861254,  0.09483654,  0.08122073,  0.06728455,  0.05342846,
        0.03909182,  0.02481123,  0.01394422,  0.        ])

    _tau_a_norm = average_shear_stress / undrained_shear_strength
    _tau_cy_norm = cyclic_shear_stress / undrained_shear_strength

    if _tau_a_norm < -0.5 or _tau_a_norm > 1:
        raise ValueError("Normalised average shear stress should be between -0.5 and 1")
    if _tau_cy_norm < 0 or _tau_cy_norm > 1:
        raise ValueError("Normalised cyclic shear stress should be between 0 and 1")

    _tau_cy_interp = [
        np.interp(_tau_a_norm, _Nf_10_tau_avg, _Nf_10_tau_cyc),
        np.interp(_tau_a_norm, _Nf_100_tau_avg, _Nf_100_tau_cyc),
        np.interp(_tau_a_norm, _Nf_1000_tau_avg, _Nf_1000_tau_cyc),
    ]

    fit_coeff = np.polyfit(
        x=_tau_cy_interp,
        y=[1, 2, 3],
        deg=1)
    fit_func = np.poly1d(fit_coeff)

    Nf_tau_cy_norm_0 = fit_func(0)
    Nf_tau_cy_norm_1 = fit_func(1)

    _tau_cy__Su_C = [0, _tau_cy_interp[2], _tau_cy_interp[1], _tau_cy_interp[0], 1]
    _Nf_interpolated = [Nf_tau_cy_norm_0, 3, 2, 1, Nf_tau_cy_norm_1]

    _Nf = 10 ** (np.interp(
        _tau_cy_norm,
        _tau_cy__Su_C,
        _Nf_interpolated))

    if _Nf < 10 or _Nf > 1000:
        warnings.warn(
            "Interpolated number of cycles to failure is greater outside the bounds of the published diagram")

    return {
        'Nf [-]': _Nf,
        'tau_cy / Su_C [-]': _tau_cy__Su_C,
        'Nf interpolated [-]': _Nf_interpolated,
    }

def plotcycliccontours_triaxialclay_andersen():
    """
    Returns a Plotly figure with the cyclic contours for triaxial tests on normally consolidated Drammen clay
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)

    _Nf_10_tau_avg = np.array([-0.5, -0.48814229, -0.46640316, -0.4486166, -0.42490119,
                               -0.40316206, -0.37821453, -0.35898525, -0.33975597, -0.32052668,
                               -0.3012974, -0.28206811, -0.26283883, -0.24360955, -0.22438026,
                               -0.20515098, -0.18592169, -0.16669241, -0.14746312, -0.12823384,
                               -0.10900456, -0.08977527, -0.07054599, -0.05306482, -0.01581028,
                               0.02173913, 0.06324111, 0.11857708, 0.16205534, 0.21146245,
                               0.24505929, 0.27272727, 0.31027668, 0.35573123, 0.38932806,
                               0.41368235, 0.43291163, 0.45214092, 0.4713702, 0.49059948,
                               0.50982877, 0.52905805, 0.54828734, 0.56751662, 0.58674591,
                               0.60597519, 0.62520447, 0.64443376, 0.66366304, 0.68289233,
                               0.70212161, 0.72135089, 0.74058018, 0.75980946, 0.77903875,
                               0.79826803, 0.8157492, 0.83148225, 0.8472153, 0.86207429,
                               0.87605923, 0.89004416, 0.90315503, 0.91539185, 0.92675461,
                               0.93724331, 0.94773201, 0.95734665, 0.96503837, 0.97113735,
                               0.97757486, 0.9843176, 0.98764734, 0.99407115, 0.99604743,
                               1.])
    _Nf_10_tau_cyc = np.array([0., 0.00996016, 0.03386454, 0.0498008, 0.07370518,
                               0.0936255, 0.11870426, 0.13696546, 0.15546694, 0.1742888,
                               0.19263009, 0.21169223, 0.23059417, 0.24877528, 0.26815779,
                               0.28657918, 0.30516076, 0.32390252, 0.34232391, 0.36122585,
                               0.37988752, 0.39870938, 0.41753123, 0.4338702, 0.44621514,
                               0.4561753, 0.47011952, 0.48406375, 0.49800797, 0.51394422,
                               0.52589641, 0.53386454, 0.54183267, 0.55577689, 0.56772908,
                               0.57363248, 0.56914728, 0.56522272, 0.56121807, 0.55673286,
                               0.55144672, 0.54616059, 0.53991333, 0.53286515, 0.52485585,
                               0.51676646, 0.50883725, 0.5002673, 0.49153716, 0.48232647,
                               0.47239494, 0.46158238, 0.45012908, 0.43675355, 0.42193635,
                               0.40543719, 0.38717599, 0.3686745, 0.34850887, 0.32902847,
                               0.30920545, 0.2880609, 0.26685342, 0.24696747, 0.22697664,
                               0.20774097, 0.18777112, 0.16780126, 0.14789014, 0.1285468,
                               0.10776584, 0.08511096, 0.07235711, 0.04780876, 0.02589641,
                               0.])

    _Nf_100_tau_avg = np.array([-0.5, -0.49802372, -0.46442688, -0.42885375, -0.39328063,
                                -0.36758893, -0.3432522, -0.32402292, -0.30514325, -0.28556435,
                                -0.26633506, -0.24710578, -0.22787649, -0.20864721, -0.18941793,
                                -0.17018864, -0.15095936, -0.14047066, -0.02859119, -0.01111002,
                                0.02766798, 0.06521739, 0.10869565, 0.15612648, 0.20750988,
                                0.26679842, 0.31422925, 0.35770751, 0.40118577, 0.42417105,
                                0.44340033, 0.46262962, 0.4818589, 0.50108819, 0.52031747,
                                0.53954675, 0.55877604, 0.57800532, 0.59723461, 0.61646389,
                                0.63569317, 0.65492246, 0.67415174, 0.69338103, 0.71261031,
                                0.73183959, 0.75087464, 0.77029816, 0.78952745, 0.80875673,
                                0.82711196, 0.84459312, 0.86207429, 0.87955546, 0.89616257,
                                0.91189562, 0.92762867, 0.94336172, 0.95909477, 0.97482782,
                                0.98968681, 1.])
    _Nf_100_tau_cyc = np.array([0., 0.00199203, 0.02589641, 0.05179283, 0.07569721,
                                0.09163347, 0.10813198, 0.12110705, 0.13379378, 0.14705718,
                                0.15995215, 0.17284712, 0.18598237, 0.19911763, 0.2120126,
                                0.22498766, 0.23796273, 0.24445026, 0.32037842, 0.33216099,
                                0.34661355, 0.35657371, 0.3685259, 0.37848606, 0.39442231,
                                0.40836653, 0.42430279, 0.43426295, 0.44621514, 0.44476285,
                                0.43763457, 0.43034611, 0.42337802, 0.41608956, 0.40888119,
                                0.40103207, 0.39326305, 0.38509357, 0.37604306, 0.3664319,
                                0.35690083, 0.34688921, 0.33623684, 0.32486363, 0.31228903,
                                0.29867322, 0.2840963, 0.26751705, 0.25005677, 0.23163538,
                                0.21273344, 0.19370334, 0.17502565, 0.15564315, 0.13676968,
                                0.11807241, 0.0987878, 0.07832849, 0.05777128, 0.03711619,
                                0.01956915, 0.])

    _Nf_1000_tau_avg = np.array([-0.5, -0.49604743, -0.46640316, -0.43478261, -0.41501976,
                                 -0.40118577, -0.38870323, -0.36947395, -0.35024467, -0.33101538,
                                 -0.3117861, -0.29255681, -0.27332753, -0.25409825, -0.23486896,
                                 -0.21563968, -0.19641039, -0.17718111, -0.15795182, -0.13872254,
                                 -0.11949326, -0.10026397, -0.08890121, -0.06126482, -0.03162055,
                                 -0.00592885, 0.01511173, 0.02647449, 0.06126482, 0.09881423,
                                 0.14229249, 0.18577075, 0.23913043, 0.27470356, 0.31620553,
                                 0.35375494, 0.39130435, 0.42094862, 0.42591917, 0.44514845,
                                 0.46437773, 0.48360702, 0.5028363, 0.52206559, 0.54129487,
                                 0.56052415, 0.57975344, 0.59898272, 0.61821201, 0.63744129,
                                 0.65667057, 0.67589986, 0.69512914, 0.71435843, 0.73358771,
                                 0.752817, 0.77204628, 0.79127556, 0.81050485, 0.82973413,
                                 0.84896342, 0.8681927, 0.88742198, 0.90665127, 0.92588055,
                                 0.94510984, 0.96346506, 0.98418972, 1.])
    _Nf_1000_tau_cyc = np.array([0., -0.00199203, 0.01195219, 0.02988048, 0.03984064,
                                 0.04581673, 0.05310809, 0.06295953, 0.07305125, 0.08298278,
                                 0.09283422, 0.10260556, 0.11269728, 0.12254872, 0.13248025,
                                 0.14241178, 0.15226322, 0.16219475, 0.17212629, 0.18245828,
                                 0.19190926, 0.20184079, 0.20612576, 0.22111554, 0.23705179,
                                 0.24900398, 0.26199063, 0.26779737, 0.27689243, 0.28486056,
                                 0.29282869, 0.30278884, 0.31075697, 0.32270916, 0.32868526,
                                 0.33864542, 0.34462151, 0.35258964, 0.35209525, 0.34224381,
                                 0.33199191, 0.32206038, 0.31228903, 0.3023575, 0.29234588,
                                 0.28225416, 0.27216244, 0.26191054, 0.25149845, 0.24092617,
                                 0.23083445, 0.21986171, 0.20888897, 0.19767595, 0.18606247,
                                 0.17388833, 0.16139382, 0.14841876, 0.13528351, 0.12198807,
                                 0.10861254, 0.09483654, 0.08122073, 0.06728455, 0.05342846,
                                 0.03909182, 0.02481123, 0.01394422, 0.])


    trace_N10 = go.Scatter(
        x=_Nf_10_tau_avg,
        y=_Nf_10_tau_cyc,
        showlegend=True, mode='lines', name=r'$ N_f = 10 $')
    fig.append_trace(trace_N10, 1, 1)

    trace_N100 = go.Scatter(
        x=_Nf_100_tau_avg,
        y=_Nf_100_tau_cyc,
        showlegend=True, mode='lines', name=r'$ N_f = 100 $')
    fig.append_trace(trace_N100, 1, 1)

    trace_N1000 = go.Scatter(
        x=_Nf_1000_tau_avg,
        y=_Nf_1000_tau_cyc,
        showlegend=True, mode='lines', name=r'$ N_f = 1000 $')
    fig.append_trace(trace_N1000, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ \tau_{avg} / S_u^{C} $', range=(-0.5, 1), dtick=0.25)
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{C} $', range=(0, 0.75), dtick=0.25)
    fig['layout'].update(height=500, width=700,
                         title=r'$ \text{Contour diagram for cyclic triaxial tests om Drammen clay} $',
                         hovermode='closest')

    return fig


# Shear strain DSS


STRAINACCUMULATION_DSSCLAY_ANDERSEN = {
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'cycle_no': {'type': 'int', 'min_value': 1.0, 'max_value': 1500.0},
}

STRAINACCUMULATION_DSSCLAY_ANDERSEN_ERRORRETURN = {
    'cyclic strain [%]': np.nan,
    'shear strains interpolation [%]': None,
    'shearstress ratios interpolation [-]': None,
}


@Validator(STRAINACCUMULATION_DSSCLAY_ANDERSEN, STRAINACCUMULATION_DSSCLAY_ANDERSEN_ERRORRETURN)
def strainaccumulation_dssclay_andersen(
        cyclic_shear_stress, undrained_shear_strength, cycle_no,
        **kwargs):
    """
    Calculates the strain accumulation for a normally consolidated clay sample under symmetrical cyclic loading (no average shear stress) in a DSS test. The contours are based on cyclic DSS tests on Drammen clay.

    Strain contours for cyclic shear strains of 0.5, 1, 3 and 15% are defined and logarithmic interpolation is used to obtain the accumulated strain for a sample tested at a certain ratio of cyclic shear stress to DSS shear strength with a given number of cycles.

    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0
    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a DSS test (:math:`S_u^{DSS}`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param cycle_no: Number of applied cycles (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= cycle_no <= 1500.0

    :returns: Dictionary with the following keys:

        - 'cyclic strain [%]': Accumulated cyclic shear strain (:math:`\\gamma_{cy}`)  [:math:`%`]
        - 'shear strains interpolation [%]': List of shear strains used for interpolation [:math:`%`]
        - 'shearstress ratios interpolation [-]': List of ratios of cyclic shear stress to undrained DSS shear strength used for interpolation [:math:`-`]

    .. figure:: images/strainaccumulation_dssclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Strain contours for symmetrical cyclic DSS tests

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    gamma_cyc__0_5_N = np.array([1.0e+00, 1.34691510e+00, 1.95374983e+00, 3.11015790e+00,
                                 5.08421772e+00, 1.00090813e+01, 1.70260074e+01, 3.05427700e+01,
                                 6.01282407e+01, 9.31992345e+01, 2.09524007e+02, 3.70876480e+02,
                                 7.20425342e+02, 1.38095466e+03, 1.59819914e+03])
    gamma_cyc__0_5_tau_cyc_norm = np.array([0.62833926, 0.62845238, 0.62592255, 0.62342871, 0.61827135,
                                            0.60249075, 0.5893274, 0.57618462, 0.56040402, 0.54720467,
                                            0.52078027, 0.49961093, 0.47312996, 0.44664386, 0.44402661])

    gamma_cyc__1_N = np.array([1.0e+00, 1.38504825e+00, 1.98250413e+00, 2.76334481e+00,
                               3.90287080e+00, 6.04947969e+00, 1.11432274e+01, 1.84569631e+01,
                               2.97687575e+01, 5.19943446e+01, 8.27536813e+01, 1.37055175e+02,
                               2.39381640e+02, 3.66138292e+02, 5.38084620e+02, 7.90836777e+02,
                               1.16228559e+03, 1.55662838e+03])
    gamma_cyc__1_tau_cyc_norm = np.array([0.78342, 0.7808696, 0.77566082, 0.77311557, 0.75988023,
                                          0.74668088, 0.72285316, 0.7016581, 0.67777896, 0.64858305,
                                          0.62469877, 0.59280849, 0.56361259, 0.54773429, 0.52114535,
                                          0.50257782, 0.48133649, 0.46808058])

    gamma_cyc__3_N = np.array([1.0e+00, 1.50170392e+00, 2.48733090e+00, 4.06559252e+00,
                               6.30095355e+00, 1.01624041e+01, 1.82267733e+01, 2.78755196e+01,
                               4.20694563e+01, 6.02051754e+01, 8.61590202e+01, 1.46533554e+02,
                               1.93646242e+02, 2.92255899e+02, 3.96628631e+02, 5.10396006e+02,
                               8.12283497e+02, 1.10237259e+03, 1.49606058e+03, 1.64173510e+03])
    gamma_cyc__3_tau_cyc_norm = np.array([0.92780553, 0.91191695, 0.89072189, 0.8721955, 0.84562713,
                                          0.81907418, 0.78454095, 0.75796743, 0.73138877, 0.70478954,
                                          0.67819031, 0.64363651, 0.62235405, 0.59844919, 0.57985082,
                                          0.55588426, 0.52397855, 0.50538018, 0.4867818, 0.47879638])

    gamma_cyc__15_N = np.array([1.0e+00, 1.33557684e+00, 1.83639606e+00, 2.45881198e+00,
                                3.66106504e+00, 4.83769294e+00, 6.83084249e+00, 1.05848501e+01,
                                1.59726591e+01, 2.11061042e+01, 3.22752665e+01, 4.68042668e+01,
                                7.54768773e+01, 1.26666111e+02, 1.74184428e+02, 2.45989906e+02,
                                3.38280109e+02, 5.24237446e+02, 8.23301963e+02, 1.24246027e+03,
                                1.55703390e+03])
    gamma_cyc__15_tau_cyc_norm = np.array([1.21122896, 1.17390365, 1.13124616, 1.08857839, 1.03793034,
                                           1.00595265, 0.96330544, 0.91802042, 0.87807273, 0.84609504,
                                           0.8061525, 0.7742108, 0.73161501, 0.69170846, 0.66242,
                                           0.63848944, 0.61187478, 0.57728499, 0.54537414, 0.51344787,
                                           0.49749245])

    _shear_strains_interpolation = np.array([0.5, 1, 3, 15])
    _shearstress_ratios_interpolation = np.array([
        np.interp(np.log10(cycle_no), np.log10(gamma_cyc__0_5_N), gamma_cyc__0_5_tau_cyc_norm),
        np.interp(np.log10(cycle_no), np.log10(gamma_cyc__1_N), gamma_cyc__1_tau_cyc_norm),
        np.interp(np.log10(cycle_no), np.log10(gamma_cyc__3_N), gamma_cyc__3_tau_cyc_norm),
        np.interp(np.log10(cycle_no), np.log10(gamma_cyc__15_N), gamma_cyc__15_tau_cyc_norm)
    ])

    fit_coeff = np.polyfit(
        x=_shearstress_ratios_interpolation,
        y=np.log(_shear_strains_interpolation),
        deg=1)

    _strain_ratio_0 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0)
    _strain_ratio_1_5 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 1.5)

    _shearstress_ratios_interpolation = np.append(np.append(1e-7, _shearstress_ratios_interpolation), 1.5)
    _shear_strains_interpolation = np.append(
        np.append(_strain_ratio_0, _shear_strains_interpolation), _strain_ratio_1_5)

    _cyclic_strain = 10 ** (np.interp(
        cyclic_shear_stress / undrained_shear_strength,
        _shearstress_ratios_interpolation,
        np.log10(_shear_strains_interpolation)))

    if _cyclic_strain < 0.5:
        warnings.warn(
            "Cyclic strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _cyclic_strain > 15:
        warnings.warn(
            "Cyclic strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    return {
        'cyclic strain [%]': _cyclic_strain,
        'shear strains interpolation [%]': _shear_strains_interpolation,
        'shearstress ratios interpolation [-]': _shearstress_ratios_interpolation,
    }

def plotstrainaccumulation_dssclay_andersen():
    """
    Returns a Plotly figure with the strain accumulation contours for DSS tests on normally consolidated Drammen clay
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=True)

    gamma_cyc__0_5_N = np.array([1.0e+00, 1.34691510e+00, 1.95374983e+00, 3.11015790e+00,
                                 5.08421772e+00, 1.00090813e+01, 1.70260074e+01, 3.05427700e+01,
                                 6.01282407e+01, 9.31992345e+01, 2.09524007e+02, 3.70876480e+02,
                                 7.20425342e+02, 1.38095466e+03, 1.59819914e+03])
    gamma_cyc__0_5_tau_cyc_norm = np.array([0.62833926, 0.62845238, 0.62592255, 0.62342871, 0.61827135,
                                            0.60249075, 0.5893274, 0.57618462, 0.56040402, 0.54720467,
                                            0.52078027, 0.49961093, 0.47312996, 0.44664386, 0.44402661])

    gamma_cyc__1_N = np.array([1.0e+00, 1.38504825e+00, 1.98250413e+00, 2.76334481e+00,
                               3.90287080e+00, 6.04947969e+00, 1.11432274e+01, 1.84569631e+01,
                               2.97687575e+01, 5.19943446e+01, 8.27536813e+01, 1.37055175e+02,
                               2.39381640e+02, 3.66138292e+02, 5.38084620e+02, 7.90836777e+02,
                               1.16228559e+03, 1.55662838e+03])
    gamma_cyc__1_tau_cyc_norm = np.array([0.78342, 0.7808696, 0.77566082, 0.77311557, 0.75988023,
                                          0.74668088, 0.72285316, 0.7016581, 0.67777896, 0.64858305,
                                          0.62469877, 0.59280849, 0.56361259, 0.54773429, 0.52114535,
                                          0.50257782, 0.48133649, 0.46808058])

    gamma_cyc__3_N = np.array([1.0e+00, 1.50170392e+00, 2.48733090e+00, 4.06559252e+00,
                               6.30095355e+00, 1.01624041e+01, 1.82267733e+01, 2.78755196e+01,
                               4.20694563e+01, 6.02051754e+01, 8.61590202e+01, 1.46533554e+02,
                               1.93646242e+02, 2.92255899e+02, 3.96628631e+02, 5.10396006e+02,
                               8.12283497e+02, 1.10237259e+03, 1.49606058e+03, 1.64173510e+03])
    gamma_cyc__3_tau_cyc_norm = np.array([0.92780553, 0.91191695, 0.89072189, 0.8721955, 0.84562713,
                                          0.81907418, 0.78454095, 0.75796743, 0.73138877, 0.70478954,
                                          0.67819031, 0.64363651, 0.62235405, 0.59844919, 0.57985082,
                                          0.55588426, 0.52397855, 0.50538018, 0.4867818, 0.47879638])

    gamma_cyc__15_N = np.array([1.0e+00, 1.33557684e+00, 1.83639606e+00, 2.45881198e+00,
                                3.66106504e+00, 4.83769294e+00, 6.83084249e+00, 1.05848501e+01,
                                1.59726591e+01, 2.11061042e+01, 3.22752665e+01, 4.68042668e+01,
                                7.54768773e+01, 1.26666111e+02, 1.74184428e+02, 2.45989906e+02,
                                3.38280109e+02, 5.24237446e+02, 8.23301963e+02, 1.24246027e+03,
                                1.55703390e+03])
    gamma_cyc__15_tau_cyc_norm = np.array([1.21122896, 1.17390365, 1.13124616, 1.08857839, 1.03793034,
                                           1.00595265, 0.96330544, 0.91802042, 0.87807273, 0.84609504,
                                           0.8061525, 0.7742108, 0.73161501, 0.69170846, 0.66242,
                                           0.63848944, 0.61187478, 0.57728499, 0.54537414, 0.51344787,
                                           0.49749245])


    trace__0_5 = go.Scatter(
        x=gamma_cyc__0_5_N,
        y=gamma_cyc__0_5_tau_cyc_norm,
        showlegend=True, mode='lines', name=r'$ \gamma_{cy} = 0.5 \text{%} $')
    fig.append_trace(trace__0_5, 1, 1)

    trace__1 = go.Scatter(
        x=gamma_cyc__1_N,
        y=gamma_cyc__1_tau_cyc_norm,
        showlegend=True, mode='lines', name=r'$ \gamma_{cy} = 1 \text{%} $')
    fig.append_trace(trace__1, 1, 1)

    trace__3 = go.Scatter(
        x=gamma_cyc__3_N,
        y=gamma_cyc__3_tau_cyc_norm,
        showlegend=True, mode='lines', name=r'$ \gamma_{cy} = 3 \text{%} $')
    fig.append_trace(trace__3, 1, 1)

    trace__15 = go.Scatter(
        x=gamma_cyc__15_N,
        y=gamma_cyc__15_tau_cyc_norm,
        showlegend=True, mode='lines', name=r'$ \gamma_{cy} = 15 \text{%} $')
    fig.append_trace(trace__15, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 4), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{DSS} $', range=(0, 1.5), dtick=0.25)
    fig['layout'].update(height=500, width=500,
                         title=r'$ \text{Strain accumulation diagram for cyclic DSS tests om Drammen clay} $',
                         hovermode='closest')

    return fig

# Cyclic strain accumulation triaxial clay
STRAINACCUMULATION_TRIAXIALCLAY_ANDERSEN = {
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'cycle_no': {'type': 'int', 'min_value': 1.0, 'max_value': 1500.0},
}

STRAINACCUMULATION_TRIAXIALCLAY_ANDERSEN_ERRORRETURN = {
    'cyclic strain [%]': np.nan,
    'average strain [%]': np.nan,
    'cyclic shear strains interpolation [%]': None,
    'shearstress ratios interpolation cyclic [-]': None,
    'average shear strains interpolation [%]': None,
    'shearstress ratios interpolation average [-]': None,
}


@Validator(STRAINACCUMULATION_TRIAXIALCLAY_ANDERSEN, STRAINACCUMULATION_TRIAXIALCLAY_ANDERSEN_ERRORRETURN)
def strainaccumulation_triaxialclay_andersen(
        cyclic_shear_stress, undrained_shear_strength, cycle_no,
        **kwargs):
    """
    Calculates the cyclic and average strain accumulation for a normally consolidated clay sample under symmetrical cyclic loading (no average shear stress) in a cyclic triaxial test. The contours are based on cyclic triaxial tests on Drammen clay.

    Strain contours for cyclic shear strains of 0.05, 0.1, 0.25, 0.5, 1, 5 and 15% are defined and logarithmic interpolation is used to obtain the accumulated strain for a sample tested at a certain ratio of cyclic shear stress to triaxial compression shear strength with a given number of cycles.

    Similarly strain contours for average shear strains of -0.5, -0.75, -1, -1.5 and -4% are defined for linear interpolation of the average strains.

    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0
    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a triaxial compression test (:math:`S_u^{C}`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param cycle_no: Number of applied cycles (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= cycle_no <= 1500.0

    :returns: Dictionary with the following keys:

        - 'cyclic strain [%]': Accumulated cyclic shear strain (:math:`\\gamma_{cy}`)  [:math:`%`]
        - 'average strain [%]': Accumulated average shear strain (:math:`\\gamma_{a}`)  [:math:`%`]
        - 'cyclic shear strains interpolation [%]': List of cyclic shear strains used for interpolation [:math:`%`]
        - 'shearstress ratios interpolation cyclic [-]': List of ratios of cyclic shear stress to undrained triaxial compression shear strength used for interpolation of cyclic strain [:math:`-`]
        - 'average shear strains interpolation [%]': List of average shear strains used for interpolation [:math:`%`]
        - 'shearstress ratios interpolation average [-]': List of ratios of cyclic shear stress to undrained triaxial compression shear strength used for interpolation of average strain [:math:`-`]

    .. figure:: images/strainaccumulation_triaxialclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Strain contours for symmetrical cyclic triaxial tests

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    # Cyclic strain contours
    N__0_05 = np.array(
        [1.0, 1.1865345093719009, 1.331829401415281, 1.5054187499994125, 1.6779733026972448, 1.883446424584047,
         2.1140803781420363, 2.3729561865462725, 2.663532153974665, 2.989690064856397, 3.355787040364022,
         3.7667137448965087, 4.2279597201296815, 4.745686719427144, 5.326811022281069, 5.9790958284157965,
         6.711254965840776, 7.533069298281567, 8.455517387067891, 9.490922152981767, 10.653115497311543,
         11.957623081272919,
         13.421871732252884, 15.06542224760208, 16.910230705985647, 18.980941777133037, 21.305217948291432,
         23.914109065496568, 26.84246712633733, 30.129411865402428, 33.818853352132976, 37.960078582430036,
         42.60840990025414, 47.82594404502555, 53.6823816977097, 60.255958607435744, 67.6344907375061, 75.9165473928299,
         85.2127680005106, 95.64734013436764, 107.35965852822204, 120.50618724058916, 135.26254984731116,
         151.82587558486662, 170.41743278632754, 191.28558478985138, 214.70910780749085, 241.0009150774991,
         270.51223705082907, 303.6373134546492, 340.8186599138337, 382.5529794868592, 429.397798087913,
         481.97891243735944,
         540.9987500372303, 607.2457528520682, 681.6049100510847, 765.0695805177987, 858.7547630632699,
         963.9119916187759,
         1081.9460543918071, 1214.4337603354245, 1363.1450036309734, 1530.0664075831264, 1717.4278637844102,
         1927.7323210841287, 2163.7892223104764, 2428.752035424657, 2726.160380483212, 3059.987315179724,
         3434.692410650144,
         3855.281326577929, 4327.372681458558, 4857.273111338863, 5452.061519735669, 6119.68364422255,
         6869.058202993392,
         7710.196039407392, 8654.333856159132, 10000.0])

    tau_cy__Su__0_05 = np.array(
        [0.12285263083927245, 0.12176974655034055, 0.12176974655034055, 0.1225289687573582, 0.12496644284771764,
         0.12229478135709525, 0.12171450851337985, 0.1220131220597217, 0.12192671008111013, 0.12100407079048968,
         0.12027667881863148, 0.12059388734771236, 0.12052552344058287, 0.12050638154658665, 0.12039153018260895,
         0.11987469904470972, 0.11987469904470972, 0.11987469904470972, 0.11987469904470972, 0.11975984768073213,
         0.11924301654283274, 0.11924301654283274, 0.11924301654283274, 0.11924301654283274, 0.11918559086084403,
         0.11797965153907895, 0.11797965153907895, 0.11797965153907895, 0.11797965153907895, 0.11797965153907895,
         0.11746282040117972, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203,
         0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11711826630924682,
         0.11671628653532505, 0.11671628653532505, 0.11671628653532505, 0.11671628653532505, 0.11654400948935872,
         0.11562519857753772, 0.11551034721356003, 0.11545292153157127, 0.11545292153157127, 0.11551034721356003,
         0.11487866471168308, 0.11482123902969432, 0.11482123902969432, 0.11482123902969432, 0.11470638766571672,
         0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736,
         0.11401727948185104, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11367272538991813,
         0.11292619152406358, 0.11292619152406358, 0.11292619152406358, 0.11292619152406358, 0.11281134016008587,
         0.11212223197622018, 0.11229450902218664, 0.11229450902218664, 0.11229450902218664, 0.11194995493025384,
         0.11166282652030968, 0.11166282652030968, 0.11166282652030968, 0.11166282652030968, 0.10932326169854334])

    N__0_1 = np.array(
        [1.0, 1.1332698462678803, 1.2905248010344652, 1.4628213061518611, 1.6291382630133944, 1.8086803310017676,
         2.048515490402444, 2.2993626715643063, 2.5809268810286943, 2.89696951576793, 3.251712567674005,
         3.6498950248588318,
         4.09683617947157, 4.5985066877577845, 5.16160833164694, 5.793663547397762, 6.5031159173085475,
         7.2994429669542455,
         8.193282774800924, 9.196576085566702, 10.322725825811917, 11.586776152716189, 13.005613427942928,
         14.598191801387115, 16.385786418364486, 18.3922776328193, 20.64447002333557, 23.17245047366496,
         26.0099901013442,
         29.194995403739252, 32.770014648344905, 36.7828062721736, 41.28697688344144, 46.34269738857255,
         52.01750681121296,
         58.38721454141777, 65.53691306809583, 73.56211472373589, 82.57002762710842, 92.6809878691753,
         104.03006707467938,
         116.76887681472002, 131.06759397533816, 147.11723413887043, 165.13220334801971, 185.353162341481,
         208.05024152425239, 233.5266496211961, 262.1227242216096, 294.2204783257493, 330.24870363031386,
         370.6886987273237,
         416.08069873902014, 467.03109228182734, 524.2205221703108, 588.4129780778753, 660.4660026224419,
         741.3421472194908,
         832.1218307404387, 934.0177727542723, 1048.3911941651404, 1176.7700016690947, 1320.869198954903,
         1482.6137973207508, 1664.1645317682282, 1867.9467261130726, 2096.6826927197667, 2353.4280996858106,
         2641.612791302339, 2965.086607108889, 3328.170811635915, 3735.7158218813756, 4193.166003698952,
         4706.632402708219,
         5282.974381334157, 5929.891252559608, 6656.025134519524, 7471.076399964792, 8385.933262876733, 10000.0])

    tau_cy__Su__0_1 = np.array(
        [0.1919767446160904, 0.1891248589309531, 0.18853667280933195, 0.18813360832808368, 0.18923343775079612,
         0.1875609831267792, 0.18470829401007785, 0.18498174963859604, 0.18294723976242103, 0.1819518612746148,
         0.18126275309074916, 0.18034394217892813, 0.17925285422114068, 0.17856374603727498, 0.17764493512545398,
         0.17655384716766664, 0.17609444171175614, 0.174888502389991, 0.17379741443220353, 0.1729360292023715,
         0.1725914751104386, 0.17161523851662885, 0.17092613033276305, 0.1702944478308861, 0.16977761669298685,
         0.1691459341911099, 0.16851425168923295, 0.16793999486934486, 0.16748058941343436, 0.16702118395752394,
         0.16598752168172526, 0.16541326486183716, 0.1647815823599602, 0.16426475122206094, 0.16334594031023994,
         0.1628291091723406, 0.1623697037164301, 0.16156574416858682, 0.16070435893875468, 0.15995782507290013,
         0.15938356825301192, 0.1589815884790904, 0.15817762893124698, 0.15748852074738118, 0.1571439666554484,
         0.15668456119953778, 0.15582317596970574, 0.15513406778583996, 0.15461723664794058, 0.15461723664794058,
         0.15444495960197424, 0.15398555414606374, 0.153928128464075, 0.1533538716441868, 0.1533538716441868,
         0.1526647634603211, 0.15214793232242174, 0.15214793232242174, 0.1520905066404329, 0.15203308095844414,
         0.15111427004662314, 0.15111427004662314, 0.1507122902727014, 0.15019545913480215, 0.15019545913480215,
         0.1497934793608804, 0.1495637766329252, 0.1494489252689476, 0.14893209413104835, 0.14893209413104835,
         0.14893209413104835, 0.14893209413104835, 0.14893209413104835, 0.1481855602651937, 0.14766872912729445,
         0.14766872912729445, 0.1470370466254175, 0.1470370466254175, 0.14697962094342876, 0.14711361420140254])

    N__0_25 = np.array(
        [1.0, 1.2245107913529747, 1.3744559988714349, 1.532382403201714, 1.7316785989162955, 1.943728103668216,
         2.1817437389097534, 2.4489051391955456, 2.7487812953574497, 3.0853782324083907, 3.4631925257558964,
         3.887271370644703, 4.363280007292049, 4.8975774024433525, 5.497301198373059, 6.1704630641584615,
         6.926055723017713,
         7.774173085482791, 8.726145093258367, 9.794689075136311, 10.994079636919247, 12.340339354900891,
         13.8514528203644,
         15.547606894504366, 17.451460383321756, 19.58844673505989, 21.98711380389605, 24.67950521876832,
         27.70158845201801,
         31.093735306386968, 34.9012612391628, 39.17503072826625, 43.97213676732497, 49.3566635670615,
         55.400542651871135,
         62.184513788125145, 69.79920358117506, 78.34633614996076, 87.940092052542, 98.7086336163459, 110.7958170499137,
         124.36311420812856, 139.59176968365608, 156.19236939496108, 175.87182153850137, 197.4078806434623,
         221.58109809314925, 248.71440224233916, 279.1702650411075, 313.35554427275935, 351.72691873906683,
         394.79698900107354, 443.14112517485177, 497.4051532614391, 558.3139826921132, 626.6812903437906,
         703.4203903926436,
         789.5564352155737, 886.2401103305348, 994.7630063254878, 1116.574873128519, 1253.3029871177525,
         1406.7738897949216,
         1579.0377884280927, 1772.3959446299912, 1989.431416754153, 2233.0435667943725, 2506.4867927628266,
         2813.41400397004, 3157.925419989917, 3544.623345211966, 3978.673650710158, 4465.8747847604245,
         5012.73523391374,
         5626.560469421717, 6315.550540526286, 7088.909618355409, 7956.96895381415, 8931.324892057308, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [0.29341511001489784, 0.2908309543254013, 0.2885995563966932, 0.2870201450502795, 0.2845342282953281,
         0.28112601406929216, 0.2782465263009961, 0.2744072092766012, 0.27165077654113823, 0.26866464107772003,
         0.2660804853882236, 0.2633240526527606, 0.2606824712812753, 0.2580983155917788, 0.2559735653581928,
         0.2538488151246068, 0.2518963419369872, 0.2499438687493676, 0.2478765441977704, 0.24575179396418434,
         0.2437418950945759, 0.2417894219069563, 0.23977952303734795, 0.23771219848575065, 0.2355874482521647,
         0.233692400746534, 0.2315676505129479, 0.2296151773253283, 0.22743300140975345, 0.2256528052681004,
         0.2242745889003689, 0.2226092441226933, 0.22117360207297296, 0.21991023706921908, 0.21870429774745412,
         0.2171538043337562, 0.21594786501199115, 0.2146270743262485, 0.213306283640506, 0.2118706415907856,
         0.21054985090504288, 0.20917163453731136, 0.2076785668056024, 0.20641520180184847, 0.20532411384406116,
         0.20383104611235206, 0.20279738383655346, 0.20262510679058696, 0.2021082756526877, 0.20159144451478828,
         0.2010171876949003, 0.2006726336029674, 0.2000983767830793, 0.19969639700915756, 0.19906471450728055,
         0.19877758609733664, 0.1976864981395492, 0.1973419440476164, 0.19693996427369465, 0.19682511290971694,
         0.1961934304078401, 0.19533204517800795, 0.1950449167680639, 0.19498749108607505, 0.19378155176431,
         0.19360927471834355, 0.1932072949444218, 0.19314986926243305, 0.1932647206264108, 0.192862740852489,
         0.1925181867605562, 0.1925181867605562, 0.1925181867605562, 0.1925181867605562, 0.19234590971458976,
         0.19188650425867929, 0.19188650425867929, 0.1937399865087752, 0.19349442335436595, 0.19230762592659725])

    N__0_5 = np.array(
        [1.0, 1.1497360022240832, 1.2905248010344652, 1.4628853151854362, 1.6339213689285486, 1.8072337574998276,
         2.0449332087551237, 2.2993626715643063, 2.5809268810286943, 2.89696951576793, 3.251712567674005,
         3.6498950248588318, 4.09683617947157, 4.5985066877577845, 5.16160833164694, 5.793663547397762,
         6.5031159173085475,
         7.2994429669542455, 8.193282774800924, 9.196576085566702, 10.322725825811917, 11.586776152716189,
         13.005613427942928, 14.598191801387115, 16.385786418364486, 18.3922776328193, 20.64447002333557,
         23.17245047366496,
         26.0099901013442, 29.194995403739252, 32.770014648344905, 36.7828062721736, 41.28697688344144,
         46.34269738857255,
         52.01750681121296, 58.38721454141777, 65.53691306809583, 73.56211472373589, 82.57002762710842,
         92.6809878691753,
         104.03006707467938, 116.76887681472002, 131.06759397533816, 147.11723413887043, 165.13220334801971,
         185.353162341481, 208.05024152425239, 233.5266496211961, 262.1227242216096, 294.2204783257493,
         330.24870363031386,
         370.6886987273237, 416.08069873902014, 467.03109228182734, 524.2205221703108, 588.4129780778753,
         660.4660026224419,
         741.3421472194908, 832.1218307404387, 934.0177727542723, 1048.3911941651404, 1176.7700016690947,
         1320.869198954903,
         1482.6137973207508, 1664.1645317682282, 1867.9467261130726, 2096.6826927197667, 2353.4280996858106,
         2641.612791302339, 2843.1147766141357, 3290.760656089363, 4072.7606582952885, 4934.262184499117,
         5789.841390942728,
         7165.710514730556, 8408.212977420439, 10000.0])

    tau_cy__Su__0_5 = np.array(
        [0.3875952408295072, 0.3848367957410856, 0.3800704641360142, 0.3771984622155488, 0.3730109336301897,
         0.3697865453928312, 0.3638534173604203, 0.3591273179146945, 0.35342494769320576, 0.3490605958620561,
         0.3446962440309064, 0.3401021894718015, 0.33636952014252874, 0.3328665535412112, 0.3294210126218825,
         0.32591804602056496, 0.3226447821472028, 0.31919924122787396, 0.3156388489445677, 0.3123655850712056,
         0.30886261846988805, 0.30541707755055936, 0.3020289623132195, 0.2986982727578684, 0.2952527318385397,
         0.2918646166011999, 0.2892804609117033, 0.2865814538582292, 0.28376759544077745, 0.2811834397512809,
         0.2783121556518404, 0.27567057428035496, 0.2732012699548362, 0.2703299858553956, 0.2676884044839103,
         0.26487454606645844, 0.26246266742292845, 0.25970623468746545, 0.2568923762700137, 0.2542507948985284,
         0.2523557473928977, 0.2502884228413005, 0.2484508010176585, 0.24621119942009484, 0.24425872623247524,
         0.2423636787268444, 0.2402963541752472, 0.23851615803359416, 0.2364488334819969, 0.2344389346123884,
         0.2322567586968137, 0.2304765625551605, 0.2286963664135073, 0.22731815004577585, 0.2263993391339549,
         0.22507854844821215, 0.2241597375363912, 0.22283894685064853, 0.22128845343695056, 0.22054191957109606,
         0.21927855456734224, 0.2183023179735324, 0.2169241016058011, 0.21594786501199115, 0.21479935137221493,
         0.2135359863684612, 0.21273202682061776, 0.21215777000072955, 0.211468661816864, 0.2110475401489459,
         0.2109264389042782, 0.2109264389042782, 0.20964399302349435, 0.2083615471427105, 0.2083615471427105,
         0.2057966553811429, 0.2057966553811429])

    N__1 = np.array(
        [1.0, 1.1865345093719009, 1.331829401415281, 1.4949161111320235, 1.6779733026972448, 1.883446424584047,
         2.1140803781420363, 2.3729561865462725, 2.663532153974665, 2.989690064856397, 3.355787040364022,
         3.7667137448965087, 4.2279597201296815, 4.745686719427144, 5.326811022281069, 5.9790958284157965,
         6.711254965840776, 7.533069298281567, 8.455517387067891, 9.490922152981767, 10.653115497311543,
         11.957623081272919, 13.421871732252884, 15.06542224760208, 16.910230705985647, 18.980941777133037,
         21.305217948291432, 23.914109065496568, 26.84246712633733, 30.129411865402428, 33.818853352132976,
         37.960078582430036, 42.60840990025414, 47.82594404502555, 53.6823816977097, 60.255958607435744,
         67.6344907375061, 75.9165473928299, 85.2127680005106, 95.64734013436764, 107.35965852822204,
         120.50618724058916, 135.26254984731116, 151.82587558486662, 170.41743278632754, 191.28558478985138,
         214.70910780749085, 241.0009150774991, 270.51223705082907, 303.6373134546492, 340.8186599138337,
         382.5529794868592, 429.397798087913, 481.97891243735944, 541.4359499509518, 607.2457528520682,
         681.6049100510847, 737.459743083853, 868.1456482321752, 1074.44746406809, 1329.773817784096,
         1699.2577497015247, 2103.0609134339356, 2630.7166535542683, 3397.701212647088, 4435.329876687908,
         5789.841390942728, 7639.009269809079, 10000.0])

    tau_cy__Su__1 = np.array(
        [0.4713759312200817, 0.4667832439391194, 0.4608109730122829, 0.4559776447782247, 0.4494406879784984,
         0.4436543668790544, 0.4369218893049375, 0.4311218954240675, 0.4261832867730297, 0.4210149753940367,
         0.41619121810697657, 0.4111951837739499, 0.4064288521688786, 0.4016625205638073, 0.3966664862307807,
         0.3918427289437205, 0.3869615459746716, 0.3821377886876114, 0.3772566057185625, 0.3729496795694016,
         0.3686427534202408, 0.36410612454312463, 0.3598566240759525, 0.3554922722448029, 0.3510704947316644,
         0.3467061429005148, 0.342341791069365, 0.33797743923821544, 0.3336130874070657, 0.32947843830387136,
         0.32580319465658736, 0.32235765373725866, 0.3186249844079859, 0.31517944348865723, 0.3115041998413733,
         0.30805865892204465, 0.3044408409567495, 0.3008804486734432, 0.2973774820721258, 0.2938170897888195,
         0.29065867727943484, 0.2873854134060726, 0.2842844265786768, 0.28124086543326976, 0.2779676015599075,
         0.2748666147325117, 0.27165077654113823, 0.26843493834976484, 0.2652765258403802, 0.262290390376962,
         0.25970623468746545, 0.2580983155917788, 0.2559735653581928, 0.2542972823553764, 0.2543120068892197,
         0.2513877144679435, 0.250221152756685, 0.2499223341186217, 0.24555247768544164, 0.24298758592387396,
         0.23785780240073864, 0.23401046475838705, 0.2301631271160356, 0.2301631271160356, 0.22503334359290028,
         0.22246845183133254, 0.22118600595054885, 0.21733866830819729, 0.2160562224274135])

    N__5 = np.array(
        [1.0, 1.1057532154694976, 1.1990605690444205, 1.3458893165917682, 1.5168798916123598, 1.6956874050284159,
         1.9033296746016315, 2.136398394819973, 2.398007114739408, 2.6916506473154294, 3.0212517563697388,
         3.391213560523237, 3.8064783541558285, 4.272593631178242, 4.795786193622877, 5.3830453350653205,
         6.042216210118048,
         6.782104637313132, 7.612594735428312, 8.544780964752203, 9.591116337218699, 10.765578775339751,
         12.083857842313964,
         13.563564337825818, 15.224465559511357, 17.088749372932266, 19.181320617750877, 21.530133809775826,
         24.166566583422536, 27.12583886338028, 30.447483365171703, 34.17587370998946, 38.360817208859686,
         43.05822023509576, 48.33083507370677, 54.249098224871204, 60.892071360314034, 68.34849750275971,
         76.71798653788906,
         86.11234589597359, 96.65707417967165, 108.49303769125844, 121.77835225590118, 136.80095982561224,
         153.64365916707808, 173.62890027624505, 194.93579285028736, 216.45057045021017, 243.2043505704545,
         272.5249490863361, 303.8366695054138, 342.5071158144816, 384.4481921365666, 435.8113290418528,
         489.06029911189927,
         551.7993676617061, 610.9770850018531, 660.4660026224419, 744.6861860281357, 835.5270987436363,
         942.7126380697476,
         1058.7174281457048, 1173.2444758060851, 1315.1021917354185, 1469.3282380640617, 1647.860944420293,
         1838.7528737488406, 2194.675777400952, 2687.4063148588807, 3434.1148442300123, 4341.768532021323,
         5914.60744927765,
         7720.877583648423, 10000.0])

    tau_cy__Su__5 = np.array(
        [0.5856780339313605, 0.5871392697188125, 0.5737098638022862, 0.5630861126343559, 0.5576092825384771,
         0.5438839643994395, 0.5312148591305658, 0.5212227904645126, 0.511977255664314, 0.5025020181361601,
         0.4930842062899951, 0.48395352285377397, 0.4756267989653963, 0.4671277980310522, 0.45868622277869703,
         0.4502446475263417, 0.4418604979559753, 0.4343951592974297, 0.42859516541655984, 0.4225654688077346,
         0.4166506235628871, 0.4107932040000283, 0.4048783587551807, 0.3991357905562996, 0.39413975622327296,
         0.3890862962082575, 0.38409026187523104, 0.3793239302701597, 0.3742130445731554, 0.3692744359221176,
         0.36422097590710223, 0.3595120699840197, 0.35486058974292606, 0.3500368324558658, 0.3455002035787497,
         0.34061902060970073, 0.33591011468661824, 0.3312586344455245, 0.3266071542044308, 0.3221853766912923,
         0.3181655789520754, 0.31403092984888104, 0.3101834091556307, 0.30656117383018266, 0.3025741234542333,
         0.29885842660430384, 0.29603470555385025, 0.2915238908880663, 0.288487002059995, 0.28506480459996675,
         0.2827013919451852, 0.2795275854924775, 0.2765066568217368, 0.2727926468177041, 0.2704385629318966,
         0.26683377521666496, 0.26554419332476153, 0.2603889622399991, 0.2610286661069793, 0.25855841987873995,
         0.2558598399879796, 0.2535170587560885, 0.2521745830390043, 0.2508228576852857, 0.2491610564367963,
         0.2464283402801149, 0.2461322391073601, 0.24042269416230636, 0.23785780240073864, 0.2352929106391709,
         0.2327280188776032, 0.2288806812352517, 0.2263157894736841, 0.22375089771211645])

    N__15 = np.array(
        [1.0, 1.1497360022240832, 1.2703553872138815, 1.4184469053064708, 1.5921402014962855, 1.7871027894928069,
         2.005939161143921, 2.251572736536749, 2.5272849177661842, 2.8367589249605314, 3.1841290001667035,
         3.5740356364063173, 4.011687569703847, 4.502931362234112, 5.054329506145568, 5.673248091443044,
         6.367955209079157,
         7.1477314038145385, 8.022993652379116, 9.005434523150093, 10.1081783763714, 11.345956691582439,
         12.73530486444342,
         14.294783101952369, 16.045224366978154, 18.010012684383273, 20.21539552660888, 22.69083445185993,
         25.4693986790415,
         28.588206857187195, 32.08892293094692, 36.01831272636747, 40.42886869235664, 45.379511143709756,
         50.93637537355583,
         57.17369515020483, 64.17479440096957, 72.03320033079872, 80.85389284579588, 90.75470697258905,
         101.86790700837318,
         114.34195343058953, 127.53734832665909, 141.43625475500417, 156.68522204103215, 175.87182153850137,
         199.49188483654282, 227.4754744078551, 259.38444312898747, 294.2204783257493, 337.25827690916964,
         382.5529794868592, 436.2153406200673, 494.80024562013926, 564.2080162899272, 643.3519151690132,
         733.5976710741156,
         840.9064110660272, 953.8424259623092, 1093.3679900647378, 1240.210280702683, 1421.6249822411414,
         1629.5765497017624, 1848.433106233945, 2118.817044747381, 2403.379928534568, 2754.940030238924,
         3157.925419989917,
         3600.9012779914533, 4084.5120996234837, 4681.983798586977, 5338.745917540053, 6087.634899687736,
         6978.118139172122,
         7956.96895381415, 10000.0])

    tau_cy__Su__15 = np.array(
        [0.6684048134018362, 0.6544886564665475, 0.6406969218422347, 0.6269434710059144, 0.6145969493783199,
         0.6024227047966919, 0.5904781629430191, 0.5783039183613912, 0.5664168021897071, 0.5553336455658666,
         0.5449970228078805, 0.5346604000498945, 0.5243812029738972, 0.5140445802159112, 0.5038802345038915,
         0.4934287603819279, 0.4830347119419529, 0.4744208596436313, 0.46689809530309706, 0.4593179052805739,
         0.4517377152580508, 0.44438722796348296, 0.436749612258971, 0.4292268479184368, 0.4218189349418801,
         0.4152724071951556, 0.4090130078583751, 0.4027536085215947, 0.39626450645685896, 0.3899476814380897,
         0.3839179848292645, 0.37863482208629373, 0.37369621343525616, 0.3685853277382519, 0.36370414476920293,
         0.35859325907219874, 0.3537120761031497, 0.3486586160881343, 0.3437774331190854, 0.3386091217400924,
         0.3337279387710433, 0.3290190328479609, 0.32567685815621195, 0.3212234965179797, 0.3167069666295597,
         0.3114008336137936, 0.30634737359877817, 0.3016939791682848, 0.2980091645740027, 0.2947244155642427,
         0.2906816475522304, 0.2870178890413442, 0.283375186613854, 0.2794798111856128, 0.2756476040075595,
         0.2728471449159051, 0.26958345198954103, 0.26730939498278405, 0.26440365547415023, 0.26137157946514106,
         0.2585921764568826, 0.25694980195200257, 0.25417039894374405, 0.2521490149377379, 0.2501276309317317,
         0.24760090092422404, 0.245958526419344, 0.24393714241333786, 0.2406945055703697, 0.2385257289805922,
         0.23663068147496136, 0.23572526988893794, 0.2336828297995358, 0.2318930627108845, 0.2309605790176377,
         0.22873465020149986])

    # Average shear strain contours
    N__min0_5 = np.array(
        [1.0, 1.2076819171550277, 1.35494265326115, 1.5366925486068341, 1.7088298995028626, 1.905167773142429,
         2.1468135020679053, 2.4085888355446423, 2.702284186829562, 3.0317917772536167, 3.4014784327353893,
         3.816243455493788, 4.281583552445862, 4.803665680758549, 5.389408775945934, 6.046575445620196,
         6.7838748440750685,
         7.611078091055592, 8.53914776431058, 9.64742174762404, 10.748583400593693, 12.059230087594043,
         13.588786426871279,
         15.108179062992349, 17.106905371261533, 19.29530515567808, 21.227686722208645, 24.147758952660176,
         27.226491201169118, 29.978349769328137, 33.96524405612947, 38.10685350699588, 42.753477107478254,
         47.96669461161624, 53.815594604856294, 60.37769010610285, 67.73994581525304, 75.99993061989628,
         85.26711063486663,
         99.72008664647265, 120.41666620968843, 132.30303154863387, 162.5163845427891, 209.68414326770167,
         307.3040228417602, 450.3715015501375, 10000.0])

    tau_cy__Su__min0_5 = np.array(
        [0.010839011326454706, 0.01089085080651242, 0.009979695249829, 0.01003605793727047, 0.01001058177765679,
         0.0088466611196365, 0.009672923792148014, 0.009628954954874702, 0.008595084296389555, 0.008951984791673695,
         0.009202532198728576, 0.009174925375029286, 0.00914731855133022, 0.009119711727631152, 0.009092104903931864,
         0.009064498080232795, 0.008796915057451837, 0.007569427238343306, 0.007541820414644019, 0.005879202445113307,
         0.007512513288737033, 0.005614182913104226, 0.005435545251960106, 0.004954408708292224, 0.0051921061066362295,
         0.0050992551491473295, 0.005076354034033281, 0.005170681083028096, 0.005033107931903924, 0.004993533562935859,
         0.005365721499811738, 0.005338114676112671, 0.004819647445200914, 0.005147914416730792, 0.0055702629663101355,
         0.005542656142610847, 0.006030452746485615, 0.0061173800178024864, 0.006662443669185336, 0.005198135226639922,
         0.006006952723005997, 0.005984365321797469, 0.002523138467207309, 0.002461995684134521, 0.0010913782969206311,
         0.0009996641223113392, 0.0])

    N__min0_75 = np.array(
        [1.0, 1.1703744478768865, 1.3130860346497983, 1.4731994000040771, 1.6528364592280498, 1.854377866937737,
         2.08049456689417, 2.3341831889009974, 2.618805761882665, 2.9381342694438928, 3.2964006383866256,
         3.698352822661314,
         4.149317726009574, 4.655271797185682, 5.222920233325705, 5.8597858411133625, 6.603023387372847,
         7.412049572754518,
         8.275358112123852, 9.355917229836805, 10.390631248651713, 11.741835973147946, 13.07146150192738,
         14.579240947102337, 15.808554901463149, 17.430877489300066, 19.018403698027424, 21.213814093608626,
         23.800556376214338, 26.7027174518336, 29.95875844419536, 33.61183029916742, 37.71034564614689,
         42.308620390336046,
         47.467593538656814, 53.255634798855226, 59.74945065456746, 67.03510092418111, 75.20913927551955,
         84.3798928111122,
         94.66889768185835, 106.21250975466879, 119.16371167959043, 133.69414030471458, 149.99636130734967,
         168.28642118619175, 188.8067104349728, 211.8291758420296, 237.65892448704983, 266.63826719725404,
         299.1512550492434, 335.6287690368665, 376.55423035634425, 422.4700069846627, 473.9846014549328,
         531.7807150852908,
         596.6242955331776, 669.3746875784484, 750.996021659378, 842.5699910890552, 945.3101872832622,
         1060.578182978589,
         1189.9015765849472, 1334.9942368067032, 1497.779015825923, 1680.4132320559845, 1885.3172601778733,
         2115.2066073508468, 2373.12790069007, 2662.4992629382136, 2987.1556114127557, 3351.399480557073,
         3760.0580416251546, 4218.547075157965, 4732.942744051923, 5310.062118397695, 5957.5535192514635,
         6683.997878626583,
         7499.022458987134, 8413.428439320318, 9507.935290267791, 10000.0])

    tau_cy__Su__min0_75 = np.array(
        [0.20648178350778207, 0.2024184758234364, 0.20071103560616388, 0.19890815030971096, 0.19716253206076664,
         0.1949015103842484, 0.19304135804028744, 0.1915248079813756, 0.1893783203998736, 0.1877472362459456,
         0.18571528275946012, 0.1837978633679911, 0.18165137578648904, 0.17996302458505298, 0.1782746733836167,
         0.17612818580211487, 0.17268547793312994, 0.17179416460745012, 0.17003868834793698, 0.16857692858109252,
         0.16758504611776948, 0.16613846246083394, 0.1630009964855157, 0.16359322665114462, 0.157936053081837,
         0.1571458864634503, 0.15532501183484304, 0.1554324224060215, 0.153744071204585, 0.15176938476560786,
         0.14996649946915494, 0.14839268236273506, 0.14647526297126578, 0.14501597995986248, 0.14332762875842595,
         0.14163927755698946, 0.14029452864060232, 0.13860617743916606, 0.1369750932852376, 0.13528674208380131,
         0.1337129249773814, 0.13202457377594512, 0.13067982485955776, 0.1293923429906787, 0.12764672474173389,
         0.12635924287285485, 0.1247854257664347, 0.1233261427550314, 0.1215805245060868, 0.12029304263720753,
         0.11877649257829592, 0.11731720956689218, 0.11625879588804587, 0.1149713140191666, 0.11362656510277945,
         0.1123390832339004, 0.11093706727000474, 0.1097068524486342, 0.10819030238972216, 0.10707462166336756,
         0.1057298727469802, 0.10484326021065836, 0.10361304538928763, 0.10255463171044088, 0.10178255326913566,
         0.10060960549527276, 0.09966572591144264, 0.09855004518508824, 0.09743436445873388, 0.09649048487490376,
         0.0956038723385817, 0.0944881916122271, 0.09394518136095442, 0.09334490406217344, 0.09263009266837607,
         0.091972548322087, 0.09137227102330624, 0.09082926077203313, 0.08988538118820323, 0.08922783684191392,
         0.08836516258833793, 0.08947943655832225])

    N__min1 = np.array(
        [1.0, 1.1342194735100033, 1.272522442364956, 1.4276896175228573, 1.6017773644875204, 1.7970927951666715,
         2.016224343058584, 2.2620760666757853, 2.5379061358147244, 2.847370010713784, 3.194568886334925,
         3.5841040437806986, 4.021137829143174, 4.5114620679119035, 5.100947784983413, 5.85593545935241,
         6.722668325492858,
         7.508286350276546, 8.450296576088531, 9.480697773981188, 10.63674268379077, 11.933751883925517,
         13.388914093420711,
         15.02151396682401, 16.85318766563534, 18.90820959328212, 21.213814093608626, 23.800556376214338,
         26.7027174518336,
         29.95875844419536, 33.61183029916742, 37.71034564614689, 42.308620390336046, 47.467593538656814,
         53.255634798855226, 59.74945065456746, 67.03510092418111, 75.20913927551955, 84.3798928111122,
         94.66889768185835,
         106.21250975466879, 119.16371167959043, 133.69414030471458, 149.99636130734967, 168.28642118619175,
         188.8067104349728, 211.8291758420296, 237.65892448704983, 266.63826719725404, 299.1512550492434,
         335.6287690368665,
         376.55423035634425, 422.4700069846627, 473.9846014549328, 531.7807150852908, 596.6242955331776,
         669.3746875784484,
         750.996021659378, 842.5699910890552, 945.3101872832622, 1060.578182978589, 1189.9015765849472,
         1334.9942368067032,
         1497.779015825923, 1680.4132320559845, 1885.3172601778733, 2115.2066073508468, 2373.12790069007,
         2662.4992629382136, 2987.1556114127557, 3351.399480557073, 3760.0580416251546, 4218.547075157965,
         4732.942744051923, 5310.062118397695, 5957.5535192514635, 6683.997878626583, 7499.022458987134,
         8413.428439320318,
         9507.935290267791, 10000.0])

    tau_cy__Su__min1 = np.array(
        [0.29017207839562104, 0.2860740723507864, 0.2827249767716125, 0.2791468130024064, 0.27585498447074075,
         0.2724486218440594, 0.2689277251223614, 0.26552136249567937, 0.2623440680590305, 0.2591667736223813,
         0.2555313428056667, 0.25246858246403425, 0.2492340209798771, 0.2462285276857525, 0.24250856202772295,
         0.23863873338241026, 0.23476890473709802, 0.2311615142500512, 0.2289550978060844, 0.2258923374644517,
         0.22288684417032711, 0.21999588497121916, 0.21687585758207814, 0.21409943247798635, 0.21143754146891086,
         0.20831751407977014, 0.20548382192817027, 0.20287919796660286, 0.2001027728625111, 0.19744088185343567,
         0.19443538855931108, 0.19205983278777672, 0.1892261406361768, 0.18685058486464226, 0.18407415976055064,
         0.1814695357989835, 0.17892217888492426, 0.17648935606588154, 0.1739419991518223, 0.17139464223776346,
         0.16907635351373718, 0.16715893412226812, 0.16524151473079907, 0.16372496467188702, 0.16192207937543435,
         0.16000465998396507, 0.15837357583003708, 0.15639888939105995, 0.15459600409460705, 0.15307945403569545,
         0.15121930169173448, 0.14964548458531435, 0.14789986633636953, 0.1459251798973924, 0.14440862983848102,
         0.14277754568455234, 0.14091739334059186, 0.13957264442420425, 0.13788429322276796, 0.13642501021136444,
         0.13479392605743626, 0.1333919100935408, 0.1316462918445962, 0.13041607702322544, 0.12924312924936232,
         0.12789838033297518, 0.12643909732157166, 0.12532341659521706, 0.12386413358381355, 0.12274845285745872,
         0.12111736870353075, 0.12028802321471675, 0.11900054134583772, 0.11782759357197505, 0.11665464579811236,
         0.11553896507175798, 0.11448055139291125, 0.11302126838150772, 0.1120773887976776, 0.10994825273961162,
         0.10963962260800007])

    N__min1_5 = np.array(
        [1.0, 1.2058786996514277, 1.33116985999329, 1.504342371300126, 1.687776901375026, 1.8935788309633208,
         2.124475566735869, 2.3835270863065103, 2.6741664908321976, 3.000245502462977, 3.366085509600439,
         3.723304616114597,
         4.1929459947865535, 4.715949931939756, 5.264053884127996, 5.87342200082079, 6.422381480879633,
         7.093221985851589,
         7.936270964554134, 8.903993580564675, 9.989717089654713, 11.207830130197673, 12.574475843510461,
         14.10776581213578,
         15.828020085050607, 17.75803647075413, 19.923392667063663, 22.35278523161796, 25.07840988532185,
         28.13638818873424,
         31.567246245962853, 35.41645178012751, 39.735016698028005, 44.58017312955347, 50.01613190109953,
         56.11493349472425,
         62.957402770446976, 70.63422010422465, 79.24712313694826, 88.91025506073403, 99.7516773107834,
         111.91506670988188,
         125.56161955708099, 140.87218789465354, 158.04967626436942, 177.3217307163015, 198.94375570646469,
         223.20229886496892, 250.41884849159942, 280.95409410543823, 315.212706511808, 353.6487007347582,
         396.7714528877894,
         445.15245072188316, 499.4328673130128, 560.332058260048, 628.6570950028489, 705.3134606016156,
         791.3170497241789,
         887.8076318720188, 996.0639562675128, 1117.5206985811444, 1253.7874740865368, 1406.6701692166987,
         1578.1948742196148, 1770.6347340826942, 1986.5400735700428, 2228.7721956070905, 2500.541300928414,
         2805.4490315218554, 3147.536201679493, 3531.3363492149433, 3961.9358165388708, 4445.041157822073,
         4987.054765564828, 5595.159718819952, 6277.4149775272745, 7042.862184530201, 7901.645490676839,
         8865.145990997167,
         10000.0])

    tau_cy__Su__min1_5 = np.array(
        [0.3981293820657492, 0.3950402521658147, 0.388253138877265, 0.3809550749157222, 0.3745135587711073,
         0.3687019801490816, 0.3628904015270564, 0.3571933570000476, 0.3513817783780224, 0.34591380204104616,
         0.3409411856650155, 0.3336886563314567, 0.33115022030009666, 0.3273300738303431, 0.32317335163326244,
         0.32227315605439744, 0.31617180518884513, 0.3139504757865685, 0.3051925963069433, 0.30069815977760594,
         0.29671912667584244, 0.2923392242415217, 0.2881311229497254, 0.28398028870543746, 0.2799439885561654,
         0.2760794895494185, 0.2722722575901795, 0.2683504915359243, 0.2643141913866525, 0.2605069594274134,
         0.2568142615631908, 0.2528924955089353, 0.24937159878723736, 0.2455643668279985, 0.24187166896377565,
         0.2381789710995532, 0.2346580743778548, 0.2307363083235996, 0.22710087750688526, 0.2235799807851873,
         0.2203454193010299, 0.21774079533946256, 0.21542250661543627, 0.21298968379639335, 0.21078592916738348,
         0.2080667711108, 0.20592028352929811, 0.20365926185278016, 0.20134097312875388, 0.1989654173572193,
         0.1967043956807009, 0.1943861069566748, 0.19223961937517275, 0.1902649329361956, 0.18788937716466148,
         0.18614375891571644, 0.1838254701916904, 0.1819080508002211, 0.1797042961712112, 0.17784414382725022,
         0.17609852557830585, 0.1741238391393285, 0.17232095384287582, 0.17051806854642315, 0.1685433821074458,
         0.1668550309060095, 0.1650521456095566, 0.1635355955506448, 0.16201904549173318, 0.16038796133780495,
         0.1585850760413523, 0.15724032712496494, 0.15543744182851227, 0.15397815881710852, 0.15274794399573774,
         0.15123139393682594, 0.14982937797293028, 0.14842736200903486, 0.14719714718766366, 0.14556606303373565,
         0.1428137355034198])

    N__min4 = np.array(
        [1.0, 1.1012737577522174, 1.2110551407838497, 1.3682348666691897, 1.557717189980799, 1.7234559734636448,
         1.9471388829875944, 2.1524345888831977, 2.400728861236578, 2.726620261193849, 3.056556910956348,
         3.4733846154249237, 3.9252388488063032, 4.396230027854009, 4.905213660218264, 5.503339179943854,
         6.082823773147818,
         6.651667016193109, 7.883325493618918, 9.442774131174305, 10.861600073824892, 12.186027639925847,
         13.671951520190976, 15.339064040690939, 17.209458744564888, 19.30792318848289, 21.66226744174,
         24.3036926414425,
         27.26720448809405, 30.59207716146492, 34.32237380482709, 38.507530475314454, 43.20301129925711,
         48.47104351498918,
         54.3814421443436, 61.012535221860034, 68.45220184338028, 76.79903679085352, 86.16365716763647,
         96.67016836057745,
         108.45780875666522, 121.68279501098316, 136.52039232053002, 153.16723714037934, 171.84394312416023,
         192.7980248242953, 216.30717789857098, 242.68295929423843, 272.27491618158666, 305.47521835600963,
         342.7238554998492, 384.51446818114965, 431.40088986509835, 484.004486636694, 543.022391904904,
         609.2367452193934,
         683.525057637993, 766.8718410127659, 860.3816553127351, 965.2937468939599, 1082.998471711822,
         1215.0557211254657,
         1363.2155944838432, 1529.441592458494, 1715.9366384945677, 1925.1722732314386, 2159.9214087943988,
         2423.295077036219, 2718.7836587377233, 3050.3031401605986, 3422.2470099711463, 3839.544484303239,
         4307.725831592646, 4832.995662905565, 5422.315159047212, 6083.494324171839, 6825.295488492801,
         7657.549431770331,
         8591.285666513302, 10000.0])

    tau_cy__Su__min4 = np.array(
        [0.6532151007005733, 0.6456295759512285, 0.637026081186364, 0.6219350950467912, 0.6065906934036011,
         0.592524189212255, 0.5820231842004857, 0.5718847859452879, 0.5591968466301058, 0.54552740705088,
         0.5342161185496164, 0.5186533484157879, 0.5086463223607629, 0.5014347839460722, 0.4879072055067,
         0.4778260503426757, 0.4721871311626917, 0.4611373209841214, 0.4483075270026908, 0.4341962821926941,
         0.4248054342616767, 0.4169895089768652, 0.4090017825495289, 0.4013003913597337, 0.39394260245498747,
         0.3867566146927659, 0.37968516102556094, 0.373014576690913, 0.3662294582612489, 0.35978794211663323,
         0.35340369301952634, 0.34742031325497624, 0.3415514675854432, 0.3359689571534505, 0.3301573785314253,
         0.32491847038448185, 0.3198513633800626, 0.3146697222806276, 0.310118018703782, 0.305337246936904,
         0.3008428104075671, 0.2966919761632789, 0.29236934077646626, 0.28839030767470275, 0.2846976098104801,
         0.2810621789937657, 0.27754128227206776, 0.27419218669289397, 0.2709003581612286, 0.2675512625820551,
         0.2643167010978977, 0.2615402759938059, 0.2589356520322388, 0.2561019598806389, 0.253268267729039,
         0.2505491096724555, 0.2485171561859698, 0.2463134015569597, 0.2439951128329336, 0.2416768241089071,
         0.2394730694798972, 0.2379565194209854, 0.23592456593449995, 0.23417894768555536, 0.23260513057913526,
         0.23051591004514194, 0.2291138940812465, 0.2275973440223347, 0.22625259510594756, 0.2246787779995274,
         0.22321949498812368, 0.22216108130927714, 0.22064453125036526, 0.2197006516665354, 0.21835590275014805,
         0.21683935269123625, 0.2157236719648816, 0.21443619009600254, 0.21332050936964772, 0.2121484742180564])

    _shear_strains_interpolation = np.array([0.05, 0.1, 0.25, 0.5, 1, 5, 15])
    _shearstress_ratios_interpolation = np.array([
        np.interp(np.log10(cycle_no), np.log10(N__0_05), tau_cy__Su__0_05),
        np.interp(np.log10(cycle_no), np.log10(N__0_1), tau_cy__Su__0_1),
        np.interp(np.log10(cycle_no), np.log10(N__0_25), tau_cy__Su__0_25),
        np.interp(np.log10(cycle_no), np.log10(N__0_5), tau_cy__Su__0_5),
        np.interp(np.log10(cycle_no), np.log10(N__1), tau_cy__Su__1),
        np.interp(np.log10(cycle_no), np.log10(N__5), tau_cy__Su__5),
        np.interp(np.log10(cycle_no), np.log10(N__15), tau_cy__Su__15)
    ])

    fit_coeff = np.polyfit(
        x=_shearstress_ratios_interpolation,
        y=np.log(_shear_strains_interpolation),
        deg=1)

    _strain_ratio_0 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0)
    _strain_ratio_1 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 1)

    _shearstress_ratios_interpolation = np.append(np.append(1e-7, _shearstress_ratios_interpolation), 1)
    _shear_strains_interpolation = np.append(
        np.append(_strain_ratio_0, _shear_strains_interpolation), _strain_ratio_1)

    _cyclic_strain = 10 ** (np.interp(
        cyclic_shear_stress / undrained_shear_strength,
        _shearstress_ratios_interpolation,
        np.log10(_shear_strains_interpolation)))

    if _cyclic_strain < 0.05:
        warnings.warn(
            "Cyclic strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _cyclic_strain > 15:
        warnings.warn(
            "Cyclic strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    # Average strain interpolation

    _average_strains_interpolation = np.array([-0.5, -0.75, -1, -1.5, -4])
    _average_shearstress_ratios_interpolation = np.array([
        np.interp(np.log10(cycle_no), np.log10(N__min0_5), tau_cy__Su__min0_5),
        np.interp(np.log10(cycle_no), np.log10(N__min0_75), tau_cy__Su__min0_75),
        np.interp(np.log10(cycle_no), np.log10(N__min1), tau_cy__Su__min1),
        np.interp(np.log10(cycle_no), np.log10(N__min1_5), tau_cy__Su__min1_5),
        np.interp(np.log10(cycle_no), np.log10(N__min4), tau_cy__Su__min4),
    ])

    fit_coeff = np.polyfit(
        x=_average_shearstress_ratios_interpolation,
        y=_average_strains_interpolation,
        deg=1)

    _strain_ratio_0 = fit_coeff[1] + fit_coeff[0] * 0
    _strain_ratio_1 = fit_coeff[1] + fit_coeff[0] * 1

    _average_shearstress_ratios_interpolation = np.append(np.append(1e-7, _average_shearstress_ratios_interpolation), 1)
    _average_strains_interpolation = np.append(
        np.append(_strain_ratio_0, _average_strains_interpolation), _strain_ratio_1)

    _average_strain = np.interp(
        cyclic_shear_stress / undrained_shear_strength,
        _average_shearstress_ratios_interpolation,
        _average_strains_interpolation)

    if _average_strain > -0.05:
        warnings.warn(
            "Average strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _average_strain < -4:
        warnings.warn(
            "Average strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    return {
        'cyclic strain [%]': _cyclic_strain,
        'average strain [%]': _average_strain,
        'shear strains interpolation [%]': _shear_strains_interpolation,
        'shearstress ratios interpolation [-]': _shearstress_ratios_interpolation,
        'average shear strains interpolation [%]': _average_strains_interpolation,
        'shearstress ratios interpolation average [-]': _average_shearstress_ratios_interpolation,
    }

def plotstrainaccumulation_triaxialclay_andersen():
    """
    Returns a Plotly figure with the strain accumulation contours for triaxial tests on normally consolidated Drammen clay with symmetrical loading
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=2, cols=1, print_grid=False, shared_yaxes=True)

    # Cyclic strain contours
    N__0_05 = np.array(
        [1.0, 1.1865345093719009, 1.331829401415281, 1.5054187499994125, 1.6779733026972448, 1.883446424584047,
         2.1140803781420363, 2.3729561865462725, 2.663532153974665, 2.989690064856397, 3.355787040364022,
         3.7667137448965087, 4.2279597201296815, 4.745686719427144, 5.326811022281069, 5.9790958284157965,
         6.711254965840776, 7.533069298281567, 8.455517387067891, 9.490922152981767, 10.653115497311543,
         11.957623081272919,
         13.421871732252884, 15.06542224760208, 16.910230705985647, 18.980941777133037, 21.305217948291432,
         23.914109065496568, 26.84246712633733, 30.129411865402428, 33.818853352132976, 37.960078582430036,
         42.60840990025414, 47.82594404502555, 53.6823816977097, 60.255958607435744, 67.6344907375061, 75.9165473928299,
         85.2127680005106, 95.64734013436764, 107.35965852822204, 120.50618724058916, 135.26254984731116,
         151.82587558486662, 170.41743278632754, 191.28558478985138, 214.70910780749085, 241.0009150774991,
         270.51223705082907, 303.6373134546492, 340.8186599138337, 382.5529794868592, 429.397798087913,
         481.97891243735944,
         540.9987500372303, 607.2457528520682, 681.6049100510847, 765.0695805177987, 858.7547630632699,
         963.9119916187759,
         1081.9460543918071, 1214.4337603354245, 1363.1450036309734, 1530.0664075831264, 1717.4278637844102,
         1927.7323210841287, 2163.7892223104764, 2428.752035424657, 2726.160380483212, 3059.987315179724,
         3434.692410650144,
         3855.281326577929, 4327.372681458558, 4857.273111338863, 5452.061519735669, 6119.68364422255,
         6869.058202993392,
         7710.196039407392, 8654.333856159132, 10000.0])

    tau_cy__Su__0_05 = np.array(
        [0.12285263083927245, 0.12176974655034055, 0.12176974655034055, 0.1225289687573582, 0.12496644284771764,
         0.12229478135709525, 0.12171450851337985, 0.1220131220597217, 0.12192671008111013, 0.12100407079048968,
         0.12027667881863148, 0.12059388734771236, 0.12052552344058287, 0.12050638154658665, 0.12039153018260895,
         0.11987469904470972, 0.11987469904470972, 0.11987469904470972, 0.11987469904470972, 0.11975984768073213,
         0.11924301654283274, 0.11924301654283274, 0.11924301654283274, 0.11924301654283274, 0.11918559086084403,
         0.11797965153907895, 0.11797965153907895, 0.11797965153907895, 0.11797965153907895, 0.11797965153907895,
         0.11746282040117972, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203,
         0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11734796903720203, 0.11711826630924682,
         0.11671628653532505, 0.11671628653532505, 0.11671628653532505, 0.11671628653532505, 0.11654400948935872,
         0.11562519857753772, 0.11551034721356003, 0.11545292153157127, 0.11545292153157127, 0.11551034721356003,
         0.11487866471168308, 0.11482123902969432, 0.11482123902969432, 0.11482123902969432, 0.11470638766571672,
         0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736,
         0.11401727948185104, 0.11418955652781736, 0.11418955652781736, 0.11418955652781736, 0.11367272538991813,
         0.11292619152406358, 0.11292619152406358, 0.11292619152406358, 0.11292619152406358, 0.11281134016008587,
         0.11212223197622018, 0.11229450902218664, 0.11229450902218664, 0.11229450902218664, 0.11194995493025384,
         0.11166282652030968, 0.11166282652030968, 0.11166282652030968, 0.11166282652030968, 0.10932326169854334])

    N__0_1 = np.array(
        [1.0, 1.1332698462678803, 1.2905248010344652, 1.4628213061518611, 1.6291382630133944, 1.8086803310017676,
         2.048515490402444, 2.2993626715643063, 2.5809268810286943, 2.89696951576793, 3.251712567674005,
         3.6498950248588318,
         4.09683617947157, 4.5985066877577845, 5.16160833164694, 5.793663547397762, 6.5031159173085475,
         7.2994429669542455,
         8.193282774800924, 9.196576085566702, 10.322725825811917, 11.586776152716189, 13.005613427942928,
         14.598191801387115, 16.385786418364486, 18.3922776328193, 20.64447002333557, 23.17245047366496,
         26.0099901013442,
         29.194995403739252, 32.770014648344905, 36.7828062721736, 41.28697688344144, 46.34269738857255,
         52.01750681121296,
         58.38721454141777, 65.53691306809583, 73.56211472373589, 82.57002762710842, 92.6809878691753,
         104.03006707467938,
         116.76887681472002, 131.06759397533816, 147.11723413887043, 165.13220334801971, 185.353162341481,
         208.05024152425239, 233.5266496211961, 262.1227242216096, 294.2204783257493, 330.24870363031386,
         370.6886987273237,
         416.08069873902014, 467.03109228182734, 524.2205221703108, 588.4129780778753, 660.4660026224419,
         741.3421472194908,
         832.1218307404387, 934.0177727542723, 1048.3911941651404, 1176.7700016690947, 1320.869198954903,
         1482.6137973207508, 1664.1645317682282, 1867.9467261130726, 2096.6826927197667, 2353.4280996858106,
         2641.612791302339, 2965.086607108889, 3328.170811635915, 3735.7158218813756, 4193.166003698952,
         4706.632402708219,
         5282.974381334157, 5929.891252559608, 6656.025134519524, 7471.076399964792, 8385.933262876733, 10000.0])

    tau_cy__Su__0_1 = np.array(
        [0.1919767446160904, 0.1891248589309531, 0.18853667280933195, 0.18813360832808368, 0.18923343775079612,
         0.1875609831267792, 0.18470829401007785, 0.18498174963859604, 0.18294723976242103, 0.1819518612746148,
         0.18126275309074916, 0.18034394217892813, 0.17925285422114068, 0.17856374603727498, 0.17764493512545398,
         0.17655384716766664, 0.17609444171175614, 0.174888502389991, 0.17379741443220353, 0.1729360292023715,
         0.1725914751104386, 0.17161523851662885, 0.17092613033276305, 0.1702944478308861, 0.16977761669298685,
         0.1691459341911099, 0.16851425168923295, 0.16793999486934486, 0.16748058941343436, 0.16702118395752394,
         0.16598752168172526, 0.16541326486183716, 0.1647815823599602, 0.16426475122206094, 0.16334594031023994,
         0.1628291091723406, 0.1623697037164301, 0.16156574416858682, 0.16070435893875468, 0.15995782507290013,
         0.15938356825301192, 0.1589815884790904, 0.15817762893124698, 0.15748852074738118, 0.1571439666554484,
         0.15668456119953778, 0.15582317596970574, 0.15513406778583996, 0.15461723664794058, 0.15461723664794058,
         0.15444495960197424, 0.15398555414606374, 0.153928128464075, 0.1533538716441868, 0.1533538716441868,
         0.1526647634603211, 0.15214793232242174, 0.15214793232242174, 0.1520905066404329, 0.15203308095844414,
         0.15111427004662314, 0.15111427004662314, 0.1507122902727014, 0.15019545913480215, 0.15019545913480215,
         0.1497934793608804, 0.1495637766329252, 0.1494489252689476, 0.14893209413104835, 0.14893209413104835,
         0.14893209413104835, 0.14893209413104835, 0.14893209413104835, 0.1481855602651937, 0.14766872912729445,
         0.14766872912729445, 0.1470370466254175, 0.1470370466254175, 0.14697962094342876, 0.14711361420140254])

    N__0_25 = np.array(
        [1.0, 1.2245107913529747, 1.3744559988714349, 1.532382403201714, 1.7316785989162955, 1.943728103668216,
         2.1817437389097534, 2.4489051391955456, 2.7487812953574497, 3.0853782324083907, 3.4631925257558964,
         3.887271370644703, 4.363280007292049, 4.8975774024433525, 5.497301198373059, 6.1704630641584615,
         6.926055723017713,
         7.774173085482791, 8.726145093258367, 9.794689075136311, 10.994079636919247, 12.340339354900891,
         13.8514528203644,
         15.547606894504366, 17.451460383321756, 19.58844673505989, 21.98711380389605, 24.67950521876832,
         27.70158845201801,
         31.093735306386968, 34.9012612391628, 39.17503072826625, 43.97213676732497, 49.3566635670615,
         55.400542651871135,
         62.184513788125145, 69.79920358117506, 78.34633614996076, 87.940092052542, 98.7086336163459, 110.7958170499137,
         124.36311420812856, 139.59176968365608, 156.19236939496108, 175.87182153850137, 197.4078806434623,
         221.58109809314925, 248.71440224233916, 279.1702650411075, 313.35554427275935, 351.72691873906683,
         394.79698900107354, 443.14112517485177, 497.4051532614391, 558.3139826921132, 626.6812903437906,
         703.4203903926436,
         789.5564352155737, 886.2401103305348, 994.7630063254878, 1116.574873128519, 1253.3029871177525,
         1406.7738897949216,
         1579.0377884280927, 1772.3959446299912, 1989.431416754153, 2233.0435667943725, 2506.4867927628266,
         2813.41400397004, 3157.925419989917, 3544.623345211966, 3978.673650710158, 4465.8747847604245,
         5012.73523391374,
         5626.560469421717, 6315.550540526286, 7088.909618355409, 7956.96895381415, 8931.324892057308, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [0.29341511001489784, 0.2908309543254013, 0.2885995563966932, 0.2870201450502795, 0.2845342282953281,
         0.28112601406929216, 0.2782465263009961, 0.2744072092766012, 0.27165077654113823, 0.26866464107772003,
         0.2660804853882236, 0.2633240526527606, 0.2606824712812753, 0.2580983155917788, 0.2559735653581928,
         0.2538488151246068, 0.2518963419369872, 0.2499438687493676, 0.2478765441977704, 0.24575179396418434,
         0.2437418950945759, 0.2417894219069563, 0.23977952303734795, 0.23771219848575065, 0.2355874482521647,
         0.233692400746534, 0.2315676505129479, 0.2296151773253283, 0.22743300140975345, 0.2256528052681004,
         0.2242745889003689, 0.2226092441226933, 0.22117360207297296, 0.21991023706921908, 0.21870429774745412,
         0.2171538043337562, 0.21594786501199115, 0.2146270743262485, 0.213306283640506, 0.2118706415907856,
         0.21054985090504288, 0.20917163453731136, 0.2076785668056024, 0.20641520180184847, 0.20532411384406116,
         0.20383104611235206, 0.20279738383655346, 0.20262510679058696, 0.2021082756526877, 0.20159144451478828,
         0.2010171876949003, 0.2006726336029674, 0.2000983767830793, 0.19969639700915756, 0.19906471450728055,
         0.19877758609733664, 0.1976864981395492, 0.1973419440476164, 0.19693996427369465, 0.19682511290971694,
         0.1961934304078401, 0.19533204517800795, 0.1950449167680639, 0.19498749108607505, 0.19378155176431,
         0.19360927471834355, 0.1932072949444218, 0.19314986926243305, 0.1932647206264108, 0.192862740852489,
         0.1925181867605562, 0.1925181867605562, 0.1925181867605562, 0.1925181867605562, 0.19234590971458976,
         0.19188650425867929, 0.19188650425867929, 0.1937399865087752, 0.19349442335436595, 0.19230762592659725])

    N__0_5 = np.array(
        [1.0, 1.1497360022240832, 1.2905248010344652, 1.4628853151854362, 1.6339213689285486, 1.8072337574998276,
         2.0449332087551237, 2.2993626715643063, 2.5809268810286943, 2.89696951576793, 3.251712567674005,
         3.6498950248588318, 4.09683617947157, 4.5985066877577845, 5.16160833164694, 5.793663547397762,
         6.5031159173085475,
         7.2994429669542455, 8.193282774800924, 9.196576085566702, 10.322725825811917, 11.586776152716189,
         13.005613427942928, 14.598191801387115, 16.385786418364486, 18.3922776328193, 20.64447002333557,
         23.17245047366496,
         26.0099901013442, 29.194995403739252, 32.770014648344905, 36.7828062721736, 41.28697688344144,
         46.34269738857255,
         52.01750681121296, 58.38721454141777, 65.53691306809583, 73.56211472373589, 82.57002762710842,
         92.6809878691753,
         104.03006707467938, 116.76887681472002, 131.06759397533816, 147.11723413887043, 165.13220334801971,
         185.353162341481, 208.05024152425239, 233.5266496211961, 262.1227242216096, 294.2204783257493,
         330.24870363031386,
         370.6886987273237, 416.08069873902014, 467.03109228182734, 524.2205221703108, 588.4129780778753,
         660.4660026224419,
         741.3421472194908, 832.1218307404387, 934.0177727542723, 1048.3911941651404, 1176.7700016690947,
         1320.869198954903,
         1482.6137973207508, 1664.1645317682282, 1867.9467261130726, 2096.6826927197667, 2353.4280996858106,
         2641.612791302339, 2843.1147766141357, 3290.760656089363, 4072.7606582952885, 4934.262184499117,
         5789.841390942728,
         7165.710514730556, 8408.212977420439, 10000.0])

    tau_cy__Su__0_5 = np.array(
        [0.3875952408295072, 0.3848367957410856, 0.3800704641360142, 0.3771984622155488, 0.3730109336301897,
         0.3697865453928312, 0.3638534173604203, 0.3591273179146945, 0.35342494769320576, 0.3490605958620561,
         0.3446962440309064, 0.3401021894718015, 0.33636952014252874, 0.3328665535412112, 0.3294210126218825,
         0.32591804602056496, 0.3226447821472028, 0.31919924122787396, 0.3156388489445677, 0.3123655850712056,
         0.30886261846988805, 0.30541707755055936, 0.3020289623132195, 0.2986982727578684, 0.2952527318385397,
         0.2918646166011999, 0.2892804609117033, 0.2865814538582292, 0.28376759544077745, 0.2811834397512809,
         0.2783121556518404, 0.27567057428035496, 0.2732012699548362, 0.2703299858553956, 0.2676884044839103,
         0.26487454606645844, 0.26246266742292845, 0.25970623468746545, 0.2568923762700137, 0.2542507948985284,
         0.2523557473928977, 0.2502884228413005, 0.2484508010176585, 0.24621119942009484, 0.24425872623247524,
         0.2423636787268444, 0.2402963541752472, 0.23851615803359416, 0.2364488334819969, 0.2344389346123884,
         0.2322567586968137, 0.2304765625551605, 0.2286963664135073, 0.22731815004577585, 0.2263993391339549,
         0.22507854844821215, 0.2241597375363912, 0.22283894685064853, 0.22128845343695056, 0.22054191957109606,
         0.21927855456734224, 0.2183023179735324, 0.2169241016058011, 0.21594786501199115, 0.21479935137221493,
         0.2135359863684612, 0.21273202682061776, 0.21215777000072955, 0.211468661816864, 0.2110475401489459,
         0.2109264389042782, 0.2109264389042782, 0.20964399302349435, 0.2083615471427105, 0.2083615471427105,
         0.2057966553811429, 0.2057966553811429])

    N__1 = np.array(
        [1.0, 1.1865345093719009, 1.331829401415281, 1.4949161111320235, 1.6779733026972448, 1.883446424584047,
         2.1140803781420363, 2.3729561865462725, 2.663532153974665, 2.989690064856397, 3.355787040364022,
         3.7667137448965087, 4.2279597201296815, 4.745686719427144, 5.326811022281069, 5.9790958284157965,
         6.711254965840776, 7.533069298281567, 8.455517387067891, 9.490922152981767, 10.653115497311543,
         11.957623081272919, 13.421871732252884, 15.06542224760208, 16.910230705985647, 18.980941777133037,
         21.305217948291432, 23.914109065496568, 26.84246712633733, 30.129411865402428, 33.818853352132976,
         37.960078582430036, 42.60840990025414, 47.82594404502555, 53.6823816977097, 60.255958607435744,
         67.6344907375061, 75.9165473928299, 85.2127680005106, 95.64734013436764, 107.35965852822204,
         120.50618724058916, 135.26254984731116, 151.82587558486662, 170.41743278632754, 191.28558478985138,
         214.70910780749085, 241.0009150774991, 270.51223705082907, 303.6373134546492, 340.8186599138337,
         382.5529794868592, 429.397798087913, 481.97891243735944, 541.4359499509518, 607.2457528520682,
         681.6049100510847, 737.459743083853, 868.1456482321752, 1074.44746406809, 1329.773817784096,
         1699.2577497015247, 2103.0609134339356, 2630.7166535542683, 3397.701212647088, 4435.329876687908,
         5789.841390942728, 7639.009269809079, 10000.0])

    tau_cy__Su__1 = np.array(
        [0.4713759312200817, 0.4667832439391194, 0.4608109730122829, 0.4559776447782247, 0.4494406879784984,
         0.4436543668790544, 0.4369218893049375, 0.4311218954240675, 0.4261832867730297, 0.4210149753940367,
         0.41619121810697657, 0.4111951837739499, 0.4064288521688786, 0.4016625205638073, 0.3966664862307807,
         0.3918427289437205, 0.3869615459746716, 0.3821377886876114, 0.3772566057185625, 0.3729496795694016,
         0.3686427534202408, 0.36410612454312463, 0.3598566240759525, 0.3554922722448029, 0.3510704947316644,
         0.3467061429005148, 0.342341791069365, 0.33797743923821544, 0.3336130874070657, 0.32947843830387136,
         0.32580319465658736, 0.32235765373725866, 0.3186249844079859, 0.31517944348865723, 0.3115041998413733,
         0.30805865892204465, 0.3044408409567495, 0.3008804486734432, 0.2973774820721258, 0.2938170897888195,
         0.29065867727943484, 0.2873854134060726, 0.2842844265786768, 0.28124086543326976, 0.2779676015599075,
         0.2748666147325117, 0.27165077654113823, 0.26843493834976484, 0.2652765258403802, 0.262290390376962,
         0.25970623468746545, 0.2580983155917788, 0.2559735653581928, 0.2542972823553764, 0.2543120068892197,
         0.2513877144679435, 0.250221152756685, 0.2499223341186217, 0.24555247768544164, 0.24298758592387396,
         0.23785780240073864, 0.23401046475838705, 0.2301631271160356, 0.2301631271160356, 0.22503334359290028,
         0.22246845183133254, 0.22118600595054885, 0.21733866830819729, 0.2160562224274135])

    N__5 = np.array(
        [1.0, 1.1057532154694976, 1.1990605690444205, 1.3458893165917682, 1.5168798916123598, 1.6956874050284159,
         1.9033296746016315, 2.136398394819973, 2.398007114739408, 2.6916506473154294, 3.0212517563697388,
         3.391213560523237, 3.8064783541558285, 4.272593631178242, 4.795786193622877, 5.3830453350653205,
         6.042216210118048,
         6.782104637313132, 7.612594735428312, 8.544780964752203, 9.591116337218699, 10.765578775339751,
         12.083857842313964,
         13.563564337825818, 15.224465559511357, 17.088749372932266, 19.181320617750877, 21.530133809775826,
         24.166566583422536, 27.12583886338028, 30.447483365171703, 34.17587370998946, 38.360817208859686,
         43.05822023509576, 48.33083507370677, 54.249098224871204, 60.892071360314034, 68.34849750275971,
         76.71798653788906,
         86.11234589597359, 96.65707417967165, 108.49303769125844, 121.77835225590118, 136.80095982561224,
         153.64365916707808, 173.62890027624505, 194.93579285028736, 216.45057045021017, 243.2043505704545,
         272.5249490863361, 303.8366695054138, 342.5071158144816, 384.4481921365666, 435.8113290418528,
         489.06029911189927,
         551.7993676617061, 610.9770850018531, 660.4660026224419, 744.6861860281357, 835.5270987436363,
         942.7126380697476,
         1058.7174281457048, 1173.2444758060851, 1315.1021917354185, 1469.3282380640617, 1647.860944420293,
         1838.7528737488406, 2194.675777400952, 2687.4063148588807, 3434.1148442300123, 4341.768532021323,
         5914.60744927765,
         7720.877583648423, 10000.0])

    tau_cy__Su__5 = np.array(
        [0.5856780339313605, 0.5871392697188125, 0.5737098638022862, 0.5630861126343559, 0.5576092825384771,
         0.5438839643994395, 0.5312148591305658, 0.5212227904645126, 0.511977255664314, 0.5025020181361601,
         0.4930842062899951, 0.48395352285377397, 0.4756267989653963, 0.4671277980310522, 0.45868622277869703,
         0.4502446475263417, 0.4418604979559753, 0.4343951592974297, 0.42859516541655984, 0.4225654688077346,
         0.4166506235628871, 0.4107932040000283, 0.4048783587551807, 0.3991357905562996, 0.39413975622327296,
         0.3890862962082575, 0.38409026187523104, 0.3793239302701597, 0.3742130445731554, 0.3692744359221176,
         0.36422097590710223, 0.3595120699840197, 0.35486058974292606, 0.3500368324558658, 0.3455002035787497,
         0.34061902060970073, 0.33591011468661824, 0.3312586344455245, 0.3266071542044308, 0.3221853766912923,
         0.3181655789520754, 0.31403092984888104, 0.3101834091556307, 0.30656117383018266, 0.3025741234542333,
         0.29885842660430384, 0.29603470555385025, 0.2915238908880663, 0.288487002059995, 0.28506480459996675,
         0.2827013919451852, 0.2795275854924775, 0.2765066568217368, 0.2727926468177041, 0.2704385629318966,
         0.26683377521666496, 0.26554419332476153, 0.2603889622399991, 0.2610286661069793, 0.25855841987873995,
         0.2558598399879796, 0.2535170587560885, 0.2521745830390043, 0.2508228576852857, 0.2491610564367963,
         0.2464283402801149, 0.2461322391073601, 0.24042269416230636, 0.23785780240073864, 0.2352929106391709,
         0.2327280188776032, 0.2288806812352517, 0.2263157894736841, 0.22375089771211645])

    N__15 = np.array(
        [1.0, 1.1497360022240832, 1.2703553872138815, 1.4184469053064708, 1.5921402014962855, 1.7871027894928069,
         2.005939161143921, 2.251572736536749, 2.5272849177661842, 2.8367589249605314, 3.1841290001667035,
         3.5740356364063173, 4.011687569703847, 4.502931362234112, 5.054329506145568, 5.673248091443044,
         6.367955209079157,
         7.1477314038145385, 8.022993652379116, 9.005434523150093, 10.1081783763714, 11.345956691582439,
         12.73530486444342,
         14.294783101952369, 16.045224366978154, 18.010012684383273, 20.21539552660888, 22.69083445185993,
         25.4693986790415,
         28.588206857187195, 32.08892293094692, 36.01831272636747, 40.42886869235664, 45.379511143709756,
         50.93637537355583,
         57.17369515020483, 64.17479440096957, 72.03320033079872, 80.85389284579588, 90.75470697258905,
         101.86790700837318,
         114.34195343058953, 127.53734832665909, 141.43625475500417, 156.68522204103215, 175.87182153850137,
         199.49188483654282, 227.4754744078551, 259.38444312898747, 294.2204783257493, 337.25827690916964,
         382.5529794868592, 436.2153406200673, 494.80024562013926, 564.2080162899272, 643.3519151690132,
         733.5976710741156,
         840.9064110660272, 953.8424259623092, 1093.3679900647378, 1240.210280702683, 1421.6249822411414,
         1629.5765497017624, 1848.433106233945, 2118.817044747381, 2403.379928534568, 2754.940030238924,
         3157.925419989917,
         3600.9012779914533, 4084.5120996234837, 4681.983798586977, 5338.745917540053, 6087.634899687736,
         6978.118139172122,
         7956.96895381415, 10000.0])

    tau_cy__Su__15 = np.array(
        [0.6684048134018362, 0.6544886564665475, 0.6406969218422347, 0.6269434710059144, 0.6145969493783199,
         0.6024227047966919, 0.5904781629430191, 0.5783039183613912, 0.5664168021897071, 0.5553336455658666,
         0.5449970228078805, 0.5346604000498945, 0.5243812029738972, 0.5140445802159112, 0.5038802345038915,
         0.4934287603819279, 0.4830347119419529, 0.4744208596436313, 0.46689809530309706, 0.4593179052805739,
         0.4517377152580508, 0.44438722796348296, 0.436749612258971, 0.4292268479184368, 0.4218189349418801,
         0.4152724071951556, 0.4090130078583751, 0.4027536085215947, 0.39626450645685896, 0.3899476814380897,
         0.3839179848292645, 0.37863482208629373, 0.37369621343525616, 0.3685853277382519, 0.36370414476920293,
         0.35859325907219874, 0.3537120761031497, 0.3486586160881343, 0.3437774331190854, 0.3386091217400924,
         0.3337279387710433, 0.3290190328479609, 0.32567685815621195, 0.3212234965179797, 0.3167069666295597,
         0.3114008336137936, 0.30634737359877817, 0.3016939791682848, 0.2980091645740027, 0.2947244155642427,
         0.2906816475522304, 0.2870178890413442, 0.283375186613854, 0.2794798111856128, 0.2756476040075595,
         0.2728471449159051, 0.26958345198954103, 0.26730939498278405, 0.26440365547415023, 0.26137157946514106,
         0.2585921764568826, 0.25694980195200257, 0.25417039894374405, 0.2521490149377379, 0.2501276309317317,
         0.24760090092422404, 0.245958526419344, 0.24393714241333786, 0.2406945055703697, 0.2385257289805922,
         0.23663068147496136, 0.23572526988893794, 0.2336828297995358, 0.2318930627108845, 0.2309605790176377,
         0.22873465020149986])

    # Average shear strain contours
    N__min0_5 = np.array(
        [1.0, 1.2076819171550277, 1.35494265326115, 1.5366925486068341, 1.7088298995028626, 1.905167773142429,
         2.1468135020679053, 2.4085888355446423, 2.702284186829562, 3.0317917772536167, 3.4014784327353893,
         3.816243455493788, 4.281583552445862, 4.803665680758549, 5.389408775945934, 6.046575445620196,
         6.7838748440750685,
         7.611078091055592, 8.53914776431058, 9.64742174762404, 10.748583400593693, 12.059230087594043,
         13.588786426871279,
         15.108179062992349, 17.106905371261533, 19.29530515567808, 21.227686722208645, 24.147758952660176,
         27.226491201169118, 29.978349769328137, 33.96524405612947, 38.10685350699588, 42.753477107478254,
         47.96669461161624, 53.815594604856294, 60.37769010610285, 67.73994581525304, 75.99993061989628,
         85.26711063486663,
         99.72008664647265, 120.41666620968843, 132.30303154863387, 162.5163845427891, 209.68414326770167,
         307.3040228417602, 450.3715015501375, 10000.0])

    tau_cy__Su__min0_5 = np.array(
        [0.010839011326454706, 0.01089085080651242, 0.009979695249829, 0.01003605793727047, 0.01001058177765679,
         0.0088466611196365, 0.009672923792148014, 0.009628954954874702, 0.008595084296389555, 0.008951984791673695,
         0.009202532198728576, 0.009174925375029286, 0.00914731855133022, 0.009119711727631152, 0.009092104903931864,
         0.009064498080232795, 0.008796915057451837, 0.007569427238343306, 0.007541820414644019, 0.005879202445113307,
         0.007512513288737033, 0.005614182913104226, 0.005435545251960106, 0.004954408708292224, 0.0051921061066362295,
         0.0050992551491473295, 0.005076354034033281, 0.005170681083028096, 0.005033107931903924, 0.004993533562935859,
         0.005365721499811738, 0.005338114676112671, 0.004819647445200914, 0.005147914416730792, 0.0055702629663101355,
         0.005542656142610847, 0.006030452746485615, 0.0061173800178024864, 0.006662443669185336, 0.005198135226639922,
         0.006006952723005997, 0.005984365321797469, 0.002523138467207309, 0.002461995684134521, 0.0010913782969206311,
         0.0009996641223113392, 0.0])

    N__min0_75 = np.array(
        [1.0, 1.1703744478768865, 1.3130860346497983, 1.4731994000040771, 1.6528364592280498, 1.854377866937737,
         2.08049456689417, 2.3341831889009974, 2.618805761882665, 2.9381342694438928, 3.2964006383866256,
         3.698352822661314,
         4.149317726009574, 4.655271797185682, 5.222920233325705, 5.8597858411133625, 6.603023387372847,
         7.412049572754518,
         8.275358112123852, 9.355917229836805, 10.390631248651713, 11.741835973147946, 13.07146150192738,
         14.579240947102337, 15.808554901463149, 17.430877489300066, 19.018403698027424, 21.213814093608626,
         23.800556376214338, 26.7027174518336, 29.95875844419536, 33.61183029916742, 37.71034564614689,
         42.308620390336046,
         47.467593538656814, 53.255634798855226, 59.74945065456746, 67.03510092418111, 75.20913927551955,
         84.3798928111122,
         94.66889768185835, 106.21250975466879, 119.16371167959043, 133.69414030471458, 149.99636130734967,
         168.28642118619175, 188.8067104349728, 211.8291758420296, 237.65892448704983, 266.63826719725404,
         299.1512550492434, 335.6287690368665, 376.55423035634425, 422.4700069846627, 473.9846014549328,
         531.7807150852908,
         596.6242955331776, 669.3746875784484, 750.996021659378, 842.5699910890552, 945.3101872832622,
         1060.578182978589,
         1189.9015765849472, 1334.9942368067032, 1497.779015825923, 1680.4132320559845, 1885.3172601778733,
         2115.2066073508468, 2373.12790069007, 2662.4992629382136, 2987.1556114127557, 3351.399480557073,
         3760.0580416251546, 4218.547075157965, 4732.942744051923, 5310.062118397695, 5957.5535192514635,
         6683.997878626583,
         7499.022458987134, 8413.428439320318, 9507.935290267791, 10000.0])

    tau_cy__Su__min0_75 = np.array(
        [0.20648178350778207, 0.2024184758234364, 0.20071103560616388, 0.19890815030971096, 0.19716253206076664,
         0.1949015103842484, 0.19304135804028744, 0.1915248079813756, 0.1893783203998736, 0.1877472362459456,
         0.18571528275946012, 0.1837978633679911, 0.18165137578648904, 0.17996302458505298, 0.1782746733836167,
         0.17612818580211487, 0.17268547793312994, 0.17179416460745012, 0.17003868834793698, 0.16857692858109252,
         0.16758504611776948, 0.16613846246083394, 0.1630009964855157, 0.16359322665114462, 0.157936053081837,
         0.1571458864634503, 0.15532501183484304, 0.1554324224060215, 0.153744071204585, 0.15176938476560786,
         0.14996649946915494, 0.14839268236273506, 0.14647526297126578, 0.14501597995986248, 0.14332762875842595,
         0.14163927755698946, 0.14029452864060232, 0.13860617743916606, 0.1369750932852376, 0.13528674208380131,
         0.1337129249773814, 0.13202457377594512, 0.13067982485955776, 0.1293923429906787, 0.12764672474173389,
         0.12635924287285485, 0.1247854257664347, 0.1233261427550314, 0.1215805245060868, 0.12029304263720753,
         0.11877649257829592, 0.11731720956689218, 0.11625879588804587, 0.1149713140191666, 0.11362656510277945,
         0.1123390832339004, 0.11093706727000474, 0.1097068524486342, 0.10819030238972216, 0.10707462166336756,
         0.1057298727469802, 0.10484326021065836, 0.10361304538928763, 0.10255463171044088, 0.10178255326913566,
         0.10060960549527276, 0.09966572591144264, 0.09855004518508824, 0.09743436445873388, 0.09649048487490376,
         0.0956038723385817, 0.0944881916122271, 0.09394518136095442, 0.09334490406217344, 0.09263009266837607,
         0.091972548322087, 0.09137227102330624, 0.09082926077203313, 0.08988538118820323, 0.08922783684191392,
         0.08836516258833793, 0.08947943655832225])

    N__min1 = np.array(
        [1.0, 1.1342194735100033, 1.272522442364956, 1.4276896175228573, 1.6017773644875204, 1.7970927951666715,
         2.016224343058584, 2.2620760666757853, 2.5379061358147244, 2.847370010713784, 3.194568886334925,
         3.5841040437806986, 4.021137829143174, 4.5114620679119035, 5.100947784983413, 5.85593545935241,
         6.722668325492858,
         7.508286350276546, 8.450296576088531, 9.480697773981188, 10.63674268379077, 11.933751883925517,
         13.388914093420711,
         15.02151396682401, 16.85318766563534, 18.90820959328212, 21.213814093608626, 23.800556376214338,
         26.7027174518336,
         29.95875844419536, 33.61183029916742, 37.71034564614689, 42.308620390336046, 47.467593538656814,
         53.255634798855226, 59.74945065456746, 67.03510092418111, 75.20913927551955, 84.3798928111122,
         94.66889768185835,
         106.21250975466879, 119.16371167959043, 133.69414030471458, 149.99636130734967, 168.28642118619175,
         188.8067104349728, 211.8291758420296, 237.65892448704983, 266.63826719725404, 299.1512550492434,
         335.6287690368665,
         376.55423035634425, 422.4700069846627, 473.9846014549328, 531.7807150852908, 596.6242955331776,
         669.3746875784484,
         750.996021659378, 842.5699910890552, 945.3101872832622, 1060.578182978589, 1189.9015765849472,
         1334.9942368067032,
         1497.779015825923, 1680.4132320559845, 1885.3172601778733, 2115.2066073508468, 2373.12790069007,
         2662.4992629382136, 2987.1556114127557, 3351.399480557073, 3760.0580416251546, 4218.547075157965,
         4732.942744051923, 5310.062118397695, 5957.5535192514635, 6683.997878626583, 7499.022458987134,
         8413.428439320318,
         9507.935290267791, 10000.0])

    tau_cy__Su__min1 = np.array(
        [0.29017207839562104, 0.2860740723507864, 0.2827249767716125, 0.2791468130024064, 0.27585498447074075,
         0.2724486218440594, 0.2689277251223614, 0.26552136249567937, 0.2623440680590305, 0.2591667736223813,
         0.2555313428056667, 0.25246858246403425, 0.2492340209798771, 0.2462285276857525, 0.24250856202772295,
         0.23863873338241026, 0.23476890473709802, 0.2311615142500512, 0.2289550978060844, 0.2258923374644517,
         0.22288684417032711, 0.21999588497121916, 0.21687585758207814, 0.21409943247798635, 0.21143754146891086,
         0.20831751407977014, 0.20548382192817027, 0.20287919796660286, 0.2001027728625111, 0.19744088185343567,
         0.19443538855931108, 0.19205983278777672, 0.1892261406361768, 0.18685058486464226, 0.18407415976055064,
         0.1814695357989835, 0.17892217888492426, 0.17648935606588154, 0.1739419991518223, 0.17139464223776346,
         0.16907635351373718, 0.16715893412226812, 0.16524151473079907, 0.16372496467188702, 0.16192207937543435,
         0.16000465998396507, 0.15837357583003708, 0.15639888939105995, 0.15459600409460705, 0.15307945403569545,
         0.15121930169173448, 0.14964548458531435, 0.14789986633636953, 0.1459251798973924, 0.14440862983848102,
         0.14277754568455234, 0.14091739334059186, 0.13957264442420425, 0.13788429322276796, 0.13642501021136444,
         0.13479392605743626, 0.1333919100935408, 0.1316462918445962, 0.13041607702322544, 0.12924312924936232,
         0.12789838033297518, 0.12643909732157166, 0.12532341659521706, 0.12386413358381355, 0.12274845285745872,
         0.12111736870353075, 0.12028802321471675, 0.11900054134583772, 0.11782759357197505, 0.11665464579811236,
         0.11553896507175798, 0.11448055139291125, 0.11302126838150772, 0.1120773887976776, 0.10994825273961162,
         0.10963962260800007])

    N__min1_5 = np.array(
        [1.0, 1.2058786996514277, 1.33116985999329, 1.504342371300126, 1.687776901375026, 1.8935788309633208,
         2.124475566735869, 2.3835270863065103, 2.6741664908321976, 3.000245502462977, 3.366085509600439,
         3.723304616114597,
         4.1929459947865535, 4.715949931939756, 5.264053884127996, 5.87342200082079, 6.422381480879633,
         7.093221985851589,
         7.936270964554134, 8.903993580564675, 9.989717089654713, 11.207830130197673, 12.574475843510461,
         14.10776581213578,
         15.828020085050607, 17.75803647075413, 19.923392667063663, 22.35278523161796, 25.07840988532185,
         28.13638818873424,
         31.567246245962853, 35.41645178012751, 39.735016698028005, 44.58017312955347, 50.01613190109953,
         56.11493349472425,
         62.957402770446976, 70.63422010422465, 79.24712313694826, 88.91025506073403, 99.7516773107834,
         111.91506670988188,
         125.56161955708099, 140.87218789465354, 158.04967626436942, 177.3217307163015, 198.94375570646469,
         223.20229886496892, 250.41884849159942, 280.95409410543823, 315.212706511808, 353.6487007347582,
         396.7714528877894,
         445.15245072188316, 499.4328673130128, 560.332058260048, 628.6570950028489, 705.3134606016156,
         791.3170497241789,
         887.8076318720188, 996.0639562675128, 1117.5206985811444, 1253.7874740865368, 1406.6701692166987,
         1578.1948742196148, 1770.6347340826942, 1986.5400735700428, 2228.7721956070905, 2500.541300928414,
         2805.4490315218554, 3147.536201679493, 3531.3363492149433, 3961.9358165388708, 4445.041157822073,
         4987.054765564828, 5595.159718819952, 6277.4149775272745, 7042.862184530201, 7901.645490676839,
         8865.145990997167,
         10000.0])

    tau_cy__Su__min1_5 = np.array(
        [0.3981293820657492, 0.3950402521658147, 0.388253138877265, 0.3809550749157222, 0.3745135587711073,
         0.3687019801490816, 0.3628904015270564, 0.3571933570000476, 0.3513817783780224, 0.34591380204104616,
         0.3409411856650155, 0.3336886563314567, 0.33115022030009666, 0.3273300738303431, 0.32317335163326244,
         0.32227315605439744, 0.31617180518884513, 0.3139504757865685, 0.3051925963069433, 0.30069815977760594,
         0.29671912667584244, 0.2923392242415217, 0.2881311229497254, 0.28398028870543746, 0.2799439885561654,
         0.2760794895494185, 0.2722722575901795, 0.2683504915359243, 0.2643141913866525, 0.2605069594274134,
         0.2568142615631908, 0.2528924955089353, 0.24937159878723736, 0.2455643668279985, 0.24187166896377565,
         0.2381789710995532, 0.2346580743778548, 0.2307363083235996, 0.22710087750688526, 0.2235799807851873,
         0.2203454193010299, 0.21774079533946256, 0.21542250661543627, 0.21298968379639335, 0.21078592916738348,
         0.2080667711108, 0.20592028352929811, 0.20365926185278016, 0.20134097312875388, 0.1989654173572193,
         0.1967043956807009, 0.1943861069566748, 0.19223961937517275, 0.1902649329361956, 0.18788937716466148,
         0.18614375891571644, 0.1838254701916904, 0.1819080508002211, 0.1797042961712112, 0.17784414382725022,
         0.17609852557830585, 0.1741238391393285, 0.17232095384287582, 0.17051806854642315, 0.1685433821074458,
         0.1668550309060095, 0.1650521456095566, 0.1635355955506448, 0.16201904549173318, 0.16038796133780495,
         0.1585850760413523, 0.15724032712496494, 0.15543744182851227, 0.15397815881710852, 0.15274794399573774,
         0.15123139393682594, 0.14982937797293028, 0.14842736200903486, 0.14719714718766366, 0.14556606303373565,
         0.1428137355034198])

    N__min4 = np.array(
        [1.0, 1.1012737577522174, 1.2110551407838497, 1.3682348666691897, 1.557717189980799, 1.7234559734636448,
         1.9471388829875944, 2.1524345888831977, 2.400728861236578, 2.726620261193849, 3.056556910956348,
         3.4733846154249237, 3.9252388488063032, 4.396230027854009, 4.905213660218264, 5.503339179943854,
         6.082823773147818,
         6.651667016193109, 7.883325493618918, 9.442774131174305, 10.861600073824892, 12.186027639925847,
         13.671951520190976, 15.339064040690939, 17.209458744564888, 19.30792318848289, 21.66226744174,
         24.3036926414425,
         27.26720448809405, 30.59207716146492, 34.32237380482709, 38.507530475314454, 43.20301129925711,
         48.47104351498918,
         54.3814421443436, 61.012535221860034, 68.45220184338028, 76.79903679085352, 86.16365716763647,
         96.67016836057745,
         108.45780875666522, 121.68279501098316, 136.52039232053002, 153.16723714037934, 171.84394312416023,
         192.7980248242953, 216.30717789857098, 242.68295929423843, 272.27491618158666, 305.47521835600963,
         342.7238554998492, 384.51446818114965, 431.40088986509835, 484.004486636694, 543.022391904904,
         609.2367452193934,
         683.525057637993, 766.8718410127659, 860.3816553127351, 965.2937468939599, 1082.998471711822,
         1215.0557211254657,
         1363.2155944838432, 1529.441592458494, 1715.9366384945677, 1925.1722732314386, 2159.9214087943988,
         2423.295077036219, 2718.7836587377233, 3050.3031401605986, 3422.2470099711463, 3839.544484303239,
         4307.725831592646, 4832.995662905565, 5422.315159047212, 6083.494324171839, 6825.295488492801,
         7657.549431770331,
         8591.285666513302, 10000.0])

    tau_cy__Su__min4 = np.array(
        [0.6532151007005733, 0.6456295759512285, 0.637026081186364, 0.6219350950467912, 0.6065906934036011,
         0.592524189212255, 0.5820231842004857, 0.5718847859452879, 0.5591968466301058, 0.54552740705088,
         0.5342161185496164, 0.5186533484157879, 0.5086463223607629, 0.5014347839460722, 0.4879072055067,
         0.4778260503426757, 0.4721871311626917, 0.4611373209841214, 0.4483075270026908, 0.4341962821926941,
         0.4248054342616767, 0.4169895089768652, 0.4090017825495289, 0.4013003913597337, 0.39394260245498747,
         0.3867566146927659, 0.37968516102556094, 0.373014576690913, 0.3662294582612489, 0.35978794211663323,
         0.35340369301952634, 0.34742031325497624, 0.3415514675854432, 0.3359689571534505, 0.3301573785314253,
         0.32491847038448185, 0.3198513633800626, 0.3146697222806276, 0.310118018703782, 0.305337246936904,
         0.3008428104075671, 0.2966919761632789, 0.29236934077646626, 0.28839030767470275, 0.2846976098104801,
         0.2810621789937657, 0.27754128227206776, 0.27419218669289397, 0.2709003581612286, 0.2675512625820551,
         0.2643167010978977, 0.2615402759938059, 0.2589356520322388, 0.2561019598806389, 0.253268267729039,
         0.2505491096724555, 0.2485171561859698, 0.2463134015569597, 0.2439951128329336, 0.2416768241089071,
         0.2394730694798972, 0.2379565194209854, 0.23592456593449995, 0.23417894768555536, 0.23260513057913526,
         0.23051591004514194, 0.2291138940812465, 0.2275973440223347, 0.22625259510594756, 0.2246787779995274,
         0.22321949498812368, 0.22216108130927714, 0.22064453125036526, 0.2197006516665354, 0.21835590275014805,
         0.21683935269123625, 0.2157236719648816, 0.21443619009600254, 0.21332050936964772, 0.2121484742180564])

    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
        [0.05, 0.1, 0.25, 0.5, 1, 5, 15],
        [N__0_05, N__0_1, N__0_25, N__0_5, N__1, N__5, N__15],
        [tau_cy__Su__0_05, tau_cy__Su__0_1, tau_cy__Su__0_25, tau_cy__Su__0_5,
         tau_cy__Su__1, tau_cy__Su__5, tau_cy__Su__15]
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$ \gamma_{cy} = %s \text{%%} $' % str(_strain))
        fig.append_trace(trace, 1, 1)

    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
        [-0.5, -0.75, -1, -1.5, -4],
        [N__min0_5, N__min0_75, N__min1, N__min1_5, N__min4],
        [tau_cy__Su__min0_5, tau_cy__Su__min0_75, tau_cy__Su__min1, tau_cy__Su__min1_5,
         tau_cy__Su__min4]
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$ \gamma_{a} = %s \text{%%} $' % str(_strain),
            line=dict(dash='dot'))
        fig.append_trace(trace, 2, 1)


    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 4), dtick=1, type='log')
    fig['layout']['xaxis2'].update(title=r'$ N $', range=(0, 4), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{C} $', range=(0, 0.8), dtick=0.2)
    fig['layout']['yaxis2'].update(title=r'$ \tau_{cyc} / S_u^{C} $', range=(0, 0.8), dtick=0.2)
    fig['layout'].update(height=900, width=500,
                         title=r'$ \text{Strain accumulation diagram for cyclic triaxial tests om Drammen clay} $',
                         hovermode='closest')

    return fig


POREPRESSUREACCUMULATION_DSSCLAY_ANDERSEN = {
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'cycle_no': {'type': 'int', 'min_value': 1.0, 'max_value': 1500.0},
}

POREPRESSUREACCUMULATION_DSSCLAY_ANDERSEN_ERRORRETURN = {
    'Excess pore pressure ratio [-]': np.nan,
    'pore pressure ratios interpolation [-]': None,
    'shearstress ratios interpolation [-]': None,
}


@Validator(POREPRESSUREACCUMULATION_DSSCLAY_ANDERSEN, POREPRESSUREACCUMULATION_DSSCLAY_ANDERSEN_ERRORRETURN)
def porepressureaccumulation_dssclay_andersen(
        cyclic_shear_stress, undrained_shear_strength, cycle_no,
        **kwargs):
    """
    Calculates the excess pore pressure accumulation for a normally consolidated clay sample under symmetrical cyclic loading (no average shear stress) in a DSS test. The contours are based on cyclic DSS tests on Drammen clay.

    Excess pore pressure contours for excess pore pressure ratios of 0.05, 0.1, 0.25 and 0.6 are defined and logarithmic interpolation is used to obtain the accumulated excess pore pressure for a sample tested at a certain ratio of cyclic shear stress to DSS shear strength with a given number of cycles.

    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0
    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a DSS test (:math:`S_u^{DSS}`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param cycle_no: Number of applied cycles (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= cycle_no <= 1500.0

    :returns: Dictionary with the following keys:

        - 'Excess pore pressure ratio [-]': Ratio of accumulated excess pore pressure to vertical effective stress (:math:`u_p / \\sigma_{vc}^{\\prime}`)  [:math:`-`]
        - 'pore pressure ratios interpolation [-]': List of excess pore pressure ratios used for interpolation [:math:`-`]
        - 'shearstress ratios interpolation [-]': List of ratios of cyclic shear stress to undrained DSS shear strength used for interpolation [:math:`-`]

    .. figure:: images/porepressureaccumulation_dssclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Excess pore pressure contours for symmetrical cyclic DSS tests

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    N__0_05 = np.array(
        [1.0, 1.0641648753957078, 1.194264500848553, 1.3451090320040884, 1.5150063547006651, 1.7063629788907853,
         1.921889374717722, 2.164638364959092, 2.4380483667229687, 2.745992094893352, 3.0928314171847555,
         3.4834791378001313, 3.923468584826476, 4.419031987038557, 4.977188750278625, 5.605844883802597,
         6.313904984916741,
         7.111398368110871, 8.009621125243594, 8.828293508591045, 10.335535090897146, 12.283026884888596,
         15.357784018725773, 19.202231842055248, 23.766472050913414, 28.53299095926973, 34.481315346180146,
         38.83656315211013, 43.741911302549106, 49.266841581889956, 55.48961184312925, 62.49836449091558,
         70.39237497428964,
         79.28345797338804, 89.29755119803623, 100.57649923193857, 113.28006268972136, 127.58818114552368,
         143.70352188638068, 161.85435059220632, 182.2977646041086, 205.3233345786583, 231.25720611030204,
         260.46671941934,
         293.36561254101815, 330.41988171628367, 372.15438199368964, 419.1602615366464, 472.1033349392315,
         531.7335141543668, 598.8954306186932, 674.5403990312017, 759.7398922465097, 855.7007181478526,
         963.7821134725517,
         1085.514996715385, 1222.6236528175239, 1377.0501567937265, 1550.9818822461825, 1746.8824844092096,
         1967.5267965840198, 2216.0401342540176, 2495.942563603532, 2811.198761481316, 3166.27417305, 3566.198262566392,
         4016.635753207953, 4523.966865021721, 5095.377687525866, 5738.961967046114, 6463.835750553389,
         7280.266509874983,
         8199.818575258458, 10000.0])

    tau_cy__Su__0_05 = np.array(
        [0.6876472247676271, 0.6755773227492964, 0.6622886993307054, 0.6401674674235813, 0.6183792748946589,
         0.5967020954918039, 0.5760240342235539, 0.5555679992074385, 0.5364441217041296, 0.5179863229572244,
         0.5003056160927897, 0.4840680798672288, 0.4680525698938027, 0.4524811124246455, 0.4381307993422279,
         0.4244465650162135, 0.4116504356987372, 0.3996313982637312, 0.3876123608287256, 0.3789132435794991,
         0.3686844421067539, 0.3479992514894348, 0.3318885730758727, 0.3111907386990078, 0.29737616701900604,
         0.2881563375580349, 0.2741264693966225, 0.2677691013910457, 0.2620778121418723, 0.25594247038842965,
         0.2496961155089201, 0.24444887876401575, 0.2393126551451785, 0.23428744465240856, 0.2291512210335711,
         0.22401499741473385, 0.21932282630016545, 0.21551876019413507, 0.21104861533170127, 0.20624543109106572,
         0.20255237811110272, 0.19930337763540829, 0.19594336403364654, 0.19202828480154865, 0.1890013105779893,
         0.18486420509375698, 0.1820592571223316, 0.17936532227697355, 0.17644936117948107, 0.17353340008198814,
         0.17028439960629393, 0.16836755664340666, 0.16467450366344338, 0.1630907000787576, 0.16017473898126489,
         0.15781384351410854, 0.1557859874251537, 0.15353610508406468, 0.15050913086050488, 0.14948039290615522,
         0.14600936617832638, 0.1443145494675735, 0.14228669337861846, 0.13948174540719327, 0.13756490244430575,
         0.13575907260748554, 0.13406425589673265, 0.13148133417744168, 0.12978651746668854, 0.12831372700806987,
         0.1253977659105776, 0.12314788356948835, 0.1220081324890716, 0.11752761456992933])

    N__0_1 = np.array(
        [1.0, 1.0846288365197625, 1.187825048557288, 1.3234677323737931, 1.4746000185482309, 1.6518977314928245,
         1.8605447595564395, 2.0955454664766977, 2.3602285188333467, 2.6583430186701698, 2.994111607635995,
         3.37229027857555, 3.798235741771234, 4.277981300044622, 4.818322307450329, 5.426912422041512,
         6.1123720160789725,
         6.8844102792598125, 7.753962744496312, 8.73334618422844, 9.83643307129818, 11.07884807553596,
         12.47818937933441,
         14.054277947033016, 15.829438278888686, 17.828814626371074, 20.08072714781255, 22.61707304917843,
         25.473778391914284, 30.635029445176198, 35.67553150110145, 41.969398530012604, 48.38099011429564,
         56.09288463952192, 63.17783514972992, 71.15766785497354, 80.14541306391024, 90.26837765785272,
         101.66994832858327,
         114.51162257858057, 128.9752962517513, 145.2658399963893, 163.6140011546596, 184.27967218240917,
         207.5555718948268,
         233.77139168093674, 263.29846541596993, 296.55502921856475, 334.0121455546078, 376.20037559966073,
         423.7172943706254, 477.2359550746949, 537.5144225688365, 605.4065109664441, 681.8738774839574,
         767.9996438306861,
         865.0037380819548, 974.2601743454874, 1097.316514978768, 1235.915791029918, 1392.0211913940466,
         1567.8438703945635,
         1765.8742676697962, 1988.9173839953753, 2240.1325127065697, 2523.0779894961747, 2841.7615944463164,
         3200.6973202135073, 3604.969310459668, 4060.303874184876, 4573.150596008435, 5150.773691284568,
         5801.354900051996,
         6534.109377258198, 7359.4162207850695, 8288.965486139592, 10000.0])

    tau_cy__Su__0_1 = np.array(
        [0.9770642201834859, 0.9609889442966528, 0.934405420302152, 0.9072690878076748, 0.8807433275065675,
         0.8544826523525787, 0.8273658297724289, 0.8008040728226156, 0.7752414340074073, 0.7515660183353421,
         0.7290007339239489, 0.7074345676471611, 0.6873115720092473, 0.6678546551277367, 0.6492858432547642,
         0.6318271626424641, 0.6152565870387015, 0.5991300639392083, 0.5841136721003872, 0.5690972802615661,
         0.5554130459355517, 0.5420618509877391, 0.5288216691659937, 0.515803513596383, 0.5040065024135116,
         0.4914323993481698, 0.4795243750392313, 0.468171416360629, 0.4565964314298918, 0.4372212051020097,
         0.4257153839151149, 0.41420703397631065, 0.4049973195229764, 0.3943435260636401, 0.3860989349149204,
         0.3789644750268728, 0.3710529232563549, 0.3638074502422399, 0.35611792472385617, 0.3488724517097417,
         0.34262609683023193, 0.33615771569858804, 0.3295783214408769, 0.3234429796874345, 0.3179737166903953,
         0.3121714143151546, 0.3061470856877795, 0.3015659276992784, 0.2963186909543738, 0.2906274017052006,
         0.28560219121243025, 0.28146508572819817, 0.2772169671178988, 0.27230276975119616, 0.26794363801482945,
         0.2636955194045301, 0.2601134795506341, 0.25642042657067066, 0.25183926858216976, 0.24870128123254265,
         0.24456417574831055, 0.2413151752726161, 0.23795516167085465, 0.2345951480690933, 0.2307910819630625,
         0.22820816024377175, 0.22407105475953948, 0.22137711991418166, 0.2182391325645545, 0.21554519771919645,
         0.21251822349563645, 0.20971327552421126, 0.20635326192244974, 0.2029932483206882, 0.20074336597959894,
         0.19838247051244265, 0.19698467111048834])

    N__0_25 = np.array(
        [1.0, 1.0628227939348285, 1.1437106681574871, 1.2606106218406172, 1.3894590512629252, 1.5397796874097833,
         1.7156135356183893, 1.921889374717722, 2.164638364959092, 2.4380483667229687, 2.745992094893352,
         3.0928314171847555, 3.4834791378001313, 3.923468584826476, 4.419031987038557, 4.977188750278625,
         5.605844883802597,
         6.313904984916741, 7.111398368110871, 8.009621125243594, 9.02129612899057, 10.160753245924004,
         11.444132311850744,
         12.889611744452548, 14.517666031411045, 16.351355741207193, 18.41665416445202, 20.742815212458172,
         23.36278778415059, 26.313682470614506, 29.63729720791532, 33.38070932379817, 37.59694236430961,
         42.34571714560499,
         47.69429766921194, 53.71844388271236, 60.50348478121955, 68.14552704959016, 76.75281636518021,
         84.59780804703091,
         97.49327100277662, 115.86361598999608, 137.69542628121434, 158.7309156439596, 182.2977646041086,
         205.3233345786583,
         231.25720611030204, 260.46671941934, 293.36561254101815, 330.41988171628367, 372.15438199368964,
         419.1602615366464,
         472.1033349392315, 531.7335141543668, 598.8954306186932, 674.5403990312017, 759.7398922465097,
         855.7007181478526,
         963.7821134725517, 1085.514996715385, 1222.6236528175239, 1377.0501567937265, 1550.9818822461825,
         1746.8824844092096, 1967.5267965840198, 2216.0401342540176, 2495.942563603532, 2811.198761481316,
         3166.27417305,
         3566.198262566392, 4016.635753207953, 4523.966865021721, 5095.377687525866, 5738.961967046114,
         6463.835750553389,
         7280.266509874983, 8199.818575258458, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [1.208713067578366, 1.1949389559289116, 1.172293771327121, 1.1444545814534142, 1.1190576803531855,
         1.0928453366396875, 1.0668080340932762, 1.0406139668149277, 1.015273354251854, 0.9912648992015868,
         0.9685886016641266, 0.947133448513406, 0.9267884266233576, 0.9073315097418472, 0.8892067503731437,
         0.8713040172565745, 0.8542893891485431, 0.8386069185533187, 0.8230354610841614, 0.8075750167410715,
         0.7921145723979813, 0.7777642593155637, 0.7637469856113477, 0.7495076856549971, 0.7354904119507814,
         0.7216951644986997, 0.7081219432987527, 0.6948817614770073, 0.6810865140249259, 0.6677353190771134,
         0.6537365475605751, 0.6400338110468831, 0.6261275504687344, 0.6116662242602493, 0.5982040161863695,
         0.5877391625163051, 0.5773853219723077, 0.5672535076804449, 0.557010680262515, 0.5494851303459993,
         0.5378503585770207, 0.5240459019046559, 0.5125350232139425, 0.5010317307789567, 0.4871900333474901,
         0.4783903765684343, 0.4692576804111763, 0.4610130892624567, 0.4526574849876697, 0.4447459332171515,
         0.4372784339509024, 0.4294778953064515, 0.42278748792267296, 0.4156530280346253, 0.4088516075247797,
         0.4026052526452699, 0.3966919371439621, 0.3903345691383853, 0.3841992273849431, 0.3788409775139712,
         0.3734827276429993, 0.3681244777720276, 0.3633212935313921, 0.35851810929075656, 0.35393695130225544,
         0.3490227539355526, 0.3444415959470517, 0.34074854296708845, 0.3361673849785873, 0.3318082532422204,
         0.3282262133883247, 0.3238670816519582, 0.32072909430233065, 0.3169250281963003, 0.3138980539727405,
         0.31009398786671016, 0.30662296113888154, 0.3035486094832578])

    N__0_6 = np.array(
        [1.0, 1.1578129556424808, 1.3021746170456958, 1.4666490022855916, 1.6518977314928245, 1.8605447595564395,
         2.05071313191929, 2.278164285029333, 2.5458139866962433, 2.8673693179980964, 3.2295394906155264,
         3.637454462520032,
         4.096892143710914, 4.614360237398508, 5.197188418340471, 5.853632153990756, 6.563351900483103,
         7.62129313279782,
         8.36365913289523, 9.460884166344293, 11.894730656983564, 13.459353108001784, 15.159369986834552,
         17.074111701632724, 19.230699603810226, 21.659680674141427, 24.395460205349153, 27.476789135736645,
         31.235478590071768, 35.092553201752764, 39.258787083344075, 44.21746527156313, 49.802461570989244,
         56.09288463952192, 63.17783514972992, 71.47899031152726, 78.85596302168895, 88.01933823500751,
         98.42475433577059,
         110.85653633325032, 124.40931766386194, 140.52053724644145, 158.39162047663308, 179.04183910501638,
         200.9306241481136, 226.3096635258025, 253.51988002904656, 291.9242151603233, 336.5209667633253,
         399.9305353549751,
         478.52780541318606, 537.5144225688365, 605.4065109664441, 681.8738774839574, 767.9996438306861,
         865.0037380819548,
         974.2601743454874, 1097.316514978768, 1235.915791029918, 1392.0211913940466, 1567.8438703945635,
         1765.8742676697962, 1988.9173839953753, 2240.1325127065697, 2523.0779894961747, 2841.7615944463164,
         3200.6973202135073, 3604.969310459668, 4060.303874184876, 4573.150596008435, 5150.773691284568,
         5801.354900051996,
         6534.109377258198, 7359.4162207850695, 8288.965486139592, 10000.0])

    tau_cy__Su__0_6 = np.array(
        [1.3256804871387675, 1.299874915952456, 1.2750411113470717, 1.2529669278600428, 1.2284710723229473,
         1.2053655240313603, 1.1897084212205995, 1.1703533630641176, 1.1484487148692466, 1.1285477454834671,
         1.108646776097688, 1.089522898594379, 1.0702880079650032, 1.0523852748484341, 1.0347045679839992,
         1.0172458873716992, 1.0035917617692491, 0.9834392037465988, 0.9688663180732188, 0.9583522816308372,
         0.9283481607531832, 0.907135553826926, 0.8955383662709757, 0.8792860282952726, 0.8627524570668661,
         0.8477360652280451, 0.8334967652716945, 0.8185913865589407, 0.8083977929116397, 0.7901795617963444,
         0.7748743685552842, 0.7600800029685976, 0.7449525980037093, 0.7327115343165691, 0.7224687068986391,
         0.7141489850363386, 0.7090364246343771, 0.6987921407046498, 0.6831706722505849, 0.6728168317065878,
         0.6636726983278401, 0.6524382225908579, 0.6420883494527978, 0.6351565160589114, 0.6229348521297444,
         0.6135801297203523, 0.6050704534245142, 0.5926231249304592, 0.5857069884587764, 0.5741961097680628,
         0.5606254202342145, 0.5498652047390253, 0.5417316267163728, 0.5340421011979892, 0.5266856150578072,
         0.5192181157915581, 0.5118616296513758, 0.5048381828893957, 0.4979257492534828, 0.4909023024915027,
         0.4841008819816568, 0.4776325008500129, 0.4707200672140999, 0.4644737123345899, 0.4583383705811477,
         0.4516479631973691, 0.4454016083178596, 0.4399323453208206, 0.4333529510631093, 0.42777267494000304,
         0.4217483463126279, 0.4158350308113199, 0.40981070218394505, 0.4044524523129731, 0.3986501499377326,
         0.3928478475624919])

    _porepressure_ratios_interpolation = np.array([0.05, 0.1, 0.25, 0.6])
    _shearstress_ratios_interpolation = np.array([
        np.interp(np.log10(cycle_no), np.log10(N__0_05), tau_cy__Su__0_05),
        np.interp(np.log10(cycle_no), np.log10(N__0_1), tau_cy__Su__0_1),
        np.interp(np.log10(cycle_no), np.log10(N__0_25), tau_cy__Su__0_25),
        np.interp(np.log10(cycle_no), np.log10(N__0_6), tau_cy__Su__0_6)
    ])

    fit_coeff = np.polyfit(
        x=_shearstress_ratios_interpolation,
        y=np.log(_porepressure_ratios_interpolation),
        deg=1)

    _porepressure_ratio_0 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0)
    _porepressure_ratio_1_5 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 1.5)

    _shearstress_ratios_interpolation = np.append(np.append(1e-7, _shearstress_ratios_interpolation), 1.5)
    _porepressure_ratios_interpolation = np.append(
        np.append(_porepressure_ratio_0, _porepressure_ratios_interpolation), _porepressure_ratio_1_5)

    _porepressure_ratio = 10 ** (np.interp(
        cyclic_shear_stress / undrained_shear_strength,
        _shearstress_ratios_interpolation,
        np.log10(_porepressure_ratios_interpolation)))

    if _porepressure_ratio < 0.05:
        warnings.warn(
            "Cyclic strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _porepressure_ratio > 0.6:
        warnings.warn(
            "Cyclic strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    return {
        'Excess pore pressure ratio [-]': _porepressure_ratio,
        'pore pressure ratios interpolation [-]': _porepressure_ratios_interpolation,
        'shearstress ratios interpolation [-]': _shearstress_ratios_interpolation,
    }


def plotporepressureaccumulation_dssclay_andersen():
    """
    Returns a Plotly figure with the excess pore pressure accumulation contours for cyclic DSS tests on normally consolidated Drammen clay with symmetrical loading
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=False)

    N__0_05 = np.array(
        [1.0, 1.0641648753957078, 1.194264500848553, 1.3451090320040884, 1.5150063547006651, 1.7063629788907853,
         1.921889374717722, 2.164638364959092, 2.4380483667229687, 2.745992094893352, 3.0928314171847555,
         3.4834791378001313, 3.923468584826476, 4.419031987038557, 4.977188750278625, 5.605844883802597,
         6.313904984916741,
         7.111398368110871, 8.009621125243594, 8.828293508591045, 10.335535090897146, 12.283026884888596,
         15.357784018725773, 19.202231842055248, 23.766472050913414, 28.53299095926973, 34.481315346180146,
         38.83656315211013, 43.741911302549106, 49.266841581889956, 55.48961184312925, 62.49836449091558,
         70.39237497428964,
         79.28345797338804, 89.29755119803623, 100.57649923193857, 113.28006268972136, 127.58818114552368,
         143.70352188638068, 161.85435059220632, 182.2977646041086, 205.3233345786583, 231.25720611030204,
         260.46671941934,
         293.36561254101815, 330.41988171628367, 372.15438199368964, 419.1602615366464, 472.1033349392315,
         531.7335141543668, 598.8954306186932, 674.5403990312017, 759.7398922465097, 855.7007181478526,
         963.7821134725517,
         1085.514996715385, 1222.6236528175239, 1377.0501567937265, 1550.9818822461825, 1746.8824844092096,
         1967.5267965840198, 2216.0401342540176, 2495.942563603532, 2811.198761481316, 3166.27417305, 3566.198262566392,
         4016.635753207953, 4523.966865021721, 5095.377687525866, 5738.961967046114, 6463.835750553389,
         7280.266509874983,
         8199.818575258458, 10000.0])

    tau_cy__Su__0_05 = np.array(
        [0.6876472247676271, 0.6755773227492964, 0.6622886993307054, 0.6401674674235813, 0.6183792748946589,
         0.5967020954918039, 0.5760240342235539, 0.5555679992074385, 0.5364441217041296, 0.5179863229572244,
         0.5003056160927897, 0.4840680798672288, 0.4680525698938027, 0.4524811124246455, 0.4381307993422279,
         0.4244465650162135, 0.4116504356987372, 0.3996313982637312, 0.3876123608287256, 0.3789132435794991,
         0.3686844421067539, 0.3479992514894348, 0.3318885730758727, 0.3111907386990078, 0.29737616701900604,
         0.2881563375580349, 0.2741264693966225, 0.2677691013910457, 0.2620778121418723, 0.25594247038842965,
         0.2496961155089201, 0.24444887876401575, 0.2393126551451785, 0.23428744465240856, 0.2291512210335711,
         0.22401499741473385, 0.21932282630016545, 0.21551876019413507, 0.21104861533170127, 0.20624543109106572,
         0.20255237811110272, 0.19930337763540829, 0.19594336403364654, 0.19202828480154865, 0.1890013105779893,
         0.18486420509375698, 0.1820592571223316, 0.17936532227697355, 0.17644936117948107, 0.17353340008198814,
         0.17028439960629393, 0.16836755664340666, 0.16467450366344338, 0.1630907000787576, 0.16017473898126489,
         0.15781384351410854, 0.1557859874251537, 0.15353610508406468, 0.15050913086050488, 0.14948039290615522,
         0.14600936617832638, 0.1443145494675735, 0.14228669337861846, 0.13948174540719327, 0.13756490244430575,
         0.13575907260748554, 0.13406425589673265, 0.13148133417744168, 0.12978651746668854, 0.12831372700806987,
         0.1253977659105776, 0.12314788356948835, 0.1220081324890716, 0.11752761456992933])

    N__0_1 = np.array(
        [1.0, 1.0846288365197625, 1.187825048557288, 1.3234677323737931, 1.4746000185482309, 1.6518977314928245,
         1.8605447595564395, 2.0955454664766977, 2.3602285188333467, 2.6583430186701698, 2.994111607635995,
         3.37229027857555, 3.798235741771234, 4.277981300044622, 4.818322307450329, 5.426912422041512,
         6.1123720160789725,
         6.8844102792598125, 7.753962744496312, 8.73334618422844, 9.83643307129818, 11.07884807553596,
         12.47818937933441,
         14.054277947033016, 15.829438278888686, 17.828814626371074, 20.08072714781255, 22.61707304917843,
         25.473778391914284, 30.635029445176198, 35.67553150110145, 41.969398530012604, 48.38099011429564,
         56.09288463952192, 63.17783514972992, 71.15766785497354, 80.14541306391024, 90.26837765785272,
         101.66994832858327,
         114.51162257858057, 128.9752962517513, 145.2658399963893, 163.6140011546596, 184.27967218240917,
         207.5555718948268,
         233.77139168093674, 263.29846541596993, 296.55502921856475, 334.0121455546078, 376.20037559966073,
         423.7172943706254, 477.2359550746949, 537.5144225688365, 605.4065109664441, 681.8738774839574,
         767.9996438306861,
         865.0037380819548, 974.2601743454874, 1097.316514978768, 1235.915791029918, 1392.0211913940466,
         1567.8438703945635,
         1765.8742676697962, 1988.9173839953753, 2240.1325127065697, 2523.0779894961747, 2841.7615944463164,
         3200.6973202135073, 3604.969310459668, 4060.303874184876, 4573.150596008435, 5150.773691284568,
         5801.354900051996,
         6534.109377258198, 7359.4162207850695, 8288.965486139592, 10000.0])

    tau_cy__Su__0_1 = np.array(
        [0.9770642201834859, 0.9609889442966528, 0.934405420302152, 0.9072690878076748, 0.8807433275065675,
         0.8544826523525787, 0.8273658297724289, 0.8008040728226156, 0.7752414340074073, 0.7515660183353421,
         0.7290007339239489, 0.7074345676471611, 0.6873115720092473, 0.6678546551277367, 0.6492858432547642,
         0.6318271626424641, 0.6152565870387015, 0.5991300639392083, 0.5841136721003872, 0.5690972802615661,
         0.5554130459355517, 0.5420618509877391, 0.5288216691659937, 0.515803513596383, 0.5040065024135116,
         0.4914323993481698, 0.4795243750392313, 0.468171416360629, 0.4565964314298918, 0.4372212051020097,
         0.4257153839151149, 0.41420703397631065, 0.4049973195229764, 0.3943435260636401, 0.3860989349149204,
         0.3789644750268728, 0.3710529232563549, 0.3638074502422399, 0.35611792472385617, 0.3488724517097417,
         0.34262609683023193, 0.33615771569858804, 0.3295783214408769, 0.3234429796874345, 0.3179737166903953,
         0.3121714143151546, 0.3061470856877795, 0.3015659276992784, 0.2963186909543738, 0.2906274017052006,
         0.28560219121243025, 0.28146508572819817, 0.2772169671178988, 0.27230276975119616, 0.26794363801482945,
         0.2636955194045301, 0.2601134795506341, 0.25642042657067066, 0.25183926858216976, 0.24870128123254265,
         0.24456417574831055, 0.2413151752726161, 0.23795516167085465, 0.2345951480690933, 0.2307910819630625,
         0.22820816024377175, 0.22407105475953948, 0.22137711991418166, 0.2182391325645545, 0.21554519771919645,
         0.21251822349563645, 0.20971327552421126, 0.20635326192244974, 0.2029932483206882, 0.20074336597959894,
         0.19838247051244265, 0.19698467111048834])

    N__0_25 = np.array(
        [1.0, 1.0628227939348285, 1.1437106681574871, 1.2606106218406172, 1.3894590512629252, 1.5397796874097833,
         1.7156135356183893, 1.921889374717722, 2.164638364959092, 2.4380483667229687, 2.745992094893352,
         3.0928314171847555, 3.4834791378001313, 3.923468584826476, 4.419031987038557, 4.977188750278625,
         5.605844883802597,
         6.313904984916741, 7.111398368110871, 8.009621125243594, 9.02129612899057, 10.160753245924004,
         11.444132311850744,
         12.889611744452548, 14.517666031411045, 16.351355741207193, 18.41665416445202, 20.742815212458172,
         23.36278778415059, 26.313682470614506, 29.63729720791532, 33.38070932379817, 37.59694236430961,
         42.34571714560499,
         47.69429766921194, 53.71844388271236, 60.50348478121955, 68.14552704959016, 76.75281636518021,
         84.59780804703091,
         97.49327100277662, 115.86361598999608, 137.69542628121434, 158.7309156439596, 182.2977646041086,
         205.3233345786583,
         231.25720611030204, 260.46671941934, 293.36561254101815, 330.41988171628367, 372.15438199368964,
         419.1602615366464,
         472.1033349392315, 531.7335141543668, 598.8954306186932, 674.5403990312017, 759.7398922465097,
         855.7007181478526,
         963.7821134725517, 1085.514996715385, 1222.6236528175239, 1377.0501567937265, 1550.9818822461825,
         1746.8824844092096, 1967.5267965840198, 2216.0401342540176, 2495.942563603532, 2811.198761481316,
         3166.27417305,
         3566.198262566392, 4016.635753207953, 4523.966865021721, 5095.377687525866, 5738.961967046114,
         6463.835750553389,
         7280.266509874983, 8199.818575258458, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [1.208713067578366, 1.1949389559289116, 1.172293771327121, 1.1444545814534142, 1.1190576803531855,
         1.0928453366396875, 1.0668080340932762, 1.0406139668149277, 1.015273354251854, 0.9912648992015868,
         0.9685886016641266, 0.947133448513406, 0.9267884266233576, 0.9073315097418472, 0.8892067503731437,
         0.8713040172565745, 0.8542893891485431, 0.8386069185533187, 0.8230354610841614, 0.8075750167410715,
         0.7921145723979813, 0.7777642593155637, 0.7637469856113477, 0.7495076856549971, 0.7354904119507814,
         0.7216951644986997, 0.7081219432987527, 0.6948817614770073, 0.6810865140249259, 0.6677353190771134,
         0.6537365475605751, 0.6400338110468831, 0.6261275504687344, 0.6116662242602493, 0.5982040161863695,
         0.5877391625163051, 0.5773853219723077, 0.5672535076804449, 0.557010680262515, 0.5494851303459993,
         0.5378503585770207, 0.5240459019046559, 0.5125350232139425, 0.5010317307789567, 0.4871900333474901,
         0.4783903765684343, 0.4692576804111763, 0.4610130892624567, 0.4526574849876697, 0.4447459332171515,
         0.4372784339509024, 0.4294778953064515, 0.42278748792267296, 0.4156530280346253, 0.4088516075247797,
         0.4026052526452699, 0.3966919371439621, 0.3903345691383853, 0.3841992273849431, 0.3788409775139712,
         0.3734827276429993, 0.3681244777720276, 0.3633212935313921, 0.35851810929075656, 0.35393695130225544,
         0.3490227539355526, 0.3444415959470517, 0.34074854296708845, 0.3361673849785873, 0.3318082532422204,
         0.3282262133883247, 0.3238670816519582, 0.32072909430233065, 0.3169250281963003, 0.3138980539727405,
         0.31009398786671016, 0.30662296113888154, 0.3035486094832578])

    N__0_6 = np.array(
        [1.0, 1.1578129556424808, 1.3021746170456958, 1.4666490022855916, 1.6518977314928245, 1.8605447595564395,
         2.05071313191929, 2.278164285029333, 2.5458139866962433, 2.8673693179980964, 3.2295394906155264,
         3.637454462520032,
         4.096892143710914, 4.614360237398508, 5.197188418340471, 5.853632153990756, 6.563351900483103,
         7.62129313279782,
         8.36365913289523, 9.460884166344293, 11.894730656983564, 13.459353108001784, 15.159369986834552,
         17.074111701632724, 19.230699603810226, 21.659680674141427, 24.395460205349153, 27.476789135736645,
         31.235478590071768, 35.092553201752764, 39.258787083344075, 44.21746527156313, 49.802461570989244,
         56.09288463952192, 63.17783514972992, 71.47899031152726, 78.85596302168895, 88.01933823500751,
         98.42475433577059,
         110.85653633325032, 124.40931766386194, 140.52053724644145, 158.39162047663308, 179.04183910501638,
         200.9306241481136, 226.3096635258025, 253.51988002904656, 291.9242151603233, 336.5209667633253,
         399.9305353549751,
         478.52780541318606, 537.5144225688365, 605.4065109664441, 681.8738774839574, 767.9996438306861,
         865.0037380819548,
         974.2601743454874, 1097.316514978768, 1235.915791029918, 1392.0211913940466, 1567.8438703945635,
         1765.8742676697962, 1988.9173839953753, 2240.1325127065697, 2523.0779894961747, 2841.7615944463164,
         3200.6973202135073, 3604.969310459668, 4060.303874184876, 4573.150596008435, 5150.773691284568,
         5801.354900051996,
         6534.109377258198, 7359.4162207850695, 8288.965486139592, 10000.0])

    tau_cy__Su__0_6 = np.array(
        [1.3256804871387675, 1.299874915952456, 1.2750411113470717, 1.2529669278600428, 1.2284710723229473,
         1.2053655240313603, 1.1897084212205995, 1.1703533630641176, 1.1484487148692466, 1.1285477454834671,
         1.108646776097688, 1.089522898594379, 1.0702880079650032, 1.0523852748484341, 1.0347045679839992,
         1.0172458873716992, 1.0035917617692491, 0.9834392037465988, 0.9688663180732188, 0.9583522816308372,
         0.9283481607531832, 0.907135553826926, 0.8955383662709757, 0.8792860282952726, 0.8627524570668661,
         0.8477360652280451, 0.8334967652716945, 0.8185913865589407, 0.8083977929116397, 0.7901795617963444,
         0.7748743685552842, 0.7600800029685976, 0.7449525980037093, 0.7327115343165691, 0.7224687068986391,
         0.7141489850363386, 0.7090364246343771, 0.6987921407046498, 0.6831706722505849, 0.6728168317065878,
         0.6636726983278401, 0.6524382225908579, 0.6420883494527978, 0.6351565160589114, 0.6229348521297444,
         0.6135801297203523, 0.6050704534245142, 0.5926231249304592, 0.5857069884587764, 0.5741961097680628,
         0.5606254202342145, 0.5498652047390253, 0.5417316267163728, 0.5340421011979892, 0.5266856150578072,
         0.5192181157915581, 0.5118616296513758, 0.5048381828893957, 0.4979257492534828, 0.4909023024915027,
         0.4841008819816568, 0.4776325008500129, 0.4707200672140999, 0.4644737123345899, 0.4583383705811477,
         0.4516479631973691, 0.4454016083178596, 0.4399323453208206, 0.4333529510631093, 0.42777267494000304,
         0.4217483463126279, 0.4158350308113199, 0.40981070218394505, 0.4044524523129731, 0.3986501499377326,
         0.3928478475624919])

    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
        [0.05, 0.1, 0.25, 0.6],
        [N__0_05, N__0_1, N__0_25, N__0_6],
        [tau_cy__Su__0_05, tau_cy__Su__0_1, tau_cy__Su__0_25, tau_cy__Su__0_6]
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$ u_p / \sigma_{vc}^{\prime} = %s $' % str(_strain))
        fig.append_trace(trace, 1, 1)


    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 4), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{DSS} $', range=(0, 1.5), dtick=0.25)
    fig['layout'].update(height=500, width=500,
                         title=r'$ \text{Excess pore pressure accumulation diagram for cyclic DSS tests om Drammen clay} $',
                         hovermode='closest')

    return fig


POREPRESSUREACCUMULATION_TRIAXIALCLAY_ANDERSEN = {
    'cyclic_shear_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 100.0},
    'undrained_shear_strength': {'type': 'float', 'min_value': 1.0, 'max_value': 100.0},
    'cycle_no': {'type': 'int', 'min_value': 1.0, 'max_value': 1500.0},
}

POREPRESSUREACCUMULATION_TRIAXIALCLAY_ANDERSEN_ERRORRETURN = {
    'Excess pore pressure ratio [-]': np.nan,
    'pore pressure ratios interpolation [-]': None,
    'shearstress ratios interpolation [-]': None,
}


@Validator(POREPRESSUREACCUMULATION_TRIAXIALCLAY_ANDERSEN, POREPRESSUREACCUMULATION_TRIAXIALCLAY_ANDERSEN_ERRORRETURN)
def porepressureaccumulation_triaxialclay_andersen(
        cyclic_shear_stress, undrained_shear_strength, cycle_no,
        **kwargs):
    """
    Calculates the excess pore pressure accumulation for a normally consolidated clay sample under symmetrical cyclic loading (no average shear stress) in a cyclic triaxial test. The contours are based on cyclic triaxial tests on Drammen clay.

    The accumulated excess pore pressure is corrected for the change in total octahedral normal stress, making the ratio independent on whether average shear stress is applied by increasing or decreasing the normal stress.

    Excess pore pressure contours for excess pore pressure ratios of 0.01, 0.125, 0.15, 0.2 and 0.25 are defined and logarithmic interpolation is used to obtain the accumulated excess pore pressure for a sample tested at a certain ratio of cyclic shear stress to triaxial compression shear strength with a given number of cycles.

    :param cyclic_shear_stress: Magnitude of the applied cyclic shear stress (:math:`\\tau_{cy}`) [:math:`kPa`] - Suggested range: 0.0 <= cyclic_shear_stress <= 100.0
    :param undrained_shear_strength: Undrained shear strength of the normally consolidated clay measured with a DSS test (:math:`S_u^{DSS}`) [:math:`kPa`] - Suggested range: 1.0 <= undrained_shear_strength <= 100.0
    :param cycle_no: Number of applied cycles (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= cycle_no <= 1500.0

    :returns: Dictionary with the following keys:

        - 'Excess pore pressure ratio [-]': Ratio of accumulated excess pore pressure to vertical effective stress (:math:`u_p / \\sigma_{vc}^{\\prime}`)  [:math:`-`]
        - 'pore pressure ratios interpolation [-]': List of excess pore pressure ratios used for interpolation [:math:`-`]
        - 'shearstress ratios interpolation [-]': List of ratios of cyclic shear stress to undrained DSS shear strength used for interpolation [:math:`-`]

    .. figure:: images/porepressureaccumulation_triaxialclay_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Excess pore pressure contours for symmetrical cyclic triaxial tests

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    N__0_01 = np.array([1, 1e4])

    tau_cy__Su__0_01 = np.array([0, 0])

    N__0_125 = np.array(
        [1.0, 1.1632657336143657, 1.3096093567567073, 1.4743636107769007, 1.6598446289101516, 1.8686599234975128,
         2.1037450426781783, 2.3684048386447345, 2.6663599276149945, 3.0017989946597448, 3.3794376786933755,
         3.8045848654390007, 4.28321732032178, 4.822063710487791, 5.428699197139137, 6.111651927974611,
         6.880522926818272,
         7.74612106585736, 8.72061501794396, 9.817704325122428, 11.052811988282523, 12.443301285384596,
         14.008719866311026,
         15.771074555855822, 17.75514072806293, 19.988810601131032, 22.50348534925293, 25.334516543740115,
         28.521703129259876, 32.10985092173141, 36.149402493362594, 40.697146299820545, 45.81701501852214,
         51.58098432117196, 58.070084715588806, 65.37554068140442, 73.6000531137271, 82.85924310350308,
         93.28327735139798,
         105.01869806544188, 118.23048306734013, 133.10436506864698, 149.8494427214407, 168.70112014993305,
         189.9244162872686, 213.81769054053345, 240.71683715977247, 271.0000072759177, 305.0929249905436,
         343.47487225162604, 386.6854266513305, 435.3320468665337, 490.09861237901885, 551.7550375276652,
         621.168095048025,
         699.3136012577777, 787.2901341887384, 886.3344775163949, 997.8390073994392, 1123.3712666553447,
         1264.6960014479241,
         1423.7999702809625, 1602.919874065518, 1804.5737999047665, 2031.5966206365906, 2287.1798477844627,
         2574.9164981738313, 2898.85160495371, 3263.5390831129835, 3674.1057489129403, 4136.3233932283065,
         4656.689922011599, 5242.5207045622055, 5902.051413784245, 6644.553804172619, 7480.466055146918,
         8421.539512113728,
         10000.0])

    tau_cy__Su__0_125 = np.array(
        [0.20765835172614835, 0.2023287798998683, 0.2009129352454748, 0.198032539536952, 0.1945643079695468,
         0.1912724271598064, 0.18803932993595415, 0.18468866554032548, 0.182690023620126, 0.17804612033495634,
         0.17481302311110414, 0.17140357512958726, 0.16846439583517614, 0.16576035088431784, 0.16293873876168308,
         0.16005834305316036, 0.15741308168819024, 0.15476782032322034, 0.15212255895825022, 0.14912459607795092,
         0.14642055112709262, 0.1438928569338992, 0.1411300283971526, 0.13824963268862978, 0.13566315490954806,
         0.1328415427869133, 0.13072533369493733, 0.1284915574311848, 0.12637534833920885, 0.12461184076256215,
         0.12261319884236264, 0.1207321240939394, 0.11855713141607528, 0.11649970590998747, 0.11461863116156434,
         0.11267877282725293, 0.1107389144929416, 0.10879905615863028, 0.10656527989487796, 0.10456663797467823,
         0.10321461549924926, 0.1017450258520437, 0.10021665261894984, 0.09862949579996794, 0.09663085387976833,
         0.09551396574789206, 0.09380924175713368, 0.09216330135226347, 0.09081127887683428, 0.08951803998729335,
         0.08845993544130548, 0.08734304730942921, 0.08599102483400001, 0.0854031889751179, 0.08381603215613588,
         0.08228765892304203, 0.08187617382182455, 0.08040658417461899, 0.07970118114396019, 0.07823159149675463,
         0.07782010639553716, 0.07646808392010795, 0.07558633013178473, 0.07470457634346139, 0.07352890462569682,
         0.0720005313926032, 0.0716478298772738, 0.0702958074018446, 0.06976675512885068, 0.06935527002763309,
         0.06806203113809217, 0.06782689679453935, 0.06759176245098653, 0.06700392659210419, 0.0660046056320045,
         0.06547555335901034, 0.0646525831565753, 0.06409861325115562])

    N__0_15 = np.array(
        [1.0, 1.1384712199630116, 1.281695591109064, 1.4429381783773023, 1.6244657476095168, 1.828830233131807,
         2.058904613125093, 2.3179232982650904, 2.9336002702864774, 3.2828318881488414, 3.7234917594891135,
         4.191922367456229, 4.719283476322464, 5.312988785955322, 5.981384670217731, 6.733867511200224,
         7.581015794583108,
         8.534738823139538, 9.608444139010127, 10.81722600838967, 12.178077618364194, 13.710129968984615,
         15.434920818947655, 17.376697465751885, 19.562757616845, 22.02383314376637, 24.794522114141923,
         27.913775175084663,
         31.42544312563128, 35.378893376047614, 39.829703960255706, 44.84044485788365, 50.481557609849645,
         56.83234559321106, 63.98208887667084, 72.03129932950047, 81.31182201997234, 91.29498402606652,
         102.78026924596607,
         115.7104506777407, 130.2673022193054, 146.65546567402941, 165.10532762748014, 185.87625824712018,
         209.26025753635133, 235.58606030234347, 265.22375754573585, 298.58999924020316, 336.15385164313693,
         378.44339148014365, 426.0531297051329, 479.65236920001223, 539.9946139074518, 607.9281616713238,
         684.4080297001171,
         770.5093803028079, 867.4426356376148, 976.5704939584504, 1099.4270865752874, 1237.7395448390946,
         1393.4522803423608, 1568.754319668882, 1766.1100779677395, 1988.2940039697114, 2238.429583489484,
         2520.033249729213, 2837.063817679215, 3193.978137569996, 3595.793749757862, 4048.159421853376,
         4557.4345597112815,
         5130.778608650124, 5776.25170171383, 6502.927969899287, 7321.023107277608, 8242.037984333196, 10000.0])

    tau_cy__Su__0_15 = np.array(
        [0.3600691208087202, 0.357688758809364, 0.35292728835241793, 0.34828338506724843, 0.34346313102441417,
         0.33881922773924456, 0.33388140652463394, 0.3295510156975349, 0.32148687810937426, 0.3160292836718106,
         0.3102504049975685, 0.3056652852982872, 0.30078624766956463, 0.29602477721261866, 0.2914396575133373,
         0.286854537814056, 0.28238698528655115, 0.2780957835167108, 0.2736870145750941, 0.2692194620475893,
         0.2648106931059726, 0.26028435699257946, 0.2557580208791863, 0.2515256026952344, 0.2468816994100648,
         0.24294319915555385, 0.23959253475992515, 0.2361243031925201, 0.23294998955455604, 0.2297168923307038,
         0.2261898771774105, 0.22248651126645247, 0.2193121976284884, 0.2154716700171244, 0.21290478676667224,
         0.20973047312870816, 0.2081236440250308, 0.20291157716567446, 0.1996784799418222, 0.1966805170615228,
         0.19291836756467665, 0.18939135241138327, 0.18586433725808996, 0.18233732210479647, 0.17863395619383848,
         0.17510694104054525, 0.17152114230136362, 0.16805291073395856, 0.16464346275244168, 0.1608813132555954,
         0.15741308168819024, 0.15476782032322034, 0.15212255895825022, 0.1492421632497274, 0.14659690188475727,
         0.14424555844922848, 0.14142394632659372, 0.13854355061807078, 0.13601585642487735, 0.13436991602000714,
         0.13225370692803118, 0.13054898293727268, 0.12837399025940846, 0.12625778116743236, 0.12443549000489752,
         0.12220171374114495, 0.12032063899272195, 0.1177929447995284, 0.11567673570755233, 0.11350174302968817,
         0.11197336979659434, 0.1104449965635006, 0.10891662333040676, 0.10697676499609544, 0.10656527989487796,
         0.1053896081771134, 0.10455488125750068])

    N__0_2 = np.array(
        [1.0, 1.1508017026790995, 1.2955772993651322, 1.4585662627389278, 1.6420599093876485, 1.8486378129677052,
         2.0812040681319908, 2.35931032574401, 2.6425308989888032, 2.9696356510231716, 3.3612841078598,
         3.728213731467251,
         4.237323977433493, 4.770396795936289, 5.370532371816123, 6.046167476318695, 6.806800261280613,
         7.663123785181705,
         8.627176337324977, 9.712510673417917, 10.934384541687331, 12.309975177964597, 13.85862078513663,
         15.602092391706787, 17.564899911283277, 19.77463670561227, 22.262367495060744, 25.0630650698365,
         28.21610194127936,
         31.765803844911108, 35.76207287644978, 40.2610890209029, 45.32609993691128, 51.028310099218395,
         57.44788179892611,
         64.67506207370832, 72.81145141048907, 81.97143205614597, 92.28377601558692, 103.89345534261015,
         116.96368017281652,
         131.67819314945885, 148.2438524975208, 166.89354005915743, 187.88943517198058, 211.52670041472282,
         238.13763103489453, 268.0963263915428, 301.82394908475663, 339.79464570534145, 382.54221243919716,
         430.6675992322018, 484.84735800995634, 545.8431537183949, 614.5124718924945, 691.820673281255,
         778.8545649941838,
         876.837678953927, 987.1474724412152, 1111.334692539236, 1251.1451767049166, 1408.5443959419215,
         1585.7450856059372,
         1785.2383522791424, 2009.828694017692, 2262.673426290616, 2547.327067864346, 2867.791310614406,
         3228.5712757453716,
         3634.73884727508, 4092.003973132199, 4606.794936225715, 5186.348723945695, 5838.812766519726,
         6573.359474473828,
         7400.315185378009, 8331.305332623706, 10000.0])

    tau_cy__Su__0_2 = np.array(
        [0.4524172828101951, 0.4497438543103202, 0.4443357644086038, 0.4388688909209991, 0.43334323384750617,
         0.42793514394578974, 0.4225270540440733, 0.4166185572040735, 0.4115667144942953, 0.4070865654841002,
         0.4025853756824482, 0.3977396542410592, 0.3899609474619981, 0.3844352903885052, 0.378850849729124,
         0.3732664090697429, 0.3674468340668089, 0.3617448262356513, 0.356101601990382, 0.3506347285027773,
         0.34487393708573155, 0.33952463076990325, 0.33482194389884545, 0.3303543913713406, 0.3257692716720592,
         0.32118415197277794, 0.3165990322734965, 0.31213147974599165, 0.30748757646082203, 0.30278488958976424,
         0.29796463554693, 0.2933207322617605, 0.2885004782189262, 0.283680224176092, 0.27897753730503416,
         0.2742160668480882, 0.2694545963911422, 0.2645167751765315, 0.2597553047195855, 0.25469991633319833,
         0.2499384458762524, 0.2448830574898653, 0.24018037061880734, 0.23518376581830855, 0.2303635117754743,
         0.2258959592479693, 0.22213380975112307, 0.21854801101194155, 0.2147270779292071, 0.21084736126058434,
         0.2073203461072911, 0.20338184585278016, 0.19956091277004573, 0.19621024837441706, 0.192742016807012,
         0.18945013599727145, 0.18615825518753104, 0.18274880720601416, 0.17916300846683264, 0.17604747841475665,
         0.17275559760501635, 0.1695812839670522, 0.16699480618797047, 0.16429076123711228, 0.16141036552858945,
         0.15858875340595469, 0.1558847084550965, 0.15323944709012638, 0.1505354021392683, 0.14824284228962747,
         0.1455387973387694, 0.14295231955968754, 0.14065975971004696, 0.13895503571928847, 0.13766179682974755,
         0.13666247586964775, 0.13542802056599512, 0.13303598080177434])

    N__0_25 = np.array(
        [1.0, 1.1384712199630116, 1.281695591109064, 1.4429381783773023, 1.6244657476095168, 1.828830233131807,
         2.058904613125093, 2.3179232982650904, 2.6095276014196203, 2.9378169276212382, 3.3074064039493707,
         3.7234917594891135, 4.191922367456229, 4.719283476322464, 5.312988785955322, 5.981384670217731,
         6.733867511200224,
         7.581015794583108, 8.534738823139538, 9.608444139010127, 10.81722600838967, 12.178077618364194,
         13.710129968984615,
         15.434920818947655, 17.376697465751885, 19.562757616845, 22.02383314376637, 24.794522114141923,
         27.913775175084663,
         31.42544312563128, 35.378893376047614, 39.829703960255706, 44.84044485788365, 50.481557609849645,
         56.83234559321106, 63.98208887667084, 72.03129932950047, 81.09313362833844, 91.29498402606652,
         102.78026924596607,
         115.7104506777407, 130.2673022193054, 146.65546567402941, 165.10532762748014, 185.87625824712018,
         209.26025753635133, 235.58606030234347, 265.22375754573585, 298.58999924020316, 336.15385164313693,
         378.44339148014365, 426.0531297051329, 479.65236920001223, 539.9946139074518, 607.9281616713238,
         684.4080297001171,
         770.5093803028079, 867.4426356376148, 976.5704939584504, 1099.4270865752874, 1237.7395448390946,
         1393.4522803423608, 1568.754319668882, 1766.1100779677395, 1988.2940039697114, 2238.429583489484,
         2520.033249729213, 2837.063817679215, 3193.978137569996, 3595.793749757862, 4048.159421853376,
         4557.4345597112815,
         5130.778608650124, 5776.25170171383, 6502.927969899287, 7321.023107277608, 8242.037984333196, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [0.5345013875060101, 0.5289841280876441, 0.5222827992963868, 0.5156402540910175, 0.5088801417138721,
         0.5025902980238321, 0.4957714020607984, 0.4905200683881171, 0.4838215388152354, 0.4761376843741321,
         0.469495139168763, 0.4629701611351703, 0.4566803174451305, 0.4503316901692024, 0.4439242793073862,
         0.4376932192032347, 0.4312858083414184, 0.42487839747960215, 0.4184709866177859, 0.4125338444430755,
         0.4069494037836944, 0.40130617953842496, 0.3957805224649321, 0.3901372982196628, 0.3844940739743934,
         0.3790272004867887, 0.3733251926556311, 0.3677407519962501, 0.3621563113368688, 0.3565130870915996,
         0.3511637807757713, 0.34552055653050195, 0.3400536830428973, 0.3344104587976279, 0.3290023688959114,
         0.32353549540830684, 0.3180098383348139, 0.3124253976754328, 0.3073112257031574, 0.3022558373167702,
         0.2972004489303831, 0.29208627695810785, 0.28703088857172065, 0.2819167165994453, 0.2768025446271699,
         0.27198229058433576, 0.2673971708850543, 0.26298840194343776, 0.2581681479006035, 0.2537005953730985,
         0.24929182643148184, 0.2447067067322005, 0.2402391542046957, 0.23565403450541425, 0.2314216163214623,
         0.22812973551172175, 0.22519055621731066, 0.2220162425793467, 0.2190182796990474, 0.21596153323285985,
         0.21261086883723115, 0.20973047312870816, 0.20673251024840889, 0.2034406294386684, 0.20073658448781015,
         0.19779740519339906, 0.19485822589898796, 0.1919190466045768, 0.18939135241138327, 0.18709879256174256,
         0.18457109836854899, 0.18210218776124365, 0.17980962791160304, 0.17769341881962708, 0.17557720972765098,
         0.1736373513933397, 0.17122722437192262, 0.1692873660376113])


    _porepressure_ratios_interpolation = np.array([0.01, 0.125, 0.15, 0.2, 0.25])
    _shearstress_ratios_interpolation = np.array([
        np.interp(np.log10(cycle_no), np.log10(N__0_01), tau_cy__Su__0_01),
        np.interp(np.log10(cycle_no), np.log10(N__0_125), tau_cy__Su__0_125),
        np.interp(np.log10(cycle_no), np.log10(N__0_15), tau_cy__Su__0_15),
        np.interp(np.log10(cycle_no), np.log10(N__0_2), tau_cy__Su__0_2),
        np.interp(np.log10(cycle_no), np.log10(N__0_25), tau_cy__Su__0_25)
    ])

    fit_coeff = np.polyfit(
        x=_shearstress_ratios_interpolation,
        y=np.log(_porepressure_ratios_interpolation),
        deg=1)

    _porepressure_ratio_0 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0)
    _porepressure_ratio_0_8 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0.8)

    _shearstress_ratios_interpolation = np.append(np.append(1e-7, _shearstress_ratios_interpolation), 0.8)
    _porepressure_ratios_interpolation = np.append(
        np.append(_porepressure_ratio_0, _porepressure_ratios_interpolation), _porepressure_ratio_0_8)

    _porepressure_ratio = 10 ** (np.interp(
        cyclic_shear_stress / undrained_shear_strength,
        _shearstress_ratios_interpolation,
        np.log10(_porepressure_ratios_interpolation)))

    if _porepressure_ratio < 0.01:
        warnings.warn(
            "Cyclic strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _porepressure_ratio > 0.25:
        warnings.warn(
            "Cyclic strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    return {
        'Excess pore pressure ratio [-]': _porepressure_ratio,
        'pore pressure ratios interpolation [-]': _porepressure_ratios_interpolation,
        'shearstress ratios interpolation [-]': _shearstress_ratios_interpolation,
    }


def plotporepressureaccumulation_triaxialclay_andersen():
    """
    Returns a Plotly figure with the excess pore pressure accumulation contours for cyclic triaxial tests on normally consolidated Drammen clay with symmetrical loading
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=False)

    N__0_01 = np.array([1, 1e4])

    tau_cy__Su__0_01 = np.array([0, 0])

    N__0_125 = np.array(
        [1.0, 1.1632657336143657, 1.3096093567567073, 1.4743636107769007, 1.6598446289101516, 1.8686599234975128,
         2.1037450426781783, 2.3684048386447345, 2.6663599276149945, 3.0017989946597448, 3.3794376786933755,
         3.8045848654390007, 4.28321732032178, 4.822063710487791, 5.428699197139137, 6.111651927974611,
         6.880522926818272,
         7.74612106585736, 8.72061501794396, 9.817704325122428, 11.052811988282523, 12.443301285384596,
         14.008719866311026,
         15.771074555855822, 17.75514072806293, 19.988810601131032, 22.50348534925293, 25.334516543740115,
         28.521703129259876, 32.10985092173141, 36.149402493362594, 40.697146299820545, 45.81701501852214,
         51.58098432117196, 58.070084715588806, 65.37554068140442, 73.6000531137271, 82.85924310350308,
         93.28327735139798,
         105.01869806544188, 118.23048306734013, 133.10436506864698, 149.8494427214407, 168.70112014993305,
         189.9244162872686, 213.81769054053345, 240.71683715977247, 271.0000072759177, 305.0929249905436,
         343.47487225162604, 386.6854266513305, 435.3320468665337, 490.09861237901885, 551.7550375276652,
         621.168095048025,
         699.3136012577777, 787.2901341887384, 886.3344775163949, 997.8390073994392, 1123.3712666553447,
         1264.6960014479241,
         1423.7999702809625, 1602.919874065518, 1804.5737999047665, 2031.5966206365906, 2287.1798477844627,
         2574.9164981738313, 2898.85160495371, 3263.5390831129835, 3674.1057489129403, 4136.3233932283065,
         4656.689922011599, 5242.5207045622055, 5902.051413784245, 6644.553804172619, 7480.466055146918,
         8421.539512113728,
         10000.0])

    tau_cy__Su__0_125 = np.array(
        [0.20765835172614835, 0.2023287798998683, 0.2009129352454748, 0.198032539536952, 0.1945643079695468,
         0.1912724271598064, 0.18803932993595415, 0.18468866554032548, 0.182690023620126, 0.17804612033495634,
         0.17481302311110414, 0.17140357512958726, 0.16846439583517614, 0.16576035088431784, 0.16293873876168308,
         0.16005834305316036, 0.15741308168819024, 0.15476782032322034, 0.15212255895825022, 0.14912459607795092,
         0.14642055112709262, 0.1438928569338992, 0.1411300283971526, 0.13824963268862978, 0.13566315490954806,
         0.1328415427869133, 0.13072533369493733, 0.1284915574311848, 0.12637534833920885, 0.12461184076256215,
         0.12261319884236264, 0.1207321240939394, 0.11855713141607528, 0.11649970590998747, 0.11461863116156434,
         0.11267877282725293, 0.1107389144929416, 0.10879905615863028, 0.10656527989487796, 0.10456663797467823,
         0.10321461549924926, 0.1017450258520437, 0.10021665261894984, 0.09862949579996794, 0.09663085387976833,
         0.09551396574789206, 0.09380924175713368, 0.09216330135226347, 0.09081127887683428, 0.08951803998729335,
         0.08845993544130548, 0.08734304730942921, 0.08599102483400001, 0.0854031889751179, 0.08381603215613588,
         0.08228765892304203, 0.08187617382182455, 0.08040658417461899, 0.07970118114396019, 0.07823159149675463,
         0.07782010639553716, 0.07646808392010795, 0.07558633013178473, 0.07470457634346139, 0.07352890462569682,
         0.0720005313926032, 0.0716478298772738, 0.0702958074018446, 0.06976675512885068, 0.06935527002763309,
         0.06806203113809217, 0.06782689679453935, 0.06759176245098653, 0.06700392659210419, 0.0660046056320045,
         0.06547555335901034, 0.0646525831565753, 0.06409861325115562])

    N__0_15 = np.array(
        [1.0, 1.1384712199630116, 1.281695591109064, 1.4429381783773023, 1.6244657476095168, 1.828830233131807,
         2.058904613125093, 2.3179232982650904, 2.9336002702864774, 3.2828318881488414, 3.7234917594891135,
         4.191922367456229, 4.719283476322464, 5.312988785955322, 5.981384670217731, 6.733867511200224,
         7.581015794583108,
         8.534738823139538, 9.608444139010127, 10.81722600838967, 12.178077618364194, 13.710129968984615,
         15.434920818947655, 17.376697465751885, 19.562757616845, 22.02383314376637, 24.794522114141923,
         27.913775175084663,
         31.42544312563128, 35.378893376047614, 39.829703960255706, 44.84044485788365, 50.481557609849645,
         56.83234559321106, 63.98208887667084, 72.03129932950047, 81.31182201997234, 91.29498402606652,
         102.78026924596607,
         115.7104506777407, 130.2673022193054, 146.65546567402941, 165.10532762748014, 185.87625824712018,
         209.26025753635133, 235.58606030234347, 265.22375754573585, 298.58999924020316, 336.15385164313693,
         378.44339148014365, 426.0531297051329, 479.65236920001223, 539.9946139074518, 607.9281616713238,
         684.4080297001171,
         770.5093803028079, 867.4426356376148, 976.5704939584504, 1099.4270865752874, 1237.7395448390946,
         1393.4522803423608, 1568.754319668882, 1766.1100779677395, 1988.2940039697114, 2238.429583489484,
         2520.033249729213, 2837.063817679215, 3193.978137569996, 3595.793749757862, 4048.159421853376,
         4557.4345597112815,
         5130.778608650124, 5776.25170171383, 6502.927969899287, 7321.023107277608, 8242.037984333196, 10000.0])

    tau_cy__Su__0_15 = np.array(
        [0.3600691208087202, 0.357688758809364, 0.35292728835241793, 0.34828338506724843, 0.34346313102441417,
         0.33881922773924456, 0.33388140652463394, 0.3295510156975349, 0.32148687810937426, 0.3160292836718106,
         0.3102504049975685, 0.3056652852982872, 0.30078624766956463, 0.29602477721261866, 0.2914396575133373,
         0.286854537814056, 0.28238698528655115, 0.2780957835167108, 0.2736870145750941, 0.2692194620475893,
         0.2648106931059726, 0.26028435699257946, 0.2557580208791863, 0.2515256026952344, 0.2468816994100648,
         0.24294319915555385, 0.23959253475992515, 0.2361243031925201, 0.23294998955455604, 0.2297168923307038,
         0.2261898771774105, 0.22248651126645247, 0.2193121976284884, 0.2154716700171244, 0.21290478676667224,
         0.20973047312870816, 0.2081236440250308, 0.20291157716567446, 0.1996784799418222, 0.1966805170615228,
         0.19291836756467665, 0.18939135241138327, 0.18586433725808996, 0.18233732210479647, 0.17863395619383848,
         0.17510694104054525, 0.17152114230136362, 0.16805291073395856, 0.16464346275244168, 0.1608813132555954,
         0.15741308168819024, 0.15476782032322034, 0.15212255895825022, 0.1492421632497274, 0.14659690188475727,
         0.14424555844922848, 0.14142394632659372, 0.13854355061807078, 0.13601585642487735, 0.13436991602000714,
         0.13225370692803118, 0.13054898293727268, 0.12837399025940846, 0.12625778116743236, 0.12443549000489752,
         0.12220171374114495, 0.12032063899272195, 0.1177929447995284, 0.11567673570755233, 0.11350174302968817,
         0.11197336979659434, 0.1104449965635006, 0.10891662333040676, 0.10697676499609544, 0.10656527989487796,
         0.1053896081771134, 0.10455488125750068])

    N__0_2 = np.array(
        [1.0, 1.1508017026790995, 1.2955772993651322, 1.4585662627389278, 1.6420599093876485, 1.8486378129677052,
         2.0812040681319908, 2.35931032574401, 2.6425308989888032, 2.9696356510231716, 3.3612841078598,
         3.728213731467251,
         4.237323977433493, 4.770396795936289, 5.370532371816123, 6.046167476318695, 6.806800261280613,
         7.663123785181705,
         8.627176337324977, 9.712510673417917, 10.934384541687331, 12.309975177964597, 13.85862078513663,
         15.602092391706787, 17.564899911283277, 19.77463670561227, 22.262367495060744, 25.0630650698365,
         28.21610194127936,
         31.765803844911108, 35.76207287644978, 40.2610890209029, 45.32609993691128, 51.028310099218395,
         57.44788179892611,
         64.67506207370832, 72.81145141048907, 81.97143205614597, 92.28377601558692, 103.89345534261015,
         116.96368017281652,
         131.67819314945885, 148.2438524975208, 166.89354005915743, 187.88943517198058, 211.52670041472282,
         238.13763103489453, 268.0963263915428, 301.82394908475663, 339.79464570534145, 382.54221243919716,
         430.6675992322018, 484.84735800995634, 545.8431537183949, 614.5124718924945, 691.820673281255,
         778.8545649941838,
         876.837678953927, 987.1474724412152, 1111.334692539236, 1251.1451767049166, 1408.5443959419215,
         1585.7450856059372,
         1785.2383522791424, 2009.828694017692, 2262.673426290616, 2547.327067864346, 2867.791310614406,
         3228.5712757453716,
         3634.73884727508, 4092.003973132199, 4606.794936225715, 5186.348723945695, 5838.812766519726,
         6573.359474473828,
         7400.315185378009, 8331.305332623706, 10000.0])

    tau_cy__Su__0_2 = np.array(
        [0.4524172828101951, 0.4497438543103202, 0.4443357644086038, 0.4388688909209991, 0.43334323384750617,
         0.42793514394578974, 0.4225270540440733, 0.4166185572040735, 0.4115667144942953, 0.4070865654841002,
         0.4025853756824482, 0.3977396542410592, 0.3899609474619981, 0.3844352903885052, 0.378850849729124,
         0.3732664090697429, 0.3674468340668089, 0.3617448262356513, 0.356101601990382, 0.3506347285027773,
         0.34487393708573155, 0.33952463076990325, 0.33482194389884545, 0.3303543913713406, 0.3257692716720592,
         0.32118415197277794, 0.3165990322734965, 0.31213147974599165, 0.30748757646082203, 0.30278488958976424,
         0.29796463554693, 0.2933207322617605, 0.2885004782189262, 0.283680224176092, 0.27897753730503416,
         0.2742160668480882, 0.2694545963911422, 0.2645167751765315, 0.2597553047195855, 0.25469991633319833,
         0.2499384458762524, 0.2448830574898653, 0.24018037061880734, 0.23518376581830855, 0.2303635117754743,
         0.2258959592479693, 0.22213380975112307, 0.21854801101194155, 0.2147270779292071, 0.21084736126058434,
         0.2073203461072911, 0.20338184585278016, 0.19956091277004573, 0.19621024837441706, 0.192742016807012,
         0.18945013599727145, 0.18615825518753104, 0.18274880720601416, 0.17916300846683264, 0.17604747841475665,
         0.17275559760501635, 0.1695812839670522, 0.16699480618797047, 0.16429076123711228, 0.16141036552858945,
         0.15858875340595469, 0.1558847084550965, 0.15323944709012638, 0.1505354021392683, 0.14824284228962747,
         0.1455387973387694, 0.14295231955968754, 0.14065975971004696, 0.13895503571928847, 0.13766179682974755,
         0.13666247586964775, 0.13542802056599512, 0.13303598080177434])

    N__0_25 = np.array(
        [1.0, 1.1384712199630116, 1.281695591109064, 1.4429381783773023, 1.6244657476095168, 1.828830233131807,
         2.058904613125093, 2.3179232982650904, 2.6095276014196203, 2.9378169276212382, 3.3074064039493707,
         3.7234917594891135, 4.191922367456229, 4.719283476322464, 5.312988785955322, 5.981384670217731,
         6.733867511200224,
         7.581015794583108, 8.534738823139538, 9.608444139010127, 10.81722600838967, 12.178077618364194,
         13.710129968984615,
         15.434920818947655, 17.376697465751885, 19.562757616845, 22.02383314376637, 24.794522114141923,
         27.913775175084663,
         31.42544312563128, 35.378893376047614, 39.829703960255706, 44.84044485788365, 50.481557609849645,
         56.83234559321106, 63.98208887667084, 72.03129932950047, 81.09313362833844, 91.29498402606652,
         102.78026924596607,
         115.7104506777407, 130.2673022193054, 146.65546567402941, 165.10532762748014, 185.87625824712018,
         209.26025753635133, 235.58606030234347, 265.22375754573585, 298.58999924020316, 336.15385164313693,
         378.44339148014365, 426.0531297051329, 479.65236920001223, 539.9946139074518, 607.9281616713238,
         684.4080297001171,
         770.5093803028079, 867.4426356376148, 976.5704939584504, 1099.4270865752874, 1237.7395448390946,
         1393.4522803423608, 1568.754319668882, 1766.1100779677395, 1988.2940039697114, 2238.429583489484,
         2520.033249729213, 2837.063817679215, 3193.978137569996, 3595.793749757862, 4048.159421853376,
         4557.4345597112815,
         5130.778608650124, 5776.25170171383, 6502.927969899287, 7321.023107277608, 8242.037984333196, 10000.0])

    tau_cy__Su__0_25 = np.array(
        [0.5345013875060101, 0.5289841280876441, 0.5222827992963868, 0.5156402540910175, 0.5088801417138721,
         0.5025902980238321, 0.4957714020607984, 0.4905200683881171, 0.4838215388152354, 0.4761376843741321,
         0.469495139168763, 0.4629701611351703, 0.4566803174451305, 0.4503316901692024, 0.4439242793073862,
         0.4376932192032347, 0.4312858083414184, 0.42487839747960215, 0.4184709866177859, 0.4125338444430755,
         0.4069494037836944, 0.40130617953842496, 0.3957805224649321, 0.3901372982196628, 0.3844940739743934,
         0.3790272004867887, 0.3733251926556311, 0.3677407519962501, 0.3621563113368688, 0.3565130870915996,
         0.3511637807757713, 0.34552055653050195, 0.3400536830428973, 0.3344104587976279, 0.3290023688959114,
         0.32353549540830684, 0.3180098383348139, 0.3124253976754328, 0.3073112257031574, 0.3022558373167702,
         0.2972004489303831, 0.29208627695810785, 0.28703088857172065, 0.2819167165994453, 0.2768025446271699,
         0.27198229058433576, 0.2673971708850543, 0.26298840194343776, 0.2581681479006035, 0.2537005953730985,
         0.24929182643148184, 0.2447067067322005, 0.2402391542046957, 0.23565403450541425, 0.2314216163214623,
         0.22812973551172175, 0.22519055621731066, 0.2220162425793467, 0.2190182796990474, 0.21596153323285985,
         0.21261086883723115, 0.20973047312870816, 0.20673251024840889, 0.2034406294386684, 0.20073658448781015,
         0.19779740519339906, 0.19485822589898796, 0.1919190466045768, 0.18939135241138327, 0.18709879256174256,
         0.18457109836854899, 0.18210218776124365, 0.17980962791160304, 0.17769341881962708, 0.17557720972765098,
         0.1736373513933397, 0.17122722437192262, 0.1692873660376113])

    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
        [0.01, 0.125, 0.15, 0.2, 0.25],
        [N__0_01, N__0_125, N__0_15, N__0_2, N__0_25],
        [tau_cy__Su__0_01, tau_cy__Su__0_125, tau_cy__Su__0_15, tau_cy__Su__0_2, tau_cy__Su__0_25]
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$ ( u_p - \Delta \sigma_{oct} ) / \sigma_{vc}^{\prime} = %s $' % str(_strain))
        fig.append_trace(trace, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 4), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / S_u^{C} $', range=(0, 0.8), dtick=0.2)
    fig['layout'].update(height=500, width=500,
                         title=r'$ \text{Excess pore pressure accumulation diagram for cyclic triaxial tests om Drammen clay} $',
                         hovermode='closest')

    return fig




STRAINACCUMULATION_DSSSAND_ANDERSEN = {
    'shearstress_ratio': {'type': 'float', 'min_value': 0.0, 'max_value': 2.0},
    'cycle_no': {'type': 'int', 'min_value': 1.0, 'max_value': 1000.0},
    'failure_stress_ratio': {'type': 'float', 'min_value': 0.19, 'max_value': None},
}

STRAINACCUMULATION_DSSSAND_ANDERSEN_ERRORRETURN = {
    'cyclic strain [%]': np.nan,
    'shear strains interpolation [%]': None,
    'shearstress ratios interpolation [-]': None,
}


@Validator(STRAINACCUMULATION_DSSSAND_ANDERSEN, STRAINACCUMULATION_DSSSAND_ANDERSEN_ERRORRETURN)
def strainaccumulation_dsssand_andersen(
        shearstress_ratio, cycle_no, failure_stress_ratio,
        **kwargs):
    """
    Calculates the strain accumulation as a function of the cyclic shear stress level and the number of cycles for normally consolidated sand and silt in a symmetrical cyclic DSS test (no average shear stress) for different levels of cyclic shear stress at failure. The contours have been compiled based on various tests in the NGI database and should be regarded as estimates.

    Strain contours for 0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10 and 15% cyclic shear strain are defined and logarithmic interpolation is used between the different contours to obtain the accumulated strain for a sample tested at a certain ratio of cyclic shear stress to vertical effective stress with a given number of cycles.

    To calculate the cyclic stress ratio, the vertical effective stress needs to be normalised as follows:

    .. math::
        \\sigma_{ref}^{\\prime} = p_a \cdot ( \\sigma_{vc}^{\\prime} / p_a ) ^ n

    A shear stress exponent of 0.9 is suggested by Andersen (2015).

    :param shearstress_ratio: Ratio cyclic shear stress to reference effective stress (:math:`\\tau_{cy} / \\sigma_{ref}^{\\prime}`) [:math:`-`] - Suggested range: 0.0 <= shearstress_ratio <= 2.0
    :param cycle_no: Number of applied cycles (:math:`N`) [:math:`-`] - Suggested range: 1.0 <= cycle_no <= 1000.0
    :param failure_stress_ratio: Ratio of cyclic shear stress to vertical effective stress for failure at N=10 (:math:`( \\tau_{cy} / \\sigma_{ref}^{\\prime})_{N=10}`) [:math:`-`] - Allowable value: [0.19, 0.25, 0.6, 1.0, 1.8]

    :returns: Dictionary with the following keys:

        - 'cyclic strain [%]': Accumulated cyclic shear strain (:math:`\\gamma_{cy}`)  [:math:`%`]
        - 'shear strains interpolation [%]': List of shear strains used for the interpolation [:math:`%`]
        - 'shearstress ratios interpolation [-]': Shear stress ratios used for the interpolation [:math:`-`]

    .. figure:: images/strainaccumulation_dsssand_andersen_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Cyclic strain accumulation contours based on symmetrical cyclic DSS tests on normally consolidated sand

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    # Cyclic shear strain sand

    # tau_cy / sigma_ref N = 10 = 0.19

    veryloose_N__0_1 = np.array(
        [1.0, 1.9905399433448885, 4.8829023626795545, 9.882426826518873, 20.144333637328288, 39.34612116439103,
         59.896031376285755, 86.12949526940572, 137.817060720145, 217.4099645912803, 465.8729490556514,
         756.1433585416446,
         1000.0])

    veryloose_tau_ratio__0_1 = np.array(
        [0.09929203539823006, 0.0998230088495575, 0.0982300884955752, 0.0982300884955752, 0.09716814159292034,
         0.09451327433628318, 0.09238938053097344, 0.0902654867256637, 0.08761061946902651, 0.0838938053097345,
         0.07805309734513274, 0.07539823008849555, 0.07433628318584065])

    veryloose_N__0_25 = np.array(
        [1.0, 1.7114769039291653, 2.8991079838241167, 4.911109720023127, 9.457967746751489, 17.2068400704796,
         30.643790671362506, 54.57306902524658, 95.8115252456874, 168.20569972172868, 308.17963839557683,
         576.8123855207602, 1000.0])

    veryloose_tau_ratio__0_25 = np.array(
        [0.16407079646017694, 0.1630088495575221, 0.16035398230088493, 0.15557522123893802, 0.14761061946902654,
         0.13699115044247787, 0.12477876106194688, 0.11309734513274332, 0.10247787610619466, 0.0934513274336283,
         0.0860176991150442, 0.079646017699115, 0.07592920353982299])

    veryloose_N__0_5 = np.array(
        [1.0, 1.9439266109192, 3.935372612958882, 6.388745150051343, 10.371719037920348, 15.131161770385535,
         22.07409246457936, 37.93915680374376, 53.032433545756085, 87.9527883291678, 142.77492733874138,
         223.64695836692204, 323.9194332732463, 610.6050943098223, 1000.0])

    veryloose_tau_ratio__0_5 = np.array(
        [0.2065486725663717, 0.1980530973451327, 0.1863716814159292, 0.17469026548672564, 0.16247787610619469,
         0.15238938053097342, 0.14336283185840706, 0.12849557522123892, 0.11946902654867254, 0.107787610619469,
         0.09876106194690262, 0.09238938053097344, 0.08761061946902651, 0.08123893805309731, 0.07805309734513274])

    veryloose_N__1 = np.array(
        [1.0, 1.9850529742633432, 3.5352403044924263, 6.075854079454643, 10.97713502928205, 19.412985374161373,
         34.82464429594147, 53.78774432277998, 89.84524950011499, 129.21388581984036, 173.0442219237976,
         261.57652095673933, 392.59264453190076, 655.6496542296684, 1000.0])

    veryloose_tau_ratio__1 = np.array(
        [0.2283185840707964, 0.21610619469026546, 0.20336283185840706, 0.19008849557522126, 0.1725663716814159,
         0.15504424778761058, 0.13699115044247787, 0.123716814159292, 0.1109734513274336, 0.1030088495575221,
         0.09876106194690262, 0.09292035398230086, 0.08761061946902651, 0.08283185840707963, 0.079646017699115])

    veryloose_N__5 = np.array(
        [1.0, 1.720127440491086, 2.7730154250757146, 4.162883104203345, 6.384398263586898, 8.92508108548957,
         12.745937149943687, 21.908890327307915, 33.124939621722035, 52.642837952063275, 114.46183555359389,
         278.87298224832324, 560.5144430749845, 1000.0])

    veryloose_tau_ratio__5 = np.array(
        [0.2707964601769911, 0.25061946902654864, 0.2336283185840708, 0.21876106194690265, 0.20336283185840706,
         0.19061946902654867, 0.17893805309734512, 0.1598230088495575, 0.14495575221238935, 0.1300884955752212,
         0.10991150442477872, 0.09557522123893802, 0.08707964601769907, 0.08176991150442475])

    veryloose_N__15 = np.array(
        [1.0, 1.6125255861197791, 3.0197994674209268, 5.535264774201663, 8.987847362763038, 17.944192188830627,
         31.510576236386598, 70.50413342008292, 133.88250791784452, 252.39462514654772, 533.2310463344431,
         1000.0])

    veryloose_tau_ratio__15 = np.array(
        [0.29628318584070795, 0.27185840707964604, 0.2421238938053097, 0.215575221238938, 0.19539823008849555,
         0.1693805309734513, 0.14973451327433626, 0.12318584070796455, 0.107787610619469, 0.0982300884955752,
         0.08920353982300883, 0.08336283185840708])

    # tau_cy / sigma_ref N = 10 = 0.25

    loose_N__0_1 = np.array(
        [1.0, 2.388565214174338, 6.283740687110599, 17.10744192086035, 42.754050979204905, 132.3493595877289,
         661.7225663307353, 1000.0])

    loose_tau_ratio__0_1 = np.array(
        [0.1131933001310339, 0.11329876005494464, 0.11256686818300476, 0.11013707153610296, 0.10769672889681026,
         0.10358062806657696, 0.0961193384498975, 0.09446889064069552])

    loose_N__0_25 = np.array(
        [1.0, 2.0976189117049318, 5.023182207446078, 14.39917234199389, 26.220380333303627, 47.74550977732968,
         98.02253307544092, 202.9759953082148, 366.4617697069345, 1000.0])

    loose_tau_ratio__0_25 = np.array(
        [0.1872346035102337, 0.18306999111500155, 0.17551800596375874, 0.1637328594667421, 0.15529606555388886,
         0.14771033322699467, 0.1341819341877346, 0.11980352816175455, 0.11136567964966226, 0.09957526015644992])

    loose_N__0_5 = np.array(
        [1.0, 2.0426155684560388, 3.4736018253381693, 6.271850059699696, 13.440270311253832, 23.652614783678075,
         43.44636482575841, 74.51989137269719, 105.86669460254744, 172.48431952704098, 285.8688936375835,
         477.8349543791893,
         771.7832576786642, 1000.0])

    loose_tau_ratio__0_5 = np.array(
        [0.23659617549585946, 0.2247688450292784, 0.21291936797867608, 0.20107727312274748, 0.1841499007358468,
         0.1714535804962417, 0.1570604100809142, 0.14436092604359194, 0.13674461033876384, 0.12659198346388412,
         0.11729252737344165, 0.11054731064011547, 0.10635106026771024, 0.1021294995135662])

    loose_N__1 = np.array(
        [1.0, 1.6337164807865294, 2.7784444917506765, 4.300342397664526, 7.899670188784587, 13.784854943123868,
         20.795383965900275, 37.552421150773284, 77.7714107993403, 126.71211875364176, 210.01162893201882,
         322.25195023415785, 726.8709260554684, 1000.0])

    loose_tau_ratio__1 = np.array(
        [0.2757450084499765, 0.26303919681721954, 0.247785473422781, 0.23592439578054855, 0.21812697902138467,
         0.2011742962527453, 0.1876079316408774, 0.1698084056832354, 0.14862150696958287, 0.1376178185087441,
         0.12746730083234248, 0.11900941493470728, 0.10804580124495457, 0.10468268427144334])

    loose_N__5 = np.array(
        [1.0, 1.953726391608351, 3.7145207805712888, 8.31083788748431, 18.755814548907, 30.045966955780106,
         50.66886948889224, 78.42997437784426, 120.35549459887636, 192.76192884494938, 369.5380241877583,
         702.3919720679346, 1000.0])

    loose_tau_ratio__5 = np.array(
        [0.3293597791669195, 0.30391229952727594, 0.2793106084773961, 0.2470694005394276, 0.211425000856862,
         0.1910575257519953, 0.17069637824256326, 0.15457999267053546, 0.14271786042906387, 0.13256312435570586,
         0.12072841169445105, 0.10889264443395708, 0.10553269125816343])

    loose_N__15 = np.array(
        [1.0, 1.6448823519666074, 2.750702318093394, 5.230723187553098, 9.051747478794958, 18.27564906752226,
         37.85868957106964, 69.55602858008639, 133.37046052081553, 251.3497248804099, 457.72391077080727,
         900.2809010204502, 1000.0])

    loose_tau_ratio__15 = np.array(
        [0.37106496067663103, 0.34474216366852906, 0.3167204072862263, 0.28360810037675577, 0.2564416239773684,
         0.2233366992625716, 0.1902349383454921, 0.16562902889865586, 0.1444326387918512, 0.1308936937602,
         0.11990371508946973, 0.1097742893978504, 0.1080858760160407])

    # tau_cy / sigma_ref N = 10 = 0.6

    mediumdense_N__0_1 = np.array(
        [1.0, 2.2694580602795598, 5.5689542040874285, 10.756736621795383, 20.777219296532987, 41.17418640422073,
         111.00682840881485, 231.56894892129944, 728.1897734911723, 1000.0])

    mediumdense_tau_ratio__0_1 = np.array(
        [0.13829787234042534, 0.13829787234042534, 0.13829787234042534, 0.1340425531914895, 0.1297872340425532,
         0.12765957446808507, 0.1191489361702125, 0.1127659574468085, 0.10212765957446823, 0.0978723404255324])

    mediumdense_N__0_25 = np.array(
        [1.0, 1.7547693241019482, 3.886622508776946, 11.12547923838821, 23.011889803050284, 50.102923935199385,
         102.75045605662336, 272.3368331413256, 1000.0])

    mediumdense_tau_ratio__0_25 = np.array(
        [0.2297872340425533, 0.22340425531914887, 0.2085106382978723, 0.1914893617021276, 0.18085106382978733,
         0.17021276595744705, 0.15957446808510636, 0.14468085106383022, 0.1297872340425532])

    mediumdense_N__0_5 = np.array(
        [1.0, 2.2470710813070065, 4.851609330699583, 8.244321889977359, 14.49691332893356, 26.377870412720533,
         48.82447257874753, 94.3156903102061, 195.08923477802864, 393.3038968549156, 734.1747639832159,
         1000.0])

    mediumdense_tau_ratio__0_5 = np.array(
        [0.3297872340425529, 0.2978723404255321, 0.26808510638297894, 0.2489361702127661, 0.2297872340425533,
         0.2127659574468086, 0.1936170212765962, 0.17872340425531918, 0.16382978723404262, 0.15319148936170235,
         0.14468085106383022, 0.14255319148936207])

    mediumdense_N__1 = np.array(
        [1.0, 1.5018618197683704, 2.3635361536496644, 3.8819593161369164, 6.267555195254916, 10.293315821398958,
         16.760376033036945, 28.72642267030004, 57.428413259105255, 106.29789079022852, 179.08091070340294,
         306.90197553971433, 530.4532878150872, 734.0946414395609, 1000.0])

    mediumdense_tau_ratio__1 = np.array(
        [0.4659574468085106, 0.4276595744680849, 0.3872340425531915, 0.3489361702127658, 0.31489361702127683,
         0.2851063829787237, 0.2595744680851064, 0.2361702127659573, 0.2085106382978723, 0.18936170212765946,
         0.17872340425531918, 0.1680851063829789, 0.16170212765957448, 0.1574468085106382, 0.15319148936170235])

    mediumdense_N__2_5 = np.array(
        [1.0, 1.4374463248101872, 2.224176587012671, 3.3828415138852166, 5.324091657204813, 9.284951730118568,
         15.512940351711679, 27.281611691595803, 44.42203840553084, 97.5648177960141, 174.51139229566462,
         306.84615578668013, 534.9298328765615, 1000.0])

    mediumdense_tau_ratio__2_5 = np.array(
        [0.6000000000000001, 0.5553191489361704, 0.4957446808510642, 0.44680851063829774, 0.3978723404255322,
         0.34468085106382995, 0.30638297872340425, 0.2723404255319153, 0.246808510638298, 0.21702127659574488,
         0.20212765957446832, 0.18936170212765946, 0.17872340425531918, 0.1680851063829789])

    mediumdense_N__5 = np.array(
        [1.0, 1.3880351477865598, 2.1113116730947583, 3.1839604451378345, 5.320799975172985, 8.446357812325733,
         15.373889603188418, 25.466980716059545, 42.18473849881988, 73.55203471499435, 160.1712166288081,
         306.79592668945236, 495.24235447838464, 885.7774626435547, 1000.0])

    mediumdense_tau_ratio__5 = np.array(
        [0.6957446808510639, 0.6468085106382979, 0.5872340425531917, 0.5340425531914894, 0.4702127659574469,
         0.41702127659574506, 0.3595744680851065, 0.32340425531914896, 0.2914893617021277, 0.2638297872340427,
         0.23191489361702144, 0.2085106382978723, 0.1957446808510639, 0.18723404255319176, 0.18297872340425547])

    mediumdense_N__7_5 = np.array(
        [1.0, 1.4477325184299636, 2.074084360248001, 3.3495325347638394, 5.743224362670379, 9.19553487840328,
         14.472673600063965, 25.239624791302823, 39.38097794455178, 82.18020521200758, 175.93610014289803,
         288.9166650753737, 516.7863553963209, 792.4905670294444, 1000.0])

    mediumdense_tau_ratio__7_5 = np.array(
        [0.7765957446808507, 0.7212765957446807, 0.6680851063829789, 0.6042553191489359, 0.5340425531914894,
         0.4765957446808513, 0.4255319148936172, 0.3723404255319149, 0.3361702127659574, 0.28936170212765955,
         0.25106382978723385, 0.23191489361702144, 0.21489361702127674, 0.204255319148936, 0.20000000000000015])

    mediumdense_N__10 = np.array(
        [1.0, 1.3743929290877581, 1.9523241547312111, 3.1803138535956847, 5.3150927678654005, 9.511970322853324,
         16.590214905266407, 32.61472605059545, 58.355067525964785, 97.49385626630985, 172.9188431828693,
         296.37400540054335, 539.2969903474461, 819.9853401496034, 1000.0])

    mediumdense_tau_ratio__10 = np.array(
        [0.8531914893617021, 0.802127659574468, 0.7446808510638299, 0.6680851063829789, 0.5957446808510638,
         0.5191489361702128, 0.4531914893617022, 0.3872340425531915, 0.3361702127659574, 0.302127659574468,
         0.274468085106383, 0.25106382978723385, 0.2276595744680852, 0.21489361702127674, 0.2085106382978723])

    mediumdense_N__15 = np.array(
        [1.0, 1.4828330896058797, 2.2175514750666228, 3.3162499076425744, 5.356243754146089, 8.217522256036148,
         14.960905099891455, 32.04489850247886, 63.54030487370965, 108.92660788558041, 194.8657989766849,
         339.7744765302462, 739.9127909397106, 1000.0])

    mediumdense_tau_ratio__15 = np.array(
        [0.997872340425532, 0.9191489361702128, 0.84468085106383, 0.7723404255319148, 0.6936170212765957,
         0.6297872340425532, 0.5446808510638297, 0.4489361702127659, 0.37872340425531936, 0.33191489361702153,
         0.2978723404255321, 0.2659574468085109, 0.2340425531914896, 0.22553191489361699])

    # tau_cy / sigma_ref N = 10 = 1.0

    dense_N__0_1 = np.array(
        [1.0, 1.5318411662295215, 2.738911850994098, 4.726061159783651, 8.058588768059021, 14.154860655978684,
         27.01397477491591, 46.890501720790475, 79.01292246271748, 149.90278908589355, 277.73363584493217,
         508.5026581587102, 856.8532992123108, 1000.0])

    dense_tau_ratio__0_1 = np.array(
        [0.15952732644017684, 0.15066469719350015, 0.14771048744460913, 0.13884785819793244, 0.13884785819793244,
         0.12998522895125575, 0.1329394387001468, 0.1240768094534701, 0.12112259970457905, 0.12112259970457905,
         0.11521418020679343, 0.11225997045790237, 0.10930576070900956, 0.10635155096011673])

    dense_N__0_25 = np.array(
        [1.0, 1.4954916115123271, 2.4903169233871267, 4.559722222685198, 9.179682770621564, 17.519817717704772,
         29.875023699319105, 51.2453464451344, 75.34385180744849, 126.20994130592287, 238.03326455121072,
         448.93714944741515, 783.8865952283917, 1000.0])

    dense_tau_ratio__0_25 = np.array(
        [0.2688330871491882, 0.2511078286558348, 0.2422451994091581, 0.22451994091580474, 0.2038404726735603,
         0.1920236336779908, 0.17725258493353024, 0.16838995568685355, 0.15952732644017684, 0.15066469719350015,
         0.14475627769571453, 0.13589364844903962, 0.12998522895125575, 0.12703101920236293])

    dense_N__0_5 = np.array(
        [1.0, 1.5309157141079104, 2.5494843424111147, 4.296351726123814, 7.413852352632262, 13.257031509982962,
         22.207498518982696, 35.68872622563735, 64.19270675468924, 105.63550583128021, 172.80457368951636,
         345.82628664720613, 533.144137131611, 817.0713272457774, 1000.0])

    dense_tau_ratio__0_5 = np.array(
        [0.3840472673559816, 0.3545051698670605, 0.3220088626292465, 0.2924667651403237, 0.26587887740029537,
         0.23338257016248146, 0.2186115214180191, 0.2008862629246675, 0.18611521418020693, 0.17429837518463742,
         0.16543574593796073, 0.15361890694239122, 0.15066469719350015, 0.14475627769571453, 0.14180206794682348])

    dense_N__1 = np.array(
        [1.0, 1.9286229739813456, 3.573929487838579, 6.3157473263440576, 11.772482906981196, 24.27081782700117,
         44.44056666840199, 92.70930436603636, 146.35979719322594, 246.63031402693963, 373.52130455805985,
         652.1977065525173, 1000.0])

    dense_tau_ratio__1 = np.array(
        [0.5435745937961602, 0.4579025110782862, 0.3899556868537655, 0.3367799113737071, 0.2924667651403237,
         0.2511078286558348, 0.22451994091580474, 0.2008862629246675, 0.189069423929098, 0.17725258493353024,
         0.16838995568685355, 0.16543574593796073, 0.15657311669128582])

    dense_N__2_5 = np.array(
        [1.0, 1.4495439432517472, 2.0816879615674897, 3.1346793771186245, 5.28348505831838, 9.011125464140898,
         17.51015336723321, 39.93305755873251, 83.80803133377964, 187.73691609106876, 351.99595993515646,
         558.9992983516075, 1000.0])

    dense_tau_ratio__2_5 = np.array(
        [0.862629246676514, 0.7769571639586417, 0.6971935007385524, 0.6233382570162478, 0.5317577548005907,
         0.4549483013293934, 0.37813884785819774, 0.2983751846381093, 0.248153618906942, 0.2097488921713424,
         0.189069423929098, 0.17725258493353024, 0.16543574593796073])

    dense_N__5 = np.array(
        [1.0, 1.8039524165073502, 2.8152805783010555, 5.006074139509686, 8.438228081272191, 13.644111454508055,
         25.4362592645546, 43.124241931124715, 83.79408611106098, 164.75418675944553, 325.8512889631112,
         675.7590552732977, 1000.0])

    dense_tau_ratio__5 = np.array(
        [1.1964549483013291, 0.9985228951255536, 0.8714918759231907, 0.7237813884785815, 0.6115214180206792,
         0.5258493353028069, 0.4313146233382561, 0.3663220088626282, 0.3042836041358932, 0.2570162481536187,
         0.2215657311669137, 0.1949778434268836, 0.1920236336779908])

    dense_N__7_5 = np.array(
        [1.0, 1.48225601791807, 2.2192073460403177, 3.2062624913155364, 4.604831563628727, 7.670794347362198,
         11.689568487911274, 19.7035670077324, 34.824467272433694, 58.00144342791514, 101.29195689151608,
         189.929922238378, 345.71726458650164, 589.5422818581796, 1000.0])

    dense_tau_ratio__7_5 = np.array(
        [1.38847858197932, 1.2496307237813884, 1.1196454948301326, 1.0073855243722305, 0.9039881831610036,
         0.774002954209748, 0.676514032496307, 0.5701624815361894, 0.4697193500738557, 0.3958641063515511,
         0.3367799113737071, 0.2924667651403237, 0.2599704579025115, 0.23338257016248146, 0.21270310192023698])

    dense_N__10 = np.array(
        [1.0, 1.4129798949862502, 1.889929333916628, 2.6824105019814386, 3.992175064961803, 6.973828449005313,
         10.317100648315002, 15.445091506108891, 24.24574724777027, 51.20048362074438, 94.88372489923368,
         187.6728006748644, 337.58753740789007, 562.1907168565426, 1000.0])

    dense_tau_ratio__10 = np.array(
        [1.5184638109305757, 1.3943870014771047, 1.29394387001477, 1.1787296898079758, 1.0605612998522895,
         0.9039881831610036, 0.8035450516986701, 0.7060561299852282, 0.5997045790251105, 0.4638109305760701,
         0.38109305760709056, 0.3249630723781394, 0.2865583456425416, 0.2570162481536187, 0.23338257016248146])

    dense_N__15 = np.array(
        [1.0, 1.7690610431837857, 2.8954012024140523, 4.627472250473347, 7.4396105115945375, 11.271228891816651,
         19.228434774566832, 31.467367831486932, 52.72914184801593, 97.1465745936483, 181.0896526764028,
         320.0025096620405,
         520.4246119699965, 846.3665016664467, 1000.0])

    dense_tau_ratio__15 = np.array(
        [1.7961595273264406, 1.5864106351550964, 1.4062038404726724, 1.2496307237813884, 1.0960118168389943,
         0.9689807976366316, 0.8035450516986701, 0.6617429837518456, 0.5406203840472656, 0.4313146233382561,
         0.36927621861152105, 0.33087149187592324, 0.2954209748892165, 0.2629246676514025, 0.2599704579025115])

    # Very dense

    verydense_N__0_1 = np.array(
        [1.0, 1.9187520990478333, 3.6382449755244655, 7.19076838234002, 13.961753486779294, 28.762962774867876,
         54.538939578901555, 116.42271847199005, 209.29300664290955, 418.58520738574873, 789.0127987039548, 1000.0])

    verydense_tau_ratio__0_1 = np.array(
        [0.19145802650957397, 0.17967599410898316, 0.1649484536082486, 0.15316642120765955, 0.15316642120765955,
         0.14432989690721776, 0.14432989690721776, 0.13549337260677596, 0.13254786450662692, 0.12076583210603785,
         0.11192930780559608, 0.1089837997054488])

    verydense_N__0_25 = np.array(
        [1.0, 1.7765244276932377, 2.8033956239563613, 4.450107450706342, 8.191794966530754, 13.879284141942199,
         29.627711919708695, 52.323453136326826, 91.3164217144668, 160.31504121414685, 252.98075162281282,
         387.55754136286913, 637.4699119514506, 1000.0])

    verydense_tau_ratio__0_25 = np.array(
        [0.3181148748159064, 0.29160530191457923, 0.2709867452135484, 0.2503681885125193, 0.23269513991163396,
         0.2179675994108976, 0.20618556701030852, 0.19734904270986675, 0.1826215022091322, 0.1649484536082486,
         0.15611192930780504, 0.14432989690721776, 0.13549337260677596, 0.12076583210603785])

    verydense_N__0_5 = np.array(
        [1.0, 1.829935058289986, 3.009949274584684, 5.28426468041387, 7.766457760052918, 13.961753486779294,
         25.85362473999301, 45.12050252978594, 78.74562151399782, 126.49061934271386, 198.42602289337088,
         298.62670677675686, 485.40734819497084, 1000.0])

    verydense_tau_ratio__0_5 = np.array(
        [0.4683357879234169, 0.4123711340206189, 0.3740795287187044, 0.33578792341679, 0.3063328424153173,
         0.2768777614138447, 0.2503681885125193, 0.2268041237113394, 0.21207658321060485, 0.19734904270986675,
         0.1826215022091322, 0.1678939617083941, 0.14727540500736325, 0.14432989690721776])

    verydense_N__1 = np.array(
        [1.0, 1.6158639902500942, 2.5650209056800453, 4.3717145900084375, 7.766457760052918, 13.634787396553907,
         32.57350512147631, 62.871853054020576, 118.51039134343159, 211.78759301164973, 346.29890230000063,
         629.9613335204734, 1000.0])

    verydense_tau_ratio__1 = np.array(
        [0.6686303387334309, 0.592047128129602, 0.5213549337260677, 0.4536082474226806, 0.3917525773195879,
         0.3446244477172318, 0.2857142857142865, 0.2533136966126648, 0.22385861561119214, 0.2002945508100158,
         0.1855670103092777, 0.1649484536082486, 0.1590574374079523])

    verydense_N__2_5 = np.array(
        [1.0, 1.504977534812715, 2.346915358771387, 3.860298839373432, 7.106070348181026, 12.111289755953514,
         20.76462944173765, 40.07891463057112, 76.9015015749672, 139.06719655383736, 232.8449886486524,
         408.7824867737063, 730.5271542664458, 1000.0])

    verydense_tau_ratio__2_5 = np.array(
        [1.0515463917525771, 0.9425625920471284, 0.8335787923416795, 0.724594992636229, 0.6067746686303384,
         0.5154639175257731, 0.4329896907216497, 0.3505154639175263, 0.2857142857142865, 0.23858615611193024,
         0.21207658321060485, 0.18851251840942496, 0.17083946980854134, 0.1678939617083941])

    verydense_N__5 = np.array(
        [1.0, 1.3447604875886854, 1.9881825636611845, 2.9921700934164344, 5.069606734204699, 9.006280202112789,
         16.876124757881488, 31.810676038110035, 63.621229586575204, 115.0514080832087, 223.38633544498575,
         438.90150181193053, 789.0127987039548, 1000.0])

    verydense_tau_ratio__5 = np.array(
        [1.4050073637702507, 1.2960235640648, 1.157584683357879, 1.0250368188512518, 0.8689248895434467,
         0.7216494845360817,
         0.5832106038291602, 0.4653902798232696, 0.35640648011782083, 0.29160530191457923, 0.23858615611193024,
         0.20618556701030852, 0.17967599410898316, 0.17967599410898316])

    verydense_N__7_5 = np.array(
        [1.0, 2.5498698292923123, 3.2126326106632246, 4.3717145900084375, 6.349571608355332, 10.444025524245689,
         15.999876781703348, 25.700912439493624, 56.5124430431384, 108.43336871134291, 228.74320347570304,
         460.2038602746067, 1000.0])

    verydense_tau_ratio__7_5 = np.array(
        [2.5449189985272462, 1.7790868924889551, 1.608247422680412, 1.4197349042709868, 1.2135493372606767,
         0.98379970544919, 0.8247422680412377, 0.6656848306332837, 0.4712812960235642, 0.3652430044182626,
         0.29160530191457923, 0.23858615611193024, 0.2002945508100158])

    verydense_N__10 = np.array(
        [1.0, 9.16777932442606, 12.32846739442066, 17.38349997394775, 25.39818863530829, 34.357423105917356,
         49.02239343792874, 77.35844275341829, 139.89352019350324, 268.42098551447515, 497.04755483626417,
         862.3380114501625, 1000.0])

    verydense_tau_ratio__10 = np.array(
        [3.8880706921944035, 1.8939617083946985, 1.6229749631811483, 1.316642120765831, 1.0338733431516935,
         0.8512518409425631, 0.6892488954344635, 0.5390279823269513, 0.4212076583210607, 0.3387334315169355,
         0.2768777614138447, 0.2415316642120775, 0.2356406480117812])

    verydense_N__15 = np.array(
        [1.0, 28.022886286880574, 36.45436781521715, 46.42199946066167, 61.2541483493394, 81.40191996271128,
         108.17671540913874, 146.85724231997148, 236.45931777976287, 451.5615770668075, 732.2603572554892,
         1000.0])

    verydense_tau_ratio__15 = np.array(
        [8.415706863033874, 1.8727540500736384, 1.4804123711340225, 1.2047128129602385, 0.9608247422680432,
         0.7664212076583237, 0.6356406480117851, 0.5508100147275421, 0.44123711340206384, 0.3316642120765856,
         0.2786450662739348, 0.26450662739322617])

    if failure_stress_ratio == 0.19:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 5, 15])
        _shearstress_ratios_interpolation = np.array([
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__0_1), veryloose_tau_ratio__0_1),
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__0_25), veryloose_tau_ratio__0_25),
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__0_5), veryloose_tau_ratio__0_5),
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__1), veryloose_tau_ratio__1),
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__5), veryloose_tau_ratio__5),
            np.interp(np.log10(cycle_no), np.log10(veryloose_N__15), veryloose_tau_ratio__15),
        ])
        _max_shearstress_ratio = 0.3
    elif failure_stress_ratio == 0.25:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 5, 15])
        _shearstress_ratios_interpolation = np.array([
            np.interp(np.log10(cycle_no), np.log10(loose_N__0_1), loose_tau_ratio__0_1),
            np.interp(np.log10(cycle_no), np.log10(loose_N__0_25), loose_tau_ratio__0_25),
            np.interp(np.log10(cycle_no), np.log10(loose_N__0_5), loose_tau_ratio__0_5),
            np.interp(np.log10(cycle_no), np.log10(loose_N__1), loose_tau_ratio__1),
            np.interp(np.log10(cycle_no), np.log10(loose_N__5), loose_tau_ratio__5),
            np.interp(np.log10(cycle_no), np.log10(loose_N__15), loose_tau_ratio__15),
        ])
        _max_shearstress_ratio = 0.4
    elif failure_stress_ratio == 0.6:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _shearstress_ratios_interpolation = np.array([
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__0_1), mediumdense_tau_ratio__0_1),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__0_25), mediumdense_tau_ratio__0_25),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__0_5), mediumdense_tau_ratio__0_5),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__1), mediumdense_tau_ratio__1),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__2_5), mediumdense_tau_ratio__2_5),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__5), mediumdense_tau_ratio__5),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__7_5), mediumdense_tau_ratio__7_5),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__10), mediumdense_tau_ratio__10),
            np.interp(np.log10(cycle_no), np.log10(mediumdense_N__15), mediumdense_tau_ratio__15)]
        )
        _max_shearstress_ratio = 1
    elif failure_stress_ratio == 1:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _shearstress_ratios_interpolation = np.array([
            np.interp(np.log10(cycle_no), np.log10(dense_N__0_1), dense_tau_ratio__0_1),
            np.interp(np.log10(cycle_no), np.log10(dense_N__0_25), dense_tau_ratio__0_25),
            np.interp(np.log10(cycle_no), np.log10(dense_N__0_5), dense_tau_ratio__0_5),
            np.interp(np.log10(cycle_no), np.log10(dense_N__1), dense_tau_ratio__1),
            np.interp(np.log10(cycle_no), np.log10(dense_N__2_5), dense_tau_ratio__2_5),
            np.interp(np.log10(cycle_no), np.log10(dense_N__5), dense_tau_ratio__5),
            np.interp(np.log10(cycle_no), np.log10(dense_N__7_5), dense_tau_ratio__7_5),
            np.interp(np.log10(cycle_no), np.log10(dense_N__10), dense_tau_ratio__10),
            np.interp(np.log10(cycle_no), np.log10(dense_N__15), dense_tau_ratio__15)]
        )
        _max_shearstress_ratio = 2
    elif failure_stress_ratio == 1.8:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _shearstress_ratios_interpolation = np.array([
            np.interp(np.log10(cycle_no), np.log10(verydense_N__0_1), verydense_tau_ratio__0_1),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__0_25), verydense_tau_ratio__0_25),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__0_5), verydense_tau_ratio__0_5),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__1), verydense_tau_ratio__1),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__2_5), verydense_tau_ratio__2_5),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__5), verydense_tau_ratio__5),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__7_5), verydense_tau_ratio__7_5),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__10), verydense_tau_ratio__10),
            np.interp(np.log10(cycle_no), np.log10(verydense_N__15), verydense_tau_ratio__15)]
        )
        _max_shearstress_ratio = 10
    else:
        raise ValueError("Stress ratio for failure at 10 cycles should be selected from [0.19, 0.25, 0.6, 1.0, 1.8]")

    fit_coeff = np.polyfit(
        x=_shearstress_ratios_interpolation,
        y=np.log(_shear_strains_interpolation),
        deg=1)

    _strain_ratio_0 = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * 0)
    _strain_ratio_max = np.exp(fit_coeff[1]) * np.exp(fit_coeff[0] * _max_shearstress_ratio)

    _shearstress_ratios_interpolation = np.append(
        np.append(1e-7, _shearstress_ratios_interpolation), _max_shearstress_ratio)
    _shear_strains_interpolation = np.append(
        np.append(_strain_ratio_0, _shear_strains_interpolation), _strain_ratio_max)

    _cyclic_strain = 10 ** (np.interp(
        shearstress_ratio,
        _shearstress_ratios_interpolation,
        np.log10(_shear_strains_interpolation)))

    if _cyclic_strain < 0.5:
        warnings.warn(
            "Cyclic strain is below the lowest contour, value is extrapolated and should be treated with caution")

    if _cyclic_strain > 15:
        warnings.warn(
            "Cyclic strain is above the cyclic failure contour, " +
            "value is extrapolated and should be treated with caution")

    return {
        'cyclic strain [%]': _cyclic_strain,
        'shear strains interpolation [%]': _shear_strains_interpolation,
        'shearstress ratios interpolation [-]': _shearstress_ratios_interpolation,
    }

def plotstrainaccumulation_dsssand_andersen(failure_stress_ratio):
    """
    Returns a Plotly figure with the cyclic strain accumulation contours for cyclic DSS tests on normally consolidated sand or silt with symmetrical loading
    
    :param failure_stress_ratio: Ratio of cyclic shear stress to vertical effective stress for failure at N=10 (:math:`( \\tau_{cy} / \\sigma_{ref}^{\\prime})_{N=10}`) [:math:`-`] - Allowable value: [0.19, 0.25, 0.6, 1.0, 1.8] 
    
    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=False)


    veryloose_N__0_1 = np.array(
        [1.0, 1.9905399433448885, 4.8829023626795545, 9.882426826518873, 20.144333637328288, 39.34612116439103,
         59.896031376285755, 86.12949526940572, 137.817060720145, 217.4099645912803, 465.8729490556514,
         756.1433585416446,
         1000.0])

    veryloose_tau_ratio__0_1 = np.array(
        [0.09929203539823006, 0.0998230088495575, 0.0982300884955752, 0.0982300884955752, 0.09716814159292034,
         0.09451327433628318, 0.09238938053097344, 0.0902654867256637, 0.08761061946902651, 0.0838938053097345,
         0.07805309734513274, 0.07539823008849555, 0.07433628318584065])

    veryloose_N__0_25 = np.array(
        [1.0, 1.7114769039291653, 2.8991079838241167, 4.911109720023127, 9.457967746751489, 17.2068400704796,
         30.643790671362506, 54.57306902524658, 95.8115252456874, 168.20569972172868, 308.17963839557683,
         576.8123855207602, 1000.0])

    veryloose_tau_ratio__0_25 = np.array(
        [0.16407079646017694, 0.1630088495575221, 0.16035398230088493, 0.15557522123893802, 0.14761061946902654,
         0.13699115044247787, 0.12477876106194688, 0.11309734513274332, 0.10247787610619466, 0.0934513274336283,
         0.0860176991150442, 0.079646017699115, 0.07592920353982299])

    veryloose_N__0_5 = np.array(
        [1.0, 1.9439266109192, 3.935372612958882, 6.388745150051343, 10.371719037920348, 15.131161770385535,
         22.07409246457936, 37.93915680374376, 53.032433545756085, 87.9527883291678, 142.77492733874138,
         223.64695836692204, 323.9194332732463, 610.6050943098223, 1000.0])

    veryloose_tau_ratio__0_5 = np.array(
        [0.2065486725663717, 0.1980530973451327, 0.1863716814159292, 0.17469026548672564, 0.16247787610619469,
         0.15238938053097342, 0.14336283185840706, 0.12849557522123892, 0.11946902654867254, 0.107787610619469,
         0.09876106194690262, 0.09238938053097344, 0.08761061946902651, 0.08123893805309731, 0.07805309734513274])

    veryloose_N__1 = np.array(
        [1.0, 1.9850529742633432, 3.5352403044924263, 6.075854079454643, 10.97713502928205, 19.412985374161373,
         34.82464429594147, 53.78774432277998, 89.84524950011499, 129.21388581984036, 173.0442219237976,
         261.57652095673933, 392.59264453190076, 655.6496542296684, 1000.0])

    veryloose_tau_ratio__1 = np.array(
        [0.2283185840707964, 0.21610619469026546, 0.20336283185840706, 0.19008849557522126, 0.1725663716814159,
         0.15504424778761058, 0.13699115044247787, 0.123716814159292, 0.1109734513274336, 0.1030088495575221,
         0.09876106194690262, 0.09292035398230086, 0.08761061946902651, 0.08283185840707963, 0.079646017699115])

    veryloose_N__5 = np.array(
        [1.0, 1.720127440491086, 2.7730154250757146, 4.162883104203345, 6.384398263586898, 8.92508108548957,
         12.745937149943687, 21.908890327307915, 33.124939621722035, 52.642837952063275, 114.46183555359389,
         278.87298224832324, 560.5144430749845, 1000.0])

    veryloose_tau_ratio__5 = np.array(
        [0.2707964601769911, 0.25061946902654864, 0.2336283185840708, 0.21876106194690265, 0.20336283185840706,
         0.19061946902654867, 0.17893805309734512, 0.1598230088495575, 0.14495575221238935, 0.1300884955752212,
         0.10991150442477872, 0.09557522123893802, 0.08707964601769907, 0.08176991150442475])

    veryloose_N__15 = np.array(
        [1.0, 1.6125255861197791, 3.0197994674209268, 5.535264774201663, 8.987847362763038, 17.944192188830627,
         31.510576236386598, 70.50413342008292, 133.88250791784452, 252.39462514654772, 533.2310463344431,
         1000.0])

    veryloose_tau_ratio__15 = np.array(
        [0.29628318584070795, 0.27185840707964604, 0.2421238938053097, 0.215575221238938, 0.19539823008849555,
         0.1693805309734513, 0.14973451327433626, 0.12318584070796455, 0.107787610619469, 0.0982300884955752,
         0.08920353982300883, 0.08336283185840708])

    # tau_cy / sigma_ref N = 10 = 0.25

    loose_N__0_1 = np.array(
        [1.0, 2.388565214174338, 6.283740687110599, 17.10744192086035, 42.754050979204905, 132.3493595877289,
         661.7225663307353, 1000.0])

    loose_tau_ratio__0_1 = np.array(
        [0.1131933001310339, 0.11329876005494464, 0.11256686818300476, 0.11013707153610296, 0.10769672889681026,
         0.10358062806657696, 0.0961193384498975, 0.09446889064069552])

    loose_N__0_25 = np.array(
        [1.0, 2.0976189117049318, 5.023182207446078, 14.39917234199389, 26.220380333303627, 47.74550977732968,
         98.02253307544092, 202.9759953082148, 366.4617697069345, 1000.0])

    loose_tau_ratio__0_25 = np.array(
        [0.1872346035102337, 0.18306999111500155, 0.17551800596375874, 0.1637328594667421, 0.15529606555388886,
         0.14771033322699467, 0.1341819341877346, 0.11980352816175455, 0.11136567964966226, 0.09957526015644992])

    loose_N__0_5 = np.array(
        [1.0, 2.0426155684560388, 3.4736018253381693, 6.271850059699696, 13.440270311253832, 23.652614783678075,
         43.44636482575841, 74.51989137269719, 105.86669460254744, 172.48431952704098, 285.8688936375835,
         477.8349543791893,
         771.7832576786642, 1000.0])

    loose_tau_ratio__0_5 = np.array(
        [0.23659617549585946, 0.2247688450292784, 0.21291936797867608, 0.20107727312274748, 0.1841499007358468,
         0.1714535804962417, 0.1570604100809142, 0.14436092604359194, 0.13674461033876384, 0.12659198346388412,
         0.11729252737344165, 0.11054731064011547, 0.10635106026771024, 0.1021294995135662])

    loose_N__1 = np.array(
        [1.0, 1.6337164807865294, 2.7784444917506765, 4.300342397664526, 7.899670188784587, 13.784854943123868,
         20.795383965900275, 37.552421150773284, 77.7714107993403, 126.71211875364176, 210.01162893201882,
         322.25195023415785, 726.8709260554684, 1000.0])

    loose_tau_ratio__1 = np.array(
        [0.2757450084499765, 0.26303919681721954, 0.247785473422781, 0.23592439578054855, 0.21812697902138467,
         0.2011742962527453, 0.1876079316408774, 0.1698084056832354, 0.14862150696958287, 0.1376178185087441,
         0.12746730083234248, 0.11900941493470728, 0.10804580124495457, 0.10468268427144334])

    loose_N__5 = np.array(
        [1.0, 1.953726391608351, 3.7145207805712888, 8.31083788748431, 18.755814548907, 30.045966955780106,
         50.66886948889224, 78.42997437784426, 120.35549459887636, 192.76192884494938, 369.5380241877583,
         702.3919720679346, 1000.0])

    loose_tau_ratio__5 = np.array(
        [0.3293597791669195, 0.30391229952727594, 0.2793106084773961, 0.2470694005394276, 0.211425000856862,
         0.1910575257519953, 0.17069637824256326, 0.15457999267053546, 0.14271786042906387, 0.13256312435570586,
         0.12072841169445105, 0.10889264443395708, 0.10553269125816343])

    loose_N__15 = np.array(
        [1.0, 1.6448823519666074, 2.750702318093394, 5.230723187553098, 9.051747478794958, 18.27564906752226,
         37.85868957106964, 69.55602858008639, 133.37046052081553, 251.3497248804099, 457.72391077080727,
         900.2809010204502, 1000.0])

    loose_tau_ratio__15 = np.array(
        [0.37106496067663103, 0.34474216366852906, 0.3167204072862263, 0.28360810037675577, 0.2564416239773684,
         0.2233366992625716, 0.1902349383454921, 0.16562902889865586, 0.1444326387918512, 0.1308936937602,
         0.11990371508946973, 0.1097742893978504, 0.1080858760160407])

    # tau_cy / sigma_ref N = 10 = 0.6

    mediumdense_N__0_1 = np.array(
        [1.0, 2.2694580602795598, 5.5689542040874285, 10.756736621795383, 20.777219296532987, 41.17418640422073,
         111.00682840881485, 231.56894892129944, 728.1897734911723, 1000.0])

    mediumdense_tau_ratio__0_1 = np.array(
        [0.13829787234042534, 0.13829787234042534, 0.13829787234042534, 0.1340425531914895, 0.1297872340425532,
         0.12765957446808507, 0.1191489361702125, 0.1127659574468085, 0.10212765957446823, 0.0978723404255324])

    mediumdense_N__0_25 = np.array(
        [1.0, 1.7547693241019482, 3.886622508776946, 11.12547923838821, 23.011889803050284, 50.102923935199385,
         102.75045605662336, 272.3368331413256, 1000.0])

    mediumdense_tau_ratio__0_25 = np.array(
        [0.2297872340425533, 0.22340425531914887, 0.2085106382978723, 0.1914893617021276, 0.18085106382978733,
         0.17021276595744705, 0.15957446808510636, 0.14468085106383022, 0.1297872340425532])

    mediumdense_N__0_5 = np.array(
        [1.0, 2.2470710813070065, 4.851609330699583, 8.244321889977359, 14.49691332893356, 26.377870412720533,
         48.82447257874753, 94.3156903102061, 195.08923477802864, 393.3038968549156, 734.1747639832159,
         1000.0])

    mediumdense_tau_ratio__0_5 = np.array(
        [0.3297872340425529, 0.2978723404255321, 0.26808510638297894, 0.2489361702127661, 0.2297872340425533,
         0.2127659574468086, 0.1936170212765962, 0.17872340425531918, 0.16382978723404262, 0.15319148936170235,
         0.14468085106383022, 0.14255319148936207])

    mediumdense_N__1 = np.array(
        [1.0, 1.5018618197683704, 2.3635361536496644, 3.8819593161369164, 6.267555195254916, 10.293315821398958,
         16.760376033036945, 28.72642267030004, 57.428413259105255, 106.29789079022852, 179.08091070340294,
         306.90197553971433, 530.4532878150872, 734.0946414395609, 1000.0])

    mediumdense_tau_ratio__1 = np.array(
        [0.4659574468085106, 0.4276595744680849, 0.3872340425531915, 0.3489361702127658, 0.31489361702127683,
         0.2851063829787237, 0.2595744680851064, 0.2361702127659573, 0.2085106382978723, 0.18936170212765946,
         0.17872340425531918, 0.1680851063829789, 0.16170212765957448, 0.1574468085106382, 0.15319148936170235])

    mediumdense_N__2_5 = np.array(
        [1.0, 1.4374463248101872, 2.224176587012671, 3.3828415138852166, 5.324091657204813, 9.284951730118568,
         15.512940351711679, 27.281611691595803, 44.42203840553084, 97.5648177960141, 174.51139229566462,
         306.84615578668013, 534.9298328765615, 1000.0])

    mediumdense_tau_ratio__2_5 = np.array(
        [0.6000000000000001, 0.5553191489361704, 0.4957446808510642, 0.44680851063829774, 0.3978723404255322,
         0.34468085106382995, 0.30638297872340425, 0.2723404255319153, 0.246808510638298, 0.21702127659574488,
         0.20212765957446832, 0.18936170212765946, 0.17872340425531918, 0.1680851063829789])

    mediumdense_N__5 = np.array(
        [1.0, 1.3880351477865598, 2.1113116730947583, 3.1839604451378345, 5.320799975172985, 8.446357812325733,
         15.373889603188418, 25.466980716059545, 42.18473849881988, 73.55203471499435, 160.1712166288081,
         306.79592668945236, 495.24235447838464, 885.7774626435547, 1000.0])

    mediumdense_tau_ratio__5 = np.array(
        [0.6957446808510639, 0.6468085106382979, 0.5872340425531917, 0.5340425531914894, 0.4702127659574469,
         0.41702127659574506, 0.3595744680851065, 0.32340425531914896, 0.2914893617021277, 0.2638297872340427,
         0.23191489361702144, 0.2085106382978723, 0.1957446808510639, 0.18723404255319176, 0.18297872340425547])

    mediumdense_N__7_5 = np.array(
        [1.0, 1.4477325184299636, 2.074084360248001, 3.3495325347638394, 5.743224362670379, 9.19553487840328,
         14.472673600063965, 25.239624791302823, 39.38097794455178, 82.18020521200758, 175.93610014289803,
         288.9166650753737, 516.7863553963209, 792.4905670294444, 1000.0])

    mediumdense_tau_ratio__7_5 = np.array(
        [0.7765957446808507, 0.7212765957446807, 0.6680851063829789, 0.6042553191489359, 0.5340425531914894,
         0.4765957446808513, 0.4255319148936172, 0.3723404255319149, 0.3361702127659574, 0.28936170212765955,
         0.25106382978723385, 0.23191489361702144, 0.21489361702127674, 0.204255319148936, 0.20000000000000015])

    mediumdense_N__10 = np.array(
        [1.0, 1.3743929290877581, 1.9523241547312111, 3.1803138535956847, 5.3150927678654005, 9.511970322853324,
         16.590214905266407, 32.61472605059545, 58.355067525964785, 97.49385626630985, 172.9188431828693,
         296.37400540054335, 539.2969903474461, 819.9853401496034, 1000.0])

    mediumdense_tau_ratio__10 = np.array(
        [0.8531914893617021, 0.802127659574468, 0.7446808510638299, 0.6680851063829789, 0.5957446808510638,
         0.5191489361702128, 0.4531914893617022, 0.3872340425531915, 0.3361702127659574, 0.302127659574468,
         0.274468085106383, 0.25106382978723385, 0.2276595744680852, 0.21489361702127674, 0.2085106382978723])

    mediumdense_N__15 = np.array(
        [1.0, 1.4828330896058797, 2.2175514750666228, 3.3162499076425744, 5.356243754146089, 8.217522256036148,
         14.960905099891455, 32.04489850247886, 63.54030487370965, 108.92660788558041, 194.8657989766849,
         339.7744765302462, 739.9127909397106, 1000.0])

    mediumdense_tau_ratio__15 = np.array(
        [0.997872340425532, 0.9191489361702128, 0.84468085106383, 0.7723404255319148, 0.6936170212765957,
         0.6297872340425532, 0.5446808510638297, 0.4489361702127659, 0.37872340425531936, 0.33191489361702153,
         0.2978723404255321, 0.2659574468085109, 0.2340425531914896, 0.22553191489361699])

    # tau_cy / sigma_ref N = 10 = 1.0

    dense_N__0_1 = np.array(
        [1.0, 1.5318411662295215, 2.738911850994098, 4.726061159783651, 8.058588768059021, 14.154860655978684,
         27.01397477491591, 46.890501720790475, 79.01292246271748, 149.90278908589355, 277.73363584493217,
         508.5026581587102, 856.8532992123108, 1000.0])

    dense_tau_ratio__0_1 = np.array(
        [0.15952732644017684, 0.15066469719350015, 0.14771048744460913, 0.13884785819793244, 0.13884785819793244,
         0.12998522895125575, 0.1329394387001468, 0.1240768094534701, 0.12112259970457905, 0.12112259970457905,
         0.11521418020679343, 0.11225997045790237, 0.10930576070900956, 0.10635155096011673])

    dense_N__0_25 = np.array(
        [1.0, 1.4954916115123271, 2.4903169233871267, 4.559722222685198, 9.179682770621564, 17.519817717704772,
         29.875023699319105, 51.2453464451344, 75.34385180744849, 126.20994130592287, 238.03326455121072,
         448.93714944741515, 783.8865952283917, 1000.0])

    dense_tau_ratio__0_25 = np.array(
        [0.2688330871491882, 0.2511078286558348, 0.2422451994091581, 0.22451994091580474, 0.2038404726735603,
         0.1920236336779908, 0.17725258493353024, 0.16838995568685355, 0.15952732644017684, 0.15066469719350015,
         0.14475627769571453, 0.13589364844903962, 0.12998522895125575, 0.12703101920236293])

    dense_N__0_5 = np.array(
        [1.0, 1.5309157141079104, 2.5494843424111147, 4.296351726123814, 7.413852352632262, 13.257031509982962,
         22.207498518982696, 35.68872622563735, 64.19270675468924, 105.63550583128021, 172.80457368951636,
         345.82628664720613, 533.144137131611, 817.0713272457774, 1000.0])

    dense_tau_ratio__0_5 = np.array(
        [0.3840472673559816, 0.3545051698670605, 0.3220088626292465, 0.2924667651403237, 0.26587887740029537,
         0.23338257016248146, 0.2186115214180191, 0.2008862629246675, 0.18611521418020693, 0.17429837518463742,
         0.16543574593796073, 0.15361890694239122, 0.15066469719350015, 0.14475627769571453, 0.14180206794682348])

    dense_N__1 = np.array(
        [1.0, 1.9286229739813456, 3.573929487838579, 6.3157473263440576, 11.772482906981196, 24.27081782700117,
         44.44056666840199, 92.70930436603636, 146.35979719322594, 246.63031402693963, 373.52130455805985,
         652.1977065525173, 1000.0])

    dense_tau_ratio__1 = np.array(
        [0.5435745937961602, 0.4579025110782862, 0.3899556868537655, 0.3367799113737071, 0.2924667651403237,
         0.2511078286558348, 0.22451994091580474, 0.2008862629246675, 0.189069423929098, 0.17725258493353024,
         0.16838995568685355, 0.16543574593796073, 0.15657311669128582])

    dense_N__2_5 = np.array(
        [1.0, 1.4495439432517472, 2.0816879615674897, 3.1346793771186245, 5.28348505831838, 9.011125464140898,
         17.51015336723321, 39.93305755873251, 83.80803133377964, 187.73691609106876, 351.99595993515646,
         558.9992983516075, 1000.0])

    dense_tau_ratio__2_5 = np.array(
        [0.862629246676514, 0.7769571639586417, 0.6971935007385524, 0.6233382570162478, 0.5317577548005907,
         0.4549483013293934, 0.37813884785819774, 0.2983751846381093, 0.248153618906942, 0.2097488921713424,
         0.189069423929098, 0.17725258493353024, 0.16543574593796073])

    dense_N__5 = np.array(
        [1.0, 1.8039524165073502, 2.8152805783010555, 5.006074139509686, 8.438228081272191, 13.644111454508055,
         25.4362592645546, 43.124241931124715, 83.79408611106098, 164.75418675944553, 325.8512889631112,
         675.7590552732977, 1000.0])

    dense_tau_ratio__5 = np.array(
        [1.1964549483013291, 0.9985228951255536, 0.8714918759231907, 0.7237813884785815, 0.6115214180206792,
         0.5258493353028069, 0.4313146233382561, 0.3663220088626282, 0.3042836041358932, 0.2570162481536187,
         0.2215657311669137, 0.1949778434268836, 0.1920236336779908])

    dense_N__7_5 = np.array(
        [1.0, 1.48225601791807, 2.2192073460403177, 3.2062624913155364, 4.604831563628727, 7.670794347362198,
         11.689568487911274, 19.7035670077324, 34.824467272433694, 58.00144342791514, 101.29195689151608,
         189.929922238378, 345.71726458650164, 589.5422818581796, 1000.0])

    dense_tau_ratio__7_5 = np.array(
        [1.38847858197932, 1.2496307237813884, 1.1196454948301326, 1.0073855243722305, 0.9039881831610036,
         0.774002954209748, 0.676514032496307, 0.5701624815361894, 0.4697193500738557, 0.3958641063515511,
         0.3367799113737071, 0.2924667651403237, 0.2599704579025115, 0.23338257016248146, 0.21270310192023698])

    dense_N__10 = np.array(
        [1.0, 1.4129798949862502, 1.889929333916628, 2.6824105019814386, 3.992175064961803, 6.973828449005313,
         10.317100648315002, 15.445091506108891, 24.24574724777027, 51.20048362074438, 94.88372489923368,
         187.6728006748644, 337.58753740789007, 562.1907168565426, 1000.0])

    dense_tau_ratio__10 = np.array(
        [1.5184638109305757, 1.3943870014771047, 1.29394387001477, 1.1787296898079758, 1.0605612998522895,
         0.9039881831610036, 0.8035450516986701, 0.7060561299852282, 0.5997045790251105, 0.4638109305760701,
         0.38109305760709056, 0.3249630723781394, 0.2865583456425416, 0.2570162481536187, 0.23338257016248146])

    dense_N__15 = np.array(
        [1.0, 1.7690610431837857, 2.8954012024140523, 4.627472250473347, 7.4396105115945375, 11.271228891816651,
         19.228434774566832, 31.467367831486932, 52.72914184801593, 97.1465745936483, 181.0896526764028,
         320.0025096620405,
         520.4246119699965, 846.3665016664467, 1000.0])

    dense_tau_ratio__15 = np.array(
        [1.7961595273264406, 1.5864106351550964, 1.4062038404726724, 1.2496307237813884, 1.0960118168389943,
         0.9689807976366316, 0.8035450516986701, 0.6617429837518456, 0.5406203840472656, 0.4313146233382561,
         0.36927621861152105, 0.33087149187592324, 0.2954209748892165, 0.2629246676514025, 0.2599704579025115])

    # Very dense

    verydense_N__0_1 = np.array(
        [1.0, 1.9187520990478333, 3.6382449755244655, 7.19076838234002, 13.961753486779294, 28.762962774867876,
         54.538939578901555, 116.42271847199005, 209.29300664290955, 418.58520738574873, 789.0127987039548, 1000.0])

    verydense_tau_ratio__0_1 = np.array(
        [0.19145802650957397, 0.17967599410898316, 0.1649484536082486, 0.15316642120765955, 0.15316642120765955,
         0.14432989690721776, 0.14432989690721776, 0.13549337260677596, 0.13254786450662692, 0.12076583210603785,
         0.11192930780559608, 0.1089837997054488])

    verydense_N__0_25 = np.array(
        [1.0, 1.7765244276932377, 2.8033956239563613, 4.450107450706342, 8.191794966530754, 13.879284141942199,
         29.627711919708695, 52.323453136326826, 91.3164217144668, 160.31504121414685, 252.98075162281282,
         387.55754136286913, 637.4699119514506, 1000.0])

    verydense_tau_ratio__0_25 = np.array(
        [0.3181148748159064, 0.29160530191457923, 0.2709867452135484, 0.2503681885125193, 0.23269513991163396,
         0.2179675994108976, 0.20618556701030852, 0.19734904270986675, 0.1826215022091322, 0.1649484536082486,
         0.15611192930780504, 0.14432989690721776, 0.13549337260677596, 0.12076583210603785])

    verydense_N__0_5 = np.array(
        [1.0, 1.829935058289986, 3.009949274584684, 5.28426468041387, 7.766457760052918, 13.961753486779294,
         25.85362473999301, 45.12050252978594, 78.74562151399782, 126.49061934271386, 198.42602289337088,
         298.62670677675686, 485.40734819497084, 1000.0])

    verydense_tau_ratio__0_5 = np.array(
        [0.4683357879234169, 0.4123711340206189, 0.3740795287187044, 0.33578792341679, 0.3063328424153173,
         0.2768777614138447, 0.2503681885125193, 0.2268041237113394, 0.21207658321060485, 0.19734904270986675,
         0.1826215022091322, 0.1678939617083941, 0.14727540500736325, 0.14432989690721776])

    verydense_N__1 = np.array(
        [1.0, 1.6158639902500942, 2.5650209056800453, 4.3717145900084375, 7.766457760052918, 13.634787396553907,
         32.57350512147631, 62.871853054020576, 118.51039134343159, 211.78759301164973, 346.29890230000063,
         629.9613335204734, 1000.0])

    verydense_tau_ratio__1 = np.array(
        [0.6686303387334309, 0.592047128129602, 0.5213549337260677, 0.4536082474226806, 0.3917525773195879,
         0.3446244477172318, 0.2857142857142865, 0.2533136966126648, 0.22385861561119214, 0.2002945508100158,
         0.1855670103092777, 0.1649484536082486, 0.1590574374079523])

    verydense_N__2_5 = np.array(
        [1.0, 1.504977534812715, 2.346915358771387, 3.860298839373432, 7.106070348181026, 12.111289755953514,
         20.76462944173765, 40.07891463057112, 76.9015015749672, 139.06719655383736, 232.8449886486524,
         408.7824867737063, 730.5271542664458, 1000.0])

    verydense_tau_ratio__2_5 = np.array(
        [1.0515463917525771, 0.9425625920471284, 0.8335787923416795, 0.724594992636229, 0.6067746686303384,
         0.5154639175257731, 0.4329896907216497, 0.3505154639175263, 0.2857142857142865, 0.23858615611193024,
         0.21207658321060485, 0.18851251840942496, 0.17083946980854134, 0.1678939617083941])

    verydense_N__5 = np.array(
        [1.0, 1.3447604875886854, 1.9881825636611845, 2.9921700934164344, 5.069606734204699, 9.006280202112789,
         16.876124757881488, 31.810676038110035, 63.621229586575204, 115.0514080832087, 223.38633544498575,
         438.90150181193053, 789.0127987039548, 1000.0])

    verydense_tau_ratio__5 = np.array(
        [1.4050073637702507, 1.2960235640648, 1.157584683357879, 1.0250368188512518, 0.8689248895434467,
         0.7216494845360817,
         0.5832106038291602, 0.4653902798232696, 0.35640648011782083, 0.29160530191457923, 0.23858615611193024,
         0.20618556701030852, 0.17967599410898316, 0.17967599410898316])

    verydense_N__7_5 = np.array(
        [1.0, 2.5498698292923123, 3.2126326106632246, 4.3717145900084375, 6.349571608355332, 10.444025524245689,
         15.999876781703348, 25.700912439493624, 56.5124430431384, 108.43336871134291, 228.74320347570304,
         460.2038602746067, 1000.0])

    verydense_tau_ratio__7_5 = np.array(
        [2.5449189985272462, 1.7790868924889551, 1.608247422680412, 1.4197349042709868, 1.2135493372606767,
         0.98379970544919, 0.8247422680412377, 0.6656848306332837, 0.4712812960235642, 0.3652430044182626,
         0.29160530191457923, 0.23858615611193024, 0.2002945508100158])

    verydense_N__10 = np.array(
        [1.0, 9.16777932442606, 12.32846739442066, 17.38349997394775, 25.39818863530829, 34.357423105917356,
         49.02239343792874, 77.35844275341829, 139.89352019350324, 268.42098551447515, 497.04755483626417,
         862.3380114501625, 1000.0])

    verydense_tau_ratio__10 = np.array(
        [3.8880706921944035, 1.8939617083946985, 1.6229749631811483, 1.316642120765831, 1.0338733431516935,
         0.8512518409425631, 0.6892488954344635, 0.5390279823269513, 0.4212076583210607, 0.3387334315169355,
         0.2768777614138447, 0.2415316642120775, 0.2356406480117812])

    verydense_N__15 = np.array(
        [1.0, 28.022886286880574, 36.45436781521715, 46.42199946066167, 61.2541483493394, 81.40191996271128,
         108.17671540913874, 146.85724231997148, 236.45931777976287, 451.5615770668075, 732.2603572554892,
         1000.0])

    verydense_tau_ratio__15 = np.array(
        [8.415706863033874, 1.8727540500736384, 1.4804123711340225, 1.2047128129602385, 0.9608247422680432,
         0.7664212076583237, 0.6356406480117851, 0.5508100147275421, 0.44123711340206384, 0.3316642120765856,
         0.2786450662739348, 0.26450662739322617])
    
    if failure_stress_ratio == 0.19:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 5, 15])
        _n_arrays = [veryloose_N__0_1, veryloose_N__0_25, veryloose_N__0_5,
                     veryloose_N__1, veryloose_N__5, veryloose_N__15]
        _tau_ratio_arrays = [veryloose_tau_ratio__0_1, veryloose_tau_ratio__0_25, veryloose_tau_ratio__0_5,
                             veryloose_tau_ratio__1, veryloose_tau_ratio__5, veryloose_tau_ratio__15]
        _max_shearstress_ratio = 0.3
    elif failure_stress_ratio == 0.25:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 5, 15])
        _n_arrays = [loose_N__0_1, loose_N__0_25, loose_N__0_5,
                     loose_N__1, loose_N__5, loose_N__15]
        _tau_ratio_arrays = [loose_tau_ratio__0_1, loose_tau_ratio__0_25, loose_tau_ratio__0_5,
                             loose_tau_ratio__1, loose_tau_ratio__5, loose_tau_ratio__15]
        _max_shearstress_ratio = 0.4
    elif failure_stress_ratio == 0.6:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _n_arrays = [mediumdense_N__0_1, mediumdense_N__0_25, mediumdense_N__0_5,
                      mediumdense_N__1, mediumdense_N__2_5, mediumdense_N__5,
                      mediumdense_N__7_5, mediumdense_N__10, mediumdense_N__15]
        _tau_ratio_arrays = [mediumdense_tau_ratio__0_1, mediumdense_tau_ratio__0_25, mediumdense_tau_ratio__0_5,
                             mediumdense_tau_ratio__1, mediumdense_tau_ratio__2_5, mediumdense_tau_ratio__5,
                             mediumdense_tau_ratio__7_5, mediumdense_tau_ratio__10, mediumdense_tau_ratio__15]
        _max_shearstress_ratio = 1
    elif failure_stress_ratio == 1:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _n_arrays = [dense_N__0_1, dense_N__0_25, dense_N__0_5,
                     dense_N__1, dense_N__2_5, dense_N__5,
                     dense_N__7_5, dense_N__10, dense_N__15]
        _tau_ratio_arrays = [dense_tau_ratio__0_1, dense_tau_ratio__0_25, dense_tau_ratio__0_5,
                             dense_tau_ratio__1, dense_tau_ratio__2_5, dense_tau_ratio__5,
                             dense_tau_ratio__7_5, dense_tau_ratio__10, dense_tau_ratio__15]
        _max_shearstress_ratio = 2
    elif failure_stress_ratio == 1.8:
        _shear_strains_interpolation = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15])
        _n_arrays = [verydense_N__0_1, verydense_N__0_25, verydense_N__0_5,
                     verydense_N__1, verydense_N__2_5, verydense_N__5,
                     verydense_N__7_5, verydense_N__10, verydense_N__15]
        _tau_ratio_arrays = [verydense_tau_ratio__0_1, verydense_tau_ratio__0_25, verydense_tau_ratio__0_5,
                             verydense_tau_ratio__1, verydense_tau_ratio__2_5, verydense_tau_ratio__5,
                             verydense_tau_ratio__7_5, verydense_tau_ratio__10, verydense_tau_ratio__15]
        _max_shearstress_ratio = 2
    else:
        raise ValueError("Stress ratio for failure at 10 cycles should be selected from [0.19, 0.25, 0.6, 1.0, 1.8]")

    
    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
        _shear_strains_interpolation,
        _n_arrays,
        _tau_ratio_arrays
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$\gamma_{cy} = %s \text{%%} $' % str(_strain))
        fig.append_trace(trace, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 3), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ \tau_{cyc} / \sigma_{ref}^{\prime} $', range=(0, _max_shearstress_ratio))
    fig['layout'].update(
        height=500, width=500,
        title=r'$ \text{Strain accumulation diagram for cyclic DSS tests on sand} $',
        hovermode='closest')

    return fig


def plotporepressureaccumulation_dsssand_andersen(failure_stress_ratio):
    """
    Returns a Plotly figure with the permanent excess pore pressure accumulation contours for cyclic DSS tests on normally consolidated sand or silt with symmetrical loading

    :param failure_stress_ratio: Ratio of cyclic shear stress to vertical effective stress for failure at N=10 (:math:`( \\tau_{cy} / \\sigma_{ref}^{\\prime})_{N=10}`) [:math:`-`] - Allowable value: [0.19, 0.25, 0.6, 1.0, 1.8]

    :return: Plotly figure object which can be further customised and to which test data can be added.
    """

    fig = subplots.make_subplots(rows=1, cols=1, print_grid=False, shared_yaxes=False)

    veryloose_ratios = [0.025, 0.05, 0.1, 0.25, 0.5, 1]

    veryloose_N__0_025 = np.array(
        [1.0, 1.3848357669923803, 2.0357271093895037, 3.193894030999909, 5.928942236129898, 11.006112193509402,
         19.9923617066251, 42.735953822977145, 77.628923972849, 175.19400701801436, 460.2574060879412, 1000.0])

    veryloose_tau_ratio__0_025 = np.array(
        [0.051140939597315416, 0.04627076556461773, 0.042601686024135776, 0.03811964550261765, 0.03361799269284105,
         0.029519024446822806, 0.028241378764953032, 0.02654206888552646, 0.025667107767415157, 0.0255722095984226,
         0.023848858849517873, 0.023759021916204937])

    veryloose_N__0_05 = np.array(
        [1.0, 1.5774571848864662, 2.421764851430612, 4.569393943561346, 8.574890717081912, 17.8391725161195,
         37.721654290991744, 86.99873974305252, 209.5503040737345, 610.3036318994721, 1000.0])

    veryloose_tau_ratio__0_05 = np.array(
        [0.09906103533902373, 0.09014819930724338, 0.08083647463846436, 0.06988997084517365, 0.061762891652651634,
         0.05483184571666572, 0.05192574745488382, 0.04860684215798439, 0.0456855601891637, 0.04354750444176153,
         0.04228251184909088])

    veryloose_N__0_1 = np.array(
        [1.0, 1.6744824380115342, 2.788717719334948, 5.148783548105401, 11.432215452969432, 29.389039247078628,
         61.47352800252821, 121.13450561611624, 234.8430383186117, 574.9405471492294, 1000.0])

    veryloose_tau_ratio__0_1 = np.array(
        [0.15463150513768142, 0.13685264950415707, 0.12189385112586132, 0.10652034774906816, 0.0931387569394286,
         0.08296156096943752, 0.07804330519778363, 0.07353469318894754, 0.07063871739852487, 0.06650748377504917,
         0.06483348007402062])

    veryloose_N__0_25 = np.array(
        [1.0, 1.5269251672721005, 2.5155238354112477, 4.520071721903168, 8.345366315362165, 16.805509702089847,
         36.31568710014062, 84.21183567285425, 198.48236072117786, 524.2761412116424, 1000.0])

    veryloose_tau_ratio__0_25 = np.array(
        [0.21704697986577184, 0.20129293483131847, 0.183516609815634, 0.1653274777648318, 0.14874592069676346,
         0.13295961028485268, 0.11756775992872096, 0.1037784233196434, 0.09200061156597797, 0.08182025232368711,
         0.07651069976855393])

    veryloose_N__0_5 = np.array(
        [1.0, 1.5774571848864662, 2.421764851430612, 3.904056585284437, 7.208023306847551, 12.536988985361376,
         22.04365381849705, 46.359951117214486, 84.67004141339133, 165.9406806916515, 395.38020784318746,
         1000.0])

    veryloose_tau_ratio__0_5 = np.array(
        [0.2532892232584868, 0.23753074964281384, 0.22217875651765898, 0.2056130158110894, 0.18540729766919556,
         0.16802733067266984, 0.14943804467594912, 0.12800908913574133, 0.11263685106786805, 0.09926981131080737,
         0.08829610337573879, 0.07932949171486267])

    veryloose_N__1 = np.array(
        [1.0, 1.5352333328811785, 2.3956242244061983, 3.8619160417804563, 6.259556989862344, 12.401664222975011,
         21.45361056064785, 43.91132993739474, 93.35750868773569, 227.32011302933896, 574.9405471492294,
         1000.0])

    veryloose_tau_ratio__1 = np.array(
        [0.29556920448973784, 0.2717621008345767, 0.2503679412896661, 0.2281646166904791, 0.2095848205106576,
         0.18091450202185824, 0.1595083220422086, 0.13606910695550858, 0.11544425523389765, 0.09883043278837186,
         0.08704439652672705, 0.08335633735244655])

    loose_ratio = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

    loose_N__0_05 = np.array(
        [1.0, 1.8134408785428524, 3.2122010600916733, 6.201789291435361, 12.35482888256747, 28.117686979742288,
         73.67954559661635, 202.35896477251575, 439.3970560760795, 1000.0])

    loose_tau_ratio__0_05 = np.array(
        [0.11572815533980575, 0.1002611012042357, 0.08789820135173798, 0.07632168725095212, 0.06863219074037374,
         0.061734363648371815, 0.06028928076084794, 0.05729608348193638, 0.054276467868700884, 0.05359047178742038])

    loose_N__0_1 = np.array(
        [1.0, 1.6507719276329036, 3.017114810529294, 5.689866029018296, 12.549876607521336, 27.03788951886361,
         63.991523363492696, 150.2694678003785, 358.44437436727236, 634.9230421295147, 1000.0])

    loose_tau_ratio__0_1 = np.array(
        [0.16931950773837026, 0.1553961649384672, 0.13915329238491492, 0.12524403936331796, 0.11135239856460376,
         0.10211831010721427, 0.09367148801268066, 0.08755388239438168, 0.0814380379983709, 0.07839552649538772,
         0.07689320388349519])

    loose_N__0_25 = np.array(
        [1.0, 1.4793400915475905, 2.329951810515372, 4.062969268445067, 7.9682058339974775, 17.301957388458945,
         30.17114810529294, 62.99697963699091, 122.58444754550987, 292.4060772343886, 675.9770831018744, 1000.0])

    loose_tau_ratio__0_25 = np.array(
        [0.23922418158201786, 0.2252867490038085, 0.20825044580939167, 0.1904488915307222, 0.17188384739008866,
         0.15177685313607636, 0.14018889109041674, 0.12551438698456718, 0.11471545252405167, 0.10316271492415717,
         0.09471325099619132, 0.09242630385487527])

    loose_N__0_5 = np.array(
        [1.0, 1.5505157798326243, 2.5001103826179305, 4.605378255822415, 7.543120063354618, 12.648552168552964,
         20.235896477251565, 40.62969268445065, 79.68205833997479, 162.51159535566703, 425.8451453924132,
         714.071165775071,
         1000.0])

    loose_tau_ratio__0_5 = np.array(
        [0.2897096184752218, 0.2742240715056249, 0.2587438082028928, 0.23784162208572746, 0.2192563238887787,
         0.20145036655438864, 0.18519252361138616, 0.16119322810030154, 0.13952138784316306, 0.12329084384562886,
         0.10631178037558052, 0.09937960944895764, 0.09863989608788493])

    loose_N__0_75 = np.array(
        [1.0, 1.625115953556669, 2.5796728079998696, 3.906939937054615, 6.974890837751715, 12.648552168552964,
         23.482714569365548, 49.80568431667222, 108.99710572808331, 193.0697728883252, 409.49150623804286,
         766.2204546142335, 1000.0])

    loose_tau_ratio__0_75 = np.array(
        [0.33242630385487515, 0.3099575105122955, 0.2905919908416441, 0.2712211875041278, 0.24720868282588115,
         0.22164454131167036, 0.1953063426017656, 0.16665301719393244, 0.1403333113180547, 0.1264170134073046,
         0.11174427052374336, 0.10404772912401206, 0.10252427184466018])

    loose_N__1 = np.array(
        [1.0, 1.5627069765469954, 2.6410018625044005, 4.4633389000179955, 7.7831690353360985, 12.648552168552964,
         22.580912953008976, 40.94915062380426, 66.0279617696055, 116.95726687403405, 196.11779729377534,
         347.38921120831185, 686.6488450043005, 1000.0])

    loose_tau_ratio__1 = np.array(
        [0.3704863175043479, 0.3425744666798758, 0.3154490016071153, 0.2898769345926072, 0.2627549919644233,
         0.2371785218941945, 0.21005922109944294, 0.18294168152697965, 0.16357792307861652, 0.144224731963983,
         0.1326323668626026, 0.1187160689518526, 0.11024899280100386, 0.10718446601941743])

    mediumdense_ratios = [0.05, 0.1, 0.3, 0.5, 0.75, 0.95, 1]

    mediumdense_N__0_05 = np.array([1.0, 53.82212827343278, 1000.0])

    mediumdense_tau_ratio__0_05 = np.array([0.10754397931695968, 0.06970136376161484, 0.060998104255411974])

    mediumdense_N__0_1 = np.array(
        [1.7110666615325072, 1.3193985940132802, 1.0173633065749466, 1.0077057574735715, 3.071076829233049,
         14.447832098477335, 65.38442896400393, 288.34142121526423, 1000.0])

    mediumdense_tau_ratio__0_1 = np.array(
        [0.8891222807960619, 0.7752179955700709, 0.6581034423047679, 0.1861955462801093, 0.1522287421380799,
         0.11816105246198337, 0.10015373750788603, 0.08857298164937255, 0.08507511455025396])

    mediumdense_N__0_3 = np.array(
        [2.289669884652019, 1.8239459193420644, 1.3439282689319951, 1.022976605157307, 1.0087439636215003,
         2.402634995145728, 4.562086266558328, 9.797993128547322, 23.192961589923282, 51.45989293685091,
         118.70817485073185, 270.3110924846991, 644.0904954868103, 1000.0])

    mediumdense_tau_ratio__0_3 = np.array(
        [0.8409005012655864, 0.7478554295239257, 0.6179103443810519, 0.5072193387028525, 0.34510381422606473,
         0.2726710122734026, 0.22436792201397626, 0.1856670265419296, 0.15817948289387873, 0.14354806894547512,
         0.1337230225306003, 0.12390098762420546, 0.11406841243813215, 0.1107572588647514])

    mediumdense_N__0_5 = np.array(
        [8.089891612859434, 6.877462515062732, 6.0783504680169, 5.441939167137974, 5.032815047014328,
         5.4020109564163965,
         7.13593627147009, 10.871329378699386, 15.726658133283493, 23.96196031902427, 40.4995194112587,
         75.43739753774507,
         130.84652908660968, 242.1552083988005, 481.2577168795256, 1000.0])

    mediumdense_tau_ratio__0_5 = np.array(
        [0.6239137865352444, 0.567771739703276, 0.5068052562869001, 0.4394152252834207, 0.378438201587366,
         0.30298033936689084, 0.2611821074235188, 0.2273764189851515, 0.20642384873795194, 0.19187976853545846,
         0.1821269983240952, 0.1707465077794792, 0.1577774465118451, 0.14960872976078046, 0.13981831569342207,
         0.13001886710062527])

    mediumdense_N__0_75 = np.array(
        [12.007792334115775, 10.476246682696745, 9.021882531841493, 8.129198857816446, 8.839818196089869,
         14.744325164843287, 26.07477604457922, 52.83480870102749, 104.32099183612353, 204.6551004891357,
         436.7578100824058, 1000.0])

    mediumdense_tau_ratio__0_75 = np.array(
        [0.5708525128779631, 0.5147044430290353, 0.4505337145903066, 0.3719062396949946, 0.3044710360643195,
         0.2529877928503779, 0.23198854422174356, 0.21256280877372855, 0.1963533643819857, 0.18496082780345047,
         0.16712818034117394, 0.14767383556260238])

    mediumdense_N__0_95 = np.array(
        [34.95016991490944, 30.689692855070177, 28.01464880531402, 41.316021698397975, 75.96100067087414,
         152.92093347310598, 290.42129877521694, 507.00671330824076, 851.3780771912169, 1000.0])

    mediumdense_tau_ratio__0_95 = np.array(
        [0.440588207836246, 0.3796232301741101, 0.3058081458292867, 0.26237918204418165, 0.2381606308507957,
         0.2155261331177094, 0.19772058923174907, 0.18314488819021868, 0.17178848971343985, 0.17014420608354808])

    mediumdense_N__1 = np.array(
        [101.74585488904283, 165.3795967685581, 272.32038902996186, 469.2510739696394, 868.3709015733681, 1000.0])

    mediumdense_tau_ratio__1 = np.array(
        [0.3392163151483394, 0.3021853011282616, 0.26675640961936065, 0.24094778194871716, 0.22154312706005985,
         0.21990336069288752])

    dense_ratios = [0.05, 0.1, 0.25, 0.35, 0.5, 0.75, 0.9, 1]

    dense_N__0_05 = np.array(
        [3.5359609705640764, 2.3713737056616546, 1.3192764829478167, 0.980854248667005, 1.0, 5.4803206579948816,
         1000.0])

    dense_tau_ratio__0_05 = np.array(
        [1.3359275053304909, 1.1424603174603192, 0.8379678986022281, 0.6984215825633733, 0.13333333333333466,
         0.10080549632788482, 0.06985015399194694])

    dense_N__0_1 = np.array(
        [3.8947992223987136, 2.1528927143713825, 1.3192764829478167, 0.980854248667005, 0.9871950864202704,
         5.105313173970846, 52.611670261399794, 1000.0])

    dense_tau_ratio__0_1 = np.array(
        [1.294613243307273, 1.0059968017057574, 0.7744758351101629, 0.6254057095475005, 0.23810116086235664,
         0.16433013503909244, 0.1219882729211097, 0.09842454394693156])

    dense_N__0_25 = np.array(
        [4.9116944635049204, 2.6975540127097815, 1.4070868214407417, 1.0064646075213344, 0.9871950864202704,
         1.4815249005317854, 2.281439493346547, 4.975404082809528, 12.106605419922511, 74.0291878788015,
         1000.0])

    dense_tau_ratio__0_25 = np.array(
        [1.2151415541340915, 0.9233534707415316, 0.6093668561952157, 0.4444414830608867, 0.3650852878464832,
         0.32045427623785905, 0.2694622127457951, 0.22148483771618288, 0.1829809286898847, 0.15357735133854433,
         0.1269959725183618])

    dense_N__0_35 = np.array(
        [9.355824995360434, 6.194091434568745, 3.4682623410984807, 2.840260728379856, 2.750208827025826,
         2.8771017678776163,
         3.5132491934040755, 5.171534222768003, 8.604021610781865, 15.267525594361608, 41.45121962992579,
         135.6633966891449,
         1000.0])

    dense_tau_ratio__0_35 = np.array(
        [1.0116708126036489, 0.8309079601990064, 0.5549840085287839, 0.4344408907841757, 0.393185856432126,
         0.3518952854773758, 0.31370824449182777, 0.26591151385927597, 0.2275823264629242, 0.2019219379294004,
         0.18558990760483415, 0.17869580668088325, 0.14921819474058218])

    dense_N__0_5 = np.array(
        [14.975217345861829, 10.305277750567273, 7.091633267391837, 6.036477442749892, 5.696354580020144,
         5.6597664115073165, 6.194091434568745, 8.277714582724503, 13.683400226563096, 30.423484605614533,
         100.21502423178413, 361.2732623927534, 1000.0])

    dense_tau_ratio__0_5 = np.array(
        [0.8717720919213471, 0.6941660743899547, 0.5197346600331674, 0.4245705993840332, 0.383327410566217,
         0.33888592750533064, 0.2975746268656714, 0.2656953328595133, 0.23054371002132254, 0.22382729211087596,
         0.2074064202795558, 0.19094408907841756, 0.17143745557924814])

    dense_N__0_75 = np.array(
        [20.801613122929545, 16.07521227040762, 12.342919894954381, 10.239086077697802, 9.001005866892035,
         8.715624428380659, 9.176700696484248, 10.574351225698, 16.494940232502046, 33.295693091189605,
         67.64315888029452, 164.59548215396748, 476.61931899259497, 1000.0])

    dense_tau_ratio__0_75 = np.array(
        [0.7827321724709789, 0.6844379294006178, 0.5670990286661937, 0.4782960199004975, 0.4180377872542049,
         0.376782752902157, 0.3323146173892457, 0.3005034352049272, 0.2939498933901916, 0.2650556740108989,
         0.2456823027718542, 0.2325752191423849, 0.2130389718076291, 0.19683131959251374])

    dense_N__0_9 = np.array(
        [37.87548554226353, 29.269675689982943, 25.89680989231187, 22.619219337553076, 21.761386736369747,
         22.619219337553076, 26.064222604320264, 35.97246090366672, 64.6597957422025, 118.49336418460686,
         277.393969229964, 657.8047647410542, 1000.0])

    dense_tau_ratio__0_9 = np.array(
        [0.6237266050698889, 0.5222577588249226, 0.4556473584458658, 0.39221748400852974, 0.3636638237384506,
         0.338249230040276, 0.322311063728975, 0.3094645818526427, 0.2869728737266062, 0.2676468846244973,
         0.2482083629471692, 0.2287639185027253, 0.2253968253968264])

    dense_N__1 = np.array(
        [95.79509745537216, 142.84029757303355, 184.8379081095041, 272.0830532738147, 368.3251236192624,
         518.2654274501964,
         782.8106335836313, 1000.0])

    dense_tau_ratio__1 = np.array(
        [0.4391731817104958, 0.3786721156124155, 0.35315683487325344, 0.3180585169391144, 0.29569710968964635,
         0.28284174366264025, 0.27312840559109297, 0.26985311537550416])

    verydense_ratios = [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    verydense_N__min0_5 = np.array(
        [5.054905390551563, 3.826851593660528, 2.8048614899377826, 1.9647202093811915, 1.44933198932027,
         1.002177039220033])

    verydense_tau_ratio__min0_5 = np.array(
        [1.7596153846153832, 1.6891025641025619, 1.6089743589743577, 1.5160256410256403, 1.4326923076923066,
         1.3397435897435876])

    verydense_N__min0_25 = np.array(
        [14.581236166116314, 9.696518186927577, 6.617983746163751, 4.345256311013824, 2.8166206477703777,
         1.7224914872929957, 1.2225005031967098, 1.0200183223073749])

    verydense_tau_ratio__min0_25 = np.array(
        [1.7019230769230766, 1.5673076923076916, 1.4551282051282044, 1.349358974358973, 1.2564102564102555,
         1.1538461538461515, 1.0929487179487154, 1.0705128205128194])

    verydense_N__0 = np.array(
        [32.89266138347117, 17.666058751849693, 10.731565799691376, 6.733987237829349, 3.9100126993906743,
         2.422203464653512, 1.406716065685001, 1.0048998011990231])

    verydense_tau_ratio__0 = np.array(
        [1.5961538461538431, 1.4006410256410238, 1.2628205128205128, 1.1442307692307685, 1.0160256410256403,
         0.9102564102564088, 0.8141025641025639, 0.7596153846153832])

    verydense_N__0_1 = np.array(
        [36.904032909294045, 25.02726122381268, 16.75344495927606, 10.112258632837511, 6.223293624563298,
         3.805432685070469,
         2.434979373821572, 1.489038829273294, 0.9906001635222234, 1.0079847993011946, 2.1188273011328835,
         4.425361629258659, 8.443488402694374, 21.40883000631555, 62.98240727486107, 235.3465967708344,
         498.008566907512,
         1000.0])

    verydense_tau_ratio__0_1 = np.array(
        [1.4038461538461533, 1.3044871794871788, 1.1923076923076898, 1.064102564102564, 0.939102564102562,
         0.8205128205128194, 0.7243589743589745, 0.6153846153846132, 0.5416666666666661, 0.2339743589743595,
         0.20192307692307665, 0.17628205128204932, 0.1538461538461533, 0.13782051282051275, 0.12499999999999822,
         0.12179487179487047, 0.11858974358974272, 0.1153846153846132])

    verydense_N__0_25 = np.array(
        [41.69028841655493, 30.954045812904372, 19.550189340904122, 14.515249140841243, 8.482399934642,
         5.75240225740087,
         3.900953345473632, 2.972266295597044, 2.150392861095529, 1.9508932389604272, 2.120890824808264,
         2.8721717424230686,
         4.342197546391367, 8.177414881060459, 16.430094683607837, 29.579575746059497, 142.20952625259778,
         73.08958708692036, 282.0823937729993, 585.4324530070704, 1000.0])

    verydense_tau_ratio__0_25 = np.array(
        [1.2756410256410238, 1.1955128205128194, 1.083333333333332, 0.9999999999999982, 0.8653846153846151,
         0.762820512820511, 0.657051282051281, 0.5801282051282044, 0.490384615384615, 0.42307692307692335,
         0.3525641025641022, 0.2788461538461533, 0.24038461538461495, 0.19871794871794712, 0.17628205128204932,
         0.1666666666666643, 0.16346153846153655, 0.15705128205128105, 0.1538461538461533, 0.14743589743589602,
         0.14664102564102555])

    verydense_N__0_5 = np.array(
        [54.6032464317418, 39.255919951720024, 26.28148643039328, 18.057437186726826, 12.897616591148848,
         8.468707923495352,
         5.8177008421429575, 4.577403510922573, 4.517691061603811, 5.3076790749817855, 7.620279993045468,
         16.226850060706685, 35.00485065500348, 68.54295090855894, 129.11514560023423, 264.52412121493353,
         521.3329860731125, 1000.0])

    verydense_tau_ratio__0_5 = np.array(
        [1.0320512820512808, 0.9647435897435894, 0.8717948717948705, 0.7916666666666661, 0.7147435897435876,
         0.6153846153846132, 0.5096153846153832, 0.4038461538461533, 0.3717948717948705, 0.31089743589743435,
         0.2788461538461533, 0.24999999999999825, 0.2275641025641022, 0.2179487179487154, 0.21474358974358765,
         0.20833333333333212, 0.20192307692307665, 0.20033333333333211])

    verydense_N__0_75 = np.array(
        [76.78263400971767, 50.427755784303706, 36.727134680268, 23.656833286325263, 16.25577844313828,
         11.992532829678323,
         9.079783304917582, 8.29271005902147, 8.13215111017109, 8.563244399859936, 10.598374469819039,
         17.094838680773467,
         33.688986591791576, 56.487435717463896, 101.03466115623269, 196.55749199999352, 397.49254868184914,
         743.8922567862396, 1000.0])

    verydense_tau_ratio__0_75 = np.array(
        [0.7852564102564088, 0.7211538461538449, 0.6602564102564088, 0.5897435897435894, 0.5256410256410255,
         0.4551282051282044, 0.397435897435896, 0.3653846153846132, 0.3397435897435894, 0.3333333333333321,
         0.3301282051282044, 0.3141025641025621, 0.2980769230769216, 0.28205128205128105, 0.262820512820511,
         0.24999999999999825, 0.23076923076922995, 0.2179487179487154, 0.21474358974358765])

    verydense_N__0_9 = np.array(
        [151.10342339230888, 122.84141845699257, 96.06765172071718, 80.66896472614657, 71.79183468978344,
         70.40475644212788,
         71.76804834426996, 80.61718814021656, 106.42807308519495, 147.96623557944426, 215.21750853612838,
         333.9305746145565, 535.1611336093466, 1000.0])

    verydense_tau_ratio__0_9 = np.array(
        [0.5512820512820493, 0.5064102564102555, 0.4615384615384599, 0.4262820512820511, 0.3846153846153833,
         0.3653846153846132, 0.3333333333333321, 0.32692307692307665, 0.31089743589743435, 0.30448717948717885,
         0.2852564102564106, 0.26602564102563875, 0.2532051282051277, 0.24999999999999825])

    verydense_N__0_95 = np.array(
        [174.11787069429826, 149.0797732308189, 126.82787695327957, 107.88169430182636, 99.82204246611258,
         98.53006194944159, 103.74464192173164, 121.92134062294777, 153.83804591735446, 228.14282094861218,
         342.7311570338762, 435.3002807460022, 574.6805449630465])

    verydense_tau_ratio__0_95 = np.array(
        [0.490384615384615, 0.4647435897435877, 0.4487179487179471, 0.4102564102564088, 0.3942307692307683,
         0.37820512820512775, 0.3589743589743577, 0.3429487179487172, 0.32692307692307665, 0.31089743589743435,
         0.2916666666666661, 0.2916666666666661, 0.2788461538461533])

    if failure_stress_ratio == 0.19:
        _shear_strains_interpolation = np.array(veryloose_ratios) # [0.025, 0.05, 0.1, 0.25, 0.5, 1]
        _n_arrays = [veryloose_N__0_025, veryloose_N__0_05, veryloose_N__0_1,
                     veryloose_N__0_25, veryloose_N__0_5, veryloose_N__1]
        _tau_ratio_arrays = [veryloose_tau_ratio__0_025, veryloose_tau_ratio__0_05, veryloose_tau_ratio__0_1,
                             veryloose_tau_ratio__0_25, veryloose_tau_ratio__0_5, veryloose_tau_ratio__1]
        _max_shearstress_ratio = 0.3
    elif failure_stress_ratio == 0.25:
        _shear_strains_interpolation = np.array(loose_ratio) # [0.05, 0.1, 0.25, 0.5, 0.75, 1]
        _n_arrays = [loose_N__0_05, loose_N__0_1, loose_N__0_25,
                     loose_N__0_5, loose_N__0_75, loose_N__1]
        _tau_ratio_arrays = [loose_tau_ratio__0_05, loose_tau_ratio__0_1, loose_tau_ratio__0_25,
                             loose_tau_ratio__0_5, loose_tau_ratio__0_75, loose_tau_ratio__1]
        _max_shearstress_ratio = 0.4
    elif failure_stress_ratio == 0.6:
        _shear_strains_interpolation = np.array(mediumdense_ratios) # [0.05, 0.1, 0.3, 0.5, 0.75, 0.95, 1]
        _n_arrays = [mediumdense_N__0_05, mediumdense_N__0_1, mediumdense_N__0_3,
                     mediumdense_N__0_5, mediumdense_N__0_75, mediumdense_N__0_95,
                     mediumdense_N__1]
        _tau_ratio_arrays = [mediumdense_tau_ratio__0_05, mediumdense_tau_ratio__0_1, mediumdense_tau_ratio__0_3,
                             mediumdense_tau_ratio__0_5, mediumdense_tau_ratio__0_75, mediumdense_tau_ratio__0_95,
                             mediumdense_tau_ratio__1]
        _max_shearstress_ratio = 1
    elif failure_stress_ratio == 1:
        _shear_strains_interpolation = np.array(dense_ratios) # [0.05, 0.1, 0.25, 0.35, 0.5, 0.75, 0.9, 1]
        _n_arrays = [dense_N__0_05, dense_N__0_1, dense_N__0_25,
                     dense_N__0_35, dense_N__0_5, dense_N__0_75,
                     dense_N__0_9, dense_N__1]
        _tau_ratio_arrays = [dense_tau_ratio__0_05, dense_tau_ratio__0_1, dense_tau_ratio__0_25,
                             dense_tau_ratio__0_35, dense_tau_ratio__0_5, dense_tau_ratio__0_75,
                             dense_tau_ratio__0_9, dense_tau_ratio__1]
        _max_shearstress_ratio = 2
    elif failure_stress_ratio == 1.8:
        _shear_strains_interpolation = np.array(verydense_ratios) # [-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        _n_arrays = [verydense_N__min0_5, verydense_N__min0_25, verydense_N__0,
                     verydense_N__0_1, verydense_N__0_25, verydense_N__0_5,
                     verydense_N__0_75, verydense_N__0_9, verydense_N__0_95]
        _tau_ratio_arrays = [verydense_tau_ratio__min0_5, verydense_tau_ratio__min0_25, verydense_tau_ratio__0,
                             verydense_tau_ratio__0_1, verydense_tau_ratio__0_25, verydense_tau_ratio__0_5,
                             verydense_tau_ratio__0_75, verydense_tau_ratio__0_9, verydense_tau_ratio__0_95]
        _max_shearstress_ratio = 2
    else:
        raise ValueError("Stress ratio for failure at 10 cycles should be selected from [0.19, 0.25, 0.6, 1.0, 1.8]")

    for i, (_strain, _n, _tau_ratio) in enumerate(zip(
            _shear_strains_interpolation,
            _n_arrays,
            _tau_ratio_arrays
    )):
        trace = go.Scatter(
            x=_n,
            y=_tau_ratio,
            showlegend=True, mode='lines', name=r'$\gamma_{cy} = %s \text{%%} $' % str(_strain))
        fig.append_trace(trace, 1, 1)

    fig['layout']['xaxis1'].update(title=r'$ N $', range=(0, 3), dtick=1, type='log')
    fig['layout']['yaxis1'].update(title=r'$ u_p / \sigma_{ref}^{\prime} $', range=(0, _max_shearstress_ratio))
    fig['layout'].update(
        height=500, width=600,
        title=r'$ \text{Pore pressure accumulation diagram for cyclic DSS tests on sand} $',
        hovermode='closest')

    return fig

# TODO: Code an interpolation function for cyclic DSS permanent pore pressure accumulation in sand

CYCLICSTRENGTH_DSSSAND_RELATIVEDENSITY = {
    'relative_density': {'type': 'float', 'min_value': 40.0, 'max_value': 110.0},
    'vertical_effective_stress': {'type': 'float', 'min_value': 100.0, 'max_value': 250.0},
    'fines_content': {'type': 'float', 'min_value': 5.0, 'max_value': 35.0},
    'stress_exponent': {'type': 'float', 'min_value': 0.2, 'max_value': 1.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
}

CYCLICSTRENGTH_DSSSAND_RELATIVEDENSITY_ERRORRETURN = {
    'cyclic strength ratio [-]': np.nan,
    'reference effective stress [kPa]': np.nan,
    'cyclic shear strength [kPa]': np.nan,
}


@Validator(CYCLICSTRENGTH_DSSSAND_RELATIVEDENSITY, CYCLICSTRENGTH_DSSSAND_RELATIVEDENSITY_ERRORRETURN)
def cyclicstrength_dsssand_relativedensity(
        relative_density, vertical_effective_stress,
        fines_content=5.0, stress_exponent=0.9, atmospheric_pressure=100.0, **kwargs):
    """
    Calculates the DSS cyclic strength of sand, defined as the ratio of cyclic shear stress to reference normal stress for failure at 10 cycles. Cyclic failure is defined as reaching an accumulated cyclic shear strain of 15%. For certain very dense samples, this strain was not reached. The underlying dataset contains symmetrical DSS tests on normally consolidated sand.

    The curves are most representative for vertical effective stresses between 100 and 250kPa. The vertical effective stress was normalised using an exponent of 0.9 for the test data. Most of the underlying cyclic DSS test data is from normally consolidated pre-sheared samples.

    The function calculates the shear stress ratio at failure for <5% fines by default but fines content of 20% or 35% can also be entered. The trend for 35% fines content needs to be treated with caution as relative density determination for samples with high fines content is not straightforward.

    :param relative_density: Relative density of the sand (:math:`D_r`) [:math:`pct`] - Suggested range: 40.0 <= relative_density <= 110.0
    :param vertical_effective_stress: Vertical effective stress at the depth considered (:math:`\\sigma_{vc}^{\\prime}`) [:math:`kPa`] - Suggested range: 100.0 <= vertical_effective_stress <= 250.0
    :param fines_content: Fines content of the soil (:math:`FC`) [:math:`pct`] - Suggested range: 5.0 <= fines_content <= 35.0 (optional, default= 5.0)
    :param stress_exponent: Stress exponent used for stress normalisation (:math:`n`) [:math:`-`] - Suggested range: 0.2 <= stress_exponent <= 1.0 (optional, default= 0.9)
    :param atmospheric_pressure: Atmospheric pressure (:math:`p_a`) [:math:`kPa`] (optional, default= 100.0)

    .. math::
        \\sigma_{ref}^{\\prime} = p_a \\cdot ( \\sigma_{ref}^{\\prime} / p_a )^n

    :returns: Dictionary with the following keys:

        - 'cyclic strength ratio [-]': Cyclic strength ratio for N=10 cycles (:math:`\\tau_f / \\sigma_{ref}^{\\prime}`)  [:math:`-`]
        - 'reference effective stress [kPa]': Reference effective normal stress (:math:`\\sigma_{ref}^{\\prime}`)  [:math:`kPa`]
        - 'cyclic shear strength [kPa]': Cyclic shear stress for failure at 10 cycles (:math:`\\tau_f`)  [:math:`kPa`]

    .. figure:: images/cyclicstrength_dsssand_relativedensity_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Underlying data and trend for cyclic strength calculation

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    # Cyclic strength vs relative density

    Dr__5 = np.array(
        [44.967320261437905, 53.72549019607843, 61.437908496732035, 67.84313725490198, 74.24836601307192,
         79.47712418300654,
         84.18300653594773, 87.97385620915036, 90.45751633986929, 93.8562091503268, 97.12418300653594, 99.3464052287582,
         103.00653594771244, 104.96732026143793])

    tau_ratio__5 = np.array(
        [0.15581344311333611, 0.17006548727918666, 0.18968305162839175, 0.21001731771866167, 0.2376378690328793,
         0.2727900361183685, 0.35672100943179674, 0.4630793815027179, 0.575536895353397, 0.7310610262689123,
         0.955891499378532, 1.1540894925374654, 1.5534075193551768, 1.8219206274460034])

    Dr__20 = np.array(
        [49.934640522875824, 55.81699346405229, 60.915032679738566, 65.62091503267975, 70.58823529411767,
         75.8169934640523,
         80.13071895424837, 84.05228758169936, 86.79738562091505, 89.28104575163401, 92.02614379084969,
         94.3790849673203,
         97.90849673202615, 100.65359477124186, 102.61437908496734])

    tau_ratio__20 = np.array(
        [0.1329164519182754, 0.1409032310514415, 0.1493614428953823, 0.1571806436159647, 0.1690447971773845,
         0.18851071747809425, 0.2102042581799257, 0.2430288942783537, 0.28299709056179345, 0.33921607551474353,
         0.41254222786197, 0.4980849366160062, 0.6752892982483564, 0.8515401346572928, 1.0280820771956511])

    Dr__35 = np.array(
        [49.803921568627466, 54.50980392156863, 59.86928104575163, 68.36601307189544, 73.98692810457517,
         79.73856209150327,
         86.01307189542486, 89.67320261437911, 90.71895424836605])

    tau_ratio__35 = np.array(
        [0.11334235682977126, 0.11587084792110924, 0.11760669086084732, 0.12560331735945068, 0.13314813905997838,
         0.14635126244097282, 0.16680058070393153, 0.18598694809510716, 0.19146701634112345])

    if fines_content == 5:
        _cyclic_strength_ratio = 10 ** (
            np.interp(relative_density, Dr__5, np.log10(tau_ratio__5))
        )
    elif fines_content == 20:
        _cyclic_strength_ratio = 10 ** (
            np.interp(relative_density, Dr__20, np.log10(tau_ratio__20))
        )
    elif fines_content == 35:
        _cyclic_strength_ratio = 10 ** (
            np.interp(relative_density, Dr__35, np.log10(tau_ratio__35))
        )
    else:
        raise ValueError("Fines content needs to be 5%, 20% or 35%.")

    _reference_effective_stress = atmospheric_pressure * (
            (vertical_effective_stress / atmospheric_pressure) ** stress_exponent
    )
    _cyclic_shear_strength = _cyclic_strength_ratio * _reference_effective_stress

    return {
        'cyclic strength ratio [-]': _cyclic_strength_ratio,
        'reference effective stress [kPa]': _reference_effective_stress,
        'cyclic shear strength [kPa]': _cyclic_shear_strength,
    }


CYCLICSTRENGTH_DSSSAND_WATERCONTENT = {
    'water_content': {'type': 'float', 'min_value': 15.0, 'max_value': 40.0},
    'vertical_effective_stress': {'type': 'float', 'min_value': 100.0, 'max_value': 250.0},
    'fines_content': {'type': 'float', 'min_value': 5.0, 'max_value': 35.0},
    'stress_exponent': {'type': 'float', 'min_value': 0.2, 'max_value': 1.0},
    'atmospheric_pressure': {'type': 'float', 'min_value': None, 'max_value': None},
}

CYCLICSTRENGTH_DSSSAND_WATERCONTENT_ERRORRETURN = {
    'cyclic strength ratio [-]': np.nan,
    'reference effective stress [kPa]': np.nan,
    'cyclic shear strength [kPa]': np.nan,
}


@Validator(CYCLICSTRENGTH_DSSSAND_WATERCONTENT, CYCLICSTRENGTH_DSSSAND_WATERCONTENT_ERRORRETURN)
def cyclicstrength_dsssand_watercontent(
        water_content, vertical_effective_stress,
        fines_content=5.0, stress_exponent=0.9, atmospheric_pressure=100.0, **kwargs):
    """
    Calculates the DSS cyclic strength of sand, defined as the ratio of cyclic shear stress to reference normal stress for failure at 10 cycles. Cyclic failure is defined as reaching an accumulated cyclic shear strain of 15%. For certain very dense samples, this strain was not reached. The underlying dataset contains symmetrical DSS tests on normally consolidated sand.

    The curves are most representative for vertical effective stresses between 100 and 250kPa. The vertical effective stress was normalised using an exponent of 0.9 for the test data. Most of the underlying cyclic DSS test data is from normally consolidated pre-sheared samples.

    The function calculates the shear stress ratio at failure for <5% fines by default but fines content of 20% or 35% can also be entered. The trend for 35% fines content is slightly more reliable when formulated in terms of water content rather than relative density.

    :param water_content: Water content of the sand after testing (:math:`w_{after}`) [:math:`pct`] - Suggested range: 15.0 <= water_content <= 40.0
    :param vertical_effective_stress: Vertical effective stress at the depth considered (:math:`\\sigma_{vc}^{\\prime}`) [:math:`kPa`] - Suggested range: 100.0 <= vertical_effective_stress <= 250.0
    :param fines_content: Fines content of the soil (:math:`FC`) [:math:`pct`] - Suggested range: 5.0 <= fines_content <= 35.0 (optional, default= 5.0)
    :param stress_exponent: Stress exponent used for stress normalisation (:math:`n`) [:math:`-`] - Suggested range: 0.2 <= stress_exponent <= 1.0 (optional, default= 0.9)
    :param atmospheric_pressure: Atmospheric pressure (:math:`p_a`) [:math:`kPa`] (optional, default= 100.0)

    .. math::
        \\sigma_{ref}^{\\prime} = p_a \\cdot ( \\sigma_{ref}^{\\prime} / p_a )^n

    :returns: Dictionary with the following keys:

        - 'cyclic strength ratio [-]': Cyclic strength ratio for N=10 cycles (:math:`\\tau_f / \\sigma_{ref}^{\\prime}`)  [:math:`-`]
        - 'reference effective stress [kPa]': Reference effective normal stress (:math:`\\sigma_{ref}^{\\prime}`)  [:math:`kPa`]
        - 'cyclic shear strength [kPa]': Cyclic shear stress for failure at 10 cycles (:math:`\\tau_f`)  [:math:`kPa`]

    .. figure:: images/cyclicstrength_dsssand_watercontent_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Underlying data and trends for cyclic strength calculation

    Reference - Andersen, K.H. (2015). Cyclic soil parameters for offshore foundation design. The 3rd McClelland Lecture. Conference: Frontiers in Offshore Geotechnics III.

    """

    # Cyclic strength vs water content

    w__5 = np.array(
        [18.992133726647, 19.376229105211404, 19.952372173058013, 20.52851524090462, 21.264698049819728,
         22.192928548017047,
         22.993127253359557, 23.82533390691577, 24.689548508685675, 25.553763110455588, 26.225930022943302,
         27.12215257292691, 27.986367174696824, 28.850581776466733, 29.842828171091448, 30.931098410357254,
         32.05137659783678, 33.13964683710259, 34.32394092100951, 35.380203212061616, 36.08437807276303])

    tau_ratio__5 = np.array(
        [1.7383243948474292, 1.3978837870801986, 1.0850783042375325, 0.8725721008703046, 0.6893923352133043,
         0.5609490762023588, 0.4784619415227204, 0.41051566215537016, 0.3542994155012633, 0.31123344221272553,
         0.28491195899294514, 0.2608165234432463, 0.24016952344364234, 0.2224636521958442, 0.2060631000915769,
         0.1919993540062132, 0.17995241426176334, 0.17268279717540358, 0.16473356625277402, 0.15901272111525705,
         0.15530962900717166])

    w__20 = np.array(
        [16.943625040970176, 17.583784005244187, 18.127919124877092, 18.768078089151093, 19.376229105211404,
         19.952372173058013, 20.52851524090462, 21.264698049819728, 21.968872910521146, 22.83308751229105,
         23.857341855129466, 24.753564405113078, 25.841834644378892, 27.12215257292691, 28.24243076040642,
         29.55475663716814, 31.123146099639463, 32.65952761389708, 34.38795681743691, 35.380203212061616,
         36.08437807276303])

    tau_ratio__20 = np.array(
        [0.9039648713693308, 0.7665087963761721, 0.6615426488402057, 0.5642633058726779, 0.4927643237726035,
         0.4354250889772613, 0.38703123256102173, 0.337989696666263, 0.30219995613282113, 0.2717964026567041,
         0.2473487701735521, 0.22776791697355656, 0.20728057324422905, 0.1908716358840055, 0.1778447027709073,
         0.1666858922153185, 0.15530962900717166, 0.1464248083194707, 0.13968433021063545, 0.13563002151049214,
         0.13404144068474758])

    w__35 = np.array(
        [19.98438012127171, 20.75257087840053, 21.32871394624713, 22.096904703375944, 22.641039823008853,
         23.34521468371026,
         23.921357751556865, 24.625532612258283, 25.23368362831859, 25.905850540806288, 26.514001556866603,
         27.21817641756801, 27.79431948541461, 28.498494346116022, 29.138653310390037, 29.810820222877744,
         30.48298713536545, 31.123146099639463, 31.76330506391347, 32.46747992461488, 33.13964683710259,
         33.747797853162886,
         34.419964765650604, 34.96409988528352, 35.700282694198634, 36.50048139954114, 36.98060062274664])

    tau_ratio__35 = np.array(
        [0.20126429761612175, 0.19313373497362207, 0.1853316262579568, 0.17995241426176334, 0.17268279717540358,
         0.16767071481872425, 0.16280410711245055, 0.15715026797250906, 0.1525890140529748, 0.14816014958200405,
         0.14385983198332253, 0.14050962072505588, 0.13723742926787116, 0.13404144068474758, 0.13091988036130467,
         0.1293864675568639, 0.12637331236058502, 0.12415958533272822, 0.12126815389652547, 0.11914385577626233,
         0.1177483708925795, 0.11568573008446925, 0.11433074885997498, 0.1136592213015499, 0.11299163799492255,
         0.11166821160606556, 0.11101232259771704])

    if fines_content == 5:
        _cyclic_strength_ratio = 10 ** (
            np.interp(water_content, w__5, np.log10(tau_ratio__5))
        )
    elif fines_content == 20:
        _cyclic_strength_ratio = 10 ** (
            np.interp(water_content, w__20, np.log10(tau_ratio__20))
        )
    elif fines_content == 35:
        _cyclic_strength_ratio = 10 ** (
            np.interp(water_content, w__35, np.log10(tau_ratio__35))
        )
    else:
        raise ValueError("Fines content needs to be 5%, 20% or 35%.")

    _reference_effective_stress = atmospheric_pressure * (
            (vertical_effective_stress / atmospheric_pressure) ** stress_exponent
    )
    _cyclic_shear_strength = _cyclic_strength_ratio * _reference_effective_stress

    return {
        'cyclic strength ratio [-]': _cyclic_strength_ratio,
        'reference effective stress [kPa]': _reference_effective_stress,
        'cyclic shear strength [kPa]': _cyclic_shear_strength,
    }


