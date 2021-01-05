
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