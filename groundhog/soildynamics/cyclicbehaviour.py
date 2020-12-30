
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

    Strain contours for cyclic shear strains of 0.5, 1, 3 and 15% are defined and spline interpolation is used to obtain the accumulated strain for a sample tested at a certain ratio of cyclic shear stress to DSS shear strength with a given number of cycles.

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