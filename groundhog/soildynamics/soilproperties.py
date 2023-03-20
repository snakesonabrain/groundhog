#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party packages
import numpy as np

# Project imports
from groundhog.general.validation import Validator

MODULUSREDUCTION_PLASTICITY_ISHIBASHI = {
    'strain': {'type': 'float', 'min_value': 0.0, 'max_value': 10.0},
    'pi': {'type': 'float', 'min_value': 0.0, 'max_value': 200.0},
    'sigma_m_eff': {'type': 'float', 'min_value': 0.0, 'max_value': 400.0},
    'multiplier_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_1': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_2': {'type': 'float', 'min_value': None, 'max_value': None},
    'multiplier_3': {'type': 'float', 'min_value': None, 'max_value': None},
    'exponent_3': {'type': 'float', 'min_value': None, 'max_value': None},
}

MODULUSREDUCTION_PLASTICITY_ISHIBASHI_ERRORRETURN = {
    'G/Gmax [-]': np.nan,
    'K [-]': np.nan,
    'm [-]': np.nan,
    'n [-]': np.nan,
    'dampingratio [pct]': np.nan,
}


@Validator(MODULUSREDUCTION_PLASTICITY_ISHIBASHI, MODULUSREDUCTION_PLASTICITY_ISHIBASHI_ERRORRETURN)
def modulusreduction_plasticity_ishibashi(
        strain, pi, sigma_m_eff,
        multiplier_1=0.000102, exponent_1=0.492, multiplier_2=0.000556, exponent_2=0.4, multiplier_3=-0.0145,
        exponent_3=1.3, **kwargs):
    """
    Calculates the modulus reduction curve (G/Gmax) as a function of shear strain. The curve depends on the plasticity of the material (plasticity index) and the mean effective stress at the depth of interest.

    The curve for cohesionless soils can be established by using a plasticity index of 0. At low plasticity, the effect of confining pressure on the modulus reduction curve is more pronounced.

    Also calculates the damping ratio of plastic and non-plastic soils based on a fit to empirical data.

    :param strain: Strain amplitude (:math:`\\gamma`) [:math:`pct`] - Suggested range: 0.0 <= strain <= 10.0
    :param PI: Plasticity index (:math:`PI`) [:math:`pct`] - Suggested range: 0.0 <= PI <= 200.0
    :param sigma_m_eff: Mean effective pressure (:math:`\\sigma_m^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= sigma_m_eff <= 400.0
    :param multiplier_1: Multiplier in equation for K (:math:``) [:math:`-`] (optional, default= 0.000102)
    :param exponent_1: Exponent in equation for K (:math:``) [:math:`-`] (optional, default= 0.492)
    :param multiplier_2: First multiplier in equation for m (:math:``) [:math:`-`] (optional, default= 0.000556)
    :param exponent_2: First exponent in equation for m (:math:``) [:math:`-`] (optional, default= 0.4)
    :param multiplier_3: Second multiplier in equation for m (:math:``) [:math:`-`] (optional, default= -0.0145)
    :param exponent_3: Second exponent in equation for m (:math:``) [:math:`-`] (optional, default= 1.3)

    .. math::
        \\frac{G}{G_{max}} = K \\left( \\gamma, \\text{PI} \\right) \\left( \\sigma_m^{\\prime} \\right)^{m \\left( \\gamma, \\text{PI} \\right) - m_0}

        K \\left( \\gamma, \\text{PI} \\right) = 0.5 \\left[ 1 + \\tanh \\left[ \\ln \\left( \\frac{0.000102 + n ( \\text{PI} )}{\\gamma} \\right)^{0.492} \\right] \\right]

        m \\left( \\gamma, \\text{PI} \\right) - m_0 = 0.272 \\left[ 1 - \\tanh \\left[ \\ln \\left( \\frac{0.000556}{\\gamma} \\right)^{0.4} \\right] \\right] \\exp \\left( -0.0145 \\text{PI}^{1.3} \\right)

        n ( \\text{PI} ) = \\begin{cases}
            0.0       & \\quad \\text{for PI } = 0 \\\\
            3.37 \\times 10^{-6} \\text{PI}^{1.404}  & \\quad \\text{for } 0 < \\text{PI} \\leq 15 \\\\
            7.0 \\times 10^{-7} \\text{PI}^{1.976}  & \\quad \\text{for } 15 < \\text{PI} \\leq 70 \\\\
            2.7 \\times 10^{-5} \\text{PI}^{1.115}  & \\quad \\text{for } \\text{PI} > 70
          \\end{cases}

        \\xi = 0.333 \\frac{1 + \\exp(-0.0145 PI^{1.3})}{2} \\left[ 0.586 \\left( \\frac{G}{G_{max}} \\right)^2 - 1.547 \\frac{G}{G_{max}} + 1 \\right]

    :returns: Dictionary with the following keys:

        - 'G/Gmax [-]': Modulus reduction ratio (:math:`G / G_{max}`)  [:math:`-`]
        - 'K [-]': Factor K in the equation (:math:`K ( \\gamma, \\text{PI} )`)  [:math:`-`]
        - 'm [-]': Exponent m in the equation (:math:`m \\left( \\gamma, \\text{PI} \\right) - m_0`)  [:math:`-`]
        - 'n [-]': Factor n in equations (:math:`n ( \\text{PI} )`)  [:math:`-`]
        - 'dampingratio [pct]': Damping ratio (:math:`\\xi`)  [:math:`pct`]

    Reference - Ishibashi, I., & Zhang, X. (1993). Unified dynamic shear moduli and damping ratios of sand and clay. Soils and foundations, 33(1), 182-191.

    """

    strain = 0.01 * strain

    if pi == 0:
        _n = 0
    elif 0 < pi <= 15:
        _n = 3.37e-6 * (pi ** 1.404)
    elif 15 < pi <= 70:
        _n = 7e-7 * (pi ** 1.976)
    else:
        _n = 2.7e-5 * (pi ** 1.115)

    _m = 0.272 * (1 - np.tanh(np.log((multiplier_2 / strain) ** exponent_2))) * \
        np.exp(multiplier_3 * (pi ** exponent_3))

    _K = 0.5 * (1 + np.tanh(np.log(((multiplier_1 + _n) / strain) ** exponent_1)))

    _G_over_Gmax = min(_K * (sigma_m_eff ** _m), 1)

    _damping = 100 * 0.333 * 0.5 * (1 + np.exp(-0.0145 * (pi ** 1.3))) * \
               (0.586 * (_G_over_Gmax ** 2) - 1.547 * _G_over_Gmax + 1)
    return {
        'G/Gmax [-]': _G_over_Gmax,
        'K [-]': _K,
        'm [-]': _m,
        'n [-]': _n,
        'dampingratio [pct]': _damping,
    }


GMAX_SHEARWAVEVELOCITY = {
    'Vs': {'type': 'float', 'min_value': 0.0, 'max_value': 600.0},
    'gamma': {'type': 'float', 'min_value': 12.0, 'max_value': 22.0},
    'g': {'type': 'float', 'min_value': 9.7, 'max_value': 10.2},
}

GMAX_SHEARWAVEVELOCITY_ERRORRETURN = {
    'rho [kg/m3]': np.nan,
    'Gmax [kPa]': np.nan,
}


@Validator(GMAX_SHEARWAVEVELOCITY, GMAX_SHEARWAVEVELOCITY_ERRORRETURN)
def gmax_shearwavevelocity(
        Vs, gamma,
        g=9.81, **kwargs):
    """
    Calculates the small-strain shear modulus (shear strain < 1e-4%) from the shear wave velocity and the bulk unit weight if the soil based on elastic theory.

    Often, the result of an in-situ or laboratory test will provide the shear wave velocity, which is then converted to the small-strain shear modulus using this function.

    :param Vs: Shear wave velocity (:math:`V_s`) [:math:`m/s`] - Suggested range: 0.0 <= Vs <= 600.0
    :param gamma: Bulk unit weight (:math:`\\gamma`) [:math:`kN/m3`] - Suggested range: 12.0 <= gamma <= 22.0
    :param g: Acceleration due to gravity (:math:`g`) [:math:`m/s2`] - Suggested range: 9.7 <= g <= 10.2 (optional, default= 9.81)

    .. math::
        G_{max} = \\rho \\cdot V_s^2

        \\rho = \\gamma / g

    :returns: Dictionary with the following keys:

        - 'rho [kg/m3]': Density of the material (:math:`\\rho`)  [:math:`kg/m3`]
        - 'Gmax [kPa]': Small-strain shear modulus (:math:`G_{max}`)  [:math:`kPa`]

    Reference - Robertson, P.K. and Cabal, K.L. (2015). Guide to Cone Penetration Testing for Geotechnical Engineering. 6th edition. Gregg Drilling & Testing, Inc.

    """

    _rho = 1000 * gamma / g
    _Gmax = 1e-3 * _rho * (Vs ** 2)

    return {
        'rho [kg/m3]': _rho,
        'Gmax [kPa]': _Gmax,
    }


DAMPINGRATIO_SANDGRAVEL_SEED = {
    'cyclic_shear_strain': {'type': 'float', 'min_value': 0.0001, 'max_value': 1.0},
}

DAMPINGRATIO_SANDGRAVEL_SEED_ERRORRETURN = {
    'D LE [pct]': np.nan,
    'D BE [pct]': np.nan,
    'D HE [pct]': np.nan,
}

@Validator(DAMPINGRATIO_SANDGRAVEL_SEED, DAMPINGRATIO_SANDGRAVEL_SEED_ERRORRETURN)
def dampingratio_sandgravel_seed(
        cyclic_shear_strain,
        **kwargs):

    """
    Damping ratios for sand are compiled from a dataset comprising several sands and gravels. Average values and upper and lower bounds are provided. The comparison of the trends proposed for sand with the datapoints measured on gravel suggests that the trend is applicable for gravels too.

    :param cyclic_shear_strain: Cyclic shear strain (:math:`\\gamma_{cyc}`) [:math:`pct`] - Suggested range: 0.0001 <= cyclic_shear_strain <= 1.0

    :returns: Dictionary with the following keys:

        - 'D LE [pct]': Low estimate damping ratio (:math:`D_{LE}`)  [:math:`pct`]
        - 'D BE [pct]': Average or best estimate damping ratio (:math:`D_{BE}`)  [:math:`pct`]
        - 'D HE [pct]': High estimate damping ratio (:math:`D_{HE}`)  [:math:`pct`]

    .. figure:: images/dampingratio_sandgravel_seed_1.png
        :figwidth: 500.0
        :width: 450.0
        :align: center

        Proposed trends and measurement data on gravels

    Reference - Seed, H. B., Wong, R. T., Idriss, I. M., & Tokimatsu, K. (1986). Moduli and damping factors for dynamic analyses of cohesionless soils. Journal of geotechnical engineering, 112(11), 1016-1032.

    """

    gamma_cyc_le = np.array(
        [0.0001, 0.00010736704570827216, 0.00012251883726624774, 0.0001456046695997414, 0.0001730404914240103,
         0.00020774487550410804, 0.00021704006888420929, 0.00023242841823261729, 0.00026473594476528894,
         0.00028545466903303663, 0.0002994757536988713, 0.00033877427235285735, 0.0003652873729599878,
         0.00038851611499061847, 0.00043055932487400233, 0.00046425569330965606, 0.0004988776087409925,
         0.0005780361825306242, 0.0006232743623828213, 0.0006674651133236992, 0.0007654682597921579,
         0.0008253752203841818, 0.0008808729401071887, 0.0009888154107768829, 0.001063466628084799,
         0.0011466954916785786, 0.0011948068117258286, 0.0013423677618519564, 0.0014474239435819526,
         0.001544747835016312, 0.0018021607982596848, 0.0019432012327136867, 0.002031677764622028, 0.002322020118922107,
         0.002503745704508012, 0.0026996934702496465, 0.0028967682011055013, 0.003160369520630482, 0.003407705879658681,
         0.003587352758504253, 0.0039619645139493676, 0.004272035178497375, 0.004497247627352388, 0.005104850422187586,
         0.005504365449960532, 0.005794543683655353, 0.006399642854956974, 0.006900490731286996, 0.007364475463440597,
         0.00859167346223717, 0.009200831362914412, 0.009989097232837956, 0.010994494638962928, 0.011534526934504477,
         0.012782733940858966, 0.013689042054936413, 0.015274674020277513, 0.016470098236609144, 0.0182524073101425,
         0.020089502548822056, 0.02467272437991229, 0.02771985351909183, 0.029845419300759967, 0.033033086887161785,
         0.03535094067449866, 0.03734224475575858, 0.04096019231635378, 0.04416581266121802, 0.0492816362557288,
         0.052236254697168326, 0.0569060794502507, 0.060317811781970765, 0.06437354416877875, 0.07158447562120226,
         0.07718680897099323, 0.08943429734581976, 0.09577527492826368, 0.11021469138190157, 0.11843396317394378,
         0.12510530005291384, 0.13864355029454234, 0.14446054728207172, 0.15845677830834998, 0.16567152579104538,
         0.17986515724109198, 0.18805465663823706, 0.19594477030621169, 0.2127320114510672, 0.22471511895922835,
         0.2456441316685955, 0.2577097853132104, 0.2648686953618417, 0.2836481402663653, 0.3320498963900685,
         0.34361965946695683, 0.3730587002567181, 0.3927255475416774, 0.4220130950950124, 0.4442607121262613,
         0.4856372698471063, 0.5112390156726162, 0.5607710125231429, 0.5943913324010771, 0.6740809134497746,
         0.7250124447179741, 0.8007223631227072, 0.8604363145928147, 0.9502879628574636, 1.0])

    damping_le = np.array(
        [0.4534883720930267, 0.4529499243609152, 0.4519499500012856, 0.4506422912233034, 0.4493346324453249,
         0.4828337734929846, 0.4963895688177154, 0.502407173529626, 0.519485771203005, 0.5027180916858249,
         0.5253259597629274, 0.5497060051741833, 0.5726665903384252, 0.5760196208360711, 0.5949526272379515,
         0.6157739981501145, 0.6371561167893276, 0.6654547137022, 0.6873456917404006, 0.6957402011768394,
         0.7467566480829397, 0.7707868403732192, 0.7902153339791624, 0.8540509495685562, 0.8587139708224107,
         0.8891618058689197, 0.8952681186568192, 0.9574928375161492, 0.9858014583105792, 1.0168619294121868,
         1.0836144945343875, 1.127967222219393, 1.150091697733842, 1.2164651242771818, 1.255469816331999, 1.295544115512854,
         1.3322938553890538, 1.3852672896808509, 1.436037660122082, 1.4546340241488522, 1.5354391867524733,
         1.5958360213280578, 1.6352897243125746, 1.7410231010658812, 1.8078375783976883, 1.8569177455165509,
         1.9543018185737682, 2.024325117283695, 2.0845323825698467, 2.257978258514367, 2.3479006000021663,
         2.4816681331904107, 2.6144286659272207, 2.6900075535795587, 2.8924545826060135, 3.018149340197983,
         3.246215051921304, 3.4167814204788307, 3.6677886130274224, 3.9455641030077793, 4.404651588779963,
         4.781935028875684, 4.965347801873111, 5.310490057916855, 5.540583659842966, 5.726458461003531, 6.069993195304786,
         6.343241847961984, 6.759558413868043, 6.996570145060081, 7.360657570938454, 7.577881570298783, 7.923406536455651,
         8.345275350010269, 8.726696936680806, 9.420148551706026, 9.702362485408274, 10.355898335818587, 10.74231359752742,
         11.022220447379462, 11.529420617872384, 11.806823440339912, 12.249383421043241, 12.467271467459032,
         12.917106208602505, 13.12659019902799, 13.435688197102575, 13.798260823546753, 14.07580986880152,
         14.566352393169891, 14.828265755570815, 14.967285396497914, 15.384448095935323, 16.285290007349026,
         16.483086151880176, 16.917793592241107, 17.153894565347766, 17.552206235202867, 17.79133266846603,
         18.22766903169697, 18.456710597771767, 18.911603318417228, 19.16150975773984, 19.725081461367047,
         19.977719644082526, 20.363309357250767, 20.60470966171605, 20.9318988258026, 21.104574241686112])

    gamma_cyc_be = np.array(
        [0.0001, 0.00011235607753106597, 0.00012004861395727114, 0.00013551207945889716, 0.000188648345808323,
         0.00020234025531060845, 0.00021853190388122838, 0.00023563461452460284, 0.0002540758149086333,
         0.00027474330647545226, 0.0002954008937595277, 0.0003185195136042041, 0.00034344743936099386,
         0.00037032627065416437, 0.0003993086889562561, 0.00043055932487400233, 0.00046425569330965606,
         0.0005005892018097682, 0.0005397662378292022, 0.000582009341086441, 0.0006275584676695892,
         0.0006766723530736442, 0.0007296299819115822, 0.0007867321726477517, 0.0008483032863554371,
         0.0009146930692049297, 0.000982647472538318, 0.001063466628084799, 0.0011466954916785786, 0.00123643799994364,
         0.0013332039227491347, 0.001437542925496226, 0.0015500476914161474, 0.001671357288225084,
         0.0018021607982596848, 0.0019432012327136867, 0.0020952797522099218, 0.0022592602176821127,
         0.0024360740974170613, 0.00262672575813104, 0.0030331108307883967, 0.003547820971844297, 0.004128194645775764,
         0.00445127478763278, 0.004799639778441663, 0.005175268457207639, 0.005580294530533349, 0.006477673843176275,
         0.008143353805556489, 0.008785091473186155, 0.009303207942506421, 0.010994494638962928, 0.011854944107167406,
         0.012782733940858966, 0.013689042054936413, 0.014760371783422216, 0.016583306222028756, 0.01763784402092674,
         0.02027382798670632, 0.024504292983635444, 0.025973417011258847, 0.027530620519893242, 0.03081321999038474,
         0.033123722073301484, 0.03535094067449866, 0.03759891840998422, 0.04012704644711407, 0.05081101695004028,
         0.057297226031172224, 0.06052475542257685, 0.06415344091522583, 0.06823297894393368, 0.0714008281326666,
         0.07587623271087471, 0.087017054005354, 0.09297411230021717, 0.11128944438907216, 0.12442159494433225,
         0.13125027575125686, 0.13807489209485932, 0.1435972726387898, 0.15176399757025846, 0.1600933225387565,
         0.1673825840529291, 0.17728991405192868, 0.18562297266826935, 0.20059360979178895, 0.21358375759392,
         0.22625970876951595, 0.2364369961823236, 0.2657774301199742, 0.2807485644436065, 0.3331891223067882,
         0.3543767262109732, 0.3769116566908065, 0.4022549635490181, 0.4293023386977912, 0.459740286851477,
         0.4923363147645632, 0.5272434323648003, 0.5685064777243272, 0.6025905676771414, 1.0])

    damping_be = np.array(
        [0.6986440349973506, 0.7362373606667489, 0.7407480828812112, 0.7729599329917427, 0.8186545698812111,
         0.8516093206127415, 0.8610737104687196, 0.8904519383891873, 0.9166213449315456, 0.9428434096098536,
         0.9892826934109812, 1.0165217070793735, 1.053387184882112, 1.0881134484327717, 1.1271181404875892,
         1.172540475298632, 1.2158235958576, 1.259106716416568, 1.3055986583536523, 1.3456729575345037,
         1.3953737208496977, 1.4397264485347063, 1.487287997597825, 1.5466152250473648, 1.6091512738750249,
         1.665269679946455, 1.7610737670214751, 1.795689813231956, 1.8667827190679205, 1.941084446281998,
         2.015386173496076, 2.093966329214308, 2.1704072706804642, 2.245778605020579, 2.3307764034950367,
         2.421122237599693, 2.510398464578305, 2.60609233431315, 2.704995025426108, 2.8060369307911444,
         3.0338432008741916, 3.208051218844247, 3.485021647551964, 3.632056659336644, 3.7753480461801945,
         3.9421707897965814, 4.038934266657467, 4.410357087416075, 4.872586516988363, 5.041283899108514,
         5.224062563648147, 5.627511939977103, 5.848349843458428, 6.076674996822021, 6.257632789259301,
         6.5034281923481885, 6.986959273630376, 7.242247227037264, 7.836749412144936, 8.58936308987851,
         8.863781359172831, 9.136238682069413, 9.653075256343936, 9.936211125087073, 10.206851707897634,
         10.4817015871874, 10.766793750723105, 11.8840614853599, 12.520696060655148, 12.764558848148788,
         13.042572185838962, 13.330364311353787, 13.578570507093573, 13.89235165841942, 14.624351308903234,
         14.939264978677109, 15.851867343610868, 16.45195445225551, 16.719807190507087, 16.998269794637665,
         17.226521036615786, 17.504654474165264, 17.77079584101518, 18.037191843938626, 18.375354933730755,
         18.583218233193925, 18.93216185095902, 19.245739648268426, 19.5441511525224, 19.81200207088116,
         20.356593802114066, 20.628656866375167, 21.420329030058628, 21.683936150001365, 21.946235972345615,
         22.21883750155392, 22.470783728706056, 22.73934319366042, 22.997894191935426, 23.2402949715018,
         23.499638731520506, 23.675504588674016, 23.915041650104733])

    gamma_cyc_he = np.array(
        [0.0001, 0.00011015945244167936, 0.00011878073476376348, 0.00012962130074026585, 0.00013810025650511809,
         0.0001489082377876431, 0.00015837736072440495, 0.00017916033092457522, 0.00019318174950993465,
         0.0002083005102252808, 0.0002200339930754089, 0.00024720861642004354, 0.00026655561958109636,
         0.00033877427235285735, 0.0003652873729599878, 0.0003911865985572175, 0.0004486240836775954,
         0.00048373423352078283, 0.0005205213923692329, 0.0005472119560568889, 0.0006105967251195091,
         0.0006583831532063964, 0.0006978556860562095, 0.0007867321726477517, 0.0008483032863554371,
         0.000905342675064425, 0.001063466628084799, 0.0011466954916785786, 0.0012217046166771036,
         0.0013332039227491347, 0.001437542925496226, 0.0015185191326326875, 0.0016486157247629332,
         0.0017776394380147169, 0.0021096817549321007, 0.0022747893477521397, 0.0024832491772025485,
         0.002644780679665561, 0.003225987328182171, 0.003419397335963503, 0.0038024279755937366, 0.0041000130158274795,
         0.004638034661967479, 0.004916102189457339, 0.005392404566036912, 0.005774731229112175, 0.006201841151930438,
         0.006924165540298022, 0.007595021510246519, 0.008050371460496924, 0.010011931835549694, 0.011455785063641071,
         0.012142602230882892, 0.013456612009179428, 0.014410698725553015, 0.015743860333017508, 0.016812061962493838,
         0.01850418719610731, 0.019680874436571974, 0.02181063714833311, 0.02327717091085021, 0.025358107025649997,
         0.026878420398955924, 0.02829539255074718, 0.029940489927209605, 0.03233902143328173, 0.03427786673884472,
         0.03696051593511605, 0.03998984577574319, 0.04224245749081527, 0.04649413871882389, 0.04962037587212595,
         0.054241807140663435, 0.057493806381175336, 0.059497088795807535, 0.06393409022925085, 0.06776717809686499,
         0.07510059254738255, 0.0925504965751776, 0.10082434014077037, 0.10871503932295457, 0.11843396317394378,
         0.12510530005291384, 0.13722627505333002, 0.14446054728207172, 0.15900042572547404, 0.17085789860592426,
         0.1780264966357266, 0.19130284068466888, 1.0])

    damping_he = np.array(
        [0.8935734930337595, 0.8897939000470387, 0.914893699463356, 0.9297414266082099, 0.955466834161644,
         0.9827058478300366, 1.0164662809107696, 1.0668734330556229, 1.0994604823542071, 1.1384651744090206,
         1.164576324510335, 1.2345022146771392, 1.2927598350006413, 1.4642200979369235, 1.526756146764587,
         1.5861352625421787, 1.7262856366181405, 1.8048657923363685, 1.8804666146001097, 1.9365630860601328,
         2.0758514063222755, 2.170475668931079, 2.2367067956656683, 2.404440140130081, 2.5204565452596555,
         2.6210414795537496, 2.9027331886815912, 3.0529770218443915, 3.183523927749576, 3.3727176164386736,
         3.5507712348784644, 3.6795646690794532, 3.9309414480374194, 4.026360275939851, 4.4869805451690254,
         4.694983163137883, 4.934766674472193, 5.093874685059003, 5.7191596994038605, 5.905365092109072,
         6.248439514045536, 6.491739167173666, 6.9336225274485335, 7.1497769196828225, 7.51434726079616,
         7.789416615047254, 8.025755323863304, 8.488469839332105, 8.902242108243207, 9.148167232152225,
         10.040707011077169, 10.641518816876305, 10.886374333659287, 11.341319951572515, 11.609058535502374,
         12.004760943127547, 12.280887851824437, 12.768306338508301, 13.047601029632936, 13.472169805166676,
         13.728248875408589, 14.146425739424538, 14.427217084511506, 14.68420213675406, 14.910263366986214,
         15.259708555291633, 15.592068121719082, 15.909017865525847, 16.356720236454017, 16.60779650533933,
         17.097110573541013, 17.39258981072291, 17.849469421930138, 18.12029262282861, 18.27298700021181,
         18.64175374434101, 18.922054852828538, 19.44190779650686, 20.52512074972923, 20.933917753476006,
         21.280434494266828, 21.704003960876555, 21.929004311591964, 22.369348677534212, 22.57368231899753,
         22.978764826023692, 23.259526672727194, 23.38275496581629, 23.694000615611877, 23.95869241126077])

    _D_LE = np.interp(np.log10(cyclic_shear_strain), np.log10(gamma_cyc_le), damping_le)
    _D_BE = np.interp(np.log10(cyclic_shear_strain), np.log10(gamma_cyc_be), damping_be)
    _D_HE = np.interp(np.log10(cyclic_shear_strain), np.log10(gamma_cyc_he), damping_he)

    return {
        'D LE [pct]': _D_LE,
        'D BE [pct]': _D_BE,
        'D HE [pct]': _D_HE,
    }


MODULUSREDUCTION_DARENDELI = {
    'mean_effective_stress': {'type': 'float', 'min_value': 0.0, 'max_value': 1600.0},
    'pi': {'type': 'float', 'min_value': 0.0, 'max_value': 60.0},
    'ocr': {'type': 'float', 'min_value': 1.0, 'max_value': 20.0},
    'N': {'type': 'float', 'min_value': 1.0, 'max_value': None},
    'frequency': {'type': 'float', 'min_value': 0.05, 'max_value': 100.0},
    'soiltype': {'type': 'string', 'options': ('sand', 'fine sand', 'silt', 'clay', 'all'), 'regex': None},
    'min_strain': {'type': 'float', 'min_value': None, 'max_value': None},
    'max_strain': {'type': 'float', 'min_value': None, 'max_value': None},
    'no_points': {'type': 'int', 'min_value': 10.0, 'max_value': None},
}

MODULUSREDUCTION_DARENDELI_ERRORRETURN = {
    'strains [pct]': np.nan,
    'G/Gmax [-]': np.nan,
    'D [pct]': np.nan,
    'sigma_ND [-]': np.nan,
    'sigma_D [pct]': np.nan,
}

@Validator(MODULUSREDUCTION_DARENDELI, MODULUSREDUCTION_DARENDELI_ERRORRETURN)
def modulusreduction_darendeli(
        mean_effective_stress, pi, ocr,N,frequency,soiltype,
        min_strain=0.0001,max_strain=1.0,no_points=250,custom_coefficients=None, **kwargs):

    """ 
    Darendeli (2001) proposed a comprehensive framework for estimating the modulus reduction curve and damping curve for sand, fine sand, silt and clay based on extensive laboratory testing. The framework is initially based on the work by Hardin and Drnevich but extends the formulation to include the effect of soil type, plasticity, overconsolidation ratio, stress ratio, loading frequency and number of cycles applied.

    The author used a Bayesian approach to calibrate the parameters of the parametric equations and also formulated expressions to estimate the standard deviation on the estimates. Parameters for individual soil types can be used as well as parameters calibrated to the entire credible dataset (using ``'all'`` for ``soiltype``).

    The formulation is based on Masing damping but takes into account the damping at small strains, which  is not zero but difficult to estimate from resonant column or cyclic DSS tests. The Masing damping at large strains is also adjusted to be in line with experimental observations (Masing damping overestimates the damping ratio at large strains).

    :param mean_effective_stress: Mean effective stress at the depth under consideration (:math:`\\sigma_0^{\\prime}`) [:math:`kPa`] - Suggested range: 0.0 <= mean_effective_stress <= 1000.0
    :param pi: Plasticity index (difference between liquid limit and plastic limit) (:math:`PI`) [:math:`pct`] - Suggested range: 0.0 <= plasticity_index <= 60.0
    :param ocr: Overconsolidation ratio of the soil (:math:`OCR`) [:math:`-`] - Suggested range: 1.0 <= OCR <= 20.0
    :param N: Number of cycles (:math:`N`) [:math:`-`] - Suggested range: N >= 1.0
    :param frequency: Loading frequency (:math:`f`) [:math:`Hz`] - Suggested range: 0.05 <= frequency <= 20.0
    :param soiltype: Soil type used for calculating modulus reduction and damping curves - Options: ('sand', 'fine sand', 'silt', 'clay', 'all')
    :param min_strain: Minimum value for the strain (:math:`\\gamma_{min}`) [:math:`pct`] (optional, default= 0.0001)
    :param max_strain: Maximum value for the strain (:math:`\\gamma_{max}`) [:math:`pct`] (optional, default= 1.0)
    :param no_points: Number of points used for the strain curve calculation (:math:``) [:math:`-`] - Suggested range: no_points >= 10.0 (optional, default= 250)
    :param custom_coefficients: Dictionary with custom calibration coefficients (:math:``) [:math:`-`] (optional, default= None)- Elementtype: float, order: ascending, unique: True, empty entries allowed: False

    .. math::
        \\frac{G}{G_{max}} = \\frac{1}{1 + \\left( \\frac{\\gamma}{\\gamma_r} \\right)^a}

        \\gamma_r = \\left( \\phi_1 + \\phi_2 \\cdot PI \\cdot OCR^{\\phi_3} \\right) \\cdot \\sigma_0^{\\prime \\phi_4}

        a = \\phi_5

        D_{\\text{adjusted}} = b \\cdot \\left( \\frac{G}{G_{max}} \\right)^{0.1} \\cdot D_{\\text{Masing}} + D_{\\text{min}}

        D_{\\text{min}} = \\left( \\phi_6 + \\phi_7 \\cdot PI \\cdot OCR^{\\phi_8} \\right) \\cdot \\sigma_0^{\\prime \\phi_9} \\cdot \\left[1 + \\phi_{10} \\cdot \\ln(f) \\right]

        b = \\phi_{11} + \\phi_{12} \\cdot \\ln(N)

        \\sigma_{\\text{NG}} = \\exp(\\phi_{13}) + \\sqrt{\\frac{0.25}{\\exp(\\phi_{14})} - \\frac{\\left(G / G_{max} - 0.5 \\right)^2}{\\exp(\\phi_{14})}}

        \\sigma_D = \\exp (\\phi_{15} ) + \\exp (\\phi_{16} ) \\cdot \\sqrt{D}

        \\rho_{i,j} = \\exp \\left( \\frac{-1}{\\exp ( \\phi_{17} )} \\right) \\cdot \\exp \\left( \\frac{- | \\ln \\gamma_i - \\ln \\gamma_j |}{\\exp ( \\phi_{18} )} \\right)

    :returns: Dictionary with the following keys:

        - 'strains [pct]': List of strains for the modulus reduction curve (:math:`\\gamma`)  [:math:`pct`]
        - 'G/Gmax [-]': Modulus ratio for given strains (:math:`G / G_{max}`)  [:math:`-`]
        - 'D [pct]': Damping ratios for given strains (:math:`D`)  [:math:`pct`]
        - 'sigma_ND [-]': Standard deviation for the modulus reduction curve (:math:`\\sigma_{ND}`)  [:math:`-`]
        - 'sigma_D [pct]': Standard deviation for the damping curve (:math:`\\sigma_D`)  [:math:`pct`]

    Reference - Darendeli, M. B. (2001). Development of a new family of normalized modulus reduction and material damping curves. The university of Texas at Austin.

    """

    if custom_coefficients is None:
        parameters = {
            'clay': {
                'phi1': 2.58e-2, 'phi2': 1.95e-3, 'phi3': 9.92e-2,
                'phi4': 2.26e-1, 'phi5': 9.75e-1, 'phi6': 9.58e-1,
                'phi7': 5.65e-3, 'phi8': -1.0e-1, 'phi9': -1.96e-1,
                'phi10': 3.68e-1, 'phi11': 4.66e-1, 'phi12': 2.23e-2,
                'phi13': -5.65e0, 'phi14': 4.0e0, 'phi15': -5.0e0,
                'phi16': -7.25e-1, 'phi17': 7.67e0, 'phi18': 2.16e0},
            'silt': {
                'phi1': 4.16e-2, 'phi2': 6.89e-4, 'phi3': 3.21e-1,
                'phi4': 2.80e-1, 'phi5': 1.0e0, 'phi6': 7.12e-1,
                'phi7': 3.03e-3, 'phi8': -1.0e-1, 'phi9': -1.89e-1,
                'phi10': 2.34e-1, 'phi11': 5.92e-1, 'phi12': -7.67e-4,
                'phi13': -5.02e0, 'phi14': 3.93e0, 'phi15': -5.2e0,
                'phi16': -6.42e-1, 'phi17': 4.06e0, 'phi18': 1.94e0},
            'fine sand': {
                'phi1': 3.34e-2, 'phi2': -5.79e-5, 'phi3': 2.49e-1,
                'phi4': 4.82e-1, 'phi5': 8.45e-1, 'phi6': 8.89e-1,
                'phi7': 2.02e-2, 'phi8': -1.0e-1, 'phi9': -3.72e-1,
                'phi10': 2.33e-1, 'phi11': 7.76e-1, 'phi12': -2.94e-2,
                'phi13': -3.98e0, 'phi14': 4.32e0, 'phi15': -5.34e0,
                'phi16': -2.66e-1, 'phi17': 4.92e0, 'phi18': 2.68e0},
            'sand': {
                'phi1': 4.74e-2, 'phi2': -2.34e-3, 'phi3': 2.5e-1,
                'phi4': 2.34e-1, 'phi5': 8.95e-1, 'phi6': 6.88e-1,
                'phi7': 1.22e-2, 'phi8': -1.0e-1, 'phi9': -1.27e-1,
                'phi10': 2.88e-1, 'phi11': 7.67e-1, 'phi12': -2.83e-2,
                'phi13': -4.14e0, 'phi14': 3.61e0, 'phi15': -5.15e0,
                'phi16': -2.32e-1, 'phi17': 5.15e0, 'phi18': 3.12e0},
            'all': {
                'phi1': 3.52e-2, 'phi2': 1.01e-3, 'phi3': 3.25e-1,
                'phi4': 3.48e-1, 'phi5': 9.19e-1, 'phi6': 8.01e-1,
                'phi7': 1.29e-2, 'phi8': -1.07e-1, 'phi9': -2.89e-1,
                'phi10': 2.92e-1, 'phi11': 6.33e-1, 'phi12': -5.66e-3,
                'phi13': -4.23e0, 'phi14': 3.62e0, 'phi15': -5.0e0,
                'phi16': -2.5e-1, 'phi17': 5.62e0, 'phi18': 2.78e0
            }
        }
        params = parameters[soiltype]
    else:
        params = custom_coefficients

    _gamma = np.logspace(np.log10(min_strain), np.log10(max_strain), no_points)

    sigma_0_eff = mean_effective_stress / 100 # Stresses in formulae are in atm

    _gamma_r = \
        (params['phi1'] + params['phi2'] * pi * (ocr ** params['phi3'])) * (sigma_0_eff ** params['phi4'])
    _a = params['phi5']
    _Dmin = \
        (params['phi6'] + params['phi7'] * pi * (ocr ** params['phi8'])) * \
        (sigma_0_eff ** params['phi9']) * \
        (1 + params['phi10'] * np.log(frequency))
    _b = params['phi11'] + params['phi12'] * np.log(N)

    _G_Gmax = 1 / (1 + ((_gamma / _gamma_r) ** _a))

    _D_masing_a1 = (100 / np.pi) * (4 * (
        (_gamma - _gamma_r * np.log((_gamma + _gamma_r) / _gamma_r)) /
        ((_gamma ** 2) / (_gamma + _gamma_r))
    ) - 2)

    c_1 = -1.1143 * (_a ** 2) + 1.8618 * _a + 0.2523
    c_2 = 0.0805 * (_a ** 2) - 0.0710 * _a - 0.0095
    c_3 = -0.0005 * (_a ** 2) + 0.0002 * _a + 0.0003
    _D_masing = c_1 * _D_masing_a1 + c_2 * (_D_masing_a1 ** 2) + c_3 * (_D_masing_a1 ** 3)
    _F = _b * (_G_Gmax ** 0.1)

    _D = _F * _D_masing + _Dmin
    _sigma_ND = np.exp(params['phi13']) + np.sqrt(
        (0.25 / np.exp(params['phi14'])) -
        (((_G_Gmax - 0.5) ** 2) / np.exp(params['phi14']))
    )
    _sigma_D = np.exp(params['phi15']) + np.exp(params['phi16']) * np.sqrt(_D)

    return {
        'strains [pct]': _gamma,
        'G/Gmax [-]': _G_Gmax,
        'D [pct]': _D,
        'sigma_ND [-]': _sigma_ND,
        'sigma_D [pct]': _sigma_D,
        'gamma_r [pct]': _gamma_r,
        'a [-]': _a,
        'Dmin [pct]': _Dmin,
        'b [-]': _b,
        'Dmasing,a=1 [pct]': _D_masing_a1,
        'Dmasing [pct]': _D_masing
    }