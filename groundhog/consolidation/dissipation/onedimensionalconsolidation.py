#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import os
import warnings
from copy import deepcopy
import json
import re

# 3rd party packages
import pandas as pd
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS

# Project imports
from groundhog.general.plotting import plot_with_log, GROUNDHOG_PLOTTING_CONFIG
from groundhog.general.parameter_mapping import map_depth_properties, merge_two_dicts, reverse_dict
from groundhog.siteinvestigation.insitutests.pcpt_correlations import *
from groundhog.general.soilprofile import SoilProfile, plot_fence_diagram
from groundhog.general.parameter_mapping import offsets, latlon_distance
from groundhog.general.agsconversion import AGSConverter


PORE_PRESSURE_FOURIER = {
    'delta_u_0': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'time': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'cv': {'type': 'float', 'min_value': 0.1, 'max_value': 1000},
    'layer_thickness': {'type': 'float', 'min_value': 0.0, 'max_value': None}
}

PORE_PRESSURE_FOURIER_ERRORRETURN = {
    'drainage length [m]': np.nan,
    'delta u [kPa]': np.nan,
    'Tv [-]': np.nan
}

@Validator(PORE_PRESSURE_FOURIER, PORE_PRESSURE_FOURIER_ERRORRETURN)
def pore_pressure_fourier(delta_u_0, depths, time, cv, layer_thickness, no_terms=1000):
    """
    The function returns the excess pore pressure distribution at the specified depths for a given time.
    Note that the Fourier series solution only applies for uniform initial excess pore pressure distributions
    as commonly observed in thin layers. In thick clay layers, where triangular distributions are more common,
    this solution does not apply. If the stress distribution is irregular, a numerical solution should be applied.
    
    As an infinite amount of terms cannot be evaluated, the sum is limited to a number of terms (1000 by default)
    which should be sufficient for convergence.
    
    :param delta_u_0: Initial excess pore pressure (:math:`\Delta u_0`) [kPa]
    :param depths: Numpy array with the depths for the excess pore pressures (:math:`z`) [:math:`m`]
    :param time: Time at which excess pore pressures are computed (:math:`t`) [:math:`s`]
    :param cv: Coefficient of consolidation (:math:`c_v`) [:math:`m^2/yr`]
    :param layer_thickness: Thickness of the layer considered (:math:`2 \\cdot H_{dr}`) [:math:`m`]
    :param no_terms: Number of terms for the Fourier series (:math:`m`) [:math:`-`]
    
    .. math::
        \\Delta u (z,t) = \\sum_{m=0}^{\\infty} \\frac{2 \\Delta u_0}{M} \\sin \\left( \\frac{M \\cdot z}{H_{dr}} \\right) \\exp \\left( -M^2 T_v \\right)
        
        M = \\frac{\\pi}{2} \\left( 2m + 1 \\right)
        
        T_v = \\frac{c_v t}{H_{dr}^2}
    
    :returns: Dictionary with the following keys:
        
        - 'drainage length [m]': Drainage length (:math:`H_{dr}`)  [:math:`m`]
        - 'delta u [kPa]': Numpy array with the calculated excess pore pressures (:math:`\\Delta u (z,t)`)  [kPa]
        - 'Tv [-]': Time factor (:math:`T_v`)  [:math:`-`]
        
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.

    """
    _drainage_length = 0.5 * layer_thickness
    _cv_m2_s = cv / (365 * 24 * 3600)
    _Tv = _cv_m2_s * time / (_drainage_length ** 2)
    
    _delta_u = np.zeros(depths.__len__())
    
    for m in range(no_terms):
        _M = 0.5 * np.pi * (2 * m + 1)
        _delta_u = _delta_u + (
            ((2 * delta_u_0) / _M) *
            np.sin(_M * depths / _drainage_length) *
            np.exp(-_M ** 2 * _Tv)
        )
    return {
        'drainage length [m]': _drainage_length,
        'delta u [kPa]': _delta_u,
        'Tv [-]': _Tv
    }

CONSOLIDATION_DEGREE = {
    'time': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'cv': {'type': 'float', 'min_value': 0.1, 'max_value': 1000},
    'drainage_length': {'type': 'float', 'min_value': 0.0, 'max_value': None},
    'distribution': {'type': 'string', 'options': ("uniform", "triangular"), 'regex': None}
}

CONSOLIDATION_DEGREE_ERRORRETURN = {
    'U [pct]': np.nan,
    'Tv [-]': np.nan
}

@Validator(CONSOLIDATION_DEGREE, CONSOLIDATION_DEGREE_ERRORRETURN)
def consolidation_degree(time, cv, drainage_length, distribution='uniform'):
    """
    Returns the degree of consolidation for a certain time and initial distribution of excess pore pressure.
    
    The average degree of consolidation can be visualised as the area between the initial excess pore pressure
    distribution and the current isochrone.
    
    The solutions are interpolated from published solutions.
    
    :param time: Time at which excess pore pressures are computed (:math:`t`) [:math:`s`]
    :param cv: Coefficient of consolidation (:math:`c_v`) [:math:`m^2/yr`]
    :param drainage_length: Drainage length (:math:`H_{dr}`) [:math:`m`]
    :param distribution: Shape of the initial excess pore pressure distribution. Choose between ``"uniform"`` (default) and ``"triangular"``
    .. math::
        T_v = \\frac{c_v t}{H_{dr}^2}
    
    :returns: Dictionary with the following keys:
        
        - 'U [pct]': Degree of consolidation (:math:`U`)  [%]
        - 'Tv [-]': Time factor (:math:`T_v`)  [:math:`-`]
        
    Reference - Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.
    """

    Tv_triangle = np.array(
        [0.0, 0.0045579717143646, 0.0116597037664448, 0.018696823727789, 0.0287530916441815, 
        0.038810074342571, 0.0493694296545485, 0.0604318723621113, 0.0709908702830904, 0.0815489151614067,
        0.0921083896037171, 0.1026659579607021, 0.113726018061608, 0.124786078162514, 0.1358461382634199,
        0.1469052236616026, 0.157963334357062, 0.1690211201516138, 0.1800784727449551, 0.1911350672361784,
        0.202191336826494, 0.2132467400143889, 0.2243016017007709, 0.2353554886844297, 0.2464091590674832,
        0.2574621796487214, 0.2685147670287491, 0.279566812907264, 0.2906185338848711, 0.3016696050606627,
        0.3127201347349414, 0.3237702312080097, 0.3348194612786574, 0.3458686913493052, 0.3569170550175323,
        0.3679654186857594, 0.3790132408524735, 0.3900607381182799, 0.4011076938825734, 0.4121546496468669,
        0.4232013888105552, 0.4342473698721254, 0.445293026032788, 0.4563384655928453, 0.4673833636513897,
        0.478428045109329, 0.4894725099666631, 0.5005165416227868, 0.5115605732789107, 0.5226043883344292,
        0.5336477701887375, 0.5446910437427431, 0.5557343172967489, 0.5667771576495443, 0.5778196731014319,
        0.5888621885533195, 0.599904487404602, 0.6109461364540689, 0.621987785503536, 0.6330292179523976,
        0.6440705421009569, 0.6551115413486083, 0.6661525405962598, 0.6771933232433061, 0.6882337809894447,
        0.6992741304352806, 0.7103145881814192, 0.7213550459275577, 0.732395070472486, 0.7434349867171116,
        0.7544752278626451, 0.7655147109060603, 0.7765543022497783, 0.7875936769928911, 0.798632726835096,
        0.8096717766773011, 0.8207106099189008, 0.8317494431605007, 0.8427880598014953, 0.8538265681421873,
        0.8648650764828794, 0.8759032599226637, 0.8869415516627508, 0.8979796268022324, 0.9092518055419344,
        0.9200553438799856, 0.9310937439203752, 0.9426729651256208, 0.9538613442499436, 0.963428808828576,
        0.975244528274066, 0.9862820619120348, 0.9973193789493984, 1.008356479386157, 1.019393796423521,
        1.030430680259674, 1.0414677806964323, 1.052504881133191, 1.063543086233036, 1.0745809406461866,
        1.085618520698571, 1.0966540636738351, 1.107689603554805, 1.1187244513452697, 1.129760035577792,
        1.1407960530115249, 1.15183217874556, 1.1628681961792926, 1.1739046468142358, 1.184941314049784,
        1.1959778729850297, 1.207012915716039, 1.2150382930382886])

    U_triangle = np.array(
        [0.0, 1.3579905913176162, 2.4610715228292293, 3.9180761288628494, 5.89130330989866, 7.919854617505578,
        10.012950739445444, 12.22591580228935, 14.291349860943685, 16.283018417503186, 18.385335227204905,
        20.340121032717025, 22.36867234032394, 24.397223647930858, 26.425774955537776, 28.378884272365923,
        30.2565515984153, 32.109071594205105, 33.928061816315434, 35.68837493448673, 37.42354072239843,
        39.09164696295123, 40.71784098640471, 42.26859301907942, 43.8025801649144, 45.2862726502302, 
        46.73643536186656, 48.144685856403584, 49.52778902068103, 50.8605975244393, 52.15149381109825,
        53.40886032407774, 54.59916728969833, 55.78947425531892, 56.9127216735806, 58.03596909184229,
        59.11730429300464, 60.17349216390743, 61.187767817710885, 62.20204347151434, 63.19955423847808,
        64.13838790150277, 65.05207423426786, 65.94899568019324, 66.8040049090193, 67.64224925100562,
        68.46372870615224, 69.25167838761938, 70.03962806908652, 70.81081286371395, 71.54846788466193,
        72.27774046219002, 73.00701303971813, 73.70275584356679, 74.37335131715585, 75.04394679074491,
        75.69777737749425, 76.30131330372441, 76.90484922995456, 77.49162026934499, 78.07000886531557,
        78.62325013102654, 79.17649139673752, 79.71296777560877, 80.22429682422043, 80.72724342941223,
        81.2385724780239, 81.74990152663555, 82.22770080156776, 82.69711763308011, 83.19168179485204,
        83.62756885268493, 84.0718383539377, 84.49934296835072, 84.90170025250416, 85.3040575366576,
        85.68964993397131, 86.07524233128501, 86.444069841759, 86.80451490881313, 87.16495997586725,
        87.50025771266178, 87.84393789287617, 88.17085318625084, 88.7008815317222, 88.79115399932073,
        89.14321662295498, 89.54965766159503, 89.93902484512826, 90.20219864166438, 90.33352358857556,
        90.61852666485092, 90.88676485428655, 91.13823815688244, 91.40647634631809, 91.64118476207425,
        91.89265806467014, 92.14413136726604, 92.48110559274456, 92.7909366681016, 93.07953222012831,
        93.21045800306712, 93.34114428762253, 93.41826276708528, 93.55238186180308, 93.72003073020036,
        93.89606204201748, 94.06371091041474, 94.26488955249148, 94.4828330814079, 94.69239416690448,
        94.784601044523, 94.84327814846203])

    Tv_uniform = np.array(
        [0.0, 0.0018052563417972, 0.00477435104525, 0.0127293795852368,
        0.0173200940363185, 0.0213585527544428, 0.0269020445325068, 0.0329469161713852, 0.0394930145035072,
        0.046540254435778, 0.0540886288771063, 0.0626414753022283, 0.0716955129651379, 0.0812507319383071,
        0.0913077146366966, 0.1018654021240143, 0.1129239460206842, 0.1239812986140256, 0.1350376765046437,
        0.1460926464913283, 0.1571469666761974, 0.1682004204586458, 0.1792531161389761, 0.1903053786180961,
        0.2013567746947955, 0.212407629269982, 0.2234578340433529, 0.234507497315211, 0.2455564024849509,
        0.2566051993543883, 0.2676532381217077, 0.278701276889027, 0.2897482326533205, 0.3007954050182191,
        0.3118421441819075, 0.3228887750452932, 0.3339349727074686, 0.3449809537690388, 0.3560263933290962,
        0.3670713996879431, 0.3781165143470928, 0.3891607626038217, 0.4002051191608532, 0.4112490425166744,
        0.422292857572193, 0.4333362394265012, 0.4443794046802044, 0.4554227865345126, 0.4664653019864003,
        0.4775080340388932, 0.4885505494907807, 0.4995930649426684, 0.5106356886948585, 0.5216774460446282,
        0.5327190950940951, 0.5437608524438646, 0.5548020682921212, 0.5658432841403779, 0.5768842833880293,
        0.5879250660350757, 0.5989656320815168, 0.6100063064282605, 0.6210467641743991, 0.6320870053199324,
        0.6431270298648607, 0.654167054409789, 0.6652067540538094, 0.6762462370972248, 0.6872859367412452,
        0.6983249865834502, 0.7093643613265629, 0.7204037360696757, 0.7314423527106703, 0.7424814025528753,
        0.7535200191938699, 0.7645584192342595, 0.7755969275749515, 0.7866352193150384, 0.7976734027548227,
        0.8087114778943045, 0.819749336433181, 0.830787086671755, 0.8418251618112368, 0.8528625871489031,
        0.863900337387477, 0.8749374378242356, 0.8859749714622045, 0.8970118552983577, 0.9080488474348136,
        0.9190859478715724, 0.930122723407423, 0.9411593906429708, 0.9521960578785194, 0.9632327251140675,
        0.9742692840493132, 0.985305734684256, 0.9963418604182914, 1.0073778778520242, 1.0184138952857569,
        1.0294498044191869, 1.0404857135526169, 1.0515215143857446, 1.0625568820176616, 1.0850144713753878,
        1.1020636883730484, 1.1182178214783312, 1.1280871869678517, 1.1398045637413845, 1.1508389566705786,
        1.1618729163985622, 1.172907850829269, 1.18394170225695, 1.1949763117867491, 1.2019994781089032])

    U_uniform = np.array(
        [0.0, 3.615729743995772, 7.707778798983043, 11.396053903722889,
        13.816484441208416, 15.840425404934408, 17.896638775826872, 19.94055789637021, 21.96032759658489,
        23.94936167092673, 25.907111268933747, 27.97280145780367, 30.011597974034856, 32.02273242698051,
        34.05128373458743, 36.0152902278614, 37.92648732759022, 39.74547754970057, 41.489025781032126,
        43.12360224790548, 44.70788405425964, 46.2251063132549, 47.68365146831111, 49.10866684968787,
        50.46662268370572, 51.78266630062426, 53.04841525702362, 54.27225199632366, 55.43741163168466,
        56.59418882362579, 57.69228891162789, 58.79038899962998, 59.80466465343344, 60.83570519407662,
        61.83321596104036, 62.82234428458423, 63.777942834448645, 64.71677649747333, 65.6136979433987,
        66.47708961564462, 67.34886373131042, 68.1535782996173, 68.96667531134403, 69.74624254939131,
        70.51742734401873, 71.25508236496671, 71.97597249907494, 72.71362752002292, 73.38422299361198,
        74.07158335404077, 74.74217882762983, 75.4127743012189, 76.09175221822782, 76.70367058787784,
        77.307206514108, 77.91912488375803, 78.48913103630873, 79.05913718885944, 79.61237845457042,
        80.14885483344166, 80.6685663254732, 81.19666026092457, 81.70798930953625, 82.20255347130818,
        82.68035274624037, 83.15815202117258, 83.61080396584521, 84.04669102367811, 84.49934296835072,
        84.90170025250416, 85.32920486691718, 85.7567094813302, 86.1255369918042, 86.52789427595764,
        86.89672178643161, 87.24878441006588, 87.60922947712, 87.9529096573344, 88.28820739412893,
        88.6151226875036, 88.92527309403854, 89.22704105715361, 89.55395635052828, 89.83057698338378,
        90.13234494649885, 90.38381824909476, 90.6688213253701, 90.90352974112628, 91.14662060030231,
        91.3980939028982, 91.62441987523452, 91.84236340415096, 92.0603069330674, 92.27825046198386,
        92.48781154748043, 92.68899018955716, 92.86502150137429, 93.03267036977157, 93.20031923816882,
        93.35958566314622, 93.51885208812362, 93.66973606968116, 93.78709027755926, 94.0089733438902,
        94.17392451834256, 94.5038268672473, 94.5038268672473, 94.61695217612572, 94.65886439322504,
        94.66724683664488, 94.75107127084354, 94.75107127084354, 94.80974837478256, 94.93548502608051])

    _cv_m2_s = np.array(cv) / (365 * 24 * 3600)
    _Tv = _cv_m2_s * time / (drainage_length ** 2)
    
    if distribution == 'uniform':
        _U = np.interp(_Tv, Tv_uniform, U_uniform)
    elif distribution == 'triangular':
        _U = np.interp(_Tv, Tv_triangle, U_triangle)
    else:
        raise ValueError("distribution must be 'uniform' or 'triangular'")
    
    return {
        'U [pct]': _U,
        'Tv [-]': _Tv
    }


class ConsolidationCalculation(object):
    """
    The consolidation equation can be discretised as follows:

    .. math::
        u_{i,j+1} = u_{i,j} + \\frac{c_v \\Delta t}{( \\Delta z )^2} \\left( u_{i-1,j} - 2 u_{i,j} + u_{i+1,j} \\right)


    At permeable boundaries, the excess pore pressure is 0 (:math:`u = 0`). At impervious boundaries, the following boundary condition applies:

    .. math::
        \\frac{\\partial u}{\partial z} = 0 = \\frac{1}{2 \\Delta z} \\left( u_{i-1,j} - u_{i+1,j} \\right) = 0

        \\implies u_{i,j+1} = u_{i,j} + \\frac{c_v \\Delta t}{( \\Delta z )^2}  \\left( 2 u_{i-1,j} - 2 u_{i,j} \\right)


    To ensure stability, the timestep needs to be chosen according to the following criterion:

    .. math::
        \\alpha = \\frac{c_v \\Delta t}{(\\Delta z)^2} < \\frac{1}{2}


    Usually, :math:`\\alpha = 0.25` is used to determine the timestep.

    The discretisation in space and time is done as follows where the number of timesteps is usually calculated from a chosen node offset:

    .. math::
        \\Delta z = \\frac{H_0}{m}, \\ \\Delta t = \\frac{t}{n}


    """
    
    def __init__(self, height, total_time, no_nodes):
        """
        Initialises the consolidation calculation with the height of the layer and the total time.
        Subdivision of the height in m elements is performed.
        """
        self.m = no_nodes
        self.H0 = height
        self.T = total_time
        self.z = np.linspace(0, height, no_nodes)
        self.dz = np.diff(self.z)[0]
        
    def set_cv(self, cv, uniform=True, cv_depths=None):
        """
        Sets the coefficient of consolidation. A constant value or an array of cv varying with depth can be
        specified. cv is specified in m2/yr and is converted to m2/s inside the routine (all calcs happen in s)
        """
        if uniform:
            self.cv = np.ones(self.z.__len__()) * cv / (365 * 24 * 3600)
            self.cv_depths = None
        else:
            cv_depths = np.array(cv_depths)
            cv = np.array(cv)
            if cv_depths is None:
                raise ValueError("Depths corresponding to the given values of cv need to be specified")
            self.cv = np.interp(self.z, cv_depths, cv / (365 * 24 * 3600))
            
        self.dt = 0.25 * (self.dz ** 2) / self.cv.max()
        self.n = int(np.ceil(self.T / self.dt))
        self.times = np.linspace(0, self.T, self.n+1)
    
    def set_top_boundary(self, freedrainage=True):
        """
        Sets the boundary condition at the top
        
        Set ``freedrainage=False`` for an impervious top surface
        """
        if freedrainage:
            self.top_boundary = "open"
        else:
            self.top_boundary = "closed"
            
    def set_bottom_boundary(self, freedrainage=True):
        """
        Sets the boundary condition at the bottom
        
        Set ``freedrainage=False`` for an impervious top surface
        """
        if freedrainage:
            self.bottom_boundary = "open"
        else:
            self.bottom_boundary = "closed"
            
    def set_initial(self, u0, u0_depths):
        """
        Sets the initial excess pore pressure distribution
        
        :param u0: NumPy array with initial excess pore pressure
        :param u0_depths: NumPy array with the depths corresponding to the defined excess pore pressures
        """
        if u0.__len__() != u0_depths.__len__():
            raise ValueError(
                "Array with excess pore pressures and corresponding depths need to be of equal length")
        self.u0 = np.interp(self.z, u0_depths, u0)
    
    def set_output_times(self, output_times):
        """
        Sets the times at which output is requested.
        These are pasted into the array with computed times
        """
        self.times = np.unique(np.sort(np.append(self.times, output_times)))
        self.dts = np.diff(self.times)
        self.output_times = output_times
        self.output_indices = []
        for _t in self.output_times:
            self.output_indices.append(np.where(self.times == _t)[0][0])
        
    def calculate(self):
        """
        Calculates the pore pressure dissipation until the specified output time
        """
        u = deepcopy(self.u0)
        self.u_steps = []
        self.u_steps.append(u)
        for j, _dt in enumerate(self.dts):
            u_previous = deepcopy(u)
            u = np.zeros(self.m)
            for i, _z in enumerate(self.z):
                if i == 0:
                    if self.top_boundary == "open":
                        u[i] = 0
                    elif self.top_boundary == "closed":
                        u[i] = u_previous[i] + ((self.cv[i] * _dt) / (self.dz ** 2)) * \
                            (-2 * u_previous[i] + 2 * u_previous[i+1])
                elif i == self.m-1:
                    if self.bottom_boundary == "open":
                        u[i] = 0
                    elif self.bottom_boundary == "closed":
                        u[i] = u_previous[i] + ((self.cv[i] * _dt) / (self.dz ** 2)) * \
                            (2 * u_previous[i-1] - 2 * u_previous[i])
                else:
                    u[i] = u_previous[i] + ((self.cv[i] * _dt) / (self.dz ** 2)) * \
                        (u_previous[i-1] - 2 * u_previous[i] + u_previous[i+1])
            self.u_steps.append(u)
            
    def plot_results(self, plot_title="", showfig=True, xtitle=r'$ \Delta u \ \text{[kPa]} $',
                     ytitle=r'$ z \ \text{[m]} $', latex_titles=True):
        self.fig = subplots.make_subplots(rows=1, cols=1, print_grid=False)
        for j, i in enumerate(self.output_indices):
            try:
                _data = go.Scatter(
                    x=self.u_steps[i], y=self.z, showlegend=True, mode='lines',
                                   name='t=%.2es' % self.times[i],
                    line=dict(color=DEFAULT_PLOTLY_COLORS[j % 10]))
                self.fig.append_trace(_data, 1, 1)
            except Exception as err:
                print(str(err), i)
        if not latex_titles:
            xtitle = 'delta u [kPa]'
            ytitle = 'z [m]'
        self.fig['layout']['xaxis1'].update(title=xtitle)
        self.fig['layout']['yaxis1'].update(title=ytitle, range=(self.z.max(), 0))
        self.fig['layout'].update(height=500, width=600,
            title=plot_title,
            hovermode='closest')
        if showfig:
            self.fig.show()