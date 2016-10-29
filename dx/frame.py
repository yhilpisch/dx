#
# DX Analytics
# Framework Classes and Functions
# dx_frame.py
#
#
# DX Analytics is a financial analytics library, mainly for
# derviatives modeling and pricing by Monte Carlo simulation
#
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.
#
import math
import numpy as np
import pandas as pd
import datetime as dt
import scipy.interpolate as sci
import scipy.optimize as sco

# Helper functions


def get_year_deltas(time_list, day_count=365.):
    ''' Return vector of floats with time deltas in years.
    Initial value normalized to zero.

    Parameters
    ==========
    time_list : list or array
        collection of datetime objects
    day_count : float
        number of days for a year
        (to account for different conventions)

    Results
    =======
    delta_list : array
        year fractions
    '''

    delta_list = []
    start = time_list[0]
    for time in time_list:
        days = (time - start).days
        delta_list.append(days / day_count)
    return np.array(delta_list)


def sn_random_numbers(shape, antithetic=True, moment_matching=True,
                      fixed_seed=False):
    ''' Return an array of shape "shape" with (pseudo-) random numbers
    which are standard normally distributed.

    Parameters
    ==========
    shape : tuple (o, n, m)
        generation of array with shape (o, n, m)
    antithetic : boolean
        generation of antithetic variates
    moment_matching : boolean
        matching of first and second moments
    fixed_seed : boolean
        flag to fix the seed

    Results
    =======
    ran : (o, n, m) array of (pseudo-)random numbers
    '''
    if fixed_seed is True:
        np.random.seed(1000)
    if antithetic is True:
        ran = np.random.standard_normal(
            (shape[0], shape[1], int(shape[2] / 2)))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching is True:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran

# Discounting classes


class constant_short_rate(object):
    ''' Class for constant short rate discounting.

    Attributes
    ==========
    name : string
        name of the object
    short_rate : float (positive)
        constant rate for discounting

    Methods
    =======
    get_forward_rates :
        get forward rates give list/array of datetime objects;
        here: constant forward rates
    get_discount_factors :
        get discount factors given a list/array of datetime objects
        or year fractions
    '''

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('Short rate negative.')

    def get_forward_rates(self, time_list, paths=None, dtobjects=True):
        ''' time_list either list of datetime objects or list of
        year deltas as decimal number (dtobjects=False)
        '''
        forward_rates = np.array(len(time_list) * (self.short_rate,))
        return time_list, forward_rates

    def get_discount_factors(self, time_list, paths=None, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(time_list)
        else:
            dlist = np.array(time_list)
        discount_factors = np.exp(self.short_rate * np.sort(-dlist))
        return time_list, discount_factors


class deterministic_short_rate(object):
    ''' Class for discounting based on deterministic short rates,
    derived from a term structure of zero-coupon bond yields

    Attributes
    ==========
    name : string
        name of the object
    yield_list : list/array of (time, yield) tuples
        input yields with time attached

    Methods
    =======
    get_interpolated_yields :
        return interpolated yield curve given a time list/array
    get_forward_rates :
        return forward rates given a time list/array
    get_discount_factors :
        return discount factors given a time list/array
    '''

    def __init__(self, name, yield_list):
        self.name = name
        self.yield_list = np.array(yield_list)
        if np.sum(np.where(self.yield_list[:, 1] < 0, 1, 0)) > 0:
            raise ValueError('Negative yield(s).')

    def get_interpolated_yields(self, time_list, dtobjects=True):
        ''' time_list either list of datetime objects or list of
        year deltas as decimal number (dtobjects=False)
        '''
        if dtobjects is True:
            tlist = get_year_deltas(time_list)
        else:
            tlist = time_list
        dlist = get_year_deltas(self.yield_list[:, 0])
        if len(time_list) <= 3:
            k = 1
        else:
            k = 3
        yield_spline = sci.splrep(dlist, self.yield_list[:, 1], k=k)
        yield_curve = sci.splev(tlist, yield_spline, der=0)
        yield_deriv = sci.splev(tlist, yield_spline, der=1)
        return np.array([time_list, yield_curve, yield_deriv]).T

    def get_forward_rates(self, time_list, paths=None, dtobjects=True):
        yield_curve = self.get_interpolated_yields(time_list, dtobjects)
        if dtobjects is True:
            tlist = get_year_deltas(time_list)
        else:
            tlist = time_list
        forward_rates = yield_curve[:, 1] + yield_curve[:, 2] * tlist
        return time_list, forward_rates

    def get_discount_factors(self, time_list, paths=None, dtobjects=True):
        discount_factors = []
        if dtobjects is True:
            dlist = get_year_deltas(time_list)
        else:
            dlist = time_list
        time_list, forward_rate = self.get_forward_rates(time_list, dtobjects)
        for no in range(len(dlist)):
            factor = 0.0
            for d in range(no, len(dlist) - 1):
                factor += ((dlist[d + 1] - dlist[d]) *
                           (0.5 * (forward_rate[d + 1] + forward_rate[d])))
            discount_factors.append(np.exp(-factor))
        return time_list, discount_factors


# Market environment class

class market_environment(object):
    ''' Class to model a market environment relevant for valuation.

    Attributes
    ==========
    name: string
        name of the market environment
    pricing_date : datetime object
        date of the market environment

    Methods
    =======
    add_constant :
        adds a constant (e.g. model parameter)
    get_constant :
        get a constant
    add_list :
        adds a list (e.g. underlyings)
    get_list :
        get a list
    add_curve :
        adds a market curve (e.g. yield curve)
    get_curve :
        get a market curve
    add_environment :
        adding and overwriting whole market environments
        with constants, lists and curves
    '''

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        for key in env.curves:
            self.curves[key] = env.curves[key]
        for key in env.lists:
            self.lists[key] = env.lists[key]
        for key in env.constants:
            self.constants[key] = env.constants[key]
