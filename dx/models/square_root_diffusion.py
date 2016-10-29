#
# DX Analytics
# Base Classes and Model Classes for Simulation
# square_root_diffusion.py
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
from ..frame import *
from .simulation_class import simulation_class


class square_root_diffusion(simulation_class):
    ''' Class to generate simulated paths based on
    the Cox-Ingersoll-Ross (1985) square-root diffusion.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for simulation
    corr : boolean
        True if correlated with other model object

    Methods
    =======
    update :
        updates parameters
    generate_paths :
        returns Monte Carlo paths given the market environment
    '''

    def __init__(self, name, mar_env, corr=False):
        super(square_root_diffusion, self).__init__(name, mar_env, corr)
        try:
            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')
        except:
            print('Error parsing market environment.')

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               kappa=None, theta=None, final_date=None):
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value
        if self.correlated is False:
            rand = sn_random_numbers((1, M, I),
                                     fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            # full truncation Euler discretization
            paths_[t] = (paths_[t - 1] + self.kappa *
                         (self.theta - np.maximum(0, paths_[t - 1])) * dt +
                         np.sqrt(np.maximum(0, paths_[t - 1])) *
                         self.volatility * np.sqrt(dt) * ran)
            paths[t] = np.maximum(0, paths_[t])
        self.instrument_values = paths


class stochastic_short_rate(object):
    ''' Class for discounting based on stochastic short rates
    based on square-root diffusion process.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : market_environment object
        containing all relevant parameters

    Methods
    =======
    get_forward_rates :
        return forward rates given a time list/array
    get_discount_factors :
        return discount factors given a time list/array
    '''

    def __init__(self, name, mar_env):
        self.name = name
        try:
            try:
                mar_env.get_curve('discount_curve')
            except:
                mar_env.add_curve('discount_curve', 0.0)  # dummy
            try:
                mar_env.get_constant('currency')
            except:
                mar_env.add_constant('currency', 'CUR')  # dummy
            self.process = square_root_diffusion('process', mar_env)
            self.process.generate_paths()
        except:
            raise ValueError('Error parsing market environment.')

    def get_forward_rates(self, time_list, paths, dtobjects=True):
        if len(self.process.time_grid) != len(time_list) \
                or self.process.paths != paths:
            self.process.paths = paths
            self.process.time_grid = time_list
            self.process.instrument_values = None
        rates = self.process.get_instrument_values()
        return time_list, rates

    def get_discount_factors(self, time_list, paths, dtobjects=True):
        discount_factors = []
        if dtobjects is True:
            dlist = get_year_deltas(time_list)
        else:
            dlist = time_list
        forward_rate = self.get_forward_rates(time_list, paths, dtobjects)[1]
        for no in range(len(dlist)):
            factor = np.zeros_like(forward_rate[0, :])
            for d in range(no, len(dlist) - 1):
                factor += ((dlist[d + 1] - dlist[d]) *
                           (0.5 * (forward_rate[d + 1] + forward_rate[d])))
            discount_factors.append(np.exp(-factor))
        return time_list, np.array(discount_factors)


def srd_forwards(initial_value, kts, time_grid):
    ''' Function for forward vols/rates in SRD model.

    Parameters
    ==========
    initial_value : float
        initial value of the process
    kts :
        (kappa, theta, sigma)
    kappa : float
        mean-reversion factor
    theta : float
        long-run mean
    sigma : float
        volatility factor (vol-vol)
    time_grid : list/array of datetime object
        dates to generate forwards for

    Returns
    =======
    forwards : array
        forward vols/rates
    '''
    kappa, theta, sigma = kts
    t = get_year_deltas(time_grid)
    g = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    sum1 = ((kappa * theta * (np.exp(g * t) - 1)) /
            (2 * g + (kappa + g) * (np.exp(g * t) - 1)))
    sum2 = initial_value * ((4 * g ** 2 * np.exp(g * t)) /
                            (2 * g + (kappa + g) * (np.exp(g * t) - 1)) ** 2)
    forwards = sum1 + sum2
    return forwards
