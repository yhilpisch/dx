#
# DX Analytics
# Base Classes and Model Classes for Simulation
# square_root_jump_diffusion.py
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
from .square_root_diffusion import *


class square_root_jump_diffusion(simulation_class):
    ''' Class to generate simulated paths based on
    the square-root jump diffusion model.

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
        returns Monte Carlo paths for the market environment
    '''

    def __init__(self, name, mar_env, corr=False):
        super(square_root_jump_diffusion, self).__init__(name, mar_env, corr)
        try:
            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')
            self.lamb = mar_env.get_constant('lambda')
            self.mu = mar_env.get_constant('mu')
            self.delt = mar_env.get_constant('delta')
        except:
            print('Error parsing market environment.')

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               kappa=None, theta=None, lamb=None, mu=None, delt=None,
               final_date=None):
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
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delt is not None:
            self.delt = delt
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None
        self.time_grid = None

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
        snr = sn_random_numbers((1, M, I),
                                fixed_seed=fixed_seed)
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            poi = np.random.poisson(self.lamb * dt, I)
            # full truncation Euler discretization
            paths_[t, :] = (paths_[t - 1, :] + self.kappa *
                            (self.theta - np.maximum(0, paths_[t - 1, :])) * dt +
                            np.sqrt(np.maximum(0, paths_[t - 1, :])) *
                            self.volatility * np.sqrt(dt) * ran +
                            ((np.exp(self.mu + self.delt * snr[t]) - 1) * poi) *
                            np.maximum(0, paths_[t - 1, :]) - rj * dt)
            paths[t, :] = np.maximum(0, paths_[t, :])
        self.instrument_values = paths


class square_root_jump_diffusion_plus(square_root_jump_diffusion):
    ''' Class to generate simulated paths based on
    the square-root jump diffusion model with term structure.

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
    srd_forward_error :
        error function for forward rate/vols calibration
    generate_shift_base :
        generates a shift base to take term structure into account
    update :
        updates parameters
    update_shift_values :
        updates shift values for term structure
    generate_paths :
        returns Monte Carlo paths for the market environment
    update_forward_rates :
        updates forward rates (vol, int. rates) for given time grid
    '''

    def __init__(self, name, mar_env, corr=False):
        super(square_root_jump_diffusion_plus,
              self).__init__(name, mar_env, corr)
        try:
            self.term_structure = mar_env.get_curve('term_structure')
        except:
            self.term_structure = None
            print('Missing Term Structure.')

        self.forward_rates = []
        self.shift_base = None
        self.shift_values = []

    def srd_forward_error(self, p0):
        if p0[0] < 0 or p0[1] < 0 or p0[2] < 0:
            return 100
        f_model = srd_forwards(self.initial_value, p0,
                               self.term_structure[:, 0])

        MSE = np.sum((self.term_structure[:, 1] -
                      f_model) ** 2) / len(f_model)
        return MSE

    def generate_shift_base(self, p0):
        # calibration
        opt = sco.fmin(self.srd_forward_error, p0)
        # shift_calculation
        f_model = srd_forwards(self.initial_value, opt,
                               self.term_structure[:, 0])
        shifts = self.term_structure[:, 1] - f_model
        self.shift_base = np.array((self.term_structure[:, 0], shifts)).T

    def update_shift_values(self, k=1):
        if self.shift_base is not None:
            t = get_year_deltas(self.shift_base[:, 0])
            tck = sci.splrep(t, self.shift_base[:, 1], k=k)
            self.generate_time_grid()
            st = get_year_deltas(self.time_grid)
            self.shift_values = np.array(list(zip(self.time_grid,
                                    sci.splev(st, tck, der=0))))
        else:
            self.shift_values = np.array(list(zip(self.time_grid,
                                    np.zeros(len(self.time_grid)))))

    def generate_paths(self, fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        self.update_shift_values()
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
        snr = sn_random_numbers((1, M, I),
                                fixed_seed=fixed_seed)
        # forward_rates = self.discount_curve.get_forward_rates(
        #    self.time_grid, dtobjects=True)
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)
        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            poi = np.random.poisson(self.lamb * dt, I)
            # full truncation Euler discretization
            paths_[t] = (paths_[t - 1] + self.kappa *
                         (self.theta - np.maximum(0, paths_[t - 1])) * dt +
                         np.sqrt(np.maximum(0, paths_[t - 1])) *
                         self.volatility * np.sqrt(dt) * ran +
                         ((np.exp(self.mu + self.delt * snr[t]) - 1) * poi) *
                         np.maximum(0, paths_[t - 1]) - rj * dt)
            paths[t] = np.maximum(0, paths_[t]) + self.shift_values[t, 1]
        self.instrument_values = paths

    def update_forward_rates(self, time_grid=None):
        if time_grid is None:
            self.generate_time_grid()
            time_grid = self.time_grid
        t = get_year_deltas(time_grid)
        g = np.sqrt(self.kappa ** 2 + 2 * self.volatility ** 2)
        sum1 = ((self.kappa * self.theta * (np.exp(g * t) - 1)) /
                (2 * g + (self.kappa + g) * (np.exp(g * t) - 1)))
        sum2 = self.initial_value * ((4 * g ** 2 * np.exp(g * t)) /
                                     (2 * g + (self.kappa + g) *
                                     (np.exp(g * t) - 1)) ** 2)
        self.forward_rates = np.array(list(zip(time_grid, sum1 + sum2)))
