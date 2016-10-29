#
# DX Analytics
# Base Classes and Model Classes for Simulation
# stoch_vol_jump_diffusion.py
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


class stoch_vol_jump_diffusion(simulation_class):
    ''' Class to generate simulated paths based on
    the Bates (1996) stochastic volatility jump-diffusion model.

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
    get_volatility_values :
        returns array with simulated volatility paths
    '''

    def __init__(self, name, mar_env, corr=False):
        super(stoch_vol_jump_diffusion, self).__init__(name, mar_env, corr)
        try:
            self.lamb = mar_env.get_constant('lambda')
            self.mu = mar_env.get_constant('mu')
            self.delt = mar_env.get_constant('delta')

            self.rho = mar_env.get_constant('rho')
            self.leverage = np.linalg.cholesky(
                np.array([[1.0, self.rho], [self.rho, 1.0]]))

            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')
            self.vol_vol = mar_env.get_constant('vol_vol')

            self.volatility_values = None
        except:
            print('Error parsing market environment.')

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               vol_vol=None, kappa=None, theta=None, rho=None, lamb=None,
               mu=None, delta=None, final_date=None):
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if vol_vol is not None:
            self.vol_vol = vol_vol
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if rho is not None:
            self.rho = rho
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delt = delta
        if final_date is not None:
            self.final_date = final_date
            self.time_grid = None
        self.instrument_values = None
        self.volatility_values = None

    def generate_paths(self, fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        va = np.zeros_like(paths)
        va_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        va[0] = self.volatility ** 2
        va_[0] = self.volatility ** 2
        if self.correlated is False:
            sn1 = sn_random_numbers((1, M, I),
                                    fixed_seed=fixed_seed)
        else:
            sn1 = self.random_numbers

        # Pseudo-random numbers for the jump component
        sn2 = sn_random_numbers((1, M, I),
                                fixed_seed=fixed_seed)
        # Pseudo-random numbers for the stochastic volatility
        sn3 = sn_random_numbers((1, M, I),
                                fixed_seed=fixed_seed)

        forward_rates = self.discount_curve.get_forward_rates(
            self.time_grid, self.paths, dtobjects=True)[1]

        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            rat = np.array([ran, sn3[t]])
            rat = np.dot(self.leverage, rat)

            va_[t] = (va_[t - 1] + self.kappa *
                      (self.theta - np.maximum(0, va_[t - 1])) * dt +
                      np.sqrt(np.maximum(0, va_[t - 1])) *
                      self.vol_vol * np.sqrt(dt) * rat[1])
            va[t] = np.maximum(0, va_[t])

            poi = np.random.poisson(self.lamb * dt, I)

            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            paths[t] = paths[t - 1] * (
                np.exp((rt - rj - 0.5 * va[t]) * dt +
                       np.sqrt(va[t]) * np.sqrt(dt) * rat[0]) +
                (np.exp(self.mu + self.delt * sn2[t]) - 1) * poi)

            # moment matching stoch vol part
            paths[t] -= np.mean(paths[t - 1] * np.sqrt(va[t]) *
                                math.sqrt(dt) * rat[0])

        self.instrument_values = paths
        self.volatility_values = np.sqrt(va)

    def get_volatility_values(self):
        if self.volatility_values is None:
            self.generate_paths(self)
        return self.volatility_values
