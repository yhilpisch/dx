#
# DX Analytics
# Base Classes and Model Classes for Simulation
# dx_models.py
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


class sabr_stochastic_volatility(simulation_class):
    ''' Class to generate simulated paths based for the SABR model.

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
    get_log_normal_implied_vol :
        returns the approximation of the lognormal Black implied volatility
        as given by Hagan et al. (2002)
    '''

    def __init__(self, name, mar_env, corr=False):
        super(sabr_stochastic_volatility, self).__init__(name, mar_env, corr)
        try:
            self.alpha = mar_env.get_constant('alpha')   # initial variance
            self.volatility = self.alpha ** 0.5  # initial volatility
            self.beta = mar_env.get_constant('beta')  # exponent of the FWR
            self.vol_vol = mar_env.get_constant('vol_vol')  # vol of var
            self.rho = mar_env.get_constant('rho')  # correlation
            self.leverage = np.linalg.cholesky(
                np.array([[1.0, self.rho], [self.rho, 1.0]]))
            self.volatility_values = None
        except:
            print('Error parsing market environment.')

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               alpha=None, beta=None, rho=None, vol_vol=None,
               final_date=None):
        ''' Updates the attributes of the object. '''
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
            self.alpha = volatility ** 2
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if rho is not None:
            self.rho = rho
        if vol_vol is not None:
            self.vol_vol = vol_vol
        if final_date is not None:
            self.final_date = final_date
            self.time_grid = None
        self.instrument_values = None
        self.volatility_values = None

    def get_log_normal_implied_vol(self, strike, expiry):
        ''' Returns the implied volatility for fixed strike and expiry.
        '''
        self.check_parameter_set()
        one_beta = 1. - self.beta
        one_beta_square = one_beta ** 2
        if self.initial_value != strike:
            fk = self.initial_value * strike
            fk_beta = fk ** (one_beta / 2.)
            log_fk = math.log(self.initial_value / strike)
            z = self.vol_vol / self.alpha * fk_beta * log_fk
            x = math.log((math.sqrt(1. - 2 * self.rho * z + z ** 2) +
                          z - self.rho) / (1 - self.rho))

            sigma_1 = ((self.alpha / fk_beta /
                       (1 + one_beta_square * log_fk ** 2 / 24) +
                        (one_beta_square ** 2 * log_fk ** 4 / 1920.)) * z / x)

            sigma_ex = (one_beta_square / 24. * self.alpha ** 2 / fk_beta ** 2 +
                        0.25 * self.rho * self.beta * self.vol_vol *
                        self.alpha / fk_beta +
                        (2. - 3. * self.rho ** 2) /
                        24. * self.vol_vol ** 2)

            sigma = sigma_1 * (1. + sigma_ex * expiry)
        else:
            f_beta = self.initial_value ** one_beta
            f_two_beta = self.initial_value ** (2. - 2 * self.beta)
            sigma = ((self.alpha / f_beta) *
                     (1 + expiry * ((one_beta_square / 24. * self.alpha ** 2 /
                      f_two_beta) + (0.25 * self.rho * self.beta *
                      self.vol_vol * self.alpha / f_beta) +
                      ((2. - 3 * self.rho ** 2) / 24. * self.vol_vol ** 2))))
        return sigma

    def calibrate_to_impl_vol(self, implied_vols, maturity, para=list()):
        ''' Calibrates the parameters alpha, beta, initial_value and vol_vol
        to a set of given implied volatilities.
        '''
        if len(para) != 4:
            para = (self.alpha, self.beta, self.initial_value, self.vol_vol)

        def error_function(para):
            self.alpha, self.beta, self.initial_value, self.vol_vol = para
            if (self.beta < 0 or self.beta > 1 or self.initial_value <= 0 or
                    self.vol_vol <= 0):
                return 10000
            e = 0
            for strike in implied_vols.columns:
                e += (self.get_log_normal_implied_vol(float(strike), maturity) -
                      float(implied_vols[strike]) / 100.) ** 2
            return e
        para = fmin(error_function, para, xtol=0.0000001,
                    ftol=0.0000001, maxiter=550, maxfun=850)
        return para

    def check_parameter_set(self):
        ''' Checks if all needed parameter are set.
        '''
        parameter = ['beta', 'initial_value', 'vol_vol', 'alpha', 'rho']
        for p in parameter:
            try:
                val = getattr(self, p)
            except:
                val = None
            if val is None:
                raise ValueError('Models %s is unset!' % p)

    def generate_paths(self, fixed_seed=True, day_count=365.):
        ''' Generates Monte Carlo Paths using Euler discretization.
        '''
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        va = np.zeros_like(paths)
        va_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        va[0] = self.alpha
        va_[0] = self.alpha
        if self.correlated is False:
            sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            sn1 = self.random_numbers

        # pseudo-random numbers for the stochastic volatility
        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            square_root_dt = np.sqrt(dt)
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            rat = np.array([ran, sn2[t]])
            rat = np.dot(self.leverage, rat)

            va_[t] = va_[t - 1] * (1 + self.vol_vol * square_root_dt * rat[1])
            va[t] = np.maximum(0, va_[t])

            F_b = np.abs(paths[t - 1]) ** self.beta
            p = paths[t - 1] + va_[t] * F_b * square_root_dt * rat[0]
            if (self.beta > 0 and self.beta < 1):
                paths[t] = np.maximum(0, p)
            else:
                paths[t] = p

        self.instrument_values = paths
        self.volatility_values = np.sqrt(va)

    def get_volatility_values(self):
        ''' Returns the volatility values for the model object.
        '''
        if self.volatility_values is None:
            self.generate_paths(self)
        return self.volatility_values
