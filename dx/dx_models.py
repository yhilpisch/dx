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

from dx_frame import *


class simulation_class(object):
    ''' Providing base methods for simulation classes.

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
    generate_time_grid :
        returns time grid for simulation
    get_instrument_values:
        returns the current instrument values (array)
    '''

    def __init__(self, name, mar_env, corr):
        try:
            self.name = name
            self.pricing_date = mar_env.pricing_date
            self.initial_value = mar_env.get_constant('initial_value')
            self.volatility = mar_env.get_constant('volatility')
            self.final_date = mar_env.get_constant('final_date')
            self.currency = mar_env.get_constant('currency')
            self.frequency = mar_env.get_constant('frequency')
            self.paths = mar_env.get_constant('paths')
            self.discount_curve = mar_env.get_curve('discount_curve')
            try:
                # if time_grid in mar_env take this
                # (for portfolio valuation)
                self.time_grid = mar_env.get_list('time_grid')
            except:
                self.time_grid = None
            try:
                # if there are special dates, then add these
                self.special_dates = mar_env.get_list('special_dates')
            except:
                self.special_dates = []
            self.instrument_values = None
            self.correlated = corr
            if corr is True:
                # only needed in a portfolio context when
                # risk factors are correlated
                self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
                self.rn_set = mar_env.get_list('rn_set')[self.name]
                self.random_numbers = mar_env.get_list('random_numbers')
        except:
            print "Error parsing market environment."

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        # pandas date_range function
        # freq = e.g. 'B' for Business Day,
        # 'W' for Weekly, 'M' for Monthly
        time_grid = pd.date_range(start=start, end=end,
                                  freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
        # enhance time_grid by start, end and special_dates
        if start not in time_grid:
            time_grid.insert(0, start)
            # insert start date if not in list
        if end not in time_grid:
            time_grid.append(end)
            # insert end date if not in list
        if len(self.special_dates) > 0:
            # add all special dates later than self.pricing_date
            add_dates = [d for d in self.special_dates
                            if d > self.pricing_date]
            time_grid.extend(add_dates)
            # delete duplicates and sort
            time_grid = sorted(set(time_grid))
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        elif fixed_seed is False:
            # also initiate re-simulation when fixed_seed is False
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        return self.instrument_values


class geometric_brownian_motion(simulation_class):
    ''' Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for simulation
    corr : boolean
        True if correlated with other model simulation object

    Methods
    =======
    update :
        updates parameters
    generate_paths :
        returns Monte Carlo paths given the market environment
    '''

    def __init__(self, name, mar_env, corr=False):
        super(geometric_brownian_motion, self).__init__(name, mar_env, corr)

    def update(self, pricing_date=None, initial_value=None,
                     volatility=None, final_date=None):
        ''' Updates model parameters. '''
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        ''' Generates Monte Carlo paths for the model. '''
        if self.time_grid is None:
            self.generate_time_grid()
            # method from generic model simulation class
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths
        # array initialization for path simulation
        paths = np.zeros((M, I))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlated is False:
            # if not correlated generate random numbers
            rand = sn_random_numbers((1, M, I),
                                     fixed_seed=fixed_seed)
        else:
            # if correlated use random number object as provided
            # in market environment
            rand = self.random_numbers

        # forward rates for drift of process
        forward_rates = self.discount_curve.get_forward_rates(
            self.time_grid, self.paths, dtobjects=True)[1]

        for t in range(1, len(self.time_grid)):
            # select the right time slice from the relevant
            # random number set
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
              # difference between two dates as year fraction
            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            paths[t] = paths[t - 1] * np.exp((rt - 0.5
                                              * self.volatility ** 2) * dt
                                    + self.volatility * np.sqrt(dt) * ran)
              # generate simulated values for the respective date
        self.instrument_values = paths


class jump_diffusion(simulation_class):
    ''' Class to generate simulated paths based on
    the Merton (1976) jump diffusion model.

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
        super(jump_diffusion, self).__init__(name, mar_env, corr)
        try:
            self.lamb = mar_env.get_constant('lambda')
            self.mu = mar_env.get_constant('mu')
            self.delt = mar_env.get_constant('delta')
        except:
            print "Error parsing market environment."

    def update(self, pricing_date=None, initial_value=None,
                volatility=None, lamb=None, mu=None, delta=None,
                final_date=None):
        if pricing_date is not None:
            self.pricing_date = pricing_date
            self.time_grid = None
            self.generate_time_grid()
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delt = delta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
            # method from generic model simulation class
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths
        # array initialization for path simulation
        paths = np.zeros((M, I))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlated is False:
            # if not correlated generate random numbers
            sn1 = sn_random_numbers((1, M, I),
                                     fixed_seed=fixed_seed)
        else:
            # if correlated use random number object as provided
            # in market environment
            sn1 = self.random_numbers

        # Standard normally distributed seudo-random numbers
        # for the jump component
        sn2 = sn_random_numbers((1, M, I),
                                fixed_seed=fixed_seed)

        forward_rates = self.discount_curve.get_forward_rates(
            self.time_grid, self.paths, dtobjects=True)[1]

        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)
        for t in range(1, len(self.time_grid)):
                        # select the right time slice from the relevant
            # random number set
            if self.correlated is False:
                ran = sn1[t]
            else:
                # only with correlation in portfolio context
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
              # difference between two dates as year fraction
            poi = np.random.poisson(self.lamb * dt, I)
              # Poisson distributed pseudo-random numbers for jump component
            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            paths[t] = paths[t - 1] * (np.exp((rt - rj-
                                        0.5 * self.volatility ** 2) * dt
                                    + self.volatility * np.sqrt(dt) * ran)
                                    + (np.exp(self.mu + self.delt *
                                        sn2[t]) - 1) * poi)
        self.instrument_values = paths


class stochastic_volatility(simulation_class):
    ''' Class to generate simulated paths based on
    the Heston (1993) stochastic volatility model.

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
    get_volatility_values :
        returns array with simulated volatility paths
    '''

    def __init__(self, name, mar_env, corr=False):
        super(stochastic_volatility, self).__init__(name, mar_env, corr)
        try:
            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')
            self.vol_vol = mar_env.get_constant('vol_vol')

            self.rho = mar_env.get_constant('rho')
            self.leverage = np.linalg.cholesky(
                np.array([[1.0, self.rho], [self.rho, 1.0]]))

            self.volatility_values = None
        except:
            print "Error parsing market environment."

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               vol_vol=None, kappa=None, theta=None, final_date=None):
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

        # Pseudo-random numbers for the stochastic volatility
        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        forward_rates = self.discount_curve.get_forward_rates(
            self.time_grid, self.paths, dtobjects=True)[1]

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            rat = np.array([ran, sn2[t]])
            rat = np.dot(self.leverage, rat)

            va_[t] = (va_[t - 1] + self.kappa
                         * (self.theta - np.maximum(0, va_[t - 1])) * dt
                         + np.sqrt(np.maximum(0, va_[t - 1]))
                         * self.vol_vol * np.sqrt(dt) * rat[1])
            va[t] = np.maximum(0, va_[t])

            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            paths[t] = paths[t - 1] * (np.exp((rt - 0.5 * va[t]) * dt
                                    + np.sqrt(va[t]) * np.sqrt(dt) * rat[0]))

            # moment matching stoch vol part
            paths[t] -= np.mean(paths[t - 1] * np.sqrt(va[t])
                            * math.sqrt(dt) * rat[0])

        self.instrument_values = paths
        self.volatility_values = np.sqrt(va)

    def get_volatility_values(self):
        if self.volatility_values is None:
            self.generate_paths(self)
        return self.volatility_values


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
            print "Error parsing market environment."

    def update(self, pricing_date=None, initial_value=None, volatility=None,
               vol_vol=None, kappa=None, theta=None, lamb=None,
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

            va_[t] = (va_[t - 1] + self.kappa
                         * (self.theta - np.maximum(0, va_[t - 1])) * dt
                         + np.sqrt(np.maximum(0, va_[t - 1]))
                         * self.vol_vol * np.sqrt(dt) * rat[1])
            va[t] = np.maximum(0, va_[t])

            poi = np.random.poisson(self.lamb * dt, I)

            rt = (forward_rates[t - 1] + forward_rates[t]) / 2
            paths[t] = paths[t - 1] * (np.exp((rt - rj - 0.5 * va[t]) * dt
                                    + np.sqrt(va[t]) * np.sqrt(dt) * rat[0])
                                    + (np.exp(self.mu + self.delt *
                                        sn2[t]) - 1) * poi)

            # moment matching stoch vol part
            paths[t] -= np.mean(paths[t - 1] * np.sqrt(va[t])
                            * math.sqrt(dt) * rat[0])

        self.instrument_values = paths
        self.volatility_values = np.sqrt(va)

    def get_volatility_values(self):
        if self.volatility_values is None:
            self.generate_paths(self)
        return self.volatility_values


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
            print "Error parsing market environment."

    def update(self, pricing_date=None, initial_value=None, volatility=None,        kappa=None, theta=None, final_date=None):
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
            paths_[t] = (paths_[t - 1] + self.kappa
                         * (self.theta - np.maximum(0, paths_[t - 1])) * dt
                         + np.sqrt(np.maximum(0, paths_[t - 1]))
                         * self.volatility * np.sqrt(dt) * ran)
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
                factor += ((dlist[d + 1] - dlist[d])
                        * (0.5 * (forward_rate[d + 1] + forward_rate[d])))
            discount_factors.append(np.exp(-factor))
        return time_list, np.array(discount_factors)

def srd_forwards(initial_value, (kappa, theta, sigma), time_grid):
    ''' Function for forward vols/rates in SRD model.

    Parameters
    ==========
    initial_value: float
        initial value of the process
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor (vol-vol)
    time_grid: list/array of datetime object
        dates to generate forwards for

    Returns
    =======
    forwards: array
        forward vols/rates
    '''
    t = get_year_deltas(time_grid)
    g = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    sum1 = ((kappa * theta * (np.exp(g * t) - 1)) /
          (2 * g + (kappa + g) * (np.exp(g * t) - 1)))
    sum2 = initial_value * ((4 * g ** 2 * np.exp(g * t)) /
            (2 * g + (kappa + g) * (np.exp(g * t) - 1)) ** 2)
    forwards = sum1 + sum2
    return forwards


class mean_reverting_diffusion(square_root_diffusion):
    ''' Class to generate simulated paths based on the
    Vasicek (1977) mean-reverting short rate model.

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
    def __init__(self, name, mar_env, corr=False, truncation=False):
        super(mean_reverting_diffusion,
               self).__init__(name, mar_env, corr)
        self.truncation = truncation

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
            if self.truncation is True:
                paths_[t] = (paths_[t - 1] + self.kappa
                             * (self.theta - np.maximum(0, paths_[t - 1])) * dt
                             + self.volatility * np.sqrt(dt) * ran)
                paths[t] = np.maximum(0, paths_[t])
            else:
                paths[t] = (paths[t - 1] + self.kappa
                             * (self.theta - paths[t - 1]) * dt
                             + self.volatility * np.sqrt(dt) * ran)
        self.instrument_values = paths


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
            print "Error parsing market environment."

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
            paths_[t, :] = (paths_[t - 1, :] + self.kappa
                         * (self.theta - np.maximum(0, paths_[t - 1, :])) * dt
                         + np.sqrt(np.maximum(0, paths_[t - 1, :]))
                         * self.volatility * np.sqrt(dt) * ran
                         + ((np.exp(self.mu + self.delt * snr[t]) - 1) * poi)
                         * np.maximum(0, paths_[t - 1, :]) - rj * dt)
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
        super(square_root_jump_diffusion_plus, self).__init__(name, mar_env, corr)
        try:
            self.term_structure = mar_env.get_curve('term_structure')
        except:
            self.term_structure = None
            print "Missing Term Structure."

        self.forward_rates = []
        self.shift_base = None
        self.shift_values = []

    def srd_forward_error(self, p0):
        if p0[0] < 0 or p0[1] < 0 or p0[2] < 0:
            return 100
        f_model = srd_forwards(self.initial_value, p0,
                               self.term_structure[:, 0])

        MSE = np.sum((self.term_structure[:, 1]
                      - f_model) ** 2) / len(f_model)
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
            self.shift_values = np.array(zip(self.time_grid,
                                         sci.splev(st, tck, der=0)))
        else:
            self.shift_values = np.array(zip(self.time_grid,
                                         np.zeros(len(self.time_grid))))

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
        forward_rates = self.discount_curve.get_forward_rates(
                        self.time_grid, dtobjects=True)
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
            paths_[t] = (paths_[t - 1] + self.kappa
                         * (self.theta - np.maximum(0, paths_[t - 1])) * dt
                         + np.sqrt(np.maximum(0, paths_[t - 1]))
                         * self.volatility * np.sqrt(dt) * ran
                         + ((np.exp(self.mu + self.delt * snr[t]) - 1) * poi)
                         * np.maximum(0, paths_[t - 1]) - rj * dt)
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
            (2 * g + (self.kappa + g) * (np.exp(g * t) - 1)) ** 2)
        self.forward_rates = np.array(zip(time_grid, sum1 + sum2))


class sabr_stochastic_volatility(simulation_class):
    ''' Class to generate simulated paths based on the SABR model.

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
    get_log_normal_implied_vol(strike, expiry)
        returns the approximation of the lognormal Black implied volatility given by
        Hagan et al. (2002)
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
            print "Error parsing market environment."

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
            print volatility
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
            fk =  self.initial_value * strike
            fk_beta = fk ** (one_beta / 2.)
            log_fk = math.log(self.initial_value / strike)
            z = self.vol_vol / self.alpha * fk_beta * log_fk
            x = math.log((math.sqrt(1. - 2 * self.rho *z + z ** 2) + z - self.rho)
                          / (1 - self.rho) )

            sigma_1 = (self.alpha / fk_beta / (1 +
                                    (one_beta_square * log_fk ** 2 / 24) +
                                    (one_beta_square ** 2 *log_fk ** 4 / 1920.))
                                     * z / x )

            sigma_ex = (one_beta_square / 24. * self.alpha ** 2 / fk_beta ** 2 +
                                    0.25 * self.rho * self.beta * self.vol_vol *
                                    self.alpha / fk_beta +
                                    (2. - 3. * self.rho ** 2) /
                                    24. * self.vol_vol ** 2)

            sigma = sigma_1 * (1. + sigma_ex * expiry)
        else:
            f_beta = self.initial_value ** one_beta
            f_two_beta = self.initial_value ** (2. -2 * self.beta)
            sigma = (self.alpha / f_beta) * ( 1 +
                            expiry * ((one_beta_square / 24. * self.alpha ** 2
                            / f_two_beta) + (0.25 * self.rho * self.beta *
                            self.vol_vol * self.alpha / f_beta) +
                            ((2. - 3 * self.rho ** 2) / 24. * self.vol_vol ** 2)))
        return sigma

    def calibrate_to_impl_vol(self, implied_vols, maturity, para = list()):
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
                e += (self.get_log_normal_implied_vol(float(strike), maturity)
                      - float(implied_vols[strike]) / 100.) ** 2
            return e
        para = fmin(error_function, para, xtol=0.0000001,
                   ftol=0.0000001, maxiter=550, maxfun=850)
        return para

    def check_parameter_set(self):
        ''' Checks if all needed parameter are set.
        '''
        parameter = ["beta", "initial_value", "vol_vol", "alpha", "rho"]
        for p in parameter:
            try:
                val = getattr(self, p)
            except:
                val = None
            if val == None:
                raise ValueError("Models %s is unset!" %p)

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
            sn1 = sn_random_numbers((1, M, I),
                                     fixed_seed=fixed_seed)
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
            p = paths[t - 1]  + va_[t] * F_b * square_root_dt * rat[0]
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

class general_underlying(object):
    ''' Needed for VAR-based portfolio modeling and valuation. '''
    def __init__(self, name, data, val_env):
        self.name = name
        self.data = data
        self.paths = val_env.get_constant('paths')
        self.frequency = 'B'
        self.discount_curve = val_env.get_curve('discount_curve')
        self.special_dates = []
        self.time_grid = val_env.get_list('time_grid')
        self.fit_model = None

    def get_instrument_values(self, fixed_seed=False):
        return self.data.values
