#
# DX Analytics
# Derivatives Instruments and Portfolio Valuation Classes
# single_risk.py
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
from ..models import *

models = {'gbm': geometric_brownian_motion,
          'jd': jump_diffusion,
          'sv': stochastic_volatility,
          'svjd': stoch_vol_jump_diffusion,
          'sabr': sabr_stochastic_volatility,
          'srd': square_root_diffusion,
          'mrd': mean_reverting_diffusion,
          'srjd': square_root_jump_diffusion,
          'srjd+': square_root_jump_diffusion_plus}


class valuation_class_single(object):
    ''' Basic class for single-risk factor instrument valuation.

    Attributes
    ==========
    name : string
        name of the object
    underlying :
        instance of simulation class
    mar_env : instance of market_environment
        market environment data for valuation
    payoff_func : string
        derivatives payoff in Python syntax
        Example: 'np.maximum(maturity_value - 100, 0)'
        where maturity_value is the NumPy vector with
        respective values of the underlying
        Example: 'np.maximum(instrument_values - 100, 0)'
        where instrument_values is the NumPy matrix with
        values of the underlying over the whole time/path grid

    Methods
    =======
    update:
        updates selected valuation parameters
    delta :
        returns the delta of the derivative
    gamma :
        returns the gamma of the derivative
    vega :
        returns the vega of the derivative
    theta :
        returns the theta of the derivative
    rho :
        returns the rho of the derivative
    '''

    def __init__(self, name, underlying, mar_env, payoff_func=''):
        try:
            self.name = name
            self.pricing_date = mar_env.pricing_date
            try:
                # strike is optional
                self.strike = mar_env.get_constant('strike')
            except:
                pass
            self.maturity = mar_env.get_constant('maturity')
            self.currency = mar_env.get_constant('currency')
            # simulation parameters and discount curve from simulation object
            self.frequency = underlying.frequency
            self.paths = underlying.paths
            self.discount_curve = underlying.discount_curve
            self.payoff_func = payoff_func
            self.underlying = underlying
            # provide pricing_date and maturity to underlying
            self.underlying.special_dates.extend([self.pricing_date,
                                                  self.maturity])
        except:
            print('Error parsing market environment.')

    def update(self, initial_value=None, volatility=None,
               strike=None, maturity=None):
        ''' Updates single parameters of the derivative. '''
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            # add new maturity date if not in time_grid
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None

    def delta(self, interval=None, accuracy=4):
        ''' Returns the delta for the derivative. '''
        if interval is None:
            interval = self.underlying.initial_value / 50.
        # forward-difference approximation
        # calculate left value for numerical delta
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        # numerical underlying value for right value
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        # reset the initial_value of the simulation object
        self.underlying.update(initial_value=initial_del - interval)
        delta = (value_right - value_left) / interval
        # correct for potential numerical errors
        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def gamma(self, interval=None, accuracy=4):
        ''' Returns the gamma for the derivative. '''
        if interval is None:
            interval = self.underlying.initial_value / 50.
        # forward-difference approximation
        # calculate left value for numerical gamma
        value_left = self.delta()
        # numerical underlying value for right value
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.delta()
        # reset the initial_value of the simulation object
        self.underlying.update(initial_value=initial_del - interval)
        gamma = (value_right - value_left) / interval
        return round(gamma, accuracy)

    def vega(self, interval=0.01, accuracy=4):
        ''' Returns the vega for the derivative. '''
        if interval < self.underlying.volatility / 50.:
            interval = self.underlying.volatility / 50.
        # forward-difference approximation
        # calculate the left value for numerical vega
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        # numerical volatility value for right value
        vola_del = self.underlying.volatility + interval
        # update the simulation object
        self.underlying.update(volatility=vola_del)
        # calculate the right value of numerical vega
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        # reset volatility value of simulation object
        self.underlying.update(volatility=vola_del - interval)
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)

    def theta(self, interval=10, accuracy=4):
        ''' Returns the theta for the derivative. '''
        # calculate the left value for numerical theta
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        # determine new pricing date
        orig_date = self.pricing_date
        new_date = orig_date + dt.timedelta(interval)
        # update the simulation object
        self.underlying.update(pricing_date=new_date)
        # calculate the right value of numerical theta
        self.pricing_date = new_date
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        # reset pricing dates of sim & val objects
        self.underlying.update(pricing_date=orig_date)
        self.pricing_date = orig_date
        # calculating the negative value by convention
        # (i.e. a decrease in time-to-maturity)
        theta = (value_right - value_left) / (interval / 365.)
        return round(theta, accuracy)

    def rho(self, interval=0.005, accuracy=4):
        ''' Returns the rho for the derivative. '''
        # calculate the left value for numerical rho
        value_left = self.present_value(fixed_seed=True, accuracy=12)
        if type(self.discount_curve) == constant_short_rate:
            # adjust constant short rate factor
            self.discount_curve.short_rate += interval
            # delete instrument values (since drift changes)
            if self.underlying.instrument_values is not None:
                iv = True
                store_underlying_values = self.underlying.instrument_values
            self.underlying.instrument_values = None
            # calculate the  right value for numerical rho
            value_right = self.present_value(fixed_seed=True, accuracy=12)
            # reset constant short rate factor
            self.discount_curve.short_rate -= interval
            if iv is True:
                self.underlying.instrument_values = store_underlying_values
            rho = (value_right - value_left) / interval
            return round(rho, accuracy)
        else:
            raise NotImplementedError(
                'Not yet implemented for this short rate model.')

    def dollar_gamma(self, key, interval=None, accuracy=4):
        ''' Returns the dollar gamma for the derivative. '''
        dollar_gamma = (0.5 * self.gamma(key, interval=interval) *
                        self.underlying_objects[key].initial_value ** 2)
        return round(dollar_gamma, accuracy)


class valuation_mcs_european_single(valuation_class_single):
    ''' Class to value European options with arbitrary payoff
    by single-factor Monte Carlo simulation.

    Methods
    =======
    generate_payoff :
        returns payoffs given the paths and the payoff function
    present_value :
        returns present value (Monte Carlo estimator)
    '''

    def generate_payoff(self, fixed_seed=False):
        '''
        Attributes
        ==========
        fixed_seed : boolean
            used same/fixed seed for valued
        '''
        try:
            # strike defined?
            strike = self.strike
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print('Maturity date not in time grid of underlying.')
        maturity_value = paths[time_index]
        # average value over whole path
        mean_value = np.mean(paths[:time_index], axis=1)
        # maximum value over whole path
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        # minimum value over whole path
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(self.payoff_func)
            return payoff
        except:
            print('Error evaluating payoff function.')

    def present_value(self, accuracy=6, fixed_seed=False, full=False):
        '''
        Attributes
        ==========
        accuracy : int
            number of decimals in returned result
        fixed_seed :
            used same/fixed seed for valuation
        '''
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)

        discount_factor = self.discount_curve.get_discount_factors(
            self.underlying.time_grid, self.paths)[1][0]

        result = np.sum(discount_factor * cash_flow) / len(cash_flow)

        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result, accuracy)


class valuation_mcs_american_single(valuation_class_single):
    ''' Class to value American options with arbitrary payoff
    by single-factor Monte Carlo simulation.
    Methods
    =======
    generate_payoff :
        returns payoffs given the paths and the payoff function
    present_value :
        returns present value (LSM Monte Carlo estimator)
        according to Longstaff-Schwartz (2001)
    '''

    def generate_payoff(self, fixed_seed=False):
        '''
        Attributes
        ==========
        fixed_seed :
            use same/fixed seed for valuation
        '''
        try:
            strike = self.strike
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index_start = int(np.where(time_grid == self.pricing_date)[0])
            time_index_end = int(np.where(time_grid == self.maturity)[0])
        except:
            print('Maturity date not in time grid of underlying.')
        instrument_values = paths[time_index_start:time_index_end + 1]
        try:
            payoff = eval(self.payoff_func)
            return instrument_values, payoff, time_index_start, time_index_end
        except:
            print('Error evaluating payoff function.')

    def present_value(self, accuracy=3, fixed_seed=False, bf=5, full=False):
        '''
        Attributes
        ==========
        accuracy : int
            number of decimals in returned result
        fixed_seed :
            used same/fixed seed for valuation
        bf : int
            number of basis functions for regression
        '''
        instrument_values, inner_values, time_index_start, time_index_end = \
            self.generate_payoff(fixed_seed=fixed_seed)
        time_list = \
            self.underlying.time_grid[time_index_start:time_index_end + 1]

        discount_factors = self.discount_curve.get_discount_factors(
            time_list, self.paths, dtobjects=True)[1]

        V = inner_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            # derive relevant discount factor for given time interval
            df = discount_factors[t] / discount_factors[t + 1]
            # regression step
            rg = np.polyfit(instrument_values[t], V * df, bf)
            # calculation of continuation values per path
            C = np.polyval(rg, instrument_values[t])
            # optimal decision step:
            # if condition is satisfied (inner value > regressed cont. value)
            # then take inner value; take actual cont. value otherwise
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
        df = discount_factors[0] / discount_factors[1]
        result = np.sum(df * V) / len(V)
        if full:
            return round(result, accuracy), df * V
        else:
            return round(result, accuracy)
