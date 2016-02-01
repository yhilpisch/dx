#
# DX Analytics
# Derivatives Instruments and Portfolio Valuation Classes
# dx_valuation.py
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

from dx_models import *
import statsmodels.api as sm


# Classes for single risk factor instrument valuation

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
                self.strike = mar_env.get_constant('strike')
                  # strike is optional
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
            print "Error parsing market environment."

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
            if not maturity in self.underlying.time_grid:
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
        dollar_gamma = (0.5 * self.gamma(key, interval=interval)
                    * self.underlying_objects[key].initial_value ** 2)
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
            print "Maturity date not in time grid of underlying."
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
            print "Error evaluating payoff function."

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
            print "Maturity date not in time grid of underlying."
        instrument_values = paths[time_index_start:time_index_end + 1]
        try:
            payoff = eval(self.payoff_func)
            return instrument_values, payoff, time_index_start, time_index_end
        except:
            print "Error evaluating payoff function."

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


# Classes for multi risk factor instrument valuation

class valuation_class_multi(object):
    ''' Basic class for multi-risk factor instrument valuation.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for valuation
    underlyings : dictionary
        instances of model classes
    correlations : list
        correlations between underlyings
    payoff_func : string
        derivatives payoff in Python syntax
        Example: 'np.maximum(maturity_value[key] - 100, 0)'
        where maturity_value[key] is the NumPy vector with
        respective values of the underlying 'key' from the
        risk_factors dictionary

    Methods
    =======
    update:
        updates selected valuation parameters
    delta :
        returns the delta of the derivative
    vega :
        returns the vega of the derivative
    '''
    def __init__(self, name, val_env, risk_factors=None, correlations=None,
                 payoff_func='', fixed_seed=False, portfolio=False):
        try:
            self.name = name
            self.val_env = val_env
            self.currency = self.val_env.get_constant('currency')
            self.pricing_date = val_env.pricing_date
            try:
                self.strike = self.val_env.get_constant('strike')
                  # strike optional
            except:
                pass
            self.maturity = self.val_env.get_constant('maturity')
            self.frequency = self.val_env.get_constant('frequency')
            self.paths = self.val_env.get_constant('paths')
            self.discount_curve = self.val_env.get_curve('discount_curve')
            self.risk_factors = risk_factors
            self.underlyings = set()
            if portfolio is False:
                self.underlying_objects = {}
            else:
                self.underlying_objects = risk_factors
            self.correlations = correlations
            self.payoff_func = payoff_func
            self.fixed_seed = fixed_seed
            self.instrument_values = {}
            try:
                self.time_grid = self.val_env.get_curve('time_grid')
            except:
                self.time_grid = None
            self.correlation_matrix = None
        except:
            print "Error parsing market environment."

        # Generating general time grid
        if self.time_grid is None:
            self.generate_time_grid()

        if portfolio is False:
            if correlations is not None:
                ul_list = sorted(self.risk_factors)
                correlation_matrix = np.zeros((len(ul_list), len(ul_list)))
                np.fill_diagonal(correlation_matrix, 1.0)
                correlation_matrix = pd.DataFrame(correlation_matrix,
                                     index=ul_list, columns=ul_list)
                for corr in correlations:
                    if corr[2] >= 1.0:
                        corr[2] = 0.999999999999
                    correlation_matrix[corr[0]].ix[corr[1]] = corr[2]
                    correlation_matrix[corr[1]].ix[corr[0]] = corr[2]
                self.correlation_matrix = correlation_matrix
                cholesky_matrix = np.linalg.cholesky(
                    np.array(correlation_matrix))

                # dictionary with index positions
                rn_set = {}
                for asset in self.risk_factors:
                    rn_set[asset] = ul_list.index(asset)

                # random numbers array
                random_numbers = sn_random_numbers((len(rn_set),
                                          len(self.time_grid),
                                          self.val_env.constants['paths']),
                                          fixed_seed=self.fixed_seed)

                # adding all to valuation environment
                self.val_env.add_list('cholesky_matrix', cholesky_matrix)
                self.val_env.add_list('rn_set', rn_set)
                self.val_env.add_list('random_numbers', random_numbers)
            self.generate_underlying_objects()


    def generate_time_grid(self):
        ''' Generats time grid for all relevant objects. '''
        start = self.val_env.get_constant('starting_date')
        end = self.val_env.get_constant('final_date')
        maturity = self.maturity
        time_grid = pd.date_range(start=start, end=end,
                            freq=self.val_env.get_constant('frequency')
                                  ).to_pydatetime()
        if start in time_grid and end in time_grid and \
                maturity in time_grid:
            self.time_grid = time_grid
        else:
            time_grid = list(time_grid)
            if maturity not in time_grid:
                time_grid.insert(0, maturity)
            if start not in time_grid:
                time_grid.insert(0, start)
            if end not in time_grid:
                time_grid.append(end)
            time_grid.sort()
            self.time_grid = np.array(time_grid)
        self.val_env.add_curve('time_grid', self.time_grid)

    def generate_underlying_objects(self):
        for asset in self.risk_factors:
            mar_env = self.risk_factors[asset]
            mar_env.add_environment(self.val_env)
            model = models[mar_env.constants['model']]
            if self.correlations is not None:
                self.underlying_objects[asset] = model(asset,
                                                       mar_env, True)
            else:
                self.underlying_objects[asset] = model(asset,
                                                       mar_env, False)

    def get_instrument_values(self, fixed_seed=True):
        for obj in self.underlying_objects.values():
            if obj.instrument_values is None:
                obj.generate_paths(fixed_seed=fixed_seed)

    def update(self, key=None, pricing_date=None, initial_value=None,
                volatility=None, short_rate=None, strike=None, maturity=None):
        ''' Updates parameters of the derivative. '''
        if key is not None:
            underlying = self.underlying_objects[key]
        if pricing_date is not None:
            self.pricing_date=pricing_date
            self.val_env.add_constant('starting_date', pricing_date)
            self.generate_time_grid()
            self.generate_underlying_objects()
        if initial_value is not None:
            underlying.update(initial_value=initial_value)
        if volatility is not None:
            underlying.update(volatility=volatility)
        if short_rate is not None:
            self.val_env.curves['discount_curve'].short_rate = short_rate
            self.discount_curve.short_rate = short_rate
            self.generate_underlying_objects()
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            for underlying in underlyings.values():
                underlying.update(final_date=self.maturity)
        self.get_instrument_values()

    def delta(self, key, interval=None, accuracy=4):
        ''' Returns the delta for the specified risk factor for the derivative. '''
        if len(self.instrument_values) == 0:
            self.get_instrument_values()
        asset = self.underlying_objects[key]
        if interval is None:
            interval = asset.initial_value / 50.
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        start_value = asset.initial_value
        initial_del = start_value + interval
        asset.update(initial_value=initial_del)
        self.get_instrument_values()
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        asset.update(initial_value=start_value)
        self.instrument_values = {}
        delta = (value_right - value_left) / interval
        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def gamma(self, key, interval=None, accuracy=4):
        ''' Returns the gamma for the specified risk factor for the derivative. '''
        if len(self.instrument_values) == 0:
            self.get_instrument_values()
        asset = self.underlying_objects[key]
        if interval is None:
            interval = asset.initial_value / 50.
        # forward-difference approximation
        # calculate left value for numerical gamma
        value_left = self.delta(key=key)
        # numerical underlying value for right value
        initial_del = asset.initial_value + interval
        asset.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.delta(key=key)
        # reset the initial_value of the simulation object
        asset.update(initial_value=initial_del - interval)
        gamma = (value_right - value_left) / interval
        return round(gamma, accuracy)

    def vega(self, key, interval=0.01, accuracy=4):
        ''' Returns the vega for the specified risk factor. '''
        if len(self.instrument_values) == 0:
            self.get_instrument_values()
        asset = self.underlying_objects[key]
        if interval < asset.volatility / 50.:
            interval = asset.volatility / 50.
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        start_vola = asset.volatility
        vola_del = start_vola + interval
        asset.update(volatility=vola_del)
        self.get_instrument_values()
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        asset.update(volatility=start_vola)
        self.instrument_values = {}
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)

    def theta(self, interval=10, accuracy=4):
        ''' Returns the theta for the derivative. '''
        if len(self.instrument_values) == 0:
            self.get_instrument_values()
        # calculate the left value for numerical theta
        value_left = self.present_value(fixed_seed=True, accuracy=10)
        # determine new pricing date
        orig_date = self.pricing_date
        new_date = orig_date + dt.timedelta(interval)
        # calculate the right value of numerical theta
        self.update(pricing_date=new_date)
        value_right = self.present_value(fixed_seed=True, accuracy=10)
        # reset pricing dates of valuation & underlying objects
        self.update(pricing_date=orig_date)
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
            orig_short_rate = self.discount_curve.short_rate
            new_short_rate = orig_short_rate + interval
            self.update(short_rate=new_short_rate)
            # self.discount_curve.short_rate += interval
            # delete instrument values (since drift changes)
            # for asset in self.underlying_objects.values():
            #    asset.instrument_values = None
            # calculate the  right value for numerical rho
            value_right = self.present_value(fixed_seed=True, accuracy=12)
            # reset constant short rate factor
            self.update(short_rate=orig_short_rate)
            # self.discount_curve.short_rate -= interval
            rho = (value_right - value_left) / interval
            return round(rho, accuracy)
        else:
            raise NotImplementedError(
                    'Not yet implemented for this short rate model.')

    def dollar_gamma(self, key, interval=None, accuracy=4):
        ''' Returns the dollar gamma for the specified risk factor. '''
        dollar_gamma = (0.5 * self.gamma(key, interval=interval)
                    * self.underlying_objects[key].initial_value ** 2)
        return round(dollar_gamma, accuracy)


class valuation_mcs_european_multi(valuation_class_multi):
    ''' Class to value European options with arbitrary payoff
    by multi-risk factor Monte Carlo simulation.

    Methods
    =======
    generate_payoff :
        returns payoffs given the paths and the payoff function
    present_value :
        returns present value (Monte Carlo estimator)
    '''
    def generate_payoff(self, fixed_seed=True):
        self.get_instrument_values(fixed_seed=True)
        paths = {key: name.instrument_values for key, name
                       in self.underlying_objects.items()}
        time_grid = self.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print "Maturity date not in time grid of underlying."
        maturity_value = {}
        mean_value = {}
        max_value = {}
        min_value = {}
        for key in paths:
            maturity_value[key] = paths[key][time_index]
            mean_value[key] = np.mean(paths[key][:time_index], axis=1)
            max_value[key] = np.amax(paths[key][:time_index], axis=1)
            min_value[key] = np.amin(paths[key][:time_index], axis=1)
        try:
            payoff = eval(self.payoff_func)
            return payoff
        except:
            print "Error evaluating payoff function."

    def present_value(self, accuracy=3, fixed_seed=True, full=False):
        cash_flow = self.generate_payoff(fixed_seed)

        discount_factor = self.discount_curve.get_discount_factors(
                          self.time_grid, self.paths)[1][0]

        result = np.sum(discount_factor * cash_flow) / len(cash_flow)
        if full:
            return round(result, accuracy), df * cash_flow
        else:
            return round(result, accuracy)


class valuation_mcs_american_multi(valuation_class_multi):
    ''' Class to value American options with arbitrary payoff
    by multi-risk factor Monte Carlo simulation.

    Methods
    =======
    generate_payoff :
        returns payoffs given the paths and the payoff function
    present_value :
        returns present value (Monte Carlo estimator)
    '''
    def generate_payoff(self, fixed_seed=True):
        self.get_instrument_values(fixed_seed=True)
        self.instrument_values = {key: name.instrument_values for key, name
                       in self.underlying_objects.items()}
        try:
            time_index_start = int(np.where(self.time_grid == self.pricing_date)[0])
            time_index_end = int(np.where(self.time_grid == self.maturity)[0])
        except:
            print "Pricing date or maturity date not in time grid of underlying."
        instrument_values = {}
        for key, obj in self.instrument_values.items():
            instrument_values[key] = \
                self.instrument_values[key][time_index_start:time_index_end
                                            + 1]
        try:
            payoff = eval(self.payoff_func)
            return instrument_values, payoff, time_index_start, time_index_end
        except:
            print "Error evaluating payoff function."
    def present_value(self, accuracy=3, fixed_seed=True, full=False):
        instrument_values, inner_values, time_index_start, time_index_end = \
                    self.generate_payoff(fixed_seed=fixed_seed)
        time_list = self.time_grid[time_index_start:time_index_end + 1]

        discount_factors = self.discount_curve.get_discount_factors(
                            time_list, self.paths, dtobjects=True)[1]

        V = inner_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            df = discount_factors[t] / discount_factors[t + 1]
            matrix = {}
            for asset_1 in instrument_values.keys():
                matrix[asset_1] = instrument_values[asset_1][t]
                for asset_2 in instrument_values.keys():
                    matrix[asset_1 + asset_2] = instrument_values[asset_1][t] \
                        * instrument_values[asset_2][t]
            rg = sm.OLS(V * df, np.array(matrix.values()).T).fit()
            C = np.sum(rg.params * np.array(matrix.values()).T, axis=1)
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
        df = discount_factors[0] / discount_factors[1]
        result = np.sum(df * V) / len(V)
        if full:
            return round(result, accuracy), df * V
        else:
            return round(result, accuracy)


# Classes for derivatives portfolio valuation
class derivatives_position(object):
    ''' Class to model a derivatives position.

    Attributes
    ==========

    name : string
        name of the object
    quantity : float
        number of derivatives instruments making up the position
    underlyings : list of strings
        names of risk_factors/risk factors for the derivative
    mar_env : instance  of market_environment
        constants, lists and curves relevant for valuation_class
    otype : string
        valuation class to use
    payoff_func : string
        payoff string for the derivative

    Methods
    =======
    get_info :
        prints information about the derivative position
    '''
    def __init__(self, name, quantity, underlyings, mar_env, otype, payoff_func):
        self.name = name
        self.quantity = quantity
        self.underlyings = underlyings
        self.mar_env = mar_env
        self.otype = otype
        self.payoff_func = payoff_func

    def get_info(self):
        print "NAME"
        print self.name, '\n'
        print "QUANTITY"
        print self.quantity, '\n'
        print "UNDERLYINGS"
        print self.underlyings, '\n'
        print "MARKET ENVIRONMENT"
        print "\n**Constants**"
        for key in self.mar_env.constants:
            print key, self.mar_env.constants[key]
        print "\n**Lists**"
        for key in self.mar_env.lists:
            print key, self.mar_env.lists[key]
        print "\n**Curves**"
        for key in self.mar_env.curves:
            print key, self.mar_env.curves[key]
        print "\nOPTION TYPE"
        print self.otype, '\n'
        print "PAYOFF FUNCTION"
        print self.payoff_func


models = {'gbm' : geometric_brownian_motion,
          'jd' : jump_diffusion,
          'sv' : stochastic_volatility,
          'svjd' : stoch_vol_jump_diffusion,
          'sabr' : sabr_stochastic_volatility,
          'srd' : square_root_diffusion,
          'mrd' : mean_reverting_diffusion,
          'srjd' : square_root_jump_diffusion,
          'srjd+' : square_root_jump_diffusion_plus}

otypes = {'European single' : valuation_mcs_european_single,
          'American single' : valuation_mcs_american_single,
          'European multi' : valuation_mcs_european_multi,
          'American multi' : valuation_mcs_american_multi}


class derivatives_portfolio(object):
    ''' Class for building and valuing portfolios of derivatives positions.

    Attributes
    ==========
    name : str
        name of the object
    positions : dict
        dictionary of positions (instances of derivatives_position class)
    val_env : market_environment
        market environment for the valuation
    risk_factors : dict
        dictionary of market environments for the risk_factors
    correlations : list or pd.DataFrame
        correlations between risk_factors
    fixed_seed : boolean
        flag for fixed rng seed

    Methods
    =======
    get_positions :
        prints information about the single portfolio positions
    get_values :
        estimates and returns positions values
    get_present_values :
        returns the full distribution of the simulated portfolio values
    get_statistics :
        returns a pandas DataFrame object with portfolio statistics
    get_port_risk :
        estimates sensitivities for point-wise parameter shocks
    '''

    def __init__(self, name, positions, val_env, risk_factors,
                 correlations=None, fixed_seed=False, parallel=False):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.risk_factors = risk_factors
        self.underlyings = set()
        if correlations is None or correlations is False:
            self.correlations = None
        else:
            self.correlations = correlations
        self.time_grid = None
        self.underlying_objects = {}
        self.valuation_objects = {}
        self.fixed_seed = fixed_seed
        self.parallel = parallel
        self.special_dates = []
        for pos in self.positions:
            # determine earliest starting_date
            self.val_env.constants['starting_date'] = \
                min(self.val_env.constants['starting_date'],
                    positions[pos].mar_env.pricing_date)
            # determine latest date of relevance
            self.val_env.constants['final_date'] = \
                max(self.val_env.constants['final_date'],
                    positions[pos].mar_env.constants['maturity'])
            # collect all underlyings
            # add to set; avoids redundancy
            for ul in positions[pos].underlyings:
                self.underlyings.add(ul)

        # generating general time grid
        start = self.val_env.constants['starting_date']
        end = self.val_env.constants['final_date']
        time_grid = pd.date_range(start=start, end=end,
                                  freq=self.val_env.constants['frequency']
                                  ).to_pydatetime()
        time_grid = list(time_grid)
        for pos in self.positions:
            maturity_date = positions[pos].mar_env.constants['maturity']
            if maturity_date not in time_grid:
                time_grid.insert(0, maturity_date)
                self.special_dates.append(maturity_date)
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        # delete duplicate entries
        # & sort dates in time_grid
        time_grid = sorted(set(time_grid))

        self.time_grid = np.array(time_grid)
        self.val_env.add_list('time_grid', self.time_grid)

        # taking care of correlations
        ul_list = sorted(self.underlyings)
        correlation_matrix = np.zeros((len(ul_list), len(ul_list)))
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = pd.DataFrame(correlation_matrix,
                                        index=ul_list, columns=ul_list)

        if self.correlations is not None:
            if isinstance(self.correlations, list):
                # if correlations are given as list of list/tuple objects
                for corr in self.correlations:
                    if corr[2] >= 1.0:
                        corr[2] = 0.999999999999
                    if corr[2] <= -1.0:
                        corr[2] = -0.999999999999
                    # fill correlation matrix
                    correlation_matrix[corr[0]].ix[corr[1]] = corr[2]
                    correlation_matrix[corr[1]].ix[corr[0]] = corr[2]
                # determine Cholesky matrix
                cholesky_matrix = np.linalg.cholesky(np.array(
                                                    correlation_matrix))
            else:
                # if correlation matrix was already given as pd.DataFrame
                cholesky_matrix = np.linalg.cholesky(np.array(
                                                    self.correlations))
        else:
            cholesky_matrix = np.linalg.cholesky(np.array(
                                                    correlation_matrix))

        # dictionary with index positions for the
        # slice of the random number array to be used by
        # respective underlying
        rn_set = {}
        for asset in self.underlyings:
            rn_set[asset] = ul_list.index(asset)

        # random numbers array, to be used by
        # all underlyings (if correlations exist)
        random_numbers = sn_random_numbers(
            (len(rn_set),
             len(self.time_grid),
             self.val_env.constants['paths']),
             fixed_seed=self.fixed_seed)

        # adding all to valuation environment which is
        # to be shared with every underlying
        self.val_env.add_list('correlation_matrix', correlation_matrix)
        self.val_env.add_list('cholesky_matrix', cholesky_matrix)
        self.val_env.add_list('random_numbers', random_numbers)
        self.val_env.add_list('rn_set', rn_set)

        for asset in self.underlyings:
            # select market environment of asset
            mar_env = self.risk_factors[asset]
            # add valuation environment to market environment
            mar_env.add_environment(val_env)
            # select the right simulation class
            model = models[mar_env.constants['model']]
            # instantiate simulation object
            if self.correlations is not None:
                corr = True
            else:
                corr = False
            self.underlying_objects[asset] = model(asset, mar_env,
                                                       corr=corr)

        for pos in positions:
            # select right valuation class (European, American)
            val_class = otypes[positions[pos].otype]
            # pick the market environment and add the valuation environment
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # instantiate valuation class single risk vs. multi risk
            if self.positions[pos].otype[-5:] == 'multi':
                underlying_objects = {}
                for obj in positions[pos].underlyings:
                    underlying_objects[obj] = self.underlying_objects[obj]
                self.valuation_objects[pos] = \
                    val_class(name=positions[pos].name,
                              val_env=mar_env,
                              risk_factors=underlying_objects,
                              payoff_func=positions[pos].payoff_func,
                              portfolio=True)
            else:
                self.valuation_objects[pos] = \
                    val_class(name=positions[pos].name,
                              mar_env=mar_env,
                              underlying=self.underlying_objects[
                                              positions[pos].underlyings[0]],
                              payoff_func=positions[pos].payoff_func)

    def get_positions(self):
        ''' Convenience method to get information about
        all derivatives positions in a portfolio. '''
        for pos in self.positions:
            bar = '\n' + 50 * '-'
            print bar
            self.positions[pos].get_info()
            print bar

    def get_values(self, fixed_seed=False):
        ''' Providing portfolio position values. '''
        res_list = []
        if self.parallel is True:
            self.underlying_objects = \
                simulate_parallel(self.underlying_objects.values())
            results = value_parallel(self.valuation_objects.values())
        # iterate over all positions in portfolio
        for pos in self.valuation_objects:
            pos_list = []
            if self.parallel is True:
                present_value = results[self.valuation_objects[pos].name]
            else:
                present_value = self.valuation_objects[pos].present_value()
            pos_list.append(pos)
            pos_list.append(self.positions[pos].name)
            pos_list.append(self.positions[pos].quantity)
            pos_list.append(self.positions[pos].otype)
            pos_list.append(self.positions[pos].underlyings)
            # calculate all present values for the single instruments
            pos_list.append(present_value)
            pos_list.append(self.valuation_objects[pos].currency)
            # single instrument value times quantity
            pos_list.append(present_value * self.positions[pos].quantity)
            res_list.append(pos_list)
        res_df = pd.DataFrame(res_list, columns=['position', 'name', 'quantity',
                                                 'otype', 'risk_facts', 'value',
                                                 'currency', 'pos_value'])
        print 'Total\n', res_df[['pos_value']].sum()
        return res_df

    def get_present_values(self, fixed_seed=False):
        ''' Get full distribution of present values. '''
        present_values = np.zeros(self.val_env.get_constant('paths'))
        if self.parallel is True:
            self.underlying_objects = \
                simulate_parallel(self.underlying_objects.values())
            results = value_parallel(self.valuation_objects.values(),
                                     full=True)
            for pos in self.valuation_objects:
                present_values += results[self.valuation_objects[pos].name] \
                                    * self.positions[pos].quantity
        else:
            for pos in self.valuation_objects:
                present_values += self.valuation_objects[pos].present_value(
                                    fixed_seed = fixed_seed, full=True)[1] \
                                    * self.positions[pos].quantity
        return present_values


    def get_statistics(self, fixed_seed=None):
        ''' Providing position statistics. '''
        res_list = []
        if fixed_seed is None:
            fixed_seed = self.fixed_seed
        if self.parallel is True:
            self.underlying_objects = \
                simulate_parallel(self.underlying_objects.values())
            results = value_parallel(self.valuation_objects.values(),
                                    fixed_seed=fixed_seed)
            delta_list = greeks_parallel(self.valuation_objects.values(),
                                        Greek='Delta')
            vega_list = greeks_parallel(self.valuation_objects.values(),
                                        Greek='Vega')
        # iterate over all positions in portfolio
        for pos in self.valuation_objects:
            pos_list = []
            if self.parallel is True:
                present_value = results[self.valuation_objects[pos].name]
            else:
                present_value = self.valuation_objects[pos].present_value(
                                fixed_seed=fixed_seed, accuracy=3)
            pos_list.append(pos)
            pos_list.append(self.positions[pos].name)
            pos_list.append(self.positions[pos].quantity)
            pos_list.append(self.positions[pos].otype)
            pos_list.append(self.positions[pos].underlyings)
            # calculate all present values for the single instruments
            pos_list.append(present_value)
            pos_list.append(self.valuation_objects[pos].currency)
            # single instrument value times quantity
            pos_list.append(present_value * self.positions[pos].quantity)
            if self.positions[pos].otype[-5:] == 'multi':
                # multiple delta and vega values for multi-risk derivatives
                delta_dict = {}
                vega_dict = {}
                for key in self.valuation_objects[pos].underlying_objects.keys():
                    # delta and vega per position and underlying
                    delta_dict[key] = round(self.valuation_objects[pos].delta(key)
                                            * self.positions[pos].quantity, 6)
                    vega_dict[key] = round(self.valuation_objects[pos].vega(key)
                                           * self.positions[pos].quantity, 6)
                pos_list.append(str(delta_dict))
                pos_list.append(str(vega_dict))
            else:
                if self.parallel is True:
                    # delta from parallel calculation
                    pos_list.append(delta_list[pos]
                                   * self.positions[pos].quantity)
                    # vega from parallel calculation
                    pos_list.append(vega_list[pos]
                                   * self.positions[pos].quantity)
                else:
                    # delta per position
                    pos_list.append(self.valuation_objects[pos].delta()
                                    * self.positions[pos].quantity)
                    # vega per position
                    pos_list.append(self.valuation_objects[pos].vega()
                                    * self.positions[pos].quantity)
            res_list.append(pos_list)
        res_df = pd.DataFrame(res_list, columns=['position', 'name',
                                                 'quantity', 'otype',
                                                 'risk_facts', 'value',
                                                 'currency', 'pos_value',
                                                 'pos_delta', 'pos_vega'])
        print 'Totals\n', res_df[['pos_value', 'pos_delta', 'pos_vega']].sum()
        return res_df

    def get_port_risk(self, Greek='Delta', low=0.8, high=1.2, step=0.1,
                      fixed_seed=None, risk_factors=None):
        ''' Calculating portfolio risk statistics. '''
        if risk_factors is None:
            risk_factors = self.underlying_objects.keys()
        if fixed_seed is None:
            fixed_seed = self.fixed_seed
        sensitivities = {}
        levels = np.arange(low, high + 0.01, step)
        if self.parallel is True:
            values = value_parallel(self.valuation_objects.values(),
                                    fixed_seed=fixed_seed)
            for key in self.valuation_objects:
                values[key] *= self.positions[key].quantity
        else:
            values = {}
            for key, obj in self.valuation_objects.items():
                values[key] = obj.present_value() \
                            * self.positions[key].quantity
        import copy
        for rf in risk_factors:
            print '\n' + rf
            in_val = self.underlying_objects[rf].initial_value
            in_vol = self.underlying_objects[rf].volatility
            results = []
            for level in levels:
                values_sens = copy.deepcopy(values)
                print level,
                if level == 1.0:
                    pass
                else:
                    for key, obj in self.valuation_objects.items():
                        if rf in self.positions[key].underlyings:

                            if self.positions[key].otype[-5:] == 'multi':
                                if Greek == 'Delta':
                                    obj.underlying_objects[rf].update(
                                            initial_value=level * in_val)
                                if Greek == 'Vega':
                                    obj.underlying_objects[rf].update(
                                            volatility=level * in_vol)

                            else:
                                if Greek == 'Delta':
                                    obj.underlying.update(
                                        initial_value=level * in_val)
                                elif Greek == 'Vega':
                                    obj.underlying.update(
                                        volatility=level * in_vol)

                            values_sens[key] = obj.present_value(
                                            fixed_seed=fixed_seed) \
                                             * self.positions[key].quantity

                            if self.positions[key].otype[-5:] == 'multi':
                                obj.underlying_objects[rf].update(
                                        initial_value=in_val)
                                obj.underlying_objects[rf].update(
                                        volatility=in_vol)

                            else:
                                obj.underlying.update(initial_value=in_val)
                                obj.underlying.update(volatility=in_vol)

                if Greek == 'Delta':
                    results.append((round(level * in_val, 2),
                                    sum(values_sens.values())))
                if Greek == 'Vega':
                    results.append((round(level * in_vol, 2),
                                    sum(values_sens.values())))

            sensitivities[rf + '_' + Greek] = pd.DataFrame(np.array(results),
                                            index=levels,
                                            columns=['factor', 'value'])
        print 2 * '\n'
        return pd.Panel(sensitivities), sum(values.values())

    def get_deltas(self, net=True, low=0.9, high=1.1, step=0.05):
        ''' Returns the deltas of the portfolio. Convenience function.'''
        deltas, benchvalue = self.dx_port.get_port_risk(Greek='Delta',
                                                low=low, high=high, step=step)
        if net is True:
            deltas.loc[:, :, 'value'] -= benchvalue
        return deltas, benchvalue

    def get_vegas(self, net=True, low=0.9, high=1.1, step=0.05):
        ''' Returns the vegas of the portfolio. Convenience function.'''
        vegas, benchvalue = self.dx_port.get_port_risk(Greek='Vega',
                                                low=low, high=high, step=step)
        if net is True:
            vegas.loc[:, :, 'value'] -= benchvalue
        return vegas, benchvalue

def risk_report(sensitivities, digits=2):
    for key in sensitivities:
        print '\n' + key
        print np.round(sensitivities[key].transpose(), digits)


import multiprocessing as mp


def simulate_parallel(objs, fixed_seed=True):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        o.generate_paths(fixed_seed=fixed_seed)
        output.put((o.name, o))
    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    results = [output.get() for o in objs]
    underlying_objects = {}
    for o in results:
        underlying_objects[o[0]] = o[1]
    return underlying_objects


def value_parallel(objs, fixed_seed=True, full=False):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if full is True:
            pvs = o.present_value(fixed_seed=fixed_seed, full=True)[1]
            output.put((o.name, pvs))
        else:
            pv = o.present_value(fixed_seed=fixed_seed)
            output.put((o.name, pv))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    res_list = [output.get() for o in objs]
    results = {}
    for o in res_list:
        results[o[0]] = o[1]
    return results

def greeks_parallel(objs, Greek='Delta'):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if Greek == 'Delta':
            output.put((o.name, o.delta()))
        elif Greek == 'Vega':
            output.put((o.name, o.vega()))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    res_list = [output.get() for o in objs]
    results = {}
    for o in res_list:
        results[o[0]] = o[1]
    return results


class var_derivatives_portfolio(derivatives_portfolio):
    ''' Class for building and valuing portfolios of derivatives positions
    with risk factors given from fitted VAR model.

    Attributes
    ==========
    name : str
        name of the object
    positions : dict
        dictionary of positions (instances of derivatives_position class)
    val_env : market_environment
        market environment for the valuation
    var_risk_factors : VAR model
        vector autoregressive model for risk factors
    fixed_seed : boolean
        flag for fixed rng seed

    Methods
    =======
    get_positions :
        prints information about the single portfolio positions
    get_values :
        estimates and returns positions values
    get_present_values :
        returns the full distribution of the simulated portfolio values
    '''

    def __init__(self, name, positions, val_env, var_risk_factors,
                 fixed_seed=False, parallel=False):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.var_risk_factors = var_risk_factors
        self.underlyings = set()

        self.time_grid = None
        self.underlying_objects = {}
        self.valuation_objects = {}
        self.fixed_seed = fixed_seed
        self.special_dates = []
        for pos in self.positions:
            # determine earliest starting_date
            self.val_env.constants['starting_date'] = \
                min(self.val_env.constants['starting_date'],
                    positions[pos].mar_env.pricing_date)
            # determine latest date of relevance
            self.val_env.constants['final_date'] = \
                max(self.val_env.constants['final_date'],
                    positions[pos].mar_env.constants['maturity'])
            # collect all underlyings
            # add to set; avoids redundancy
            for ul in positions[pos].underlyings:
                self.underlyings.add(ul)

        # generating general time grid
        start = self.val_env.constants['starting_date']
        end = self.val_env.constants['final_date']
        time_grid = pd.date_range(start=start, end=end,
                                  freq='B' # allow business day only
                                  ).to_pydatetime()
        time_grid = list(time_grid)

        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        # delete duplicate entries & sort dates in time_grid
        time_grid = sorted(set(time_grid))

        self.time_grid = np.array(time_grid)
        self.val_env.add_list('time_grid', self.time_grid)

        #
        # generate simulated paths
        #
        self.fit_model = var_risk_factors.fit(maxlags=5, ic='bic')
        sim_paths = self.fit_model.simulate(
                        paths=self.val_env.get_constant('paths'),
                        steps=len(self.time_grid),
                        initial_values=var_risk_factors.y[-1])
        symbols = sim_paths[0].columns.values
        for sym in symbols:
            df = pd.DataFrame()
            for i, path in enumerate(sim_paths):
                df[i] = path[sym]
            self.underlying_objects[sym] = general_underlying(
                                        sym, df, self.val_env)
        for pos in positions:
            # select right valuation class (European, American)
            val_class = otypes[positions[pos].otype]
            # pick the market environment and add the valuation environment
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # instantiate valuation classes
            self.valuation_objects[pos] = \
                val_class(name=positions[pos].name,
                          mar_env=mar_env,
                          underlying=self.underlying_objects[
                                          positions[pos].underlyings[0]],
                          payoff_func=positions[pos].payoff_func)

    def get_statistics(self):
        raise NotImplementedError
    def get_port_risk(self):
        raise NotImplementedError

