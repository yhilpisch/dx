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
from ..frame import *
from ..models import *
from .single_risk import models
import statsmodels.api as sm


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
                # strike optional
                self.strike = self.val_env.get_constant('strike')
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
            print('Error parsing market environment.')

        # Generating general time grid
        if self.time_grid is None:
            self.generate_time_grid()

        if portfolio is False:
            if self.correlations is not None:
                ul_list = sorted(self.risk_factors)
                if isinstance(self.correlations, list):
                    correlation_matrix = np.zeros((len(ul_list), len(ul_list)))
                    np.fill_diagonal(correlation_matrix, 1.0)
                    correlation_matrix = pd.DataFrame(
                        correlation_matrix, index=ul_list, columns=ul_list)
                    for corr in correlations:
                        if corr[2] >= 1.0:
                            corr[2] = 0.999999999999
                        correlation_matrix[corr[0]].ix[corr[1]] = corr[2]
                        correlation_matrix[corr[1]].ix[corr[0]] = corr[2]
                    self.correlation_matrix = correlation_matrix
                    cholesky_matrix = np.linalg.cholesky(
                        np.array(correlation_matrix))
                else:
                    # if correlation matrix was already given as pd.DataFrame
                    cholesky_matrix = np.linalg.cholesky(np.array(
                        self.correlations))

                # dictionary with index positions
                rn_set = {}
                for asset in self.risk_factors:
                    rn_set[asset] = ul_list.index(asset)

                # random numbers array
                random_numbers = sn_random_numbers(
                    (len(rn_set), len(self.time_grid),
                     self.val_env.constants['paths']),
                    fixed_seed=self.fixed_seed)

                # adding all to valuation environment
                self.val_env.add_list('cholesky_matrix', cholesky_matrix)
                self.val_env.add_list('rn_set', rn_set)
                self.val_env.add_list('random_numbers', random_numbers)
            self.generate_underlying_objects()

    def generate_time_grid(self):
        ''' Generates time grid for all relevant objects. '''
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
            self.pricing_date = pricing_date
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
        ''' Returns the delta for the specified risk factor
        for the derivative. '''
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
        ''' Returns the gamma for the specified risk factor
        for the derivative. '''
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
        dollar_gamma = (0.5 * self.gamma(key, interval=interval) *
                        self.underlying_objects[key].initial_value ** 2)
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
            print('Maturity date not in time grid of underlying.')
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
            print('Error evaluating payoff function.')

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
            time_index_start = int(
                np.where(self.time_grid == self.pricing_date)[0])
            time_index_end = int(np.where(self.time_grid == self.maturity)[0])
        except:
            print('Pricing or maturity date not in time grid of underlying.')
        instrument_values = {}
        for key, obj in self.instrument_values.items():
            instrument_values[key] = \
                self.instrument_values[key][
                    time_index_start:time_index_end + 1]
        try:
            payoff = eval(self.payoff_func)
            return instrument_values, payoff, time_index_start, time_index_end
        except:
            print('Error evaluating payoff function.')

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
            rg = sm.OLS(V * df, np.array(list(matrix.values())).T).fit()
            C = np.sum(rg.params * np.array(list(matrix.values())).T, axis=1)
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
        df = discount_factors[0] / discount_factors[1]
        result = np.sum(df * V) / len(V)
        if full:
            return round(result, accuracy), df * V
        else:
            return round(result, accuracy)
