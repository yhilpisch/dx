#
# DX Analytics
# Derivatives Instruments and Portfolio Valuation Classes
# derivatives_portfolio.py
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
from .single_risk import *
from .multi_risk import *
from .parallel_valuation import *


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

    def __init__(self, name, quantity, underlyings, mar_env,
                 otype, payoff_func):
        self.name = name
        self.quantity = quantity
        self.underlyings = underlyings
        self.mar_env = mar_env
        self.otype = otype
        self.payoff_func = payoff_func

    def get_info(self):
        print('NAME')
        print(self.name, '\n')
        print('QUANTITY')
        print(self.quantity, '\n')
        print('UNDERLYINGS')
        print(self.underlyings, '\n')
        print('MARKET ENVIRONMENT')
        print('\n**Constants**')
        for key in self.mar_env.constants:
            print(key, self.mar_env.constants[key])
        print('\n**Lists**')
        for key in self.mar_env.lists:
            print(key, self.mar_env.lists[key])
        print('\n**Curves**')
        for key in self.mar_env.curves:
            print(key, self.mar_env.curves[key])
        print('\nOPTION TYPE')
        print(self.otype, '\n')
        print('PAYOFF FUNCTION')
        print(self.payoff_func)


otypes = {'European single': valuation_mcs_european_single,
          'American single': valuation_mcs_american_single,
          'European multi': valuation_mcs_european_multi,
          'American multi': valuation_mcs_american_multi}


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
            print(bar)
            self.positions[pos].get_info()
            print(bar)

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
        res_df = pd.DataFrame(res_list,
                              columns=['position', 'name', 'quantity',
                                       'otype', 'risk_facts', 'value',
                                       'currency', 'pos_value'])
        print('Total\n', res_df[['pos_value']].sum())
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
                    fixed_seed=fixed_seed, full=True)[1] \
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
                    delta_dict[key] = round(
                        self.valuation_objects[pos].delta(key) *
                        self.positions[pos].quantity, 6)
                    vega_dict[key] = round(
                        self.valuation_objects[pos].vega(key) *
                        self.positions[pos].quantity, 6)
                pos_list.append(str(delta_dict))
                pos_list.append(str(vega_dict))
            else:
                if self.parallel is True:
                    # delta from parallel calculation
                    pos_list.append(delta_list[pos] *
                                    self.positions[pos].quantity)
                    # vega from parallel calculation
                    pos_list.append(vega_list[pos] *
                                    self.positions[pos].quantity)
                else:
                    # delta per position
                    pos_list.append(self.valuation_objects[pos].delta() *
                                    self.positions[pos].quantity)
                    # vega per position
                    pos_list.append(self.valuation_objects[pos].vega() *
                                    self.positions[pos].quantity)
            res_list.append(pos_list)
        res_df = pd.DataFrame(res_list, columns=['position', 'name',
                                                 'quantity', 'otype',
                                                 'risk_facts', 'value',
                                                 'currency', 'pos_value',
                                                 'pos_delta', 'pos_vega'])
        print('Totals\n',
              res_df[['pos_value', 'pos_delta', 'pos_vega']].sum())
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
            print('\n' + rf)
            in_val = self.underlying_objects[rf].initial_value
            in_vol = self.underlying_objects[rf].volatility
            results = []
            for level in levels:
                values_sens = copy.deepcopy(values)
                print(level)
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

            sensitivities[rf + '_' + Greek] = pd.DataFrame(
                np.array(results), index=levels, columns=['factor', 'value'])
        print(2 * '\n')
        return pd.Panel(sensitivities), sum(values.values())

    def get_deltas(self, net=True, low=0.9, high=1.1, step=0.05):
        ''' Returns the deltas of the portfolio. Convenience function.'''
        deltas, benchvalue = self.dx_port.get_port_risk(
            Greek='Delta', low=low, high=high, step=step)
        if net is True:
            deltas.loc[:, :, 'value'] -= benchvalue
        return deltas, benchvalue

    def get_vegas(self, net=True, low=0.9, high=1.1, step=0.05):
        ''' Returns the vegas of the portfolio. Convenience function.'''
        vegas, benchvalue = self.dx_port.get_port_risk(
            Greek='Vega', low=low, high=high, step=step)
        if net is True:
            vegas.loc[:, :, 'value'] -= benchvalue
        return vegas, benchvalue


def risk_report(sensitivities, digits=2, gross=True):
    if gross is True:
        for key in sensitivities:
            print('\n' + key)
            print(np.round(sensitivities[key].transpose(), digits))
    else:
        print(np.round(sensitivities, digits))
