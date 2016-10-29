#
# DX Analytics
# Mean Variance Portfolio
# portfolio.py
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
from .frame import *
import scipy.optimize as sco
import scipy.interpolate as sci
from pandas_datareader import data as web


class mean_variance_portfolio(object):
    '''
    Class to implement the mean variance portfolio theory of Markowitz
    '''

    def __init__(self, name, mar_env):
        self.name = name
        try:
            self.symbols = mar_env.get_list('symbols')
            self.start_date = mar_env.pricing_date
        except:
            raise ValueError('Error parsing market environment.')

        self.number_of_assets = len(self.symbols)
        try:
            self.final_date = mar_env.get_constant('final date')
        except:
            self.final_date = dt.date.today()
        try:
            self.source = mar_env.get_constant('source')
        except:
            self.source = 'google'
        try:
            self.weights = mar_env.get_constant('weights')
        except:
            self.weights = np.ones(self.number_of_assets, 'float')
            self.weights /= self.number_of_assets
        try:
            weights_sum = sum(self.weights)
        except:
            msg = 'Weights must be an iterable of numbers.'
            raise TypeError(msg)

        if round(weights_sum, 6) != 1:
            raise ValueError('Sum of weights must be one.')

        if len(self.weights) != self.number_of_assets:
            msg = 'Expected %s weights, got %s'
            raise ValueError(msg % (self.number_of_assets,
                                    len(self.weights)))

        self.load_data()
        self.make_raw_stats()
        self.apply_weights()

    def __str__(self):
        string = 'Portfolio %s \n' % self.name
        string += len(string) * '-' + '\n'
        string += 'return       %10.3f\n' % self.portfolio_return
        string += 'volatility   %10.3f\n' % math.sqrt(self.variance)
        string += 'Sharpe ratio %10.3f\n' % (self.portfolio_return /
                                             math.sqrt(self.variance))
        string += '\n'
        string += 'Positions\n'
        string += 'symbol | weight | ret. con. \n'
        string += '--------------------------- \n'
        for i in range(len(self.symbols)):
            string += '{:<6} | {:6.3f} | {:9.3f} \n'.format(
                self.symbols[i], self.weights[i], self.mean_returns[i])

        return string

    def load_data(self):
        '''
        Loads asset values from the web.
        '''

        self.data = pd.DataFrame()
        # if self.source == 'yahoo' or self.source == 'google':
        for sym in self.symbols:
            try:
                self.data[sym] = web.DataReader(sym, self.source,
                                                self.start_date,
                                                self.final_date)['Close']
            except:
                print('Can not find data for source %s and symbol %s.'
                      % (self.source, sym))
                print('Will try other source.')
                try:
                    if self.source == 'yahoo':
                        source = 'google'
                    if self.source == 'google':
                        source = 'yahoo'
                    self.data[sym] = web.DataReader(sym, source,
                                                    self.start_date,
                                                    self.final_date)['Close']
                except:
                    msg = 'Can not find data for source %s and symbol %s'
                    raise IOError(msg % (source, sym))
        self.data.columns = self.symbols
        # To do: add more sources

    def make_raw_stats(self):
        '''
        Computes returns and variances
        '''

        self.raw_returns = np.log(self.data / self.data.shift(1))
        self.mean_raw_return = self.raw_returns.mean()
        self.raw_covariance = self.raw_returns.cov()

    def apply_weights(self):
        '''
        Applies weights to the raw returns and covariances
        '''

        self.returns = self.raw_returns * self.weights
        self.mean_returns = self.returns.mean() * 252
        self.portfolio_return = np.sum(self.mean_returns)

        self.variance = np.dot(self.weights.T,
                               np.dot(self.raw_covariance * 252, self.weights))

    def test_weights(self, weights):
        '''
        Returns the theoretical portfolio return, portfolio volatility
        and Sharpe ratio for given weights.

        Please note:
        The method does not set the weight.

        Parameters
        ==========
        weight: iterable,
            the weights of the portfolio content.
         '''
        weights = np.array(weights)
        portfolio_return = np.sum(self.raw_returns.mean() * weights) * 252
        portfolio_vol = math.sqrt(
            np.dot(weights.T, np.dot(self.raw_covariance * 252, weights)))

        return np.array([portfolio_return, portfolio_vol,
                         portfolio_return / portfolio_vol])

    def set_weights(self, weights):
        '''
        Sets new weights

        Parameters
        ==========
        weights: interable
            new set of weights
        '''

        try:
            weights = np.array(weights)
            weights_sum = sum(weights).round(3)
        except:
            msg = 'weights must be an interable of numbers'
            raise TypeError(msg)

        if weights_sum != 1:
            raise ValueError('Sum of weights must be one')

        if len(weights) != self.number_of_assets:
            msg = 'Expected %s weights, got %s'
            raise ValueError(msg % (self.number_of_assets,
                                    len(weights)))
        self.weights = weights
        self.apply_weights()

    def get_weights(self):
        '''
        Returns a dictionary with entries symbol:weights
        '''

        d = dict()
        for i in range(len(self.symbols)):
            d[self.symbols[i]] = self.weights[i]
        return d

    def get_portfolio_return(self):
        '''
        Returns the average return of the weighted portfolio
        '''

        return self.portfolio_return

    def get_portfolio_variance(self):
        '''
        Returns the average variance of the weighted portfolio
        '''

        return self.variance

    def get_volatility(self):
        '''
        Returns the average volatility of the portfolio
        '''

        return math.sqrt(self.variance)

    def optimize(self, target, constraint=None, constraint_type='Exact'):
        '''
        Optimize the weights of the portfolio according to the value of the
        string 'target'

        Parameters
        ==========
        target: string
            one of:

            Sharpe: maximizes the ratio return/volatility
            Vol: minimizes the expected volatility
            Return: maximizes the expected return

        constraint: number
            only for target options 'Vol' and 'Return'.
            For target option 'Return', the function tries to optimize
            the expected return given the constraint on the volatility.
            For target option 'Vol', the optimization returns the minimum
            volatility given the constraint for the expected return.
            If constraint is None (default), the optimization is made
            without concerning the other value.

        constraint_type: string, one of 'Exact' or 'Bound'
            only relevant if constraint is not None.
            For 'Exact' (default) the value of the constraint must be hit
            (if possible), for 'Bound', constraint is only the upper/lower
            bound of the volatility or return resp.
        '''
        weights = self.get_optimal_weights(target, constraint, constraint_type)
        if weights is not False:
            self.set_weights(weights)
        else:
            raise ValueError('Optimization failed.')

    def get_capital_market_line(self, riskless_asset):
        '''
        Returns the capital market line as a lambda function and
        the coordinates of the intersection between the captal market
        line and the efficient frontier

        Parameters
        ==========

        riskless_asset: float
            the return of the riskless asset
        '''
        x, y = self.get_efficient_frontier(100)
        if len(x) == 1:
            raise ValueError('Efficient Frontier seems to be constant.')
        f_eff = sci.UnivariateSpline(x, y, s=0)
        f_eff_der = f_eff.derivative(1)

        def tangent(x, rl=riskless_asset):
            return f_eff_der(x) * x / (f_eff(x) - rl) - 1

        left_start = x[0]
        right_start = x[-1]

        left, right = self.search_sign_changing(
            left_start, right_start, tangent, right_start - left_start)
        if left == 0 and right == 0:
            raise ValueError('Can not find tangent.')

        zero_x = sco.brentq(tangent, left, right)

        opt_return = f_eff(zero_x)
        cpl = lambda x: f_eff_der(zero_x) * x + riskless_asset
        return cpl, zero_x, float(opt_return)

    def get_efficient_frontier(self, n):
        '''
        Returns the efficient frontier in form of lists containing the x and y
        coordinates of points of the frontier.

        Parameters
        ==========
        n : int >= 3
            number of points
        '''
        if type(n) is not int:
            raise TypeError('n must be an int')
        if n < 3:
            raise ValueError('n must be at least 3')

        min_vol_weights = self.get_optimal_weights('Vol')
        min_vol = self.test_weights(min_vol_weights)[1]
        min_return_weights = self.get_optimal_weights('Return',
                                                      constraint=min_vol)
        min_return = self.test_weights(min_return_weights)[0]
        max_return_weights = self.get_optimal_weights('Return')
        max_return = self.test_weights(max_return_weights)[0]

        delta = (max_return - min_return) / (n - 1)
        if delta > 0:
            returns = np.arange(min_return, max_return + delta, delta)
            vols = list()
            rets = list()
            for r in returns:
                w = self.get_optimal_weights('Vol', constraint=r,
                                             constraint_type='Exact')
                if w is not False:
                    result = self.test_weights(w)[:2]
                    rets.append(result[0])
                    vols.append(result[1])
        else:
            rets = [max_return, ]
            vols = [min_vol, ]

        return np.array(vols), np.array(rets)

    def get_optimal_weights(self, target, constraint=None,
                            constraint_type='Exact'):
        if target == 'Sharpe':
            def optimize_function(weights):
                return -self.test_weights(weights)[2]

            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        elif target == 'Vol':
            def optimize_function(weights):
                return self.test_weights(weights)[1]

            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, ]
            if constraint is not None:
                d = dict()
                if constraint_type == 'Exact':
                    d['type'] = 'eq'
                    d['fun'] = lambda x: self.test_weights(x)[0] - constraint
                    cons.append(d)
                elif constraint_type == 'Bound':
                    d['type'] = 'ineq'
                    d['fun'] = lambda x: self.test_weights(x)[0] - constraint
                    cons.append(d)
                else:
                    msg = 'Value for constraint_type must be either '
                    msg += 'Exact or Bound, not %s' % constraint_type
                    raise ValueError(msg)

        elif target == 'Return':
            def optimize_function(weights):
                return -self.test_weights(weights)[0]

            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, ]
            if constraint is not None:
                d = dict()
                if constraint_type == 'Exact':
                    d['type'] = 'eq'
                    d['fun'] = lambda x: self.test_weights(x)[1] - constraint
                    cons.append(d)
                elif constraint_type == 'Bound':
                    d['type'] = 'ineq'
                    d['fun'] = lambda x: constraint - self.test_weights(x)[1]
                    cons.append(d)
                else:
                    msg = 'Value for constraint_type must be either '
                    msg += 'Exact or Bound, not %s' % constraint_type
                    raise ValueError(msg)

        else:
            raise ValueError('Unknown target %s' % target)

        bounds = tuple((0, 1) for x in range(self.number_of_assets))
        start = self.number_of_assets * [1. / self.number_of_assets, ]
        result = sco.minimize(optimize_function, start,
                              method='SLSQP', bounds=bounds, constraints=cons)

        if bool(result['success']) is True:
            new_weights = result['x'].round(6)
            return new_weights
        else:
            return False

    def search_sign_changing(self, l, r, f, d):
        if d < 0.000001:
            return (0, 0)
        for x in np.arange(l, r + d, d):
            if f(l) * f(x) < 0:
                ret = (x - d, x)
                return ret

        ret = self.search_sign_changing(l, r, f, d / 2.)
        return ret


if __name__ == '__main__':
    ma = market_environment('ma', dt.date(2010, 1, 1))
    ma.add_constant('symbols', ['AAPL', 'GOOG', 'MSFT', 'DB'])
    ma.add_constant('final date', dt.date(2014, 3, 1))
    port = mean_variance_portfolio('My Portfolio', ma)
