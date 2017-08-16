#
# DX Analytics
# Analytical Option Pricing
# black_scholes_merton.py
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
from math import log, exp, sqrt
from scipy import stats
from scipy.optimize import fsolve
from ..frame import market_environment


class BSM_european_option(object):
    ''' Class for European options in BSM Model.

    Attributes
    ==========
    initial_value : float
        initial stock/index level
    strike : float
        strike price
    pricing_date : datetime/Timestamp object
        pricing date
    maturity : datetime/Timestamp object
        maturity date
    short_rate : float
        constant risk-free short rate
    volatility : float
        volatility factor in diffusion term

    Methods
    =======
    call_value : float
        return present value of call option
    put_value : float
        return present_values of put option
    vega : float
        return vega of option
    imp_vol: float
        return implied volatility given option quote
    '''

    def __init__(self, name, mar_env):
        try:
            self.name = name
            self.initial_value = mar_env.get_constant('initial_value')
            self.strike = mar_env.get_constant('strike')
            self.pricing_date = mar_env.pricing_date
            self.maturity = mar_env.get_constant('maturity')
            self.short_rate = mar_env.get_curve('discount_curve').short_rate
            self.volatility = mar_env.get_constant('volatility')
            try:
                self.dividend_yield = mar_env.get_constant('dividend_yield')
            except:
                self.dividend_yield = 0.0
            self.mar_env = mar_env
        except:
            print('Error parsing market environment.')

    def update_ttm(self):
        ''' Updates time-to-maturity self.ttm. '''
        if self.pricing_date > self.maturity:
            raise ValueError('Pricing date later than maturity.')
        self.ttm = (self.maturity - self.pricing_date).days / 365.

    def d1(self):
        ''' Helper function. '''
        d1 = ((log(self.initial_value / self.strike) +
               (self.short_rate - self.dividend_yield +
                0.5 * self.volatility ** 2) * self.ttm) /
              (self.volatility * sqrt(self.ttm)))
        return d1

    def d2(self):
        ''' Helper function. '''
        d2 = ((log(self.initial_value / self.strike) +
               (self.short_rate - self.dividend_yield -
                0.5 * self.volatility ** 2) * self.ttm) /
              (self.volatility * sqrt(self.ttm)))
        return d2

    def call_value(self):
        ''' Return call option value. '''
        self.update_ttm()
        call_value = (
            exp(- self.dividend_yield * self.ttm) *
            self.initial_value * stats.norm.cdf(self.d1(), 0.0, 1.0) -
            exp(-self.short_rate * self.ttm) * self.strike *
            stats.norm.cdf(self.d2(), 0.0, 1.0))
        return call_value

    def put_value(self):
        ''' Return put option value. '''
        self.update_ttm()
        put_value = (
            exp(-self.short_rate * self.ttm) * self.strike *
            stats.norm.cdf(-self.d2(), 0.0, 1.0) -
            exp(-self.dividend_yield * self.ttm) *
            self.initial_value *
            stats.norm.cdf(-self.d1(), 0.0, 1.0))
        return put_value

    def vega(self):
        ''' Return Vega of option. '''
        self.update_ttm()
        d1 = ((log(self.initial_value / self.strike) +
               (self.short_rate + (0.5 * self.volatility ** 2)) * self.ttm) /
              (self.volatility * sqrt(self.ttm)))
        vega = self.initial_value * stats.norm.pdf(d1, 0.0, 1.0) \
            * sqrt(self.ttm)
        return vega

    def imp_vol(self, price, otype='call', volatility_est=0.2):
        ''' Return implied volatility given option price. '''
        me = market_environment('iv', self.pricing_date)
        me.add_environment(self.mar_env)
        me.add_constant('volatility', volatility_est)
        option = BSM_european_option('ivc', me)
        option.update_ttm()

        def difference(volatility_est):
            option.volatility = volatility_est
            if otype == 'call':
                return option.call_value() - price
            if otype == 'put':
                return (option.put_value() - price) ** 2
            else:
                raise ValueError('No valid option type.')
        iv = fsolve(difference, volatility_est)[0]
        return iv
