#
# DX Analytics
# Fourier-based Pricing
# dx_fourier.py
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
from math import log, exp, sqrt
import numpy as np
from scipy.integrate import quad
from scipy import stats
from scipy.optimize import fsolve
from dx_frame import market_environment

#
# Black-Scholes-Merton (1973) Diffusion Model
#


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
            print "Error parsing market environment."

    def update_ttm(self):
        ''' Updates time-to-maturity self.ttm. '''
        if self.pricing_date > self.maturity:
            raise ValueError("Pricing date later than maturity.")
        self.ttm = (self.maturity - self.pricing_date).days / 365.

    def d1(self):
        ''' Helper function. '''
        d1 = ((log(self.initial_value / self.strike)
            + (self.short_rate - self.dividend_yield 
                + 0.5 * self.volatility ** 2) * self.ttm)
            / (self.volatility * sqrt(self.ttm)))
        return d1

    def d2(self):
        ''' Helper function. '''
        d2 = ((log(self.initial_value / self.strike)
            + (self.short_rate - self.dividend_yield 
                - 0.5 * self.volatility ** 2) * self.ttm)
            / (self.volatility * sqrt(self.ttm)))
        return d2
        
    def call_value(self):
        ''' Return call option value. '''
        self.update_ttm()
        call_value = (exp(- self.dividend_yield * self.ttm) 
                    * self.initial_value * stats.norm.cdf(self.d1(), 0.0, 1.0)
                    - exp(-self.short_rate * self.ttm) * self.strike 
                    * stats.norm.cdf(self.d2(), 0.0, 1.0))
        return call_value

    def put_value(self):
        ''' Return put option value. '''
        self.update_ttm()
        put_value = (exp(-self.short_rate * self.ttm) * self.strike
                    * stats.norm.cdf(-self.d2(), 0.0, 1.0) 
                    - exp(-self.dividend_yield * self.ttm) * self.initial_value
                    * stats.norm.cdf(-self.d1(), 0.0, 1.0))
        return put_value
        
    def vega(self):
        ''' Return Vega of option. '''
        self.update_ttm()
        d1 = ((log(self.initial_value / self.strike)
            + (self.short_rate + (0.5 * self.volatility ** 2)) * self.ttm)
            / (self.volatility * sqrt(self.ttm)))
        vega = self.initial_value * stats.norm.cdf(d1, 0.0, 1.0) \
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


#
# Merton (1976) Jump Diffusion Model
#

def M76_call_value(mar_env):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    initial_value : float
        initial stock/index level
    strike : float
        strike price
    maturity : datetime object
        time-to-maturity (for t=0)
    short_rate : float
        constant risk-free short rate
    volatility : float
        volatility factor diffusion term
    lamb : float
        jump intensity
    mu : float
        expected jump size
    delta : float
        standard deviation of jump

    Returns
    =======
    call_value: float
        present value of European call option
    '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') 
            - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
        lamb = mar_env.get_constant('lambda')
        mu = mar_env.get_constant('mu')
        delta = mar_env.get_constant('delta')
        volatility = mar_env.get_constant('volatility')
    except:
        print "Error parsing market environment."

    int_value = quad(lambda u: M76_int_func_sa(u, S0, K, T, r,
                        volatility, lamb, mu, delta), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


def M76_put_value(mar_env):
    ''' Valuation of European put option in M76 model via Lewis (2001)
    Fourier-based approach. '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') 
                - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
    except:
        print "Error parsing market environment."

    call_value = M76_call_value(mar_env)
    put_value = call_value + K * math.exp(-r * T) - S0
    return put_value


def M76_int_func_sa(u, S0, K, T, r, volatility, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function M76_call_value.'''
    char_func_value = M76_char_func_sa(u - 0.5 * 1j, T, r, volatility,
                                        lamb, mu, delta)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def M76_char_func_sa(u, T, r, volatility, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: characteristic function "jump component".

    Parameter definitions see function M76_call_value.'''
    omega = r - 0.5 * volatility ** 2 \
            - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    char_func_value = np.exp((1j * u * omega - 0.5 * u ** 2 * volatility ** 2
                + lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5)
                    - 1)) * T)
    return char_func_value


#
# Heston (1993) Stochastic Volatility Model
#


def H93_call_value(mar_env):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    initial_value : float
        initial stock/index level
    strike : float
        strike price
    maturity : datetime object
        time-to-maturity (for t=0)
    short_rate : float
        constant risk-free short rate
    kappa_v : float
        mean-reversion factor
    theta_v : float
        long-run mean of variance
    sigma_v : float
        volatility of variance
    rho : float
        correlation between variance and stock/index level
    volatility: float
        initial level of volatility (square root of variance)

    Returns
    =======
    call_value: float
        present value of European call option

    '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity')
            - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
        kappa_v = mar_env.get_constant('kappa') 
        theta_v = mar_env.get_constant('theta') 
        sigma_v = mar_env.get_constant('vol_vol')
        rho = mar_env.get_constant('rho')
        v0 = mar_env.get_constant('volatility') ** 2
    except:
        print "Error parsing market environment."

    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v,
                        theta_v, sigma_v, rho, v0), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


def H93_put_value(mar_env):
    ''' Valuation of European call option in Heston (1993) model via
    Lewis (2001) -- Fourier-based approach. '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') 
                - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
    except:
        print "Error parsing market environment."

    call_value = H93_call_value(mar_env)
    put_value = call_value + K * math.exp(-r * T) - S0
    return put_value


def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function H93_call_value.'''
    char_func_value = H93_char_func(u - 1j * 0.5, T, r, kappa_v,
                                    theta_v, sigma_v, rho, v0)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function B96_call_value.'''
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v)
            ** 2 - sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
          / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2)
          * ((kappa_v - rho * sigma_v * u * 1j + c2) * T
                - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2
          * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value


#
# Bates (1996) Stochastic Volatility Jump Diffusion Model
#


def B96_call_value(mar_env):
    ''' Valuation of European call option in B96 Model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    intial_value: float
        initial stock/index level
    strike: float
        strike price
    maturity: datetime object
        time-to-maturity (for t=0)
    short_rate: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    call_value: float
        present value of European call option

    '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity')
            - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
        kappa_v = mar_env.get_constant('kappa') 
        theta_v = mar_env.get_constant('theta') 
        sigma_v = mar_env.get_constant('vol_vol')
        rho = mar_env.get_constant('rho')
        v0 = mar_env.get_constant('volatility') ** 2
        lamb = mar_env.get_constant('lambda')
        mu = mar_env.get_constant('mu')
        delta = mar_env.get_constant('delta')
    except:
        print "Error parsing market environment."

    int_value = quad(lambda u: B96_int_func(u, S0, K, T, r, kappa_v, theta_v, 
                sigma_v, rho, v0, lamb, mu, delta), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


def B96_put_value(mar_env):
    ''' Valuation of European put option in Bates (1996) model via Lewis (2001)
    Fourier-based approach. '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') 
                - mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
    except:
        print "Error parsing market environment."
    
    call_value = B96_call_value(mar_env)
    put_value = call_value + K * math.exp(-r * T) - S0
    return put_value


def B96_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                            lamb, mu, delta):
    ''' Valuation of European call option in BCC97 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function B96_call_value.'''
    char_func_value = B96_char_func(u - 1j * 0.5, T, r, kappa_v, theta_v, 
                        sigma_v, rho, v0, lamb, mu, delta)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def M76_char_func(u, T, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function M76_call_value.'''
    omega = -lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    char_func_value = np.exp((1j * u * omega + lamb
            * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return char_func_value


def B96_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                    lamb, mu, delta):
    ''' Valuation of European call option in BCC97 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function B96_call_value.'''
    BCC1 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    BCC2 = M76_char_func(u, T, lamb, mu, delta)
    return BCC1 * BCC2