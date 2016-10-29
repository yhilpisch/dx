#
# DX Analytics
# Analytical Option Pricing
# stochastic_volatility.py
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
import numpy as np
from scipy.integrate import quad


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
        T = (mar_env.get_constant('maturity') -
             mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
        kappa_v = mar_env.get_constant('kappa')
        theta_v = mar_env.get_constant('theta')
        sigma_v = mar_env.get_constant('vol_vol')
        rho = mar_env.get_constant('rho')
        v0 = mar_env.get_constant('volatility') ** 2
    except:
        print('Error parsing market environment.')

    int_value = quad(lambda u:
                     H93_int_func(u, S0, K, T, r, kappa_v,
                                  theta_v, sigma_v, rho, v0),
                     0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value


def H93_put_value(mar_env):
    ''' Valuation of European call option in Heston (1993) model via
    Lewis (2001) -- Fourier-based approach. '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') -
             mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
    except:
        print('Error parsing market environment.')

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
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v) ** 2 -
                  sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
        / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2) *
          ((kappa_v - rho * sigma_v * u * 1j + c2) * T -
          2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2 *
          ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value
