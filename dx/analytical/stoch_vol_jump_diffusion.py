#
# DX Analytics
# Analytical Option Pricing
# stoch_vol_jump_diffusion.py
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
from .stochastic_volatility import H93_char_func


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
        T = (mar_env.get_constant('maturity') -
             mar_env.pricing_date).days / 365.
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
        print('Error parsing market environment.')

    int_value = quad(lambda u:
                     B96_int_func(u, S0, K, T, r, kappa_v, theta_v,
                                  sigma_v, rho, v0, lamb, mu, delta),
                     0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value


def B96_put_value(mar_env):
    ''' Valuation of European put option in Bates (1996) model via Lewis (2001)
    Fourier-based approach. '''

    try:
        S0 = mar_env.get_constant('initial_value')
        K = mar_env.get_constant('strike')
        T = (mar_env.get_constant('maturity') -
             mar_env.pricing_date).days / 365.
        r = mar_env.get_curve('discount_curve').short_rate
    except:
        print('Error parsing market environment.')

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
    char_func_value = np.exp((1j * u * omega + lamb *
        (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return char_func_value


def B96_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                  lamb, mu, delta):
    ''' Valuation of European call option in BCC97 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function B96_call_value.'''
    BCC1 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    BCC2 = M76_char_func(u, T, lamb, mu, delta)
    return BCC1 * BCC2
