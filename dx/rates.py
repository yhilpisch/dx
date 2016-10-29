#
# DX Analytics
# Rates Instruments Valuation Classes
# dx_rates.py
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
from .frame import *
from .models import *


# Classes for simple interest rates products

class interest_rate_swap(object):
    ''' Basic class for interest rate swap valuation.

    Attributes
    ==========
    name : string
        name of the object
    underlying :
        instance of simulation class
    mar_env : instance of market_environment
        market environment data for valuation

    Methods
    =======
    update:
        updates selected valuation parameters
    '''

    def __init__(self, name, underlying, mar_env):
        try:
            self.name = name
            self.pricing_date = mar_env.pricing_date

            # interest rate swap parameters
            self.fixed_rate = mar_env.get_constant('fixed_rate')
            self.trade_date = mar_env.get_constant('trade_date')
            self.effective_date = mar_env.get_constant('effective_date')
            self.payment_date = mar_env.get_constant('payment_date')
            self.payment_day = mar_env.get_constant('payment_day')
            self.termination_date = mar_env.get_constant('termination_date')
            self.notional = mar_env.get_constant('notional')
            self.currency = mar_env.get_constant('currency')
            self.tenor = mar_env.get_constant('tenor')
            self.counting = mar_env.get_constant('counting')

            # simulation parameters and discount curve from simulation object
            self.frequency = underlying.frequency
            self.paths = underlying.paths
            self.discount_curve = underlying.discount_curve
            self.underlying = underlying
            self.payoff = None

            # provide selected dates to underlying
            self.underlying.special_dates.extend([self.pricing_date,
                                                  self.effective_date,
                                                  self.payment_date,
                                                  self.termination_date])
        except:
            print('Error parsing market environment.')

        self.payment_dates = pd.date_range(self.payment_date,
                                           self.termination_date,
                                           freq=self.tenor)
        self.payment_dates = [d.replace(day=self.payment_day)
                              for d in self.payment_dates]
        self.payment_dates = pd.DatetimeIndex(self.payment_dates)
        self.underlying.time_grid = None
        self.underlying.instrument_values = None
        self.underlying.special_dates.extend(
            self.payment_dates.to_pydatetime())

    def generate_payoff(self, fixed_seed=True):
        ''' Generates the IRS payoff for simulated underlyin values. '''
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        payoff = paths - self.fixed_rate
        payoff = pd.DataFrame(payoff, index=self.underlying.time_grid)
        payoff = payoff.ix[self.payment_dates]
        return self.notional * payoff

    def present_value(self, fixed_seed=True, full=False):
        ''' Calculates the present value of the IRS. '''
        if self.payoff is None:
            self.payoff = self.generate_payoff(fixed_seed=fixed_seed)
        if not fixed_seed:
            self.payoff = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factors = self.discount_curve.get_discount_factors(
            self.payment_dates, dtobjects=True)[1]
        present_values = self.payoff.T * discount_factors[:][::-1]
        if full:
            return present_values.T
        else:
            return np.sum(np.sum(present_values)) / len(self.payoff.columns)


#
# Zero-Coupon Bond Valuation Formula CIR85/SRD Model
#


def gamma(kappa_r, sigma_r):
    ''' Help Function. '''
    return math.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


def b1(kappa_r, theta_r, sigma_r, T):
    ''' Help Function. '''
    g = gamma(kappa_r, sigma_r)
    return (((2 * g * math.exp((kappa_r + g) * T / 2)) /
             (2 * g + (kappa_r + g) * (math.exp(g * T) - 1))) **
            (2 * kappa_r * theta_r / sigma_r ** 2))


def b2(kappa_r, theta_r, sigma_r, T):
    ''' Help Function. '''
    g = gamma(kappa_r, sigma_r)
    return ((2 * (math.exp(g * T) - 1)) /
            (2 * g + (kappa_r + g) * (math.exp(g * T) - 1)))


def CIR_zcb_valuation(r0, kappa_r, theta_r, sigma_r, T):
    ''' Function to value unit zero-coupon bonds in
    Cox-Ingersoll-Ross (1985) model.

    Parameters
    ==========
    r0: float
        initial short rate
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean of short rate
    sigma_r: float
        volatility of short rate
    T: float
        time horizon/interval

    Returns
    =======
    zcb_value: float
        zero-coupon bond present value
    '''
    b_1 = b1(kappa_r, theta_r, sigma_r, T)
    b_2 = b2(kappa_r, theta_r, sigma_r, T)
    zcb_value = b_1 * math.exp(-b_2 * r0)
    return zcb_value
