#
# DX Analytics
# Helper Function for Plotting
# dx_plot.py
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

import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm

def plot_option_stats(s_list, pv, de, ve):
    ''' Plot option prices, deltas and vegas for a set of
    different initial values of the underlying.

    Parameters
    ==========
    s_list : array or list
        set of intial values of the underlying
    pv : array or list
        present values
    de : array or list
        results for deltas
    ve : array or list
        results for vega
    '''
    plt.figure(figsize=(9, 7))
    sub1 = plt.subplot(311)
    plt.plot(s_list, pv, 'ro', label='Present Value')
    plt.plot(s_list, pv, 'b')
    plt.grid(True); plt.legend(loc=0)
    plt.setp(sub1.get_xticklabels(), visible=False)
    sub2 = plt.subplot(312)
    plt.plot(s_list, de, 'go', label='Delta')
    plt.plot(s_list, de, 'b')
    plt.grid(True); plt.legend(loc=0)
    plt.setp(sub2.get_xticklabels(), visible=False)
    sub3 = plt.subplot(313)
    plt.plot(s_list, ve, 'yo', label='Vega')
    plt.plot(s_list, ve, 'b')
    plt.xlabel('Strike')
    plt.grid(True); plt.legend(loc=0)


def plot_option_stats_full(s_list, pv, de, ve, th, rh, ga):
    ''' Plot option prices, deltas and vegas for a set of
    different initial values of the underlying.

    Parameters
    ==========
    s_list : array or list
        set of intial values of the underlying
    pv : array or list
        present values
    de : array or list
        results for deltas
    ve : array or list
        results for vega
    th : array or list
        results for theta
    rh : array or list
        results for rho
    ga : array or list
        results for gamma
    '''
    plt.figure(figsize=(10, 14))
    sub1 = plt.subplot(611)
    plt.plot(s_list, pv, 'ro', label='Present Value')
    plt.plot(s_list, pv, 'b')
    plt.grid(True); plt.legend(loc=0)
    plt.setp(sub1.get_xticklabels(), visible=False)
    sub2 = plt.subplot(612)
    plt.plot(s_list, de, 'go', label='Delta')
    plt.plot(s_list, de, 'b')
    plt.grid(True); plt.legend(loc=0)
    plt.setp(sub2.get_xticklabels(), visible=False)
    sub3 = plt.subplot(613)
    plt.plot(s_list, ve, 'yo', label='Gamma')
    plt.plot(s_list, ve, 'b')
    plt.grid(True); plt.legend(loc=0)
    sub4 = plt.subplot(614)
    plt.plot(s_list, th, 'mo', label='Vega')
    plt.plot(s_list, th, 'b')
    plt.grid(True); plt.legend(loc=0)
    sub5 = plt.subplot(615)
    plt.plot(s_list, rh, 'co', label='Theta')
    plt.plot(s_list, rh, 'b')
    plt.grid(True); plt.legend(loc=0)
    sub6 = plt.subplot(616)
    plt.plot(s_list, ga, 'ko', label='Rho')
    plt.plot(s_list, ga, 'b')
    plt.xlabel('Strike')
    plt.grid(True); plt.legend(loc=0)


def plot_greeks_3d(inputs, labels):
    ''' Plot Greeks in 3d.

    Parameters
    ==========
    inputs : list of arrays
        x, y, z arrays
    labels : list of strings
        labels for x, y, z
    '''
    x, y, z = inputs
    xl, yl, zl = labels
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)
    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_calibration_results(cali, relative=False):
    ''' Plot calibration results.

    Parameters
    ==========
    cali : instance of calibration class
        instance has to have opt_parameters
    relative : boolean
        if True, then relative error reporting
        if False, absolute error reporting
    '''
    cali.update_model_values()
    mats = set(cali.option_data[:, 0])
    mats = np.sort(list(mats))
    fig, axarr = plt.subplots(len(mats), 2, sharex=True)
    fig.set_size_inches(8, 12)
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    z = 0
    for T in mats:
        strikes = strikes = cali.option_data[cali.option_data[:, 0] == T][:, 1]
        market = cali.option_data[cali.option_data[:, 0] == T][:, 2]
        model = cali.model_values[cali.model_values[:, 0] == T][:, 2]
        axarr[z, 0].set_ylabel('%s' % str(T)[:10])
        axarr[z, 0].plot(strikes, market, label='Market Quotes')
        axarr[z, 0].plot(strikes, model, 'ro', label='Model Prices')
        axarr[z, 0].grid()
        if T is mats[0]:
            axarr[z, 0].set_title('Option Quotes')
        if T is mats[-1]:
            axarr[z, 0].set_xlabel('Strike')
        wi = 2.
        if relative is True:
            axarr[z, 1].bar(strikes - wi / 2,
                           (model - market) / market * 100, width=wi)
        else:
            axarr[z, 1].bar(strikes - wi / 2, model - market, width=wi)
        axarr[z, 1].grid()
        if T is mats[0]:
            axarr[z, 1].set_title('Differences')
        if T is mats[-1]:
            axarr[z, 1].set_xlabel('Strike')
        z += 1
