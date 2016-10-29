#
# DX Analytics
# Base Classes and Model Classes for Simulation
# mean_reverting_diffusion.py
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
from .square_root_diffusion import square_root_diffusion


class mean_reverting_diffusion(square_root_diffusion):
    ''' Class to generate simulated paths based on the
    Vasicek (1977) mean-reverting short rate model.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for simulation
    corr : boolean
        True if correlated with other model object

    Methods
    =======
    update :
        updates parameters
    generate_paths :
        returns Monte Carlo paths given the market environment
    '''

    def __init__(self, name, mar_env, corr=False, truncation=False):
        super(mean_reverting_diffusion,
              self).__init__(name, mar_env, corr)
        self.truncation = truncation

    def generate_paths(self, fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value
        if self.correlated is False:
            rand = sn_random_numbers((1, M, I),
                                     fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            # full truncation Euler discretization
            if self.truncation is True:
                paths_[t] = (paths_[t - 1] + self.kappa *
                             (self.theta - np.maximum(0, paths_[t - 1])) * dt +
                             self.volatility * np.sqrt(dt) * ran)
                paths[t] = np.maximum(0, paths_[t])
            else:
                paths[t] = (paths[t - 1] + self.kappa *
                            (self.theta - paths[t - 1]) * dt +
                            self.volatility * np.sqrt(dt) * ran)
        self.instrument_values = paths
