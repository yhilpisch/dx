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
from .derivatives_portfolio import *


class var_derivatives_portfolio(derivatives_portfolio):
    ''' Class for building and valuing portfolios of derivatives positions
    with risk factors given from fitted VAR model.

    Attributes
    ==========
    name : str
        name of the object
    positions : dict
        dictionary of positions (instances of derivatives_position class)
    val_env : market_environment
        market environment for the valuation
    var_risk_factors : VAR model
        vector autoregressive model for risk factors
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
    '''

    def __init__(self, name, positions, val_env, var_risk_factors,
                 fixed_seed=False, parallel=False):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.var_risk_factors = var_risk_factors
        self.underlyings = set()

        self.time_grid = None
        self.underlying_objects = {}
        self.valuation_objects = {}
        self.fixed_seed = fixed_seed
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
                                  freq='B'  # allow business day only
                                  ).to_pydatetime()
        time_grid = list(time_grid)

        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        # delete duplicate entries & sort dates in time_grid
        time_grid = sorted(set(time_grid))

        self.time_grid = np.array(time_grid)
        self.val_env.add_list('time_grid', self.time_grid)

        #
        # generate simulated paths
        #
        self.fit_model = var_risk_factors.fit(maxlags=5, ic='bic')
        sim_paths = self.fit_model.simulate(
            paths=self.val_env.get_constant('paths'),
            steps=len(self.time_grid),
            initial_values=var_risk_factors.y[-1])
        symbols = sim_paths[0].columns.values
        for sym in symbols:
            df = pd.DataFrame()
            for i, path in enumerate(sim_paths):
                df[i] = path[sym]
            self.underlying_objects[sym] = general_underlying(
                sym, df, self.val_env)
        for pos in positions:
            # select right valuation class (European, American)
            val_class = otypes[positions[pos].otype]
            # pick the market environment and add the valuation environment
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # instantiate valuation classes
            self.valuation_objects[pos] = \
                val_class(name=positions[pos].name,
                          mar_env=mar_env,
                          underlying=self.underlying_objects[
                    positions[pos].underlyings[0]],
                payoff_func=positions[pos].payoff_func)

    def get_statistics(self):
        raise NotImplementedError

    def get_port_risk(self):
        raise NotImplementedError
