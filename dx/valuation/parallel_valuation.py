#
# DX Analytics
# Derivatives Instruments and Portfolio Valuation Classes
# parallel_valuation.py
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
import multiprocess as mp  # changed import


def simulate_parallel(objs, fixed_seed=True):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        o.generate_paths(fixed_seed=fixed_seed)
        output.put((o.name, o))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    results = [output.get() for _ in objs]
    underlying_objects = {name: obj for name, obj in results}
    return underlying_objects


def value_parallel(objs, fixed_seed=True, full=False):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if full:
            _, pvs = o.present_value(fixed_seed=fixed_seed, full=True)
            output.put((o.name, pvs))
        else:
            pv = o.present_value(fixed_seed=fixed_seed)
            output.put((o.name, pv))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    res_list = [output.get() for _ in objs]
    return {name: result for name, result in res_list}


def greeks_parallel(objs, Greek='Delta'):
    procs = []
    man = mp.Manager()
    output = man.Queue()

    def worker(o, output):
        if Greek == 'Delta':
            output.put((o.name, o.delta()))
        elif Greek == 'Vega':
            output.put((o.name, o.vega()))

    for o in objs:
        procs.append(mp.Process(target=worker, args=(o, output)))
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    res_list = [output.get() for _ in objs]
    return {name: greek for name, greek in res_list}
