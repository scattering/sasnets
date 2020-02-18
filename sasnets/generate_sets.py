#!/usr/bin/env python
"""
Program to generate sample SANS datasets for neural network training. A
modified version of compare_multi.py.

The program writes datafiles as result_<modelname>_<run_number> to the out/
directory. See example_data.dat for a sample of the file format used.

"""
from __future__ import print_function

import argparse
import os
import resource
import sys
import time
import traceback
from collections import OrderedDict, namedtuple
import sqlite3
import logging

import numpy as np  # type: ignore

from sasmodels import core as sascore
from sasmodels import compare as sascomp
from sasmodels import data as sasdata

from . import sas_io

MODELS = sascore.list_models()

# noinspection PyTypeChecker
def gen_data(model_name, data, index, count=1,
             mono=True, magnetic=False, cutoff=1e-5,
             maxdim=np.inf, precision='double'):
    r"""
    Generates the data for the given model and parameters.

    *model_name* is the name of the model.

    *data* is the data object giving $q, \Delta q$ calculation points.

    *index* is the active set of points.

    *N* is the number of comparisons to make.

    *cutoff* is the polydispersity weight cutoff to make the calculation
    a little bit faster.

    *maxdim* is maximum value for any shape dimension.

    *precision* is the name of the calculation engine to use.

    Returns list of items::
        [(seed, {par: value, ...}, (q, dq, iq, diq)), ...]
    """
    model_info = sascore.load_model_info(model_name)
    calculator = sascomp.make_engine(model_info, data, precision, cutoff)
    default_pars = sascomp.get_pars(model_info)

    # A not very clean macro for evaluating the models, wich uses name and
    # seed from the current scope even though they haven't been defined yet.
    def exec_model(fn, pars, noise=2):
        """
        Generate a random dataset for *fn* evaluated at *pars*.
        Returns *(x, dx, y, dy)*, o.

        Note that this replaces the data object within *fn*.
        """
        # TODO: support 2D data, which does not use x, dx, y, dy
        try:
            fn.simulate_data(noise=noise, **pars)
            data = fn._data
            result = (data.x, data.dx, data.y.copy(), data.dy.copy())
        except Exception:
            traceback.print_exc()
            print(f"Error when generating {model_name} for {seed}")
            result = [np.NaN]*4
        return result

    items = []
    for k in range(count):
        seed = np.random.randint(int(1e6))
        if k%100 == 0:
            print(f"generating {model_name} {k} with seed {seed}")

        # Generate parameters
        with sascomp.push_seed(seed):
            pars = sascomp.randomize_pars(model_info, default_pars, maxdim)
        sascomp.constrain_pars(model_info, pars)
        if mono:
            pars = sascomp.suppress_pd(pars)
        if not magnetic:
            pars = sascomp.suppress_magnetism(pars)
        pars.update({'scale': 1, 'background': 1e-5})

        # Evaluate model
        data = exec_model(calculator, pars) # q, dq, iq, diq
        items.append((seed, pars, data))

    print(f"Complete {model_name}")
    return items

def run_model(opts):
    model = opts.model
    tag = opts.tag
    count = opts.count
    is2D = opts.dimension.startswith('2d')
    nq = opts.npoint
    mono = opts.cutoff == 'mono'
    cutoff = float(opts.cutoff) if not mono else 0
    precision = opts.precision
    res = opts.resolution

    try:
        model_list = [model] if model in MODELS else sascore.list_models(model)
    except ValueError:
        print(f'Bad model {model}.  Use model type or one of:', file=sys.stderr)
        sascomp.print_models()
        print(f'model types: {sascore.KINDS}')
        return

    if opts.template:
        # Fetch q, dq from an actual SANS file
        data, index = sasdata.read(opts.template), None
    else:
        # Generarte
        data, index = sascomp.make_data({
            'qmin': 1e-4, 'qmax': 0.2, 'is2d': is2D, 'nq': nq, 'res': res,
            'accuracy': 'Low', 'view': 'log', 'zero': False,
            })

    # Names for columns of data gotten from simulate
    db = sas_io.sql_connect()
    for model in model_list:
        items = gen_data(model, data, index, count=count, mono=mono,
                         cutoff=cutoff, precision=precision)
        sas_io.write_sql(db, tag, model, items)
        #sas_io.write_1d(path, tag, model, items)
    sas_io.read_sql(db, tag)
    db.close()


def main():
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="""
        A script that generates SANS datasets for use in neural network training.""")
    parser.add_argument(
        "model",
        help=f"""
        model is the model name of the model or one of the model types
        listed in sasmodels.core.list_models {sascore.KINDS}. Model types can be
        combined, such as 2d+single.""")
    parser.add_argument(
        "--tag", type=str, default="train",
        help="Tag for the generated data: train or test.")
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Count is the number of distinct models to generate.")
    parser.add_argument(
        "--database", type=str, default=sas_io.DB_FILE,
        help="Path to the sqlite database file.")
    parser.add_argument(
        "--template", type=str, default="",
        help="Template file defining q and resolution.")
    parser.add_argument(
        "--resolution", type=float, default=0.03,
        help="Constant dQ/Q resolution.")
    parser.add_argument(
        "--dimension", choices=['1D', '2D'], default='1D',
        help="Choose whether to generate 1D or 2D data.")
    parser.add_argument(
        "--npoint", type=int, default=128,
        help="The number of points per model.")
    parser.add_argument(
        "--cutoff",
        default=0.,
        help="""
        CUTOFF is the cutoff value to use for the polydisperse distribution.
        Weights below the cutoff will be ignored. Use 'mono' for
        monodisperse models. The choice of polydisperse parameters, and
        the number of points in the distribution is set in compare.py
        defaults for each model. Polydispersity is given in the 'demo'
        attribute of each model.""")
    parser.add_argument(
        "--precision",
        choices=['single', 'double', 'fast', 'single!', 'double!', 'quad!'],
        default='double',
        help="""
        Precision to use in floating point calculations. If postfixed with
        an '!', builds a DLL for the CPU.""")
    parser.add_argument(
        "-v", "--verbose",
        help="Verbose output level.", choices=[0, 1, 2])

    logging.basicConfig(level=logging.INFO)
    opts = parser.parse_args()

    time_start = time.perf_counter()
    run_model(opts)
    time_end = time.perf_counter() - time_start
    print('Total computation time (s): %.2f' % (time_end / 10))
    print('Total memory usage: %.2f' %
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Units of mem are OS dependent

if __name__ == "__main__":
    main()
