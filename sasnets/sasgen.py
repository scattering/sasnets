#!/usr/bin/env python
"""
Program to generate sample SANS datasets for neural network training. A
modified version of compare_multi.py.

The program writes datafiles as result_<modelname>_<run_number> to the out/
directory. See example_data.dat for a sample of the file format used.

"""
from __future__ import print_function

from copy import deepcopy
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
def gen_data(model_name, data, count=1, noise=2,
             mono=True, magnetic=False, cutoff=1e-5,
             maxdim=np.inf, precision='double'):
    r"""
    Generates the data for the given model and parameters.

    *model_name* is the name of the model.

    *data* is the data object giving $q, \Delta q$ calculation points.

    *N* is the number of comparisons to make.

    *cutoff* is the polydispersity weight cutoff to make the calculation
    a little bit faster.

    *maxdim* is maximum value for any shape dimension.

    *precision* is the name of the calculation engine to use.

    Returns iterator *(seed, pars, data), ...* where *pars* is
    *{par: value, ...}* and *data* is *(q, dq, iq, diq)*.
    """
    assert data.x.size > 0
    model_info = sascore.load_model_info(model_name)
    calculator = sascomp.make_engine(model_info, data, precision, cutoff)
    default_pars = sascomp.get_pars(model_info)
    assert calculator._data.x.size > 0
    x, dx = calculator._data.x, calculator._data.dx

    # A not very clean macro for evaluating the models, wich uses name and
    # seed from the current scope even though they haven't been defined yet.
    def simulate(pars):
        """
        Generate a random dataset for *fn* evaluated at *pars*.
        Returns *(x, dx, y, dy)*, o.

        Note that this replaces the data object within *fn*.
        """
        # TODO: support 2D data, which does not use x, dx, y, dy
        try:
            assert calculator._data.x.size > 0
            calculator.simulate_data(noise=noise, **pars)
            assert calculator._data.x.size > 0
            data = calculator._data
            result = (x, dx, data.y.copy(), data.dy.copy())
        except Exception:
            traceback.print_exc()
            print(f"Error when generating {model_name} for {seed}")
            result = (x, dx, np.NaN*x, np.NaN*x)
            #raise
        return result

    t0 = -np.inf
    interval = 5
    for k in range(count):
        seed = np.random.randint(int(1e6))
        t1 = time.perf_counter()
        if t1 > t0 + interval:
            print(f"generating {model_name} {k+1} of {count} with seed {seed}")
            t0 = t1

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
        data = simulate(pars) # q, dq, iq, diq
        # Warning: don't check if data is bad and skip the yield.
        # If you do then a bad model will be an infinite loop.
        yield seed, pars, data

    # TODO: can free the calculator now
    print(f"Complete {model_name}")

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
    noise = opts.noise

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
        # Generate
        data, index = sascomp.make_data({
            'qmin': 1e-4, 'qmax': 0.2, 'is2d': is2D, 'nq': nq, 'res': res/100,
            'accuracy': 'Low', 'view': 'log', 'zero': False,
            })

    # Open database and lookup model counts.
    db = sas_io.sql_connect()
    model_counts = sas_io.model_counts(db, tag)
    #print(model_counts)
    for model in model_list:
        # Figure out how many more entries we need for the model
        missing = count - model_counts.get(model, 0)
        if missing <= 0:
            continue
        # TODO: should not need deepcopy(data) but something is messing with q
        seq = gen_data(model, deepcopy(data), count=missing, mono=mono,
                       cutoff=cutoff, precision=precision, noise=noise)
        # Process the missing entries batch by batch so if there is an
        # error we won't lose the entire group.
        for batch in chunk(seq, batch_size=100):
            sas_io.write_sql(db, model, batch, tag=tag)
            #sas_io.write_1d(path, model, items, tag=tag)
    #sas_io.read_sql(db, tag)
    db.close()

def chunk(seq, batch_size):
    """
    Chunk sequence in groups of *batch_size*.

    Remaining items are returned in final group.
    """
    batch = []
    for item in seq:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)
    yield batch

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
        "--resolution", type=float, default=3,
        help="Constant dQ/Q resolution %.")
    parser.add_argument(
        "--noise", type=float, default=2,
        help="Constant dI/I uncertainty %.")
    parser.add_argument(
        "--dimension", choices=('1D', '2D'), default='1D',
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
