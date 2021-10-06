#!/usr/bin/env python
"""
Program to generate sample SANS datasets for neural network training. A
modified version of compare_multi.py.

The program writes datafiles as result_<modelname>_<run_number> to the out/
directory. See example_data.dat for a sample of the file format used.

**TODOS**

* don't save q, dq with every dataset
* use hdf5 rather than sqlite

* Set background as a log q range percentage.  With 10k samples, a beta
  distribution B(alpha=2, beta=10) should have about 1 with a background
  q at 0.33 or less, and 1% will be less than 0.55. 80% will be above 0.75.
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
import fnmatch

import numpy as np  # type: ignore

from sasmodels import core as sascore
from sasmodels import compare as sascomp
from sasmodels import data as sasdata

from . import sas_io
from .util.utils import columnize

MODELS = sascore.list_models()

# Maxim @ stackoverflow
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
# (with mods by PAK)
def str2bool(value):
    """parse boolean argument"""
    if isinstance(value, bool):
       return value
    value = value.lower()
    if value in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value (1/0, Y[es]/N[o], T[rue]/F[alse]) expected.')

parser = argparse.ArgumentParser(
    description="""
    A script that generates SANS datasets for use in neural network training.""")
parser.add_argument(
    "models", nargs='*',
    help=f"A list of models or kinds ({', '.join(sascore.KINDS)})."
         f" A pattern such as *sphere will select all sphere models.")
parser.add_argument(
    "-x", "--exclude", type=str, default="",
    help="Exclude specific models separated by commas.")
parser.add_argument(
    "--tag", type=str, default="train",
    help="Tag for the generated data: train, test or validate.")
parser.add_argument(
    "--database", type=str, default=sas_io.DB_FILE,
    help="Path to the sqlite database file.")
parser.add_argument(
    "--count", type=int, default=1000,
    help="Count is the number of distinct models to generate.")
parser.add_argument(
    "--seed", type=int, default=1,
    help="Seed value for the random seq"
)
parser.add_argument(
    "--template", type=str, default="",
    help="SANS dataset defining q and resolution.")
parser.add_argument(
    "--resolution", type=float, default=3,
    help="Constant dQ/Q resolution percentage.")
parser.add_argument(
    "--noise", type=float, default=2,
    help="Constant dI/I uncertainty percentage.")
parser.add_argument(
    "--dimension", choices=('1D', '2D'), default='1D',
    help="Choose whether to generate 1D or 2D data.")
parser.add_argument(
    "--npoint", type=int, default=128,
    help="The number of points per model.")
parser.add_argument(
    "--mono", type=str2bool, default=True,
    help="Force all models to be monodisperse.")
parser.add_argument(
    "--magnetic", type=str2bool, default=False,
    help="Allow magnetic parameters in the model")
parser.add_argument(
    "--sesans", type=str2bool, default=False,
    help="Generate SESANS data instead of SANS data")
parser.add_argument(
    "--cutoff", type=float, default=0.,
    help="""
    CUTOFF is the cutoff value to use for the polydisperse distribution.
    Weights below the cutoff will be ignored.""")
parser.add_argument(
    "--precision", default='default',
    choices=['default', 'single', 'double', 'fast', 'single!', 'double!', 'quad!'],
    help="""
    Precision to use in floating point calculations. If postfixed with
    an '!', builds a DLL for the CPU. If default, use single unless the
    model requires double.""")
parser.add_argument(
    "-v", "--verbose",
    help="Verbose output level.", choices=[0, 1, 2])


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
    is2d = False
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
            # TODO: Do we need to copy? [Yes if data.y is reused.]
            result = (x, dx, data.y.copy(), data.dy.copy())
        except Exception:
            traceback.print_exc()
            print(f"Error when generating {model_name} for {seed}")
            result = (x, dx, np.NaN*x, np.NaN*x)
            #raise
        return result

    def pretty(pars):
        """
        Pretty the parameter set for displaying on one line
        """
        parlist = sascomp.parlist(model_info, pars, is2d)
        parlist = parlist.replace(os.linesep, '  ')
        parlist = parlist.replace(': ', '=')
        return parlist

    t0 = -np.inf
    interval = 5
    for k in range(count):
        seed = np.random.randint(int(1e6))
        t1 = time.perf_counter()
        if t1 > t0 + interval:
            print(f"generating {model_name} {k+1} of {count}")
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
        #print(f"{model_name} {seed} {pretty(pars)}")

        # Evaluate model
        data = simulate(pars) # q, dq, iq, diq

        # Skip data sets with NaN or negative numbers.
        # Note: some datasets will have fewer entries than others.
        if np.isnan(data[2]).any():
            print(f">>> NaN in {model_name} {seed} {pretty(pars)}")
            continue
        if (data[2] <= 0.).any():
            print(f">>> Negative values in {model_name} {seed} {pretty(pars)}")
            continue

        yield seed, pars, data

    # TODO: can free the calculator now
    print(f"Complete {model_name}")

def model_group(models, required=False):
    """
    Build a list of models from the items in *models*.  Could be individual
    model names or could be a unix-style glob pattern.
    """
    good = []
    bad = []
    for name in models:
        if name == "":
            continue
        items = fnmatch.filter(MODELS, name)
        if not items:
            try:
                items = sascore.list_models(name)
            except ValueError:
                pass
        if items:
            good.extend(items)
        else:
            bad.append(name)

    if bad:
        print(f"Bad model(s): {', '.join(bad)}.  ", file=sys.stderr)
    if bad or (required and not good):
        print(f"Use kind ({', '.join(sascore.KINDS)}) or one of:", file=sys.stderr)
        print(columnize(MODELS, indent="  "), file=sys.stderr, end='')
        print(f"Patterns such as *sphere will also work.")
        sys.exit(1)

    return sorted(set(good))

def run_model(opts):
    tag = opts.tag
    count = opts.count
    is2D = opts.dimension.startswith('2d')
    nq = opts.npoint
    mono = opts.mono
    magnetic = opts.magnetic
    cutoff = opts.cutoff if not mono else 0
    precision = opts.precision
    res = opts.resolution
    noise = opts.noise
    sesans = opts.sesans

    # Figure out which models we are using.
    include = model_group(opts.models, required=True)
    exclude = model_group(opts.exclude.split(','))
    model_list = sorted(set(include)-set(exclude))
    print("Selected models:\n", columnize(model_list, indent="  "))

    if opts.template:
        # Fetch q, dq from an actual SANS file
        data, index = sasdata.read(opts.template), None
    else:
        # Generate
        data, index = sascomp.make_data({
            'qmin': 1e-4, 'qmax': 0.2, 'is2d': is2D, 'nq': nq, 'res': res/100,
            'accuracy': 'Low', 'view': 'log', 'zero': False,
            'sesans': sesans,
            })

    # Open database and lookup model counts.
    db = sas_io.sql_connect(opts.database)
    model_counts = sas_io.model_counts(db, tag)
    #print(model_counts)
    for model in model_list:
        # Figure out how many more entries we need for the model
        missing = count - model_counts.get(model, 0)
        if missing <= 0:
            continue
        # TODO: should not need deepcopy(data) but something is messing with q
        seq = gen_data(
            model, deepcopy(data), count=missing, mono=mono, magnetic=magnetic,
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
    Generic iter tool: chunk sequence in groups of *batch_size*.

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
