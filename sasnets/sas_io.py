"""
Collection of utility IO functions used in SASNet. Contains the read from disk
functions as well as the SQL generator.
"""

from __future__ import print_function

import ast
import itertools
import logging
import multiprocessing
import os
import re

import numpy as np
import psycopg2 as psql
from keras.utils.np_utils import to_categorical
from psycopg2 import sql

gpath = ""
gpattern = ""

try: # Python 3 compatability
    xrange(1)
    nrange = xrange
except NameError:
    nrange = range


def sql_dat_gen(dname, mname, dbname="sas_data", host="127.0.0.1",
                user="sasnets", encoder=None):
    """
    A pythonic generator that gets its data from a PostgreSQL database. Yields a
    (q, iq, dq, diq) list and a label list.

    :param dname: The data table name to connect to.
    :param mname: The metadata table name to connect to.
    :param dbname: The database name.
    :param host: The database host.
    :param user: The username to connect as.
    :param encoder: LabelEncoder for transforming labels to categorical ints.
    :return: None
    """
    conn = psql.connect("dbname=" + dbname + " user=" + user + " host=" + host)
    with conn:
        with conn.cursor() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS tsm_system_rows")
            c.execute(
                sql.SQL("SELECT * FROM {}").format(
                    sql.Identifier(mname)))
            x = np.asarray(c.fetchall())
            # pprint(x)
            while True:
                c.execute(
                    sql.SQL(
                        "SELECT * FROM {} TABLESAMPLE SYSTEM_ROWS(5)").format(
                        sql.Identifier(dname)))
                x = np.asarray(c.fetchall())
                iq_list = x[:, 1]
                diq = x[:,2]
                y_list = x[:, 3]
                encoded = encoder.transform(y_list)
                yt = np.asarray(to_categorical(encoded, 64))
                q_list = np.asarray(
                    [np.transpose([np.log10(iq), np.log10(dq)]) for iq, dq in zip(iq_list,diq)])
                yield q_list, yt
    conn.close()

    # noinspection PyUnusedLocal


def read_parallel_1d(path, pattern='_eval_', typef='aggr'):
    """
    Reads all files in the folder path. Opens the files whose names match the
    regex pattern. Returns lists of Q, I(Q), and ID. Path can be a
    relative or absolute path. Uses Pool and map to speed up IO. WIP. Uses an
    excessive amount of memory currently. It is recommended to use sequential on
    systems with less than 16 GiB of memory.

    Calling parallel on 69 150k line files, a gc, and parallel on 69 5k line
    files takes around 70 seconds. Running sequential on both sets without a gc
    takes around 562 seconds. Parallel peaks at 15 + GB of memory used with two
    file reading threads. Sequential peaks at around 7 to 10 GB. Use at your own
    risk. Be prepared to kill the threads and/or press the reset button.

    typef is one of 'json' or 'aggr'. JSON mode reads in all and only json files
    in the folder specified by path. aggr mode reads in aggregated data files.
    See sasmodels/generate_sets.py for more about these formats.

    Assumes files contain 1D data.

    :param path: Path to the directory of files to read from.
    :param pattern: A regex. Only files matching this regex are opened.
    :param typef: Type of file to read (aggregate data or json data).
    :param verbosity: Controls the verbosity of output.
    """
    global gpath
    global gpattern
    # q_list, iq_list, y_list = (list() for i in range(3))
    # pattern = re.compile(pattern)
    if typef == 'aggr':
        gpattern = pattern
        gpath = path
        nlines = 0
        l = 0
        fn = os.listdir(path)
        chunked = [fn[i: i + 1] for i in nrange(0, len(fn), 1)]
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 6,
                                    maxtasksperchild=2)
        result = np.asarray(
            pool.map(read_h, chunked, chunksize=1))
        pool.close()
        pool.join()
        logging.info("IO Done")
        result = list(itertools.chain.from_iterable(result))
        q_list = result[0::3]
        iq_list = result[1::3]
        y_list = result[2::3]
    else:
        print(
            "Error: the type " + typef + " was not recognised. Valid types "
                                         "are 'aggr' and 'json'.")
        return None
    return q_list, iq_list, y_list, nlines


def read_h(l):
    """
    Read helper for parallel read.

    :param l: A list of filenames to read from.
    :return: Three lists, Q, IQ, and Y, corresponding to Q data, I(Q) data, and model labels respectively.
    """
    logging.info(os.getpid())
    if l is None:
        raise Exception("Empty args")
    global gpath # Abuse globals because pool only passes one argument
    global gpattern
    q_list, iq_list, y_list = (list() for i in range(3))
    p = re.compile(gpattern)
    for fn in l:
        if p.search(fn):
            try:
                with open(gpath + fn, 'r') as fd:
                    logging.info("Reading " + fn)
                    templ = ast.literal_eval(fd.readline().strip())
                    y_list.extend([templ[0] for i in range(templ[1])])
                    t2 = ast.literal_eval(fd.readline().strip())
                    q_list.extend([t2 for i in range(templ[1])])
                    iq_list.extend(ast.literal_eval(fd.readline().strip()))
            except Exception as e:
                logging.warning("skipped, " + str(e))
    return q_list, iq_list, y_list

    # noinspection PyCompatibility,PyUnusedLocal


def read_seq_1d(path, pattern='_eval_', typef='aggr', verbosity=False):
    """
    Reads all files in the folder path. Opens the files whose names match the
    regex pattern. Returns lists of Q, I(Q), and ID. Path can be a
    relative or absolute path. Uses a single thread only. It is recommended to
    use :meth:`read_parallel_1d`, except in hyperopt, where map() is broken.

    typef is one of 'json' or 'aggr'. JSON mode reads in all and only json files
    in the folder specified by path. aggr mode reads in aggregated data files.
    See sasmodels/generate_sets.py for more about these formats.

    Assumes files contain 1D data.

    :param path: Path to the directory of files to read from.
    :param pattern: A regex. Only files matching this regex are opened.
    :param typef: Type of file to read (aggregate data or json data).
    :param verbosity: Controls the verbosity of output.
    """
    q_list, dq_list, iq_list, diq_list, y_list = (list() for i in range(5))
    pattern = re.compile(pattern)
    n = 0
    nlines = None
    if typef == 'json':
        try:
            from ruamel.yaml import \
                safe_load  # using ruamel for better input processing.
        except ImportError:
            from json import loads as safe_load
        for fn in os.listdir(path):
            if pattern.search(fn):  # Only open JSON files
                with open(path + fn, 'r') as fd:
                    n += 1
                    data_d = safe_load(fd)
                    q_list.append(data_d['data']['Q'])
                    iq_list.append(data_d["data"]["I(Q)"])
                    y_list.append(data_d["model"])
                if (n % 100 == 0) and verbosity:
                    print("Read " + str(n) + " files.")
    if typef == 'aggr':
        nlines = 0
        for fn in sorted(os.listdir(path)):
            if pattern.search(fn):
                try:
                    with open(path + fn, 'r') as fd:
                        print("Reading " + fn)
                        templ = ast.literal_eval(fd.readline().strip())
                        y_list.extend([templ[0] for i in nrange(templ[1])])
                        t2 = ast.literal_eval(fd.readline().strip())
                        q_list.extend([t2 for i in xrange(templ[1])])
                        iq_list.extend(
                            ast.literal_eval(fd.readline().strip()))
                        #dqt = ast.literal_eval(fd.readline().strip())
                        #dq_list.extend([dqt for i in xrange(templ[1])])
                        #diqt = ast.literal_eval(fd.readline().strip())
                        #diq_list.extend([diqt for i in xrange(templ[1])])
                        nlines += templ[1]
                    if (n % 1000 == 0) and verbosity:
                        print("Read " + str(nlines) + " points.")
                except Exception as e:
                    logging.warning("skipped, " + str(e))
    else:
        print(
            "Error: the type " + typef + " was not recognised. Valid types "
                                         "are 'aggr' and 'json'.")
    return q_list, iq_list, y_list, nlines, #dq_list, diq_list, nlines
