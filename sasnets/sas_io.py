"""
Collection of utility IO functions used in SASNet. Contains the read from disk
functions as well as the SQL generator.
"""

from __future__ import print_function

from contextlib import closing
import ast
import itertools
import logging
import multiprocessing
import functools
import os
import re
import json

try:
    # ruamel has better json input processing.
    from ruamel.yaml import safe_load as json_load
except ImportError:
    from json import loads as json_load

import numpy as np

## In case we want to go back to PostgreSQL
#def sql_connect(dbname="sas_data", host="127.0.0.1", user="sasnets"):
#    import psycopg2 as pqsql
#    conn = pgsql.connect(f"dbname={dbname} user={user} host={host}")
#    conn.execute(f"CREATE EXTENSION IF NOT EXISTS tsm_system_rows")
#    return conn

DB_DTYPE = np.dtype('<f4')
DB_FILE = "sasnets.db"
def sql_connect(dbfile=DB_FILE):
    import sqlite3
    conn = sqlite3.connect(dbfile)
    return conn


def sql_iread(db, datatable, metatable, encoder=None, batch_size=5):
    """
    Generator that gets its data from a SQL database.

    Yields batches of data and label with data = [log10(iq), ...] and
    label = ['model', ...].

    Data is chosen at random with replacement and runs forever.

    :param db: The SQL database connection.
    :param datatable: The data table name to connect to.
    :param encoder: LabelEncoder for transforming labels to categorical ints.
    """
    from keras.utils.np_utils import to_categorical

    # Build query string, checking values before substituting
    batch_size = int(batch_size) # force integer
    assert datatable.isidentifier()
    # using implicit rowid column in (most) SQLite tables.
    random_row_query = f"""
        SELECT (model, iq) FROM {datatable} WHERE rowid IN
            (SELECT rowid FROM table ORDER BY RANDOM() LIMIT {batch_size})
        """
    ## PostgreSQL supports tablesample
    #random_row_query = f"""
    #    SELECT (iq, model) FROM {datatable}
    #        TABLESAMPLE SYSTEM_ROWS({batch_size})"""

    with db, closing(db.cursor()) as cursor:
        while True:
            cursor.execute(random_row_query)
            data = cursor.fetchall()
            models, iq = zip(*data)
            # Convert binary blob back into numpy array
            iq = [np.frombuffer(v, dtype=DB_DTYPE) for v in iq]
            iq = [np.log10(v) for v in iq]
            encoded = encoder.transform(models)
            categorical = np.asarray(to_categorical(encoded, 64))
            yield iq, categorical

def read_sql(db, datatable):
    assert datatable.isidentifier()
    all_rows = f"SELECT model, iq FROM {datatable}"
    with db, closing(db.cursor()) as cursor:
        cursor.execute(all_rows)
        data = cursor.fetchall()
    model, iq = zip(*data)
    # Convert binary blob back into numpy array
    iq = [np.frombuffer(v, dtype=DB_DTYPE) for v in iq]
    return iq, model

def read_1d_parallel(path, tag='train', format='csv', verbose=True):
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

    Assumes files contain 1D data.

    :param path: Path to the directory of files to read from.
    :param pattern: A regex. Only files matching this regex are opened.
    """
    parser = re.compile(f"_{tag}_.*[.]{format}")
    files = (fn+path for fn in os.listdir(path) if parser.search(fn))
    pool = multiprocessing.Pool(multiprocessing.cpu_count() / 2,
                                maxtasksperchild=2)
    reader = _read_csv if format == 'csv' else _read_json
    delayed = pool.imap_unordered(reader, files, chunksize=100)
    # TODO: try prellocating array joined iterable
    iq, labels = (np.asarray(v) for v in zip(*delayed))
    pool.join()
    pool.close()
    logging.info("IO Done")
    return iq, labels

def read_1d_serial(path, tag='train', format='csv', verbose=True):
    """
    Reads all files in the folder path. Opens the files whose names match the
    regex pattern. Returns lists of I(Q), and model. Path can be a
    relative or absolute path. Uses a single thread only. It is recommended to
    use :meth:`read_parallel_1d`, except in hyperopt, where map() is broken.

    *format* is one of 'json' or 'csv'. JSON mode reads in all and only json
    files in the folder specified by path. csv mode reads in aggregated data
    files. See sasmodels/generate_sets.py for more about these formats.

    Assumes files contain 1D data.

    :param path: Path to the directory of files to read from.
    :param tag: A regex. Only files matching this regex are opened.
    :param typef: Type of file to read (aggregate data or json data).
    :param verbose: Controls the verbose of output.
    """
    parser = re.compile(f"_{tag}_.*[.]{format}")
    files = (fn+path for fn in os.listdir(path) if parser.search(fn))
    if format == 'json':
        items = []
        for k, fn in enumerate(files):
            items.append(_read_json(path+fn))
            if (k % 100 == 0) and verbose:
                print("Read " + str(k) + " files.")
    elif format == 'csv':
        items = []
        for k, fn in enumerate(files):
            items.append(_read_csv(path+fn))
            if (k % 100 == 0) and verbose:
                print("Read " + str(k) + " files.")
    else:
        print(f"Error: the type {format} was not recognised. Valid types"
              f" are 'aggr' and 'json'.")
    iq, model = zip(*items)
    return iq, model

def _read_csv(path):
    with open(path, 'r') as fd:
        logging.info("Reading " + path)
        model = ast.literal_eval(fd.readline().strip())[0]
        fd.readline() # q
        fd.readline() # dq
        iq = ast.literal_eval(fd.readline().strip())
        #fd.readline() # iq
        return iq, model

def _read_json(path):
    with open(path, 'r') as fd:
        data_d = json_load(fd)
        return data_d["data"]["IQ"], data_d["model"]


# Jie Yang https://stackoverflow.com/a/57915246/6195051
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_sql(db, table, model, items):
    with db, closing(db.cursor()) as cursor:
        # Check that table exists before writing
        cursor.execute(f"""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='{table}'""")
        result = cursor.fetchall()
        if not result:
            # TODO: Make model+seed the primary key so no duplicates?
            cursor.execute(f"""
                CREATE TABLE {table} (
                    model TEXT, seed INTEGER, params TEXT,
                    q BLOB, dq BLOB, iq BLOB, diq BLOB)
                """)
        # Write entries to the table
        for seed, params, data in items:
            #print("key", model, seed, data[2][:3])
            q, dq, iq, diq = (np.asarray(v, DB_DTYPE).tobytes() for v in data)
            paramstr = json.dumps(params, cls=NpEncoder)
            #paramstr = json.dumps({k: float(v) for k, v in params.items()})
            #disp = lambda x: (print(x),x)[1]
            cursor.execute(f"""
                INSERT INTO {table} (model, seed, params, q, dq, iq, diq)
                VALUES ('{model}', {seed}, ?, ?, ?, ?, ?)""",
                (paramstr, q, dq, iq, diq))
        db.commit()

def write_1d(dirname, tag, model, items, format='csv'):
    """
    Write a series of *items* for *model* into file *dirname_tag_seed.format*.

    *tag* is the name of the group, such as 'trial' or 'validate'.

    *model* is the model name.

    *items* is a sequence *[(seed, params, (q, dq, iq, diq)), ...]*.

    *format* is 'csv' or 'json'
    """
    assert format in ('csv', 'json')
    writer = _write_csv if format == 'csv' else _write_json
    for seed, params, data in items:
        filename = f"{model}_{tag}_{seed}.{format}"
        path = os.path.join(dirname, filename)
        logging.info("Writing " + path)
        writer(path, model, seed, params, data)

def _write_csv(path, model, seed, params, data):
    with open(path, 'w') as fd:
        # First line contains model, seed, par, val, par, val, ...
        line = (model, seed) + (s for pair in params.items() for s in pair)
        fd.write(",".join(repr(v) for v in line))
        fd.write("\n")
        # Subsequent lines contain data q, dq, iq, diq
        for line in data:
            fd.write(",".join(repr(v) for v in line))
            fd.write("\n")

def _write_json(path, model, seed, params, data):
    with open(path, 'w') as fd:
        q, dq, iq, diq = data
        data = {'Q': q, 'dQ': dq, 'IQ': iq, 'dIQ': diq}
        value = {'model': model, 'seed': seed, 'params': params, 'data': data}
        json.dump(fd, value, cls=NpEncoder)


if __name__ == '__main__':
    raise NotImplementedError("Cannot run sas_io as main. Import a specific "
                              "function instead.")
