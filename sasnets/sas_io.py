"""
Collection of utility IO functions used in SASNet. Contains the read from disk
functions as well as the SQL generator.
"""

from __future__ import print_function

from contextlib import closing
import ast
import logging
import multiprocessing
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
def sql_connect(dbfile=DB_FILE, mode='ro'):
    """
    Open the sqlite database.

    Use *mode='ro'* for read-only access (the default). This is required
    if the database is on an NFS mounted filesystem since locking is not
    supported by SQLite.
    """
    import sqlite3
    # since we are using "with connection:" for our transactions, we
    # get commit-rollback for free.  Otherwise we might be tempted to
    # use auto-commit as follows:
    #      conn = sqlite3.connect(dbfile, isolation_level=None)
    #conn = sqlite3.connect(dbfile)
    if dbfile.startswith('file:'):
        uri = f"{dbfile}?mode={mode}"
    else:
        uri = f"file:{os.path.realpath(os.path.expanduser(dbfile))}?mode={mode}"
    conn = sqlite3.connect(uri, uri=True)
    return conn

def sql_tables(db):
    # Check that table exists before writing
    with closing(db.cursor()) as cursor:
        result = _get_tables(cursor)
    print(f"tables: {sorted(result)}")
    return len(result) > 0

def _get_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return sorted(cursor.fetchall())


def asdata(blob):
    return np.frombuffer(blob, dtype=DB_DTYPE)

def asblob(data):
    return np.asarray(data, DB_DTYPE).tobytes()

def input_encoder(iq):
    """
    Feature engineering on inputs: scale peak to 1 and take the log.
    """
    return np.log10(iq) - np.log10(iq.max())


# TODO: maybe return interleaved points log10(iq), diq/iq
# Then we can maybe learn to ignore noise in the data?
# TODO: generalize to take a set of columns a column to data transform
# Either that or give a choice amongst standard transforms, such as log/linear
def iread_sql(db, tag, batch_size=5,
              encoder=lambda y: y,
              input_encoder=input_encoder,
              random=True,
              ):
    """
    Generator that gets its data from a SQL database.

    Yields batches of data and label with data = [log10(iq), ...] and
    label = [encoder(model), ...].

    Data is chosen at random with replacement and runs forever.

    Note: batches may be different sizes if some datasets contain NaN.
    It would be better to strip these before adding to the database.

    Random batches require a table with no deleted rows. If there are a few
    deleted rows then some batches may be smaller. If there are many deleted
    rows then some batches may be empty.

    :param db: The SQL database connection.
    :param datatable: The data table name to connect to.
    :param encoder: *encoder(label)* converts model name to target encoding.
    :param input_encoder: *input_encoder(iq)* scale and transform iq
    :param random: return random batches rather than ordered batches
    """
    # Build query string, checking values before substituting
    assert tag.isidentifier()
    sql_batch = _random_sqlite if random else _ordered_sqlite
    columns = "model, iq"
    for batch in sql_batch(db, tag, columns, batch_size):
        labels, iq = zip(*batch)
        # Convert binary blob back numpy array
        iq = [asdata(v) for v in iq]
        # Normalize to peak = 1 and take the log
        if input_encoder is not None:
            iq = [input_encoder(v) for v in iq]
        yield iq, encoder(labels)

def _random_sqlite(db, table, columns, batch_size):
    with db, closing(db.cursor()) as cursor:
        # Find number of rows in the table. Assumes no deleted rows.
        cursor.execute(f"SELECT max(rowid) FROM {table} LIMIT 1")
        num_rows = cursor.fetchall()[0][0]
        # Fetch k random rows.
        random_row_query = f"""
            SELECT {columns} FROM {table}
            WHERE rowid IN (
                SELECT abs(random())%{num_rows}+1 FROM {table} LIMIT {batch_size}
            )
            """
        #print(random_row_query)
        while True:
            cursor.execute(random_row_query)
            batch = cursor.fetchall()
            yield batch
    ## PostgreSQL supports tablesample
    #random_row_query = f"""
    #    SELECT {columns} FROM {table}
    #        TABLESAMPLE SYSTEM_ROWS({batch_size})"""

def _ordered_sqlite(db, table, columns, batch_size):
    ordered_query = f"SELECT {columns} from {table}"
    batch = []  # Start with an empty batch
    with db, closing(db.cursor()) as cursor:
        while True:
            # Query from the start of the database
            cursor.execute(ordered_query)
            # If rollover then fill the last batch from the start and yield
            if batch:
                batch = batch + cursor.fetchmany(batch_size - len(batch))
                yield batch
            # yield full batches until the final unfilled batch. This breaks
            # the inner loop and goes back to the outer loop starting the
            # cycle again.
            while True:
                batch = cursor.fetchmany(batch_size)
                if len(batch) < batch_size:
                    break
                yield batch

def model_counts(db, tag='train'):
    """
    Returns {'name': count, ...} for each model appearing in table *tag*.
    """
    # Note: Not writing so don't need "with db".
    with closing(db.cursor()) as cursor:
        if not _table_exists(cursor, tag):
            return {}
        cursor.execute(f"select model, count(model) from {tag} group by model")
        counts = cursor.fetchall()
    return dict(counts)

def _table_exists(cursor, tag):
    # Check that table exists before writing
    cursor.execute(f"""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='{tag}'""")
    result = cursor.fetchall()
    return len(result) > 0

def read_sql(db, tag='train', input_encoder=input_encoder, limit=None):
    assert tag.isidentifier()
    limit_str = "" if limit is None else f" LIMIT {limit}"
    all_rows = f"SELECT model, iq FROM {tag}{limit_str}"
    # Note: Not writing so don't need "with db".
    with closing(db.cursor()) as cursor:
        if not _table_exists(cursor, tag):
            raise ValueError(f"table {tag} doesn't exist: one of {_get_tables(cursor)}.")
        cursor.execute(all_rows)
        data = cursor.fetchall()
    label, iq = zip(*data)
    # Convert binary blob back into numpy array
    iq = [asdata(v) for v in iq]
    if input_encoder is not None:
        iq = [input_encoder(v) for v in iq]
    return iq, label

def read_sql_all(db, tag='train'):
    assert tag.isidentifier()
    all_rows = f"SELECT model, q, dq, iq, diq FROM {tag}"
    # Note: Not writing so don't need "with db".
    with closing(db.cursor()) as cursor:
        if not _table_exists(cursor, tag):
            raise ValueError(f"table {tag} doesn't exist: one of {_get_tables(cursor)}.")
        cursor.execute(all_rows)
        data = cursor.fetchall()
    label, *columns = zip(*data)
    # Convert binary blob back into numpy array
    columns = [[asdata(v) for v in col] for col in columns]
    return [label] + columns

def write_sql(db, label, items, tag='train'):
    assert tag.isidentifier()
    with db, closing(db.cursor()) as cursor:
        # Check that table exists before writing
        if not _table_exists(cursor, tag):
            # TODO: Make label+seed the primary key so no duplicates?
            cursor.execute(f"""
                CREATE TABLE {tag} (
                    model TEXT, seed INTEGER, params TEXT,
                    q BLOB, dq BLOB, iq BLOB, diq BLOB)
                """)
        # Write entries to the table
        for seed, params, data in items:
            #print("key", label, seed, data[2][:3])
            q, dq, iq, diq = (asblob(v) for v in data)
            paramstr = json.dumps(params, cls=NpEncoder)
            #paramstr = json.dumps({k: float(v) for k, v in params.items()})
            #disp = lambda x: (print(x),x)[1]
            cursor.execute(f"""
                INSERT INTO {tag} (model, seed, params, q, dq, iq, diq)
                VALUES ('{label}', {seed}, ?, ?, ?, ?, ?)""",
                (paramstr, q, dq, iq, diq))

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

# ======= old method --- data in separate files =====
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
    regex pattern. Returns lists of I(Q), and label. Path can be a
    relative or absolute path. Uses a single thread only. It is recommended to
    use :meth:`read_parallel_1d`, except in hyperopt, where map() is broken.

    *format* is one of 'json' or 'csv'. JSON mode reads in all and only json
    files in the folder specified by path. csv mode reads in comma separated
    data files. See sasmodels/generate_sets.py for more about these formats.

    Assumes files contain 1D data.

    :param path: Path to the directory of files to read from.
    :param tag: A regex. Only files matching this regex are opened.
    :param typef: Type of file to read (csv data or json data).
    :param verbose: Controls the verbose of output.
    """
    iq, label = zip(*iread_1d(path, tag, format, verbose))
    return iq, label

def iread_1d(path, tag='train', format='csv', verbose=True):
    """
    Read from iterator
    """
    parser = re.compile(f"_{tag}_.*[.]{format}")
    files = (fn+path for fn in os.listdir(path) if parser.search(fn))
    if format == 'json':
        for k, fn in enumerate(files):
            yield _read_json(path+fn)
            if (k % 100 == 0) and verbose:
                print("Read " + str(k) + " files.")
    elif format == 'csv':
        items = []
        for k, fn in enumerate(files):
            yield _read_csv(path+fn)
            if (k % 100 == 0) and verbose:
                print("Read " + str(k) + " files.")
    else:
        print(f"Error: the type {format} was not recognised. Valid types"
              f" are 'csv' and 'json'.")
    iq, label = zip(*items)
    return iq, label

def _read_csv(path):
    with open(path, 'r') as fd:
        logging.info("Reading " + path)
        label = ast.literal_eval(fd.readline().strip())[0]
        fd.readline() # q
        fd.readline() # dq
        iq = ast.literal_eval(fd.readline().strip())
        #fd.readline() # iq
        return iq, label

def _read_json(path):
    with open(path, 'r') as fd:
        data_d = json_load(fd)
        return data_d["data"]["IQ"], data_d["model"]

def write_1d(dirname, label, items, tag='train', format='csv'):
    """
    Write a series of *items* for *label* into file *dirname_tag_seed.format*.

    *tag* is the name of the group, such as 'trial' or 'validate'.

    *label* is the model name.

    *items* is a sequence *[(seed, params, (q, dq, iq, diq)), ...]*.

    *format* is 'csv' or 'json'
    """
    assert format in ('csv', 'json')
    writer = _write_csv if format == 'csv' else _write_json
    for seed, params, data in items:
        filename = f"{label}_{tag}_{seed}.{format}"
        path = os.path.join(dirname, filename)
        logging.info("Writing " + path)
        writer(path, label, seed, params, data)

def _write_csv(path, label, seed, params, data):
    with open(path, 'w') as fd:
        # First line contains label, seed, par, val, par, val, ...
        line = (label, seed) + (s for pair in params.items() for s in pair)
        fd.write(",".join(repr(v) for v in line))
        fd.write("\n")
        # Subsequent lines contain data q, dq, iq, diq
        for line in data:
            fd.write(",".join(repr(v) for v in line))
            fd.write("\n")

def _write_json(path, label, seed, params, data):
    with open(path, 'w') as fd:
        q, dq, iq, diq = data
        data = {'Q': q, 'dQ': dq, 'IQ': iq, 'dIQ': diq}
        value = {'model': label, 'seed': seed, 'params': params, 'data': data}
        json.dump(fd, value, cls=NpEncoder)


if __name__ == '__main__':
    raise NotImplementedError("Cannot run sas_io as main. Import a specific "
                              "function instead.")
