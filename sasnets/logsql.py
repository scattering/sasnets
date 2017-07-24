from __future__ import print_function

import argparse
import ast
import os
import re
import sys

import numpy as np
from psycopg2 import sql
import psycopg2 as pgsql

#parser = argparse.ArgumentParser()
#parser.add_argument("key", help="DB Table identifier")
#parser.add_argument("path", help="Relative or absolute path to a folder "
                                # "containing data files")


# noinspection SqlNoDataSourceInspection,SqlResolve
def main(args):
    """
    Main function. Args should conform to the argparse args specified.
    :param args: Arguments from the command line
    :return: None
    """
    #parsed = parser.parse_args(args)
    conn = pgsql.connect(
        "dbname=sas_data user=sasnets password=sasnets host=127.0.0.1")
    c = conn.cursor()
    c.execute("SELECT id FROM new_eval_data")
    lid = c.fetchall()
    for nid in lid:
        c.execute(sql.SQL("SELECT * FROM new_eval_data WHERE id = %s"), (nid,))
        x = c.fetchall()
        niq = np.log10(x[0][1])
        ndq = np.log10(x[0][2])
        c.execute(sql.SQL("UPDATE new_eval_data SET iq = %s, diq = %s WHERE id = %s"),(np.ndarray.tolist(niq), np.ndarray.tolist(ndq), nid))
    conn.commit()
    c.close()
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
