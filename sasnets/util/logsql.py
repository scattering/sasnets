"""
Convert data tables from linear iq,diq to log(iq), diq/iq
"""
from __future__ import print_function

import argparse
import sys
from contextlib import closing

import numpy as np

from .. import sas_io
from ..sas_io import asblob, asdata

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database", type=str, default=sas_io.DB_FILE,
    help="Path to the sqlite database file.")
parser.add_argument(
    "--tag", type=str, default="train",
    help="Tag for the generated data: train or test.")


def main(args):
    """
    Main function. Args should conform to the argparse args specified.

    :param args: Arguments from the command line
    :return: None
    """
    opts = parser.parse_args(args)
    conn = sas_io.sql_connect(opts.database)
    with conn, closing(conn.cursor()) as cursor:
        cursor.execute(f"SELECT rowid FROM {opts.tag}")
        for rowid in cursor.fetchall():
            cursor.execute(
                f"SELECT iq, diq FROM {opts.tag} WHERE rowid = ?",
                (rowid,))
            iq, diq = cursor.fetchall()[0]
            iq, diq = [asblob(np.log10(asdata(v))) for v in (iq, diq)]
            cursor.execute(
                f"UPDATE {opts.tag} SET iq = ?, diq = ? WHERE id = ?",
                (iq, diq, rowid))

if __name__ == "__main__":
    main(sys.argv[1:])
