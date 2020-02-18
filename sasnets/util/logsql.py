from __future__ import print_function

import argparse
import sys
import sqlite3

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("tablename", help="Database table name to run on.")
parser.add_argument("--dbname", help="Database name to connect to.",
                    default="sasdata.db")
parser.add_argument("--user", help="Database username.", default="sasnets")
parser.add_argument("--password", help="Database password.", default="sasnets")
parser.add_argument("--host", help="Database host.", default="127.0.0.1")


def main(args):
    """
    Main function. Args should conform to the argparse args specified.

    :param args: Arguments from the command line
    :return: None
    """
    parsed = parser.parse_args(args)
    conn = sqlite3.connect(
        "dbname=" + parsed.dbname + " user=" + parsed.user + " password=" +
        parsed.password + " host=" + parsed.host)
    c = conn.cursor()
    c.execute(sqlite3.SQL("SELECT id FROM {}").format(sql.Identifier(parsed.name)))
    lid = c.fetchall()
    for nid in lid:
        c.execute(sql.SQL("SELECT * FROM {} WHERE id = %s").format(
            sql.Identifier(parsed.name)), (nid,))
        x = c.fetchall()
        niq = np.log10(x[0][1])
        ndq = np.log10(x[0][2])
        c.execute(
            sql.SQL("UPDATE {} SET iq = %s, diq = %s WHERE id = %s").format(
                sql.Identifier(parsed.name)),
            (np.ndarray.tolist(niq), np.ndarray.tolist(ndq), nid))
    conn.commit()
    c.close()
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
