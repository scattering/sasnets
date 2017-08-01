"""
Various small utility functions that are reused throughout the program.
"""
from __future__ import print_function

import os
import time

from matplotlib import pyplot as plt


def plot(q, i_q):
    """
    Method to plot Q vs I(Q) data for testing and verification purposes.

    :param q: List of Q values
    :param i_q: List of I values
    :return: None
    """
    plt.style.use("classic")
    plt.plot(q, i_q)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.autoscale(enable=True)
    plt.show()


def inepath(pathname):
    """
    Returns a normalised path given a path. Checks if the path exists and
    creates the path if it does not. If a directory is specified, appends the
    current time as the filename.

    :param pathname: The pathname to process.
    :return: A normalised path, or None.
    """
    sp = os.path.normpath(pathname)
    if sp is not None:
        if not os.path.exists(os.path.dirname(sp)):
            os.makedirs(os.path.dirname(sp))
        if os.path.isdir(sp):
            return os.path.join(sp, str(time.time()))
        else:
            return sp
    return None
