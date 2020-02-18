"""
Various small utility functions that are reused throughout the program.
"""
import os
import time

def plot(q, i_q):
    """
    Method to plot Q vs I(Q) data for testing and verification purposes.

    :param q: List of Q values
    :param i_q: List of I values
    :return: None
    """
    from matplotlib import pyplot as plt

    plt.style.use("classic")
    plt.loglog(q, i_q)
    ax = plt.gca()
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
