"""
File used for analysis of SASNet networks using various techniques, including
dendrograms and confusion matrices.
"""
from __future__ import print_function

import argparse
import ast
import logging
import os
import random
import sys

import bottleneck
import keras
import matplotlib.pyplot as plt
import numpy as np
import psycopg2 as psql
from keras.utils import to_categorical
from pandas import factorize
from psycopg2 import sql
from sasnets.sas_io import read_seq_1d
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(
    description="Test a previously trained neural network, or use it to "
                "classify new data.")
parser.add_argument("model_file",
                    help="Path to h5 model file from sasnet.py. The directory "
                         "should also contain a 'name' file containing a list "
                         "of models the network was trained on.")
parser.add_argument("data_path", help="Path to load data from.")
parser.add_argument("-v", "--verbose", help="Verbose output.",
                    action="store_true")
parser.add_argument("-c", "--classify", help="Classification mode.",
                    action="store_true")
parser.add_argument("-p", "--pattern",
                    help="Pattern to match to files to open.")


def load_from(path):
    """
    Loads a model from the specified path.

    :param path: Relative or absolute path to the .h5 model file
    :return: The loaded model.
    """
    return keras.models.load_model(os.path.normpath(path))


def predict_and_val(model, x, y, names):
    """
    Runs the model on the input datasets and compares the results with the
    correct labels provided from y.

    :param model: The model to evaluate.
    :param x: List of x values to predict on.
    :param y: List of y values to predict on.
    :param names: A list of all possible model names.
    :return: Two lists, il and nl, which are the indices of the model and its proper name respectively.
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)

    prob = model.predict(x)
    err = [0] * (len(set(names)))
    ind = 0
    il = list()
    nl = list()
    for p, e, in zip(prob, yt):
        p1 = p.argmax(axis=-1)
        e1 = e.argmax(axis=-1)
        if names[p1] != y[e1]:
            print("Predicted: " + str(names[p1]) + ", Actual: " + str(
                y[e1]) + ", Index: " + str(ind) + ".")
            err[e1] += 1
            nl.append(p1)
            il.append(ind)
        ind += 1
    print(err)
    return il, nl


def predict(model, x, names, num=5):
    """
    Runs a Keras model to predict based on input.

    :param model: The model to use.
    :param x: The x inputs to predict from.
    :param names: A list of all model names.
    :param num: The top num probabilities and models will be printed.
    :return: None
    """
    prob = model.predict(x)
    for p in prob:
        pt = bottleneck.argpartition(p, num)[-num:]
        plist = pt[np.argsort(p[pt])]
        sys.stdout.write("The " + str(num) + " five most likely models and " +
                         "respective probabilities were: ")
        for i in reversed(plist):
            sys.stdout.write(str(names[i]) + ", " + str(p[i]) + " ")
        sys.stdout.write("\n")


def cpredict(model, x, l=71, pl=5000):
    """
    Runs a Keras model to create a confusion matrix.

    :param model: Model to use.
    :param x: A list of x values to predict on.
    :param l: The number of input models.
    :param pl: The number of data iterations per model.
    :return: A confusion matrix of percentages
    """
    res = np.zeros([l, l])
    row = 0
    c = 0
    prob = model.predict(x, verbose=1)
    for p in prob:
        pt = bottleneck.nanargmax(p)
        res[row][pt] += 1
        c += 1
        if c % 5000 == 0:
            row += 1
    return np.divide(res, float(pl))


def rpredict(model, x, names):
    """
    Same as predict, but outputs names only.

    :param model: The model to use.
    :param x: List of x to predict on.
    :param names: List of all model names.
    :return: List of predicted names.
    """
    res = list()
    prob = model.predict(x, verbose=1)
    for p in prob:
        pt = bottleneck.nanargmax(p)
        res.append(names[pt])
    return res


def fit(mn, q, iq):
    """
    Fit resulting data using bumps server. Currently unimplemented.

    :param mn: Model name.
    :param q: List of q values.
    :param iq: List of I(q) values.
    :return: Bumps fit.
    """
    logging.info("Starting fit")
    return (mn, q, iq)


def tcluster(model, x, names):
    """
    Displays a t-SNE cluster coloured by the model predicted labels.

    :param model: Model to use.
    :param x: List of x values to predict on.
    :param names: List of all model names.
    :return: The tSNE object that was plotted.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None
        pass
    xt = random.sample(x, 5000)
    arr = rpredict(model, xt, names)
    t = TSNE(n_components=2, verbose=2)
    classx = t.fit_transform(xt)
    if sns is not None:
        p = np.array(sns.color_palette("hls", 69))
        plt.scatter(classx[:, 0], classx[:, 1],
                    c=p[np.asarray(factorize(arr)[0]).astype(np.int)])
    else:
        plt.scatter(classx[:, 0], classx[:, 1])
    plt.show()
    return classx


def dcluster(model, x, names):
    """
    Displays a dendrogram clustering based on the confusion matrix.

    :param model: The model to predict on.
    :param x: A list of x values to predict on.
    :param names: List of all model names.
    :return: The dendrogram object.
    """
    arr = cpredict(model, x, names)
    z = linkage(arr, 'average')
    h = dendrogram(z, leaf_rotation=90., leaf_font_size=8, labels=names,
                   color_threshold=.5, get_leaves=True)
    plt.tight_layout()
    plt.show()
    return h


def main(args):
    """
    Main method. Called from command line; uses argparse.

    :param args: Arguments from command line.
    :return: None.
    """
    parsed = parser.parse_args(args)
    with open(os.path.join(os.path.dirname(parsed.data_path), "name"),
              'r') as fd:
        n = ast.literal_eval(fd.readline().strip())
        q, iq, y, dq, diq, nlines = read_seq_1d(parsed.data_path,
        pattern=parsed.pattern,
        verbosity=parsed.verbose)
    # conn = psql.connect("dbname=sas_data user=sasnets host=127.0.0.1")
    # # conn.set_session(readonly=True)
    # with conn:
    #     with conn.cursor() as c:
    #         c.execute("SELECT model FROM train_data;")
    #         xt = set(c.fetchall())
    #         y = [i[0] for i in xt]
    #         encoder = LabelEncoder()
    #         encoder.fit(y)
    #
    #         c.execute("CREATE EXTENSION IF NOT EXISTS tsm_system_rows")
    #         c.execute(
    #             sql.SQL("SELECT * FROM {}").format(
    #                 sql.Identifier("train_metadata")))
    #         x = np.asarray(c.fetchall())
    #         q = x[0][1]
    #         dq = x[0][2]
    #         diq = x[0][3]
    #         c.execute(sql.SQL(
    #             "SELECT * FROM {}").format(
    #             sql.Identifier("eval_data")))
    #         x = np.asarray(c.fetchall())
    #         iq_list = x[:, 1]
    #         y_list = x[:, 2]
    #         encoded = encoder.transform(y_list)
    #         yt = np.asarray(to_categorical(encoded, 71))
    #         q_list = np.asarray([np.transpose([q, iq, dq, diq]) for iq in
    #                              iq_list])
    model = load_from(parsed.model_file)
    if parsed.classify:
        # tcluster(model, b, n, c)
        z = dcluster(model, iq, n)
        plt.pcolor(z, cmap='RdBu')
        plt.show()
    else:
        ilist, nlist = predict_and_val(model, q, y, n)
        for i, n1 in zip(ilist, nlist):
            plt.style.use("classic")
            plt.plot(q[i], iq[i])
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.autoscale(enable=True)
            with open("/home/chwang/out/Img/" + n[n1] + str(i), 'w') as fd:
                plt.savefig(fd, format="svg", bbox_inches="tight")
                plt.clf()


if __name__ == '__main__':
    main(sys.argv[1:])
