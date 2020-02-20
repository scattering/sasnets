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
import warnings

import numpy as np
from pandas import factorize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

try:
    # Use bottleneck for nan functions if it is available.
    from bottleneck import nanargmax, argpartition
except ImportError:
    from numpy import nanargmax, argpartition
    #warnings.warn("Using numpy since bottleneck isn't available")

from . import sas_io
from .sasnet import OnehotEncoder, fix_dims, reload_net

def columnize(items):
    try:
        from columnize import columnize as _columnize, default_opts
        return _columnize(list(items), default_opts['displaywidth'])
    except ImportError:
        return "\n".join(items)

parser = argparse.ArgumentParser(
    description="Test a previously trained neural network, or use it to "
                "classify new data.")
parser.add_argument(
    "model_file", default="savenet/out.h5",
    help="""
        Path to h5 model file from sasnet.py. The directory
        should also contain a 'name' file containing a list
        of models the network was trained on.""")
parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose output.")
parser.add_argument("-c", "--classify", action="store_true",
    help="Classification mode.")
parser.add_argument(
    "--database", type=str, default=sas_io.DB_FILE,
    help="Path to the sqlite database file.")


def predict_and_val(classifier, x, y, categories):
    """
    Runs the classifier on the input datasets and compares the results with the
    correct labels provided from y.

    :param classifier: The trained classifier.
    :param x: List of x values to predict on.
    :param y: List of y values to predict on.
    :param categories: A list of all possible sas model names.
    :return: Two lists, the indices of the sas model and its proper name respectively.
    """
    encoder = OnehotEncoder(categories)
    prediction = classifier.predict(fix_dims(x))
    error_frequency = {name: 0 for name in categories}
    errors = []
    for k, (prob, actual) in enumerate(zip(prediction, y)):
        mle = prob.argmax(axis=-1)
        predicted = encoder.label([mle])[0]
        if predicted != actual:
            actual_index = encoder.index([actual])[0]
            try:
                ratio = prob[actual_index] / prob[mle]
            except IndexError:
                print(actual, actual_index, mle, predicted, len(categories), prob)
                raise
            if len(errors) == 20:
                print("...")
            if len(errors) < 20:
                print(f"Predicted: {predicted}, Actual: {actual}, Index: {k}, Prediction ratio: {ratio:.2}.")
            error_frequency[actual] += 1
            errors.append((k, predicted, actual, ratio))
    print(columnize(f"{k}:{v:<3}" for k, v in sorted(error_frequency.items())))
    rows, predicted, _, _ = zip(*errors)
    return rows, predicted

def show_predictions(classifier, x, y, categories, rank=5):
    """
    Runs a Keras classifier to predict based on input.

    :param classifier: The trained classifier.
    :param x: The x inputs to predict from.
    :param y: The expected value, or None if no expectation.
    :param categories: A list of all model names.
    :param rank: The top num probabilities and models will be printed.
    :return: None
    """
    encoder = OnehotEncoder(categories)
    prediction = classifier.predict(fix_dims(x))
    target = (lambda k: f"{k}") if y is None else (lambda k: f"{k}({y[k]})")
    for k, prob in enumerate(prediction):
        pt = argpartition(p, rank)[-rank:]
        plist = pt[np.argsort(prob[pt])]
        labels = encoder.label(plist)
        values = prob[plist]
        print(f"{target(k)} => ", end="")
        for k, v in reversed(zip(labels, values)):
            print(f"{k}:{v:.3} ", end="")
        print("")


def confusion_matrix(classifier, x, y, categories):
    """
    Runs a Keras classifier to create a confusion matrix.

    :param classifier: The trained classifier.
    :param x: A list of x values to predict on.
    :return: A confusion matrix, with proportions in [0, 1]
    """
    encoder = OnehotEncoder(categories)
    index = encoder.index(y)
    prediction = classifier.predict(fix_dims(x), verbose=1)
    n = len(categories)
    res = np.zeros((n, n))
    weight = np.zeros(n)
    for k, (prob, row) in enumerate(zip(prediction, index)):
        mle = nanargmax(prob)
        res[row][mle] += 1
        weight[row] += 1
    return res/weight[:, None] # TODO: divide row or column?

def rpredict(classifier, x, categories):
    """
    Same as predict, but outputs names only.

    :param classifier: The trained classifier.
    :param x: List of x to predict on.
    :param categories: List of all model names.
    :return: List of predicted names.
    """
    encoder = OnehotEncoder(categories)
    prediction = classifier.predict(fix_dims(x), verbose=1)
    index = [nanargmax(prob) for prob in prediction]
    label = encoder.label(index)
    return label.tolist()


def fit(model, q, dq, iq, diq):
    """
    Fit resulting data using bumps server. Currently unimplemented.

    :param model: sasmodels name.
    :param q: List of q values.
    :param iq: List of I(q) values.
    :return: Bumps fit.
    """
    logging.info("Starting fit")
    return (model, q, iq)

def plot_tSNE(classifier, x, categories):
    """
    Displays a t-SNE cluster coloured by the classifier predicted labels.

    :param classifier: The trained classifier.
    :param x: List of x values to predict on.
    :param categories: List of all model names.
    :return: The tSNE object that was plotted.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    xt = random.sample(x, 5000)
    arr = rpredict(classifier, xt,categories)
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


def plot_dendrogram(classifier, x, y, categories):
    """
    Displays a dendrogram clustering based on the confusion matrix.

    :param classifier: The trained classifier.
    :param x: A list of x values to predict on.
    :param y: The target values for the predictions.
    :param categories: List of all model names.
    :return: The dendrogram object.
    """
    import matplotlib.pyplot as plt
    arr = confusion_matrix(classifier, x, y, categories)
    plt.subplot(211)
    plt.pcolor(arr, cmap='RdBu')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.subplot(212)
    z = linkage(arr, 'average')
    h = dendrogram(z, leaf_rotation=90., leaf_font_size=8, labels=categories,
                   color_threshold=.5, get_leaves=True)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_failures(failures, q, iq):
    import matplotlib.pyplot as plt
    index, predicted = failures
    if len(index) > 100:
        warnings.warn(f"too many failures to plot {len(index)}")
        return None
    for i, name in zip(index, predicted):
        plt.style.use("classic")
        plt.plot(q[i], iq[i])
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.autoscale(enable=True)
        with open(f"./savenet/failed-{name}-{i}.svg", "w") as fd:
            plt.savefig(fd, format="svg", bbox_inches="tight")
            plt.clf()

def main(args):
    """
    Main method. Called from command line; uses argparse.

    :param args: Arguments from command line.
    :return: None.
    """
    opts = parser.parse_args(args)
    db = sas_io.sql_connect(opts.database)
    labels, q, dq, iq, diq = sas_io.read_sql_all(db)
    log_iq = [sas_io.input_encoder(v) for v in iq]
    categories = sorted(set(labels))
    classifier = reload_net(opts.model_file)
    if opts.classify:
        # plot_tSNE(classifier, logiq, categories)
        plot_dendrogram(classifier, log_iq, labels, categories)
    else:
        failures = predict_and_val(classifier, log_iq, labels, categories)
        plot_failures(failures, q, iq)

if __name__ == '__main__':
    main(sys.argv[1:])
