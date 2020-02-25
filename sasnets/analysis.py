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
from sklearn.preprocessing import LabelEncoder

try:
    # Use bottleneck for nan functions if it is available.
    from bottleneck import nanargmax, argpartition
except ImportError:
    from numpy import nanargmax, argpartition
    #warnings.warn("Using numpy since bottleneck isn't available")

from . import sas_io
from .sasnet import OnehotEncoder, fix_dims, reload_net
from .util.utils import columnize

parser = argparse.ArgumentParser(
    description="Test a previously trained neural network, or use it to "
                "classify new data.")
parser.add_argument(
    "model_file", default="savenet/out.h5", nargs="?",
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

BATCH_SIZE = 1024


def predict_and_val(classifier, x, y, categories):
    """
    Runs the classifier on the input datasets and compares the results with the
    correct labels provided from y.

    One reported statistic is the frequency with which the target value lies
    within the 95% credible set.  The "shortest credible set" for a prediction
    is computed by sorting the probabilities from lowest to highest, resulting
    in a cumulative mass function (cmf).  If the cmf value associated the
    target bin is above 5% then the target is said to be in the
    "95% credible set".

    The "average credibility" results from accumulating the individual cmfs
    for each target and dividing by the number of times the target appears.
    If the target is always the most probable, then its cmf will always
    be 100% and so that will be the average credibility.  If there is 2-way
    confusion between targets (say 75/25 split), then the first will be 100%
    with freq 75% and somewhere between 0% and 50% the remaining 25% of
    the time, leading to an "average credibility" somewhat higher than 75%.
    Similarly for the 25% case.  So a highly biased and perhaps not very
    useful statistic.  At least order should be maintained.

    :param classifier: The trained classifier.
    :param x: List of x values to predict on.
    :param y: List of y values to predict on.
    :param categories: A list of all possible sas model names.
    :return: Two lists, the indices of the sas model and its proper name respectively.
    """
    encoder = OnehotEncoder(categories)
    yindex = encoder.index(y)
    prediction = classifier.predict(fix_dims(x), batch_size=BATCH_SIZE)
    error_freq = {name: 0 for name in categories}
    top5_freq = {name: 0 for name in categories}
    rank_avg = {name: 0 for name in categories}
    cred_68 = {name: 0 for name in categories}
    cred_95 = {name: 0 for name in categories}
    cred_avg = {name: 0 for name in categories}
    freq = {name: 0 for name in categories}
    num_rejected = {name: 0 for name in categories}
    errors = []
    for k, (prob, actual, actual_index) in enumerate(zip(prediction, y, yindex)):
        # Sort by increasing probability
        sort_index = prob.argsort(axis=-1)
        # Count frequency of targets for normalization.
        freq[actual] += 1
        # Generate cmf for each bin by sorting and accumulating.
        cdf = np.empty_like(prob)
        cdf[sort_index] = np.cumsum(prob[sort_index])
        # Track statistics for the frequency and average credible interval.
        cred_68[actual] += (cdf[actual_index] >= 1 - 0.68)
        cred_95[actual] += (cdf[actual_index] >= 1 - 0.95)
        cred_avg[actual] += cdf[actual_index]
        # Track presence in top-5 choices (arbitrary, but interesting from
        # a blind fitting perspective)
        top5_freq[actual] += (actual_index in sort_index[-5:])
        # Track rank by finding the position of actual in the sorted list.
        # Need to reverse the list since lowest to highest, and add one since
        # array indices are zero-origin. where()[0][0] because where returns
        # a tuple of dimensions with each dimension listing the position
        # along that dimension containing the item.
        rank_avg[actual] += np.where(sort_index[::-1] == actual_index)[0][0] + 1
        # Count the number below the 95% probability level
        num_rejected[actual] += np.sum(cdf < 0.05)
        # Check if prediction from mle matches actual. Note that the maximum
        # likelihood is at the end of the sorted list of probabilities.
        mle_index = sort_index[-1]
        if mle_index != actual_index:
            predicted = encoder.label([mle_index])[0]
            try:
                ratio = prob[actual_index] / prob[mle_index]
            except IndexError:
                print(actual, actual_index, mle_index, predicted, len(categories), prob)
                raise
            if len(errors) == 20:
                print("...")
            if len(errors) < 20:
                print(f"Predicted: {predicted}, Actual: {actual}, Index: {k}, Target/mle: {ratio:.2}.")
            error_freq[actual] += 1
            errors.append((k, predicted, actual, ratio))
    #print("Error rate")
    #print(columnize(f"{k}: {int(100*v/freq[k]+0.5)}%" for k, v in sorted(error_freq.items())))
    #print("Top 5 rate")
    #print(columnize(f"{k}: {int(100*v/freq[k]+0.5)}%" for k, v in sorted(top5_freq.items())))
    print("Average rank")
    print(columnize(f"{k}: {v/freq[k]:.1f}" for k, v in sorted(rank_avg.items())))
    print("Average number rejected at the 5% level")
    print(columnize(f"{k}: {v/freq[k]:.1f}" for k, v in sorted(num_rejected.items())))
    print("68% CI rate")
    print(columnize(f"{k}: {int(100*v/freq[k]+0.5)}%" for k, v in sorted(cred_68.items())))
    #print("95% CI rate")
    #print(columnize(f"{k}: {int(100*v/freq[k]+0.5)}%" for k, v in sorted(cred_95.items())))
    print("Average cred")
    print(columnize(f"{k}: {int(100*v/freq[k]+0.5)}%" for k, v in sorted(cred_avg.items())))

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
    prediction = classifier.predict(fix_dims(x), batch_size=BATCH_SIZE)
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
    prediction = classifier.predict(fix_dims(x), verbose=1, batch_size=BATCH_SIZE)
    n = len(categories)
    res = np.zeros((n, n))
    weight = np.zeros(n)
    for k, (prob, row) in enumerate(zip(prediction, index)):
        mle = nanargmax(prob)
        res[row][mle] += 1
        weight[row] += 1
    return res/weight[:, None] # TODO: divide row or column?

def rpredict(classifier, x, categories, verbose=0, output='label'):
    """
    Same as predict, but outputs names only.

    :param classifier: The trained classifier.
    :param x: List of x to predict on.
    :param categories: List of all model names.
    :param verbose: 0 or 1
    :return: List of predicted names.
    """
    batch_size = min(len(x), BATCH_SIZE)
    encoder = OnehotEncoder(categories)
    prediction = classifier.predict(
        fix_dims(x), batch_size=batch_size, verbose=verbose)
    index = [nanargmax(prob) for prob in prediction]
    if output == "label":
        label = encoder.label(index)
        return label.tolist()
    if output == "index":
        return index
    raise ValueError("output should be 'label' or 'index'")

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

def plot_tSNE(classifier, x, categories, order=None):
    """
    Displays a t-SNE cluster coloured by the classifier predicted labels.

    :param classifier: The trained classifier.
    :param x: List of x values to predict on.
    :param categories: List of all model names.
    :param order: order of categories in dendrogram.
    :return: The tSNE object that was plotted.
    """
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt

    # TODO: move to calculation function
    try:
        from umap import UMAP # umap is much faster but requies numpy
    except ImportError:
        warnings.info("umap-learn missing. Using scikit TSNE instead")
        UMAP = None
        from sklearn.manifold import TSNE
        #from tsnecuda import TSNE # tsnecuda claims to be much faster
    density=2000#len(x)/10
    xt = random.sample(x, density)
    if UMAP is not None:
        t = UMAP()
    else:
        t = TSNE(n_components=2, verbose=2, n_iter=3000, perplexity=55)
    print("mapping manifold")
    xt_reduced = t.fit_transform(xt)
    #return xt, xt_reduced
    x1, x2 = xt_reduced.T

    fig = plt.figure()
    # Label each point with the predicted model
    prediction = rpredict(classifier, xt, categories, output='index')
    # If the labels are ordered, give nearby labels similar colors
    if order is not None:
        rev = np.empty_like(order)
        rev[order] = np.arange(len(order))
        c = rev[prediction]
    else:
        c = prediction
    cm = plt.cm.get_cmap('viridis')
    h = plt.scatter(x1, x2, c=c, vmin=0, vmax=len(categories)-1, cmap=cm)
    # Associate colors with labels
    cbar = plt.colorbar(h, orientation='horizontal')
    cbar.set_ticks(np.arange(len(categories)))
    cbar.ax.set_xticklabels([categories[k] for k in order], rotation=90)
    plt.tight_layout()
    # Set the label for the mouse coordinates to the category of the nearest point
    picker = cKDTree(xt_reduced)
    lookup = lambda x, y: picker.query([[x,y]])[1][0]
    label = lambda index: categories[prediction[index]]
    plt.gca().format_coord = lambda x,y: f"= {label(lookup(x,y))} ="
    # save figure and display
    plt.savefig('tsne.png')
    plt.pause(0.1)

def plot_filters(model, x, categories,iq):
    """
    Displays filters

    :param classifier: The trained classifier.
    :param x: List of x values to predict on.
    :param categories: List of all model names.
    :return:
    """

    #adapted from https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    from tensorflow.keras.models import Model

    # summarize filter shapes
    flist=[]
    llist=[]
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        flist.append(filters)
        llist.append(layer)
        print(layer.name, filters.shape)

    print('oi')
    #filters, biases = model.layers[1].get_weights()
    filters=flist[0]
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1 #note that there are 128 filters for us in the first layer
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, i]
        #print('f',f.shape)
        # plot each channel separately # we only have one channel
        for j in range(1):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 1, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            #print('fj',f[:, j])
            plt.imshow(np.expand_dims(f[:, j],1), cmap='gray')
            ix += 1
    plt.pause(0.1)

    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=llist[0].output)
    model.summary()
    # load the image with the required shape
    #img = load_img('bird.jpg', target_size=(224, 224))

    # convert the image to an array
    #img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = x[0]
    print('img',img.shape)
    img = fix_dims(img)
    print('img',img.shape)
    #
    img = np.expand_dims(img, axis=0) #first make our 1d array to 2D
    print('img expand',img.shape)

    # prepare the image (e.g. scale pixel values for the vgg)
    #img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img, batch_size=BATCH_SIZE)
    print('features', feature_maps.shape)
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    fig = plt.figure()
    plt.plot(x[0])
    plt.savefig('sample_data.png')

    for _ in range(square):
        for _ in range(2*square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, 2*square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.plot(feature_maps[0, :, ix-1])
            #plt.imshow(np.expand_dims(feature_maps[0, :, ix-1],axis=0), cmap='gray')
            ix += 1
    # show the figure
    #plt.show()
    plt.savefig('sample_filters_layer1.png')
    plt.pause(0.1)

def plot_dendrogram(corr, categories, confusion_norm=False):
    """
    Displays a dendrogram clustering based on the confusion matrix.

    :param corr: The confusion matrix.
    :param categories: List of all model names.
    :param confusion_norm: Normalize confusion matrix by acceptance percentage.
    :return: The dendrogram object.
    """
    import matplotlib.pyplot as plt
    if confusion_norm:
        # Remove diagonal from the confusion matrix so we can see confusion
        # patterns for elements that are not correctly recognized.
        d = np.diag(corr)
        d = d + (d == 0)
        corr = corr/d
    corr[corr == 0.] = corr[corr > 0].min()/10
    #corr = np.log10(corr)
    fig = plt.figure()
    plt.subplot(212)
    z = linkage(corr, 'ward', optimal_ordering=True)
    h = dendrogram(z, leaf_rotation=90., leaf_font_size=8, labels=categories,
                   color_threshold=.5, get_leaves=True)
    order = np.asarray(h['leaves'], 'i')
    plt.gca().get_yaxis().set_visible(False)
    plt.subplot(211)
    # Reorder array rows and columns to match dendrogram order
    reorder = corr[order, :][:, order]
    plt.pcolor(reorder, cmap='RdBu')
    #plt.pcolor(np.log10(reorder))
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('dendogram.png')
    plt.pause(0.1)
    return order

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
        corr = confusion_matrix(classifier, log_iq, labels, categories)
        import matplotlib.pyplot as plt
        order = plot_dendrogram(corr, categories)
        plot_tSNE(classifier, log_iq, categories, order=order)
        plt.show()
    else:
        failures = predict_and_val(classifier, log_iq, labels, categories)
        plot_failures(failures, q, iq)

if __name__ == '__main__':
    main(sys.argv[1:])
