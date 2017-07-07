from __future__ import print_function

import argparse
import os
import sys

import keras
import bottleneck
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sasnet import read_1d

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
parser.add_argument("-c", "--classify", help="Classification mode.", action="store_true")
parser.add_argument("-p", "--pattern",
                    help="Pattern to match to files to open.")


def load_from(path):
    return keras.models.load_model(os.path.normpath(path))


def predict_and_val(model, x, y, names):
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
    prob = model.predict(x)
    for p in prob:
        pt = bottleneck.argpartition(p, num)[-num:]
        plist = pt[np.argsort(p[pt])]
        sys.stdout.write("The " + str(num) + " five most likely models and " +
                         "respective probabilities were: ")
        for i in reversed(plist):
            sys.stdout.write(str(names[i]) + ", " + str(p[i]) + " ")
        sys.stdout.write("\n")


def main(args):
    parsed = parser.parse_args(args)
    a, b, c, d, = read_1d(parsed.data_path, pattern=parsed.pattern,
                          verbosity=parsed.verbose)
    with open(os.path.join(os.path.dirname(parsed.data_path), "name"),
              'r') as fd:
        n = eval(fd.readline().strip())
    model = load_from(parsed.model_file)
    if parsed.classify:
        predict(model, b, n)
    else:
        ilist, nlist = predict_and_val(model, b, c, n)
        for i, n1 in zip(ilist, nlist):
            plt.style.use("classic")
            plt.plot(a[i], b[i])
            ax = plt.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.autoscale(enable=True)
            with open("/home/chwang/out/Img/" + n[n1] + str(i), 'w') as fd:
                plt.savefig(fd, format="svg", bbox_inches="tight")
                plt.clf()


if __name__ == '__main__':
    main(sys.argv[1:])
