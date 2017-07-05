from __future__ import print_function

import argparse
import sys

import keras
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sasnet import read_1d, plot

parser = argparse.ArgumentParser(
    description="Test a previously trained neural network.")
parser.add_argument("model_file", help="Path to model file from sasnet.py")
parser.add_argument("data_path", help="Path to load data from")
parser.add_argument("-v", "--verbose", help="Verbose output",
                    action="store_true")
parser.add_argument("-p", "--pattern",
                    help="Pattern to match to files to open.")


def load_from(path):
    return keras.models.load_model(path)


def predict_and_val(model, x, y, names):
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)

    prob = model.predict(x)
    err = [0]*(len(set(names)))
    unq = list(set(names))
    unq2 = list(set(y))
    ind = 0
    il = list()
    for p, e, in zip(prob, yt):
        p1 = p.argmax(axis=-1)
        e1 = e.argmax(axis=-1)
        #if p[0] > .65 or p[0] < .35:
            #print(#"Probabilities: " + str(p) + ", Predicted: " +
             #     str(names[p1]) + ", Actual: " + str(y[e1]) + ", Index: " + str(ind)
             #     + ".")
        if names[p1] != y[e1]:
            print("Predicted: " + str(names[p1]) + ", Actual: " + str(y[e1]) + ", Index: " + str(ind) + ".")
            err[e1] += 1
            il.append(ind)
        ind += 1
    print(err)
    return il


def main(args):
    parsed = parser.parse_args(args)
    a, b, c, d, = read_1d(parsed.data_path, pattern=parsed.pattern,
                          verbosity=parsed.verbose)
    #plot(a[989], b[989])
    #plot(a[126], b[126])
    n = None
    with open("../sasmodels/out/name", 'r') as fd:
        n = eval(fd.readline().strip())
    model = load_from(parsed.model_file)
    ilist = predict_and_val(model, b, c, n)
    for i in ilist:
        plot(a[i], b[i])


if __name__ == '__main__':
    main(sys.argv[1:])
