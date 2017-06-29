from __future__ import print_function

import argparse
import ast
import os
import random
import re
import sys
import time

import keras
import numpy as np
import ruamel.yaml as yaml  # using ruamel for better input processing.
from keras.layers import Conv1D, Dropout, Flatten, Dense, \
    Embedding, MaxPooling1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Use neural nets to classify scattering data.")
parser.add_argument("path",
                    help="Relative or absolute path to a folder containing data files")
parser.add_argument("-v", "--verbose", help="Control output verbosity",
                    action="store_true")
parser.add_argument("-s", "--save-path",
                    help="Path to save model weights and info to")


def read_1d(path, pattern='_all_', typef='aggr', verbosity=False):
    """
    Reads all files in the folder path. Opens the files whose names match the
    regex pattern. Returns lists of Q, I(Q), and ID. Path can be a
    relative or absolute path.

    typef is one of 'json' or 'aggr'. JSON mode reads in all and only json files
    in the folder specified by path. aggr mode reads in aggregated data files.
    See sasmodels/generate_sets.py for more about these formats.

    Assumes files contain 1D data.

    :type path: String
    :type pattern: String
    :type typef: String
    """
    q_list, iq_list, y_list = (list() for i in range(3))
    pattern = re.compile(pattern)
    n = 0
    nlines = None
    if typef == 'json':
        for fn in os.listdir(path):
            if pattern.search(fn):  # Only open JSON files
                with open(path + fn, 'r') as fd:
                    n += 1
                    data_d = yaml.safe_load(fd)
                    q_list.append(data_d['data']['Q'])
                    iq_list.append(data_d["data"]["I(Q)"])
                    y_list.append(data_d["model"])
                if (n % 100 == 0) and verbosity:
                    print("Read " + str(n) + " files.")
    if typef == 'aggr':
        nlines = 0
        for fn in os.listdir(path):
            if pattern.search(fn):
                with open(path + fn, 'r') as fd:
                    templ = ast.literal_eval(fd.readline().strip())
                    y_list += [templ[0] for i in range(templ[1])]
                    t2 = ast.literal_eval(fd.readline().strip())
                    q_list += [t2 for i in range(templ[1])]
                    iq_list += ast.literal_eval(fd.readline().strip())
                    nlines += templ[1]
                if (n % 1000 == 0) and verbosity:
                    print("Read " + str(nlines) + " lines.")
    else:
        print("Error: the type " + typef + " was not recognised. Valid types "
                                           "are 'aggr' and 'json'.")
    return q_list, iq_list, y_list, nlines


def plot(q, i_q):
    """
    Method to plot Q vs I(Q) data for testing and verification purposes.

    :param q: List of Q values
    :param i_q: List of I values
    :return: None
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
        pass
    if not plt:
        raise ImportError("Matplotlib isn't installed, can't plot.")
    else:
        plt.style.use("classic")
        plt.plot(q, i_q)
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.autoscale(enable=True)
        plt.show()


def oned_convnet(x, y, xevl=None, yevl=None, random_s=235, verbosity=False,
                 save_path=None):
    """
    Runs a 1D convolutional classification neural net on the input data x and y.

    :param x: List of training data x
    :param y: List of corresponding categories for each vector in x
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :param verbosity: Either true or false. Controls level of output.
    :param save_path: The path to save the model to. If it points to a directory, writes to a file named the current unix time. If it points to a file, the file is overwritten.
    :return: None
    """
    if verbosity:
        v = 1
    else:
        v = 0

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)
    xval, xtest, yval, ytest = train_test_split(x, yt, test_size=.25,
                                                random_state=random_s)

    # Begin model definitions
    model = Sequential()
    model.add(Embedding(4000, 64, input_length=100))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Dropout(.25))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(len(set(y)), activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    # Model Run
    if v: print(model.summary())
    history = model.fit(xval, yval, batch_size=6, epochs=50, verbose=v,
                        validation_data=(xtest, ytest))
    score = None
    if not (xevl is None) and not (yevl is None):
        e2 = LabelEncoder().fit(yevl)
        yv = e2.transform(yevl)
        score = model.evaluate(xevl, yv, verbose=v)
        print('Test loss: ', score[0])
        print('Test accuracy:', score[1])

    # Model Save
    if save_path:
        yaml_model = model.to_yaml()
        if save_path[-1:] == "/":  # Assumes *nix-style file paths
            base = save_path + time.time()
        else:
            base = save_path
        with open(base + ".model", 'w') as fd:
            fd.write(yaml_model)
        with open(base + ".history", 'w') as fd:
            fd.write(model.history + "\n")
            if not (score is None): fd.write(score + "\n")
            fd.write("Seed " + str(random_s))
        model.save_weights(base + ".h5")
    print("Complete.")


def trad_nn(x, y, xevl=None, yevl=None, random_s=235):
    """
    Runs a traditional MLP categorisation neural net on the input data x and y.

    :param x: List of training data x
    :param y: List of corresponding categories for each vector in x
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :return: None
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)
    xval, xtest, yval, ytest = train_test_split(x, yt, test_size=.25,
                                                random_state=random_s)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x.shape[1]))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y)), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam",
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(xval, yval, batch_size=10, epochs=10, verbose=1,
                        validation_data=(xtest, ytest))
    if xevl and yevl:
        score = model.evaluate(xtest, ytest, verbose=0)
        print('Test loss: ', score[0])
        print('Test accuracy:', score[1])


def main(args):
    parsed = parser.parse_args(args)
    time_start = time.clock()
    a, b, c, n = read_1d(parsed.path, pattern='_all_')
    at, bt, ct, dt = read_1d(parsed.path, pattern='_eval_')
    time_end = time.clock() - time_start
    if parsed.verbose: print(
        "File I/O Took " + str(time_end) + " seconds for " + str(
            n) + " lines of data.")
    r = random.randint(0, 2 ** 32 - 1)
    print("Random seed for this iter is " + str(r))
    oned_convnet(np.asarray(b), c, np.asarray(bt), ct, random_s=r)


if __name__ == '__main__':
    main(sys.argv[1:])
