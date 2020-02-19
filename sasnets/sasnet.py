"""
SASNets main file. Contains the main neural network code used for training
networks.

SASNets uses Keras and Tensorflow for the networks. You can change the backend
to Theano or CNTK through the Keras config file.
"""
# System imports
import argparse
import logging
import os
import sys
import time
import random

# Installed packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv1D, Dropout, Flatten, Dense, Embedding, \
    MaxPooling1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

# SASNets packages
from . import sas_io
from .util.utils import inepath

# Define the argparser parameters
parser = argparse.ArgumentParser(
    description="Use neural nets to classify scattering data.")
parser.add_argument("path", help="Relative or absolute path to a folder "
                                 "containing data files")
parser.add_argument("-v", "--verbose", help="Control output verbosity",
                    action="store_true")
parser.add_argument("-s", "--save-path",
                    help="Path to save model weights and info to")


def sql_net(datatable, metatable, verbosity=False, save_path=None,
            label_encoder=None, xval=None, yval=None):
    """
    A 1D convnet that uses a generator reading from a SQL database
    instead of loading all files into memory at once.

    :param dn: The data table name.
    :param mn: The metadata table name.
    :param verbosity: The verbosity level.
    :param save_path: The path to save model output and weights to.
    :param encoder: A LabelEncoder for encoding labels.
    :param xval: A list of validation data, x version.
    :param yval: A list of validation data, label version.
    :return: None
    """
    vlevel = 1 if verbosity else 0
    basename = inepath(save_path)

    tb = TensorBoard(log_dir=os.path.dirname(basename), histogram_freq=1)
    es = EarlyStopping(min_delta=0.001, patience=15, verbose=vlevel)

    # Begin model definitions
    model = Sequential()
    model.add(
        Conv1D(256, kernel_size=8, activation='relu', input_shape=[267, 2]))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.17676))
    model.add(Conv1D(256, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(.20782))
    model.add(Flatten())
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(.20582))
    model.add(Dense(64, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Model Run
    if vlevel > 0:
        print(model.summary())
    db = sas_io.sql_connect()
    seq = sas_io.iread_sql(db, datatable, metatable, encoder=label_encoder, batch_size=5)
    history = model.fit_generator(seq, 20000,
                                  epochs=60, workers=1, verbose=vlevel,
                                  validation_data=(xval, yval),
                                  max_queue_size=1, callbacks=[tb, es])

    score = None
    if xval is not None and yval is not None:
        score = model.evaluate(xval, yval, verbose=vlevel)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])

    # Model Save
    if basename is not None:
        model.save(basename + ".h5")
        with open(basename + ".history", 'w') as fd:
            fd.write(str(history.history) + "\n")
            if score is not None:
                fd.write(str(score) + "\n")
        plot_history(history, basename=basename)
    logging.info("Complete.")


def oned_convnet(x, y, xevl=None, yevl=None, random_s=235, verbosity=False,
                 save_path=None):
    """
    Runs a 1D convolutional classification neural net on the input data x and y.

    :param x: List of training data x.
    :param y: List of corresponding categories for each vector in x.
    :param xevl: List of evaluation data.
    :param yevl: List of corresponding categories for each vector in x.
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :param verbosity: Either true or false. Controls level of output.
    :param save_path: The path to save the model to. If it points to a directory, writes to a file named the current unix time. If it points to a file, the file is overwritten.
    :return: None.
    """
    vlevel = 1 if verbosity else 0
    basename = inepath(save_path)

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)
    xval, xtest, yval, ytest = train_test_split(x, yt, test_size=.25,
                                                random_state=random_s)
    #if not len(set(y)) == len(set(yevl)):
    #    raise ValueError("Differing number of categories in train (" + str(
    #        len(set(y))) + ") and test (" + str(len(set(yevl))) + ") data.")

    tb = TensorBoard(log_dir=os.path.dirname(basename), histogram_freq=1)
    es = EarlyStopping(min_delta=0.005, patience=5, verbose=vlevel)

    # Begin model definitions
    model = Sequential()
    model.add(Embedding(4000, 128, input_length=xval.shape[1]))
    model.add(Conv1D(128, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.17676))
    model.add(Conv1D(64, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.20782))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.20582))
    model.add(Dense(len(set(y)), activation='softmax'))
    if len(set(y)) == 2:
        l = 'binary_crossentropy'
    else:
        l = 'categorical_crossentropy'
    model.compile(loss=l, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Model Run
    if vlevel > 0:
        print(model.summary())
    history = model.fit(xval, yval, batch_size=5, epochs=50, verbose=vlevel,
                        validation_data=(xtest, ytest), callbacks=[tb, es])
    score = None
    if xevl is not None and yevl is not None:
        e2 = LabelEncoder()
        e2.fit(yevl)
        yv = to_categorical(e2.transform(yevl))
        score = model.evaluate(xevl, yv, verbose=vlevel)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])

    # Model Save
    if not (basename is None):
        model.save(basename + ".h5")
        with open(basename + ".history", 'w') as fd:
            fd.write(str(list(set(y))) + "\n")
            fd.write(str(history.history) + "\n")
            if score is not None:
                fd.write(str(score) + "\n")
            fd.write("Seed " + str(random_s))
        plot_history(history, basename=basename)
    logging.info("Complete.")

def trad_nn(x, y, xevl=None, yevl=None, random_s=235):
    """
    Runs a traditional MLP categorisation neural net on the input data x and y.

    :param x: List of training data x.
    :param y: List of corresponding categories for each vector in x.
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :param xevl: Evaluation data for model.
    :param yevl: Evaluation data for model.
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

    #plot_history(history, basename=basename)


def plot_history(history, basename=None):
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    with open(basename + ".svg", 'w') as fd:
        plt.savefig(fd, format='svg', bbox_inches='tight')

def read_data(path, pattern='_all_', verbosity=True):
    time_start = time.perf_counter()
    q, iq, label, n = sas_io.read_seq_1d(path, pattern=pattern, verbosity=verbosity)
    time_end = time.perf_counter() - time_start
    logging.info("File I/O Took " + str(time_end) + " seconds for " + str(n) +
                 " points of data.")
    return np.asarray(iq), label

def main(args):
    """
    Main method. Takes in arguments from command line and runs a model.

    :param args: Command line args.
    :return: None.
    """

    parsed = parser.parse_args(args)
    data, label = read_data(parsed.path, verbosity=parsed.verbose)
    seed = random.randint(0, 2 ** 32 - 1)
    logging.info(f"Random seed for this iter is {seed}")
    oned_convnet(data, label, None, None, random_s=seed,
                 verbosity=parsed.verbose, save_path=parsed.save_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
