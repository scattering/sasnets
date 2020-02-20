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
import json

# Installed packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Embedding, \
    MaxPooling1D, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# SASNets packages
from . import sas_io
from .util.utils import inepath

# Define the argparser parameters
parser = argparse.ArgumentParser(
    description="Use neural nets to classify scattering data.")
parser.add_argument(
    "--train", type=str, default="train",
    help="Training data table.")
parser.add_argument(
    "--validation", type=str, default="30",
    help="Validation data table, or percent of training data.")
parser.add_argument(
    "--test", type=str, default="10",
    help="Test data table, or percent of training data.")
parser.add_argument(
    "--database", type=str, default=sas_io.DB_FILE,
    help="Path to the sqlite database file.")
parser.add_argument(
    "--steps", type=int, default=20000,
    help="Number of steps per epochs.")
parser.add_argument(
    "--epochs", type=int, default=50,
    help="Number of epochs.")
parser.add_argument(
    "--batch", type=int, default=5,
    help="Batch size.")
parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Control output verbosity")
parser.add_argument(
    "-s", "--save-path", default="./savenet/out",
    help="Path to save model weights and info to")

class OnehotEncoder:
    def __init__(self, categories):
        self.categories = sorted(categories)
        self.label_encoder = LabelEncoder().fit(self.categories)
        self.hotlen = len(categories)

    def __call__(self, y):
        return to_categorical(self.label_encoder.transform(y), self.hotlen)

    def index(self, y):
        return self.label_encoder.transform(y)

def fix_dims(*args):
    """
    Insert extra dimension on inputs for image channel.

    Keras seems to require an extra dimension on the inputs, which is
    either at the start or the end of the input item, depending on the
    backend in use.  Usual is at the end?
    """
    order = keras.backend.image_data_format()
    if order == 'channels_last':
        fixer = lambda x: np.asarray(x)[..., None]
    elif order == 'channels_first':
        fixer = lambda x: np.asarray(x)[:, None, ...]
    else:
        raise ValueError(f"unknown image data format {order}")
    if len(args) == 0:
        return fixer
    if len(args) == 1:
        return fixer(args[0])
    return (fixer(v) for v in args)


def save_output(save_path, model, encoder, history, seed, score):
    # Create output directory.
    basename = inepath(save_path)
    if basename is not None:
        model.save(basename + ".h5")
        out = {
            'categories': encoder.categories,
            'history': history.history,
            # seed is used to split training and evaluation data.
            # for sqlite it does nothing.
            'seed': seed,
            'score': score,
        }
        with open(basename + ".history.json", 'w') as fd:
            json.dump(out, fd, cls=sas_io.NpEncoder)
        plot_history(history, basename=basename)

def sql_net(opts):
    """
    A 1D convnet that uses a generator reading from a SQL database
    instead of loading all files into memory at once.
    """
    verbose = 1 if opts.verbose else 0
    db = sas_io.sql_connect(opts.database)
    counts = model_counts(db, tag=opts.train)
    encoder = OnehotEncoder(counts.keys())

    tb = TensorBoard(log_dir=os.path.dirname(basename), histogram_freq=1)
    es = EarlyStopping(min_delta=0.001, patience=15, verbose=verbose)

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
    if verbose > 0:
        print(model.summary())
    train_seq = sas_io.iread_sql(
        db, opts.train, encoder=encoder, batch_size=opts.batch)
    validation_seq = sas_io.iread_sql(
        db, opts.validation, encoder=encoder, batch_size=opts.batch)
    history = model.fit_generator(
        train_seq, opts.steps, epochs=opts.epochs,
        workers=1, verbose=verbose, validation_data=validation_seq,
        max_queue_size=1, callbacks=[tb, es])

    score = None
    if xval is not None and yval is not None:
        score = model.evaluate(xval, yval, verbose=verbose)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])

    save_output(
        save_path=opts.save_path,
        model=model,
        encoder=encoder,
        history=history,
        seed=None,
        score=score)
    logging.info("Complete.")


def oned_convnet(x, y, test=None, seed=235, verbose=False,
                 save_path=None):
    """
    Runs a 1D convolutional classification neural net on the input data x and y.

    :param x: List of training data x.
    :param y: List of corresponding categories for each vector in x.
    :param xtest: List of evaluation data.
    :param ytest: List of corresponding categories for each vector in x.
    :param seed: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :param verbose: Either true or false. Controls level of output.
    :param save_path: The path to save the model to. If it points to a directory, writes to a file named the current unix time. If it points to a file, the file is overwritten.
    :return: None.
    """
    verbose = 1 if verbose else 0

    batch_size = 5

    # 1-hot encoding.
    categories = sorted(set(y))
    encoder = OnehotEncoder(categories)

    # Split data into train and validation.
    xtrain, xval, ytrain, yval = train_test_split(
        x, encoder(y), test_size=.25, random_state=seed)

    # We need to poke an extra dimension into our input data for some reason.
    xtrain, xval = fix_dims(xtrain, xval)

    # Check that the validation data covers all the categories
    #if categories != sorted(set(ytrain)):
    #    raise ValueError("Training data is missing categories.")
    #if categories != sorted(set(yval)):
    #    raise ValueError("Test data is missing categories.")

    tb = TensorBoard(log_dir="./tensorboard", histogram_freq=1)
    #es = EarlyStopping(min_delta=0.005, patience=5, verbose=verbose)

    # Begin model definitions
    model = Sequential()
    #model.add(Embedding(4000, 128, input_length=x.shape[1]))
    model.add(InputLayer(input_shape=(x.shape[1],1)))
    model.add(Conv1D(128, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.17676))
    model.add(Conv1D(64, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.20782))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.20582))
    model.add(Dense(len(categories), activation='softmax'))
    loss = ('binary_crossentropy' if len(categories) == 2
            else 'categorical_crossentropy')
    model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Model Run
    if verbose > 0:
        print(model.summary())
    history = model.fit(
        xtrain, ytrain, batch_size=batch_size, epochs=2,
        verbose=verbose, validation_data=(xval, yval),
        #callbacks=[tb, es],
        callbacks=[tb],
        )

    # Check the results against the validation data.
    score = None
    if test is not None:
        if categories != sorted(set(test[1])):
            raise ValueError("Validation data has missing categories.")
        score = model.evaluate(test[0], encoder(test[1]), verbose=verbose)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])

    save_output(
        save_path=save_path,
        model=model,
        encoder=encoder,
        history=history,
        seed=seed,
        score=score)
    logging.info("Complete.")

def trad_nn(x, y, xtest=None, ytest=None, seed=235):
    """
    Runs a traditional MLP categorisation neural net on the input data x and y.

    :param x: List of training data x.
    :param y: List of corresponding categories for each vector in x.
    :param seed: Random seed. Defaults to 235 for reproducibility purposes, but should be set randomly in an actual run.
    :param xevl: Evaluation data for model.
    :param yevl: Evaluation data for model.
    :return: None
    """
    verbose = 1

    categories = sorted(set(y))
    encoder = OnehotEncoder(categories)
    xtrain, xval, ytrain, yval= train_test_split(
        x, encoder(y), test_size=.25, random_state=seed)
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
    history = model.fit(xtrain, ytrain, batch_size=10, epochs=10,
                        verbose=verbose, validation_data=(xval, yval))
    if xtest and ytest:
        score = model.evaluate(xtest, ytest, verbose=verbose)
        print('Test loss: ', score[0])
        print('Test accuracy:', score[1])

    #plot_history(history, basename=basename)


def plot_history(history, basename=None):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    with open(basename + ".svg", 'w') as fd:
        plt.savefig(fd, format='svg', bbox_inches='tight')

def read_data(opts):
    time_start = time.perf_counter()
    #q, iq, label, n = sas_io.read_1d_seq(opts.path, tag=opts.train, verbose=verbose)
    db = sas_io.sql_connect(opts.database)
    iq, label = sas_io.read_sql(db, opts.train)
    db.close()
    time_end = time.perf_counter()
    logging.info(f"File I/O Took {time_end-time_start} seconds for {len(label)} points of data.")
    return np.asarray(iq), label

def main(args):
    """
    Main method. Takes in arguments from command line and runs a model.

    :param args: Command line args.
    :return: None.
    """
    opts = parser.parse_args(args)
    data, label = read_data(opts)
    #print(data.shape)
    seed = random.randint(0, 2 ** 32 - 1)
    logging.info(f"Random seed for this iter is {seed}")
    oned_convnet(data, label, seed=seed,
                 verbose=opts.verbose, save_path=opts.save_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
