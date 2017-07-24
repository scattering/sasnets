"""
SASNets main file. Contains the main neural network code used for training
networks.

SASNets uses Keras and Tensorflow for the networks. You can change the backend
to Theano or CNTK through the Keras config file.
"""
from __future__ import print_function

import argparse
import logging
import os
import sys
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import psycopg2 as psql
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv1D, Dropout, Flatten, Dense, \
    Embedding, MaxPooling1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from psycopg2 import sql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sas_io import sql_dat_gen

parser = argparse.ArgumentParser(
    description="Use neural nets to classify scattering data.")
parser.add_argument("path", help="Relative or absolute path to a folder "
                                 "containing data files")
parser.add_argument("-v", "--verbose", help="Control output verbosity",
                    action="store_true")
parser.add_argument("-s", "--save-path",
                    help="Path to save model weights and info to")


DEC2FLOAT = psql.extensions.new_type(
     psql._psycopg.DECIMAL.values,
     'DEC2FLOAT',
     lambda value, curs: float(value) if value is not None else None)
psql.extensions.register_type(DEC2FLOAT, None)


def sql_net(dn, mn, verbosity=False, save_path=None, encoder=None, xval=None,
            yval=None):
    """
    A oned convnet that uses a generator reading from a Postgres database
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
    if verbosity:
        v = 1
    else:
        v = 0
    base = None
    sp = os.path.normpath(save_path)
    if sp is not None:
        if os.path.isdir(sp):
            base = os.path.join(sp, str(time.time()))
        else:
            base = sp
        if not os.path.exists(os.path.dirname(sp)):
            os.makedirs(os.path.dirname(sp))
    tb = TensorBoard(log_dir=os.path.dirname(base), histogram_freq=1)
    es = EarlyStopping(min_delta=0.001, patience=15, verbose=v)

    # Begin model definitions
    model = Sequential()
    model.add(Conv1D(256, kernel_size=8, activation='relu', input_shape=[267,2]))
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
    if v:
        print(model.summary())
    history = model.fit_generator(sql_dat_gen(dn, mn, encoder=encoder), 20000,
                                  epochs=60, workers=1, verbose=v,
                                  validation_data=(xval, yval),
                                  max_queue_size=1, callbacks=[tb, es])
    score = None

    # Model Save
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not (base is None):
        with open(base + ".history", 'w') as fd:
            fd.write(str(history.history) + "\n")
            if score is not None:
                fd.write(str(score) + "\n")
        model.save(base + ".h5")
        with open(base + ".svg", 'w') as fd:
            plt.savefig(fd, format='svg', bbox_inches='tight')
    if xval is not None and yval is not None:
        score = model.evaluate(xval, yval, verbose=v)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])
    logging.info("Complete.")


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
    if verbosity:
        v = 1
    else:
        v = 0
    base = None
    sp = os.path.normpath(save_path)
    if sp is not None:
        if os.path.isdir(sp):
            base = os.path.join(sp, str(time.time()))
        else:
            base = sp
        if not os.path.exists(os.path.dirname(sp)):
            os.makedirs(os.path.dirname(sp))
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded = encoder.transform(y)
    yt = to_categorical(encoded)
    xval, xtest, yval, ytest = train_test_split(x, yt, test_size=.25,
                                                random_state=random_s)
    if not len(set(y)) == len(set(yevl)):
        raise ValueError("Differing number of categories in train (" + str(
            len(set(y))) + ") and test (" + str(len(set(yevl))) + ") data.")
    tb = TensorBoard(log_dir=os.path.dirname(base), histogram_freq=1)
    es = EarlyStopping(min_delta=0.005, patience=5, verbose=v)

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
    # plot_model(model, to_file="model.png")

    # Model Run
    if v:
        print(model.summary())
    history = model.fit(xval, yval, batch_size=5, epochs=50, verbose=v,
                        validation_data=(xtest, ytest), callbacks=[tb, es])
    score = None
    if not (xevl is None) and not (yevl is None):
        e2 = LabelEncoder()
        e2.fit(yevl)
        yv = to_categorical(e2.transform(yevl))
        score = model.evaluate(xevl, yv, verbose=v)
        print('\nTest loss: ', score[0])
        print('Test accuracy:', score[1])

    # Model Save
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not (base is None):
        with open(base + ".history", 'w') as fd:
            fd.write(str(list(set(y))) + "\n")
            fd.write(str(history.history) + "\n")
            if score is not None:
                fd.write(str(score) + "\n")
            fd.write("Seed " + str(random_s))
        model.save(base + ".h5")
        with open(base + ".svg", 'w') as fd:
            plt.savefig(fd, format='svg', bbox_inches='tight')
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
    model.fit(xval, yval, batch_size=10, epochs=10, verbose=1,
                        validation_data=(xtest, ytest))
    if xevl and yevl:
        score = model.evaluate(xtest, ytest, verbose=0)
        print('Test loss: ', score[0])
        print('Test accuracy:', score[1])


def main(args):
    """
    Main method. Takes in arguments from command line and runs a model.

    :param args: Command line args.
    :return: None.
    """
    parsed = parser.parse_args(args)
    # time_start = time.clock()
    # a, b, c, d, e, n = read_seq_1d(parsed.path, pattern='_all_',
    #                               verbosity=parsed.verbose)
    # gc.collect()
    # at, bt, ct, dt, et, nt = read_seq_1d(parsed.path, pattern='_eval_',
    #                                     verbosity=parsed.verbose)
    # time_end = time.clock() - time_start
    # logging.info("File I/O Took " + str(time_end) + " seconds for " + str(n) +
    #             " points of data.")
    # r = random.randint(0, 2 ** 32 - 1)
    # logging.warn("Random seed for this iter is " + str(r))
    # oned_convnet(np.asarray(b), c, np.asarray(bt), ct, random_s=r,
    #             verbosity=parsed.verbose, save_path=parsed.save_path)
    conn = psql.connect("dbname=sas_data user=sasnets host=127.0.0.1")
    # conn.set_session(readonly=True)
    with conn:
        with conn.cursor() as c:
            c.execute("SELECT model FROM new_train_data;")
            xt = set(c.fetchall())
            y = [i[0] for i in xt]
            #c.execute("SELECT model FROM new_eval_data;")
            #xt = set(c.fetchall())
            #y2 = set([i[0] for i in xt])
            #z=['adsorbed_layer', 'barbell', 'bcc_paracrystal',
             #'be_polyelectrolyte', 'binary_hard_sphere', 'broad_peak',
             #'capped_cylinder', 'core_multi_shell', 'core_shell_bicelle',
             #'core_shell_bicelle_elliptical', 'core_shell_cylinder',
             #'core_shell_ellipsoid', 'core_shell_parallelepiped',
             #'core_shell_sphere', 'correlation_length', 'cylinder', 'dab',
             #'ellipsoid', 'elliptical_cylinder', 'flexible_cylinder',
             #'flexible_cylinder_elliptical', 'fractal', 'fractal_core_shell',
             #'fuzzy_sphere', 'gauss_lorentz_gel', 'gaussian_peak', 'gel_fit',
             #'guinier', 'guinier_porod', 'hardsphere', 'hayter_msa',
             #'hollow_cylinder', 'hollow_rectangular_prism',
             #'hollow_rectangular_prism_thin_walls', 'lamellar', 'lamellar_hg',
             #'lamellar_hg_stack_caille', 'lamellar_stack_caille',
             #'lamellar_stack_paracrystal', 'line', 'linear_pearls', 'lorentz',
             #'mass_fractal', 'mass_surface_fractal', 'mono_gauss_coil',
             #'multilayer_vesicle', 'onion', 'parallelepiped', 'peak_lorentz',
             #'pearl_necklace', 'poly_gauss_coil', 'polymer_excl_volume',
             #'polymer_micelle', 'porod', 'power_law', 'pringle', 'raspberry',
             #'rectangular_prism', 'rpa', 'sphere', 'spherical_sld', 'spinodal',
             #'squarewell', 'stacked_disks', 'star_polymer', 'stickyhardsphere',
             #'surface_fractal', 'teubner_strey', 'triaxial_ellipsoid',
            # 'two_lorentzian', 'two_power_law', 'unified_power_Rg', 'vesicle']
            encoder = LabelEncoder()
            encoder.fit(y)
            #for m in z:
             #   if(not y.__contains__(m)):
              #      print(m)
            c.execute("CREATE EXTENSION IF NOT EXISTS tsm_system_rows")
            #c.execute(
            #        sql.SQL("SELECT * FROM {}").format(
            #            sql.Identifier("train_metadata")))
            #x = np.asarray(c.fetchall())
            # q = x[0][1]
            # dq = x[0][2]
            #diq = x[0][3]
            c.execute(sql.SQL(
                "SELECT * FROM {} TABLESAMPLE SYSTEM_ROWS(10000)").format(
                            sql.Identifier("new_eval_data")))
            x = np.asarray(c.fetchall())
            iq_list = x[:, 1]
            diq = x[:,2]
            y_list = x[:, 3]
            encoded = encoder.transform(y_list)
            yt = np.asarray(to_categorical(encoded, 64))
            q_list = np.asarray([np.transpose([np.log10(iq), np.log10(dq)]) for iq, dq in
                                 zip(iq_list, diq)])

    sql_net("new_train_data", "new_train_metadata",
            verbosity=parsed.verbose, save_path=parsed.save_path,
            encoder=encoder, xval=q_list, yval=yt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
