import os
import re
import sys

import keras
import ruamel.yaml as yaml  # using rumel for better input processing.
from keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import text
from sklearn.model_selection import train_test_split


def read_in_1d(path):
    """
    Reads all files in the folder path. Opens the JSON files and adds the data
    to the lists. Returns lists of Q, I(Q), and ID. Path can be a
    relative or absolute path.
    Assumes files contain 1D data.
    :type path: String
    """
    q_list, iq_list, y_list = (list() for i in range(3))
    pattern = re.compile(".json$")
    for fn in os.listdir(path):
        if (pattern.search(fn)):  # Only open JSON files
            with open(path + fn, 'r') as fd:
                data_d = yaml.safe_load(fd)
                q_list.append(data_d['data']['Q'])
                iq_list.append(data_d["data"]["I(Q)"])
                y_list.append(data_d["model"])
    return q_list, iq_list, y_list


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


def oned_convnet(x, y, random_s=235):
    """
    Runs a 1D convolutional classification neural net on the input data x and y.
    :param x: List of training data x
    :param y: List of corresponding categories for each vector in x
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes,
    but should be set randomly in an actual run.
    :return: None
    """
    xval, yval, xtest, ytest = \
        train_test_split(x, keras.preprocessing.text.one_hot(' '.join(y), len(set(y)) + 1),
                         test_size=.25, random_state=random_s)
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(len(set(y)), activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(x, y, batch_size=32, epochs=50, verbose=1,
              validation_data=(xval, yval))
    score = model.evaluate(xtest, ytest, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy:', score[1])


def trad_nn(x, y, random_s=235):
    """
    Runs a traditional MLP categorisation neural net on the input data x and y.
    :param x: List of training data x
    :param y: List of corresponding categories for each vector in x
    :param random_s: Random seed. Defaults to 235 for reproducibility purposes,
    but should be set randomly in an actual run.
    :return: None
    """
    yt = keras.preprocessing.text.one_hot(' '.join(y), len(set(y)) + 1)
    xval, yval, xtest, ytest = train_test_split(x, yt, test_size=.25,
                                                random_state=random_s)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(100,)))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(len(set(y)), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(x, y, batch_size=32, epochs=50, verbose=1,
              validation_data=(xval, yval))
    score = model.evaluate(xtest, ytest, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy:', score[1])


def main(args):
    a, b, c = read_in_1d(args[0])
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    # prep = ' '.join(c)
    # print(keras.preprocessing.text.one_hot(prep, len(set(c)) + 1))
    # Q = np.asfarray(data["Q"])
    # IQ = np.asfarray(data["I(Q)"])
    # newIQ = ''.join(IQ)
    # ntypes = keras.preprocessing.text.text_to_word_sequence(newIQ)
    # print(ntypes)
    # scaled_IQ = np.log10(np.divide(IQ, np.nanmin(IQ)))
    # scaled_Q = np.log10(Q)


if __name__ == '__main__':
    main(sys.argv[1:])
