from __future__ import print_function

import logging

import keras
from hyperas import optim
from hyperas.distributions import uniform
from hyperopt import Trials, tpe
from keras import backend as K
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential


def data():
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sasnet import read_seq_1d
    a, b, c, d = read_seq_1d("../sasmodels/out2/", pattern="_eval_",
                             verbosity=True)
    encoder = LabelEncoder()
    encoder.fit(c)
    encoded = encoder.transform(c)
    yt = np_utils.to_categorical(encoded)
    xtrain, xtest, ytrain, ytest = train_test_split(b, yt, test_size=.25,
                                                    random_state=235)
    return xtrain, ytrain, xtest, ytest


def model(xtrain, ytrain, xtest, ytest):
    from hyperopt import STATUS_OK
    from numpy import asarray
    model = Sequential()
    model.add(Embedding(3000, 64, input_length=100))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.17676))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.20782))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.20582))
    model.add(Dense(55, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(asarray(xtrain), ytrain, batch_size={{choice([3,5,7,10,15])}}, epochs=20, verbose=2,
              validation_data=(asarray(xtest), ytest))
    score, acc = model.evaluate(asarray(xtest), ytest, verbose=1)
    m = model.to_yaml()
    print("Test Accuracy: ", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': m}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    K.clear_session()
    best_run, best_model = optim.minimize(model=model, data=data,
                                          algo=tpe.suggest, max_evals=8,
                                          trials=Trials())
    print(best_run)
    print(best_model)
