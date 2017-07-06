from __future__ import print_function

import keras
from hyperopt import Trials, STATUS_OK, rand, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def data():
    from keras.utils import np_utils
    from numpy import asarray
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sasnet import read_1d
    a, b, c, d = read_1d("../sasmodels/out/", verbosity=True)
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
    model.add(Embedding({{choice([2800, 3000, 3200])}}, 64, input_length=100))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size={{choice([4,6,8])}}))
    model.add(Dropout(.25))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size={{choice([2,3,4])}}))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.25))
    model.add(Dense(55, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(asarray(xtrain), ytrain, batch_size=5, nb_epoch=25, show_accuracy=True, verbose=2, validation_data=(asarray(xtest), ytest))
    score, acc = model.evaluate(asarray(xtest), ytest, show_accuracy=True, verbose=0)
    print("Test Accuracy: ", acc)
    return {'loss': -acc,'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
   # parsed = parser.parse_args(sys.argv[1:])
    best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())
    xt, yt, xe, ye = data()
    best_model.save("./out/full5/full5.h5")
    print(best_run)
    print("Evaluation of best model: ", best_model.evaluate(xt, yt))