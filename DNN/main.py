import json
import math
import os
import time

import pandas as pd
import numpy as np
from sklearn import datasets

#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

from preprocessor import Preprocessor


def main():
    #np.random.seed(7)
   # input_matrix = np.random.uniform(low = 0.0, high = 1.0, size = (800, 10))
    #labels = np.random.randint(low = 0, high = 2, size = (800, 1))
    #new_labels = labels.astype(float)
    #dataset = np.concatenate((new_labels, input_matrix), axis = 1)
    #dataset = np.loadtxt("pima_indians.csv", delimiter = ',')

    full_file = '../titanic_full.csv'
    columns = [
        'pclass',
        'name',
        'sex',
        'age',
        'sibSp',
        'parch',
        'ticket',
        'fare',
        'cabin',
        'embarked',
        'survived',
	    'boat',
	    'body'
    ]
    # preprocessor_test = Preprocessor(train_file)
    
    preprocessor = Preprocessor(full_file)
    data = preprocessor.get_matrix(['pclass', 'name', 'sex', 'age', 'ticket', 'cabin', 'fare', 'survived'])
    
    ###TODO:VERIFY BELOW INDEXING IS PROPER. seeing should be df[:size, : -1]... ?
    TRAIN_SIZE = math.ceil(data.shape[0]*.70)
    train_data, train_labels = data[:TRAIN_SIZE,:-1], data[:TRAIN_SIZE,-1:]
    # train, train_labels, train_pid = data[:TEST_SIZE, :-1], data[:TEST_SIZE, -2], data[:TEST_SIZE, -1]
    test_data, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]
    print(train_data.shape)
    print(data.shape)
    print(test_data.shape)
    #print(train_labels_pid)
    # print(train_labels_pid)
    #
    print(train_labels)

######DATA EXISTS in train and test sets. STILL must normalize.


######
    #data_samples = dataset[:, 0:8]
    #data_labels = dataset[:, 8]
    #data_labels = dataset[:, 0]
    #data_samples = dataset[:, 1: ]

    #create the model.
    model = Sequential()
    model.add(Dense(units = 12, input_dim = 8, activation='sigmoid', use_bias = True))
    model.add(Dense(8, activation='sigmoid', use_bias = True))
    #ADD MORE LAYERS HERE AS DESIRED#
    model.add(Dense(1, activation='sigmoid'))

    #compile the model.
    #sgd = keras.optimizer.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs = 1, batch_size=5)
    ####WHY are we predicting on the training data??
    predictions = model.predict(train_labels)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)

if __name__ == '__main__':
    main()