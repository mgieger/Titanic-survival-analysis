import json
import time
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from preprocessor import Preprocessor

#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers


def main():
    np.random.seed(7)
   # input_matrix = np.random.uniform(low = 0.0, high = 1.0, size = (800, 10))
    #labels = np.random.randint(low = 0, high = 2, size = (800, 1))
    #new_labels = labels.astype(float)
    #dataset = np.concatenate((new_labels, input_matrix), axis = 1)
    dataset = np.loadtxt("pima_indians.csv", delimiter = ',')

    data_samples = dataset[:, 0:8]
    data_labels = dataset[:, 8]
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
    model.fit(data_samples, data_labels, epochs = 1, batch_size=5)
    predictions = model.predict(data_samples)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)

if __name__ == '__main__':
    main()