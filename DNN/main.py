import json
import math
import os
import time

from preprocessor import Preprocessor

import pandas as pd
import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#from keras.utils.vis_utils import plot_model

def main():
    np.random.seed(79)
   
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
    
    preprocessor = Preprocessor(full_file)
    data = preprocessor.get_matrix_scaled(['pclass', 'name', 'sex', 'age', 'ticket', 'cabin', 'fare', 'survived'])
    
    TRAIN_SIZE = math.ceil(data.shape[0]*.70)
    train_data, train_labels = data[:TRAIN_SIZE,:-1], data[:TRAIN_SIZE,-1:]
    test_data, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]
    #print(train_data.shape)
    #print(test_data.shape)

    #create the model.
    model = Sequential()
    model.add(Dense(units = 5, input_dim = 7, activation='sigmoid', use_bias=True))
    # model.add(Dense(6, activation='sigmoid', use_bias=True))
    # model.add(Dense(4, activation='sigmoid', use_bias=True))
    # model.add(Dense(6, activation='sigmoid', use_bias=True))
    # model.add(Dense(4, activation='sigmoid', use_bias=True))
    # model.add(Dense(6, activation='sigmoid', use_bias=True))
    model.add(Dense(1, activation='sigmoid'))
    
    #Create visualization and png model overview to verify structure.
    print(model.summary())

    #compile the model.
    sgd = optimizers.SGD(lr=0.05, momentum=0.9)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    #Train and Evaluate the model.
    model.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
    score = model.evaluate(test_data, test_labels, batch_size = 1, verbose=1)
    print(model.metrics_names)
    print(score)
    
    ##Attempt to loop and get training & test accuracy on graph together.

if __name__ == '__main__':
    main()