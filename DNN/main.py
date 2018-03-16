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
    np.random.seed(7)
   
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
    # train, train_labels, train_pid = data[:TEST_SIZE, :-1], data[:TEST_SIZE, -2], data[:TEST_SIZE, -1]
    test_data, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]
    print(train_data.shape)
    print(test_data.shape)

    #create the model.
    model = Sequential()
    model.add(Dense(units = 12, input_dim = 7, activation='sigmoid', use_bias=True))
    model.add(Dense(8, activation='sigmoid', use_bias=True))
    #ADD MORE LAYERS HERE AS DESIRED#
    model.add(Dense(1, activation='sigmoid'))
    
    #Create visualization and png model overview to verify structure.
    print(model.summary())
#    plot_model(model, to_file='titanic_model.png', show_shapes = True, show_layer_names = True)

    #compile the model.
    sgd = optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    model.fit(train_data, train_labels, epochs = 10, batch_size = 1, verbose = 2, shuffle = True)
  
    score = model.evaluate(test_data, test_labels, batch_size= 1)
    print(score)

    ####WHY are we predicting on the training data??
    predictions = model.predict(test_data) ####<<___TESTDATA NOT TRAIN?!?
    rounded = [round(x[0]) for x in predictions]
    #print(rounded)
    

if __name__ == '__main__':
    main()