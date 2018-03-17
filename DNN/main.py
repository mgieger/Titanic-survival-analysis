import json
import math
import os
import time

from preprocessor import Preprocessor

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
import numpy as np
#from numpy import *
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
#from keras.utils.vis_utils import plot_model
#from sklearn import datasets

def main():
    np.random.seed(79)
   
    full_file = '../titanic_full.csv'
    columns = [
        'pclass',
        'name',
        'sex',
        'age',
        'sibsp',
        'parch',
        'ticket',
        'fare',
        'cabin',
        'embarked',
	    #'boat',
	    #'body',
        'survived'
    ]
    preprocessor = Preprocessor(full_file)
    data = preprocessor.get_matrix_scaled(columns)
    data.shuffle()
    TRAIN_SIZE = math.ceil(data.shape[0]*.70)
    train_data, train_labels = data[:TRAIN_SIZE,:-1], data[:TRAIN_SIZE,-1:]
    test_data, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]

    #create the model.
    model = Sequential()
    model.add(Dense(units = 10, input_dim = 10, activation='sigmoid', use_bias=True))
    model.add(Dropout(0.312, seed=None))
    model.add(Dense(8, activation='sigmoid', use_bias=True))
    model.add(Dropout(0.20, seed=None))
    #model.add(Dense(6, activation='sigmoid', use_bias=True))
    #model.add(Dropout(0.25))
    model.add(Dense(4, activation='sigmoid', use_bias=True))
    #model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    
    #Create visualization and png model overview to verify structure.
    print(model.summary())

    #compile the model.
    sgd = optimizers.SGD(lr=0.05, momentum=0.9)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    #Train and Evaluate the model.
    model.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
    score = model.evaluate(test_data, test_labels, batch_size = 1, verbose=0)
    print(model.metrics_names)
    print(score, '\n')
    test_predictions = model.predict_classes(test_data, batch_size = 1)
    
    #Convert test labels into binary class matrices for confusion matrix
    #test_label_matrix = np.utils.to_categorically(test_labels, 2)

    target_names = ['Class 0: Perished', 'Class 1: Survived']
    print(classification_report(test_labels, test_predictions, target_names=target_names))
    print(confusion_matrix(test_labels, test_predictions))
    ##Attempt to loop and get training & test accuracy on graph together.

if __name__ == '__main__':
    main()