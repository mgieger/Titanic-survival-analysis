import json
import math
import os
import sys
import time

from preprocessor import Preprocessor

from keras import optimizers
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_data(directory, file_name, results, file_type='.json'):
    '''
    Saves performance data to:
        directory/file_name: raw data
        results (dict | pd.dataframe):
        file_type (string): .json or .csv
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if file_type == '.json':
        with open(file_name + file_type, 'w') as f:
            json.dump(results, f)
    elif file_type == '.csv':
        results.to_csv(file_name + file_type, index=False)


def main():
    #Reading in the data, preprocessing it, and creating training and test sets.
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
        'survived'
    ]

    preprocessor = Preprocessor(full_file)
    data = preprocessor.get_matrix_scaled(columns)
    TRAIN_SIZE = math.ceil(data.shape[0]*.70)
    
    train_data, train_labels = data[:TRAIN_SIZE,:-1], data[:TRAIN_SIZE,-1:]
    test_data, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]

    ####Sequence of creating neural networks to analyze Titanic dataset. 
    # We tested various models of:
    # 1-4 layers
    # 4, 8, and 10 nodes per layer
    # With and without dropout; using dropout values of 0.2, 0.3, 0.4
    # For the sake of expediting the script run time only the best sequences 
    # we found for 1-3 layers were included. 
    # All 4 layer models would always predict the passenger to perish, 
    # so these models were not kept, as they were not useful.

    target_names = ['Class 0: Perished', 'Class 1: Survived']
    sgd = optimizers.SGD(lr=0.05, momentum=0.9)
    drop_rates = [0.0, 0.2, 0.3]
    node_nums = [10]

    for node in node_nums:
    #     for nodes in node_nums:
        ##Create the test input models.
        np.random.seed(79)
        
        model1 = Sequential()
        model1.add(Dense(units = 10, input_dim = 10, activation='relu', use_bias=True))
        model1.add(Dropout(0.3, seed=None))
        model1.add(Dense(units = node, input_dim = 10, activation='sigmoid', use_bias=True))
        model1.add(Dense(1, activation='sigmoid'))
        
        #Create visualization overview to verify model structure.
        print('MODEL SPECIFICATIONS:\n2 layers, \nLayer 1: 10 nodes Dropout: 0.3 ',
                'Layer 2: Hidden nodes: ', node, ' No dropout.')
        print(model1.summary())

        #Compile the model.
        model1.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
        
        #Train and Evaluate the model.
        model1.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
        score1 = model1.evaluate(test_data, test_labels, batch_size = 1, verbose=0)
        print(model1.metrics_names)
        print(score1, '\n')
        test_predictions1 = model1.predict_classes(test_data, batch_size = 1)

        print(classification_report(test_labels, test_predictions1, target_names=target_names))
        print(confusion_matrix(test_labels, test_predictions1)) 


    ##Create the 1 layer model.
    # np.random.seed(79)
    
    # model1 = Sequential()
    # model1.add(Dense(units = 10, input_dim = 10, activation='sigmoid', use_bias=True))
    # model1.add(Dropout(0.30, seed=None))
    # model1.add(Dense(1, activation='sigmoid'))
    
    # #Create visualization overview to verify model structure.
    # print(model1.summary())

    # #Compile the model.
    # model1.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # #Train and Evaluate the model.
    # model1.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
    # score1 = model1.evaluate(test_data, test_labels, batch_size = 1, verbose=0)
    # print(model1.metrics_names)
    # print(score1, '\n')
    # test_predictions1 = model1.predict_classes(test_data, batch_size = 1)

    # print(classification_report(test_labels, test_predictions1, target_names=target_names))
    # print(confusion_matrix(test_labels, test_predictions1))
    

    # ##Create 2 layer model.
    # np.random.seed(79)
    
    # model2 = Sequential()
    # model2.add(Dense(units = 10, input_dim = 10, activation='sigmoid', use_bias=True))
    # model2.add(Dropout(0.312))
    # model2.add(Dense(4, activation='sigmoid', use_bias=True))
    # model2.add(Dense(1, activation='sigmoid'))
    
    # #Create visualization overview to verify model structure.
    # print(model2.summary())

    # #Compile the model.
    # model2.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # #Train and Evaluate the model.
    # model2.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
    # score2 = model2.evaluate(test_data, test_labels, batch_size = 1, verbose=0)
    # print(model2.metrics_names)
    # print(score2, '\n')
    # test_predictions2 = model2.predict_classes(test_data, batch_size = 1)

    # print(classification_report(test_labels, test_predictions2, target_names=target_names))
    # print(confusion_matrix(test_labels, test_predictions2))


    ##Create 3 layer model.
    # np.random.seed(79)
    
    # model3 = Sequential()
    # model3.add(Dense(units = 10, input_dim = 10, activation='sigmoid', use_bias=True))
    # model3.add(Dropout(0.31075, seed=None))
    # model3.add(Dense(8, activation='sigmoid', use_bias=True))
    # model3.add(Dropout(0.2, seed=None))
    # model3.add(Dense(4, activation='sigmoid', use_bias=True))
    # model3.add(Dense(1, activation='sigmoid'))
    
    # #Create visualization overview to verify model structure.
    # print(model3.summary())

    # #Compile the model.
    # model3.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # #Train and Evaluate the model.
    # model3.fit(train_data, train_labels, epochs = 5, batch_size = 1, verbose = 2, shuffle = True)
    # score3 = model3.evaluate(test_data, test_labels, batch_size = 1, verbose=0)
    # print(model3.metrics_names)
    # print(score3, '\n')
    # test_predictions3 = model3.predict_classes(test_data, batch_size = 1)

    # print(classification_report(test_labels, test_predictions3, target_names=target_names))
    # print(confusion_matrix(test_labels, test_predictions3))


if __name__ == '__main__':
    main()