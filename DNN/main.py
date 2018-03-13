import json
import time
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from preprocessor import Preprocessor

#import keras
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)


def main():
    #files
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    model_file = directory + '/model-set-{}'
    results_file = directory + '/results-set-{}'
    train_file = './data/titanic_train.csv'

    feature_set = ['pclass', 'sex', 'age', 'fare', 'sibSp', 'parch', 'cabin', 'embarked']

    preprocessor = Preprocessor(train_file)
    dataset = preprocessor.get_matrix()
    # for set_num, features in enumerate(feature_set):
    #     dataset = preprocessor.get_matrix(features + ['Survived'])
    #     data, labels = dataset[:,:-1], dataset[:,-1]
    #     svc = SVC(cache_size=cache_size, max_iter=max_iterations)
    #     clf = GridSearchCV(svc, param_grid=parameters, cv=cv_size, return_train_score=True)
    #     results = clf.fit(data, labels)

    data = 


if __name__ == 'main':
    main()