import json
import time
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from preprocessor import Preprocessor
from graph import plot_svm

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

def get_predictions(model, features, test_file, pca=None):
    '''
    Returns:
        pd.dataframe: predictions with cols ['PassengerId', 'Survived']
    '''
    preprocessor = Preprocessor(test_file)
    data = preprocessor.get_matrix_scaled(features)
    
    if pca:
        data = pca.transform(data)
    
    predictions = model.predict(data)
    df = preprocessor.get_dataframe()
    return pd.DataFrame(data={'PassengerId': df['PassengerId'], 'Survived': predictions.astype(int).tolist()})

def main():
    #files
    directory = './save/{}'.format(time.ctime().replace(' ', '-'))
    model_file = directory + '/model-set-{}'
    results_file = directory + '/results-set-{}'
    predictions_file = directory + '/predictions-set-{}'
    graph_file = directory + '/{}.png'
    train_file = './data/titanic_train.csv'
    test_file = './data/titanic_test.csv'    

    #data preprocessing, set pca = None to turn off PCA
    save_graph = True
    pca_n_components = 2
    preprocessor = Preprocessor(train_file)
    pca = PCA(n_components=pca_n_components)

    # grid search for hyper params
    # set if data not scaled: 
    max_iterations = 1000000
    cv_size = 6
    cache_size = 5000
    feature_set = [
         ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare'],
    ]

    
    parameters = {
        'kernel': ['linear'],
        'C': [0.1, 1.0, 15.0],
        'gamma': [1e-6, 1e-4, 1.0, 5.0, 10.0],
       # 'degree': [1, 3, 6, 9]
    }

    for set_num, features in enumerate(feature_set):
        data, labels = preprocessor.get_matrix_scaled(features), preprocessor.get_labels()

        if pca:
            data = pca.fit_transform(data)
        
        svc = SVC(cache_size=cache_size)
        clf = GridSearchCV(svc, param_grid=parameters, scoring="accuracy", cv=cv_size, refit=True, return_train_score=True)
        results = clf.fit(data, y=labels)
        
        #test best model, fit, and save predictions for submission
        top_model = clf.best_params_
        top_model['cv_score'] = clf.best_score_
        top_svc = clf.best_estimator_
        save_data(directory,
            predictions_file.format(set_num),
            get_predictions(top_svc, features, test_file, pca=pca),
            file_type='.csv'
        )

        #graph / save training and testing predictions of top model
        if save_graph:
            x_axis_title, y_axis_title = 'x', 'y'
            if pca != None and len(features) == 2:
                x_axis_title, y_axis_title = features[0], features[1]
            title = 'c={},gamma={},degree={},type={},feature-set={}'.format(
                top_model.get('C', 'None'),
                top_model.get('gamma', 'None'),
                top_model.get('degree', 'None'),            
                top_model.get('kernel', 'N/A'),
                set_num
            )
            plot_svm(top_svc,
                graph_file.format(title),
                title, x_axis_title,
                y_axis_title,
                data,
                labels
            )

        #save parameter search scores as table, and model details w/ feature set as JSON
        top_model['training_feature_set'] = features
        save_data(directory,
            results_file.format(set_num),
            pd.DataFrame(results.cv_results_),
            file_type='.csv'
        )
        save_data(directory,
            model_file.format(set_num),
            top_model,
            file_type='.json'
        )
        


if __name__ == '__main__':
    main()