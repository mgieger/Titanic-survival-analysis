import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier



def main():
    full_file = '../titanic_full.csv'

    # columns = [
    #     'PassengerId',
    #     'Pclass',
    #     'Name',
    #     'Sex',
    #     'Age',
    #     'SibSp',
    #     'Parch',
    #     'Ticket',
    #     'Fare',
    #     'Cabin',
    #     'Embarked',
    #     'Survived'
    # ]

    preprocessor = Preprocessor(full_file)
    #data = preprocessor.get_matrix_split(['Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Fare', 'Survived','PassengerId'])
    #data = preprocessor.get_matrix_split(['pclass', 'name', 'sex', 'age', 'ticket', 'cabin', 'fare', 'survived'])
    data = preprocessor.get_matrix_split(['pclass', 'sex', 'fare', 'survived'])
    training_data = data["training"]
    testing_data = data["test"]

    train, train_labels = training_data[:,:-1], training_data[:,-1:]
    train_labels = train_labels.reshape(train_labels.shape[0],)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train, train_labels)
    print(clf.feature_importances_)
    # print(test[0].reshape(1,-1))
    # print("predict: ", clf.predict(test[0].reshape(1,-1)))
    # print("label: ", test_labels[0])
    # print("Decision Path:\n", clf.decision_path(test[0].reshape(1,-1)))
    #
    test, test_labels = testing_data[:, :-1], testing_data[:, -1:]
    test_labels = test_labels.reshape(test_labels.shape[0], )
    confusion_matrix = np.zeros((2,2))
    print(np.shape(test))
    print(np.shape(test_labels))
    for i in range(test_labels.shape[0]):
        x = clf.predict(test[i].reshape(1,-1))
        y = test_labels[i]
        #print(int(y))
        confusion_matrix[int(x)][int(y)] += 1

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print(accuracy)

if __name__ == '__main__':
    main()