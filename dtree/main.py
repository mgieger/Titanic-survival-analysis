import numpy as np

from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
import math


def main():
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

#pclass,survived,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home.dest

    # preprocessor = Preprocessor(train_file)

    preprocessor = Preprocessor(full_file)
    # preprocessor_test = Preprocessor(train_file)

    data = preprocessor.get_matrix(['pclass', 'name', 'sex', 'age', 'ticket', 'cabin', 'fare', 'survived'])
    # np.random.shuffle(data) TODO: put in
    # data_split = data.shape[0]/2
    
    
    
    # data_test = preprocessor_test.get_matrix(['Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Fare', 'Survived','PassengerId'])
    #
    print(data[:10,:])
    #
    #
    TRAIN_SIZE = math.ceil(data.shape[0]*.70)
    train, train_labels = data[:TRAIN_SIZE,:-1], data[:TRAIN_SIZE,-1:]
    # train, train_labels, train_pid = data[:TEST_SIZE, :-1], data[:TEST_SIZE, -2], data[:TEST_SIZE, -1]
    test, test_labels = data[TRAIN_SIZE:,:-1], data[TRAIN_SIZE:,-1]
    print(train.shape)
    print(data.shape)
    print(test.shape)
    #print(train_labels_pid)
    # print(train_labels_pid)
    #
    print(train_labels)
    #
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train, train_labels)
    print(clf.feature_importances_)
    # print(test[0].reshape(1,-1))
    print("predict: ", clf.predict(test[0].reshape(1,-1)))
    print("label: ", test_labels[0])
    print("Decision Path:\n", clf.decision_path(test[0].reshape(1,-1)))

    confusion_matrix = np.zeros((2,2))
    for i in range(test_labels.shape[0]):
        x = clf.predict(test[i].reshape(1,-1))
        y = test_labels[i]
        # print(i)
        print(x)
        print(y)
        #confusion_matrix[x][y] += 1

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print(accuracy)

if __name__ == '__main__':
    main()