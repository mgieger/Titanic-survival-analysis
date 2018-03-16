from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from dtree.decisiontree import DecisionTreeRunner
import pprint

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
    default_rfc = RandomForestClassifier(random_state=0)
    warmstart_rfc = RandomForestClassifier(warm_start=True)
    
    
    preprocessor = Preprocessor(full_file)
    
	#Experiment Category 1:  Consi
    #TODO: normalize data.
    data_perm_1 = preprocessor.get_matrix_split(['pclass', 'name', 'sex', 'age', 'ticket', 'cabin', 'fare', 'survived'])
    
    
    # train_perm_1= data_perm_1["training"]
    # test_perm_1 = data_perm_1["test"]
    
    dtr1 = DecisionTreeRunner(data_perm_1, default_rfc)
    dtr1.run()
    dtr1.print_accuracy()
    # pprint.pprint(dtr1.results)
    # pprint.pprint(dtr1.failed_predictions)



	#compared failed predictions, are there any outliers who are consistent across all models?
	

if __name__ == '__main__':
    main()