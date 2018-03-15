import pandas as pd
import numpy as np
import sklearn as sk
import re
pd.options.mode.chained_assignment = None

class Preprocessor(object):
    '''Reads and preprocesses the Titanic Dataset'''

    def __init__(self, filename):
        self.preprocessing_map = {
            'survived': Preprocessor.survived,
            'pclass': Preprocessor.pclass,
            'name': Preprocessor.name,
            'sex': Preprocessor.sex,
            'age': Preprocessor.age,
            'sibsp': Preprocessor.sibsp,
            'parch': Preprocessor.parch,
            'ticket': Preprocessor.ticket,
            'fare': Preprocessor.fare,
            'cabin': Preprocessor.cabin,
            'embarked': Preprocessor.embarked
            #No preprocessing for boat, body, or home dest.
        }
        self.dataset_df = pd.read_csv(filename)
        ###datashaping -> Moved labels to 0 column and kept only cols of interest.
       # cols = ['survived', 'pclass', 'name', 'sex', 'age', 'sibsp',
        #        'parch', 'ticket', 'fare', 'cabin', 'embarked']
        #self.dataset_df.loc[:, cols]
        # dataset = self.dataset_df.values
        # print(np.shape(dataset))
        # labels = dataset[:, 1]
        # pclass = dataset[:, 0]
        # remaining_data = dataset[:, 2:]
        # print(np.shape(labels))
        # print(np.shape(pclass))
        # print(np.shape(remaining_data))
        self.dataset_df = self.dataset_df.iloc[:, : 11]

        self.processed_df = self._preprocess(self.dataset_df)

    def get_matrix(self, cols):
        '''
        Args:
            cols (list): list of dataframe columns to retrieve
        Returns:
            np.array: preprocessed data where np.array[:,n] == col[n]
        '''
        return np.nan_to_num(self.processed_df[cols].as_matrix())
    
    def get_matrix_scaled(self, cols, range=(0,1)):
        '''
        Args:
            cols (list): list of dataframe columns to retrieve
            range (tuple): range to scale features to
        Returns:
            np.array: preprocessed data where np.array[:,n] == col[n] and
                      the data in col[n] is scaled to the given range
        '''
        # can also use l-1, l-2 normalization, standarization, etc.
        return sk.preprocessing.minmax_scale(self.get_matrix(cols),
            feature_range=(0,1),
            axis=0,
            copy=False
        )

    def get_dataframe(self):
        '''
        Returns:
            pd.dataframe: a copy of the original dataset
        '''
        return self.dataset_df.copy(deep=True)

    def _preprocess(self, original_df):
        '''
        Args:
            original_df (pd.dataframe): dataframe
        Returns:
            pd.dataframe: a preprocessed copy of original_df according to self.preprocessing_map rules
        '''
        _processed_df = original_df.copy(deep=True)
        for field in self.preprocessing_map.keys():
            #index_val = _processed_df.columns.get_loc(field)
           # _processed_df.iloc[:, index_val] = _processed_df.iloc[:, index_val].apply(self.preprocessing_map[field])
            _processed_df[field] = _processed_df[field].apply(self.preprocessing_map[field])
        return _processed_df

    # preprocessing functions (as class funcs meh...)
    # Explictly cast all of the input values to try and fix a potential type error
    # in the values of the titanic_full.csv file.
    def pclass(self, pclass):
        '''passenger.pclass -> (int)'''
        return int(pclass)

    def survived(self, survived):
        return int(survived)

    def name(self, name):
        '''passenger.name -> (int)'''              
        if 'sir.' in name.lower():
            return 5
        elif 'dr.' in name.lower():
            return 4
        elif 'mr.' in name.lower():
            return 3
        elif 'mrs.' in name.lower():
            return 2
        elif 'miss' in name.lower():
            return 1
        else:
            return 0

    def sex(self, sex):
        '''passenger.sex -> (int)'''
        if sex == 'female':
            return 1
        elif sex == 'male':
            return 0
        else:
            return -1

    def age(self, age):
        return int(age)

    def sibsp(self, sibsp):
        return int(sibsp)

    def parch(self, parch):
        return int(parch)

    def ticket(self, ticket):
        '''passenger.ticket -> (int)'''        
        try:
            re.sub('[^0-9]', '', ticket)
            return int(ticket)
        except:
            return 0
    
    def fare(self, fare):
        return int(fare)

    def cabin(self, cabin):
        '''passenger.cabin -> (int)'''
        if 'T' or 'A' in cabin:
            return 1
        elif 'B' in cabin:
            return 2
        elif 'C' in cabin:
            return 3
        elif 'D' in cabin:
            return 4
        elif 'E' in cabin:
            return 5
        elif 'F' in cabin:
            return 6
        else:
            return 7

    def embarked(self, embarked):
        '''passenger.embarked -> (int)'''
        if embarked == 'S':
            return 0
        elif embarked == 'Q':
            return 1
        elif embarked == 'C':
            return 2
        else:
            return 3

    def zero(self, item):
        '''item -> 0 used for unknown fields'''
        return 0

        
    