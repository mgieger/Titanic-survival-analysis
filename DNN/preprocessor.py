import pandas as pd
import numpy as np
import sklearn as sk

pd.options.mode.chained_assignment = None

class Preprocessor(object):
    '''Reads and preprocesses the Titanic Dataset'''

    def __init__(self, filename):
        self.preprocessing_map = {
            'name': Preprocessor.name,
            'sex': Preprocessor.sex,
            'embarked': Preprocessor.embarked,
            'ticket': Preprocessor.ticket,
            'cabin': Preprocessor.cabin
        }
        self.dataset_df = pd.read_csv(filename)
        self.processed_df = self._preprocess(self.dataset_df)

        
    def get_matrix(self, cols):
        '''
        Args:
            cols (list): list of dataframe columns to retrieve
        Returns:
            np.array: preprocessed data where np.array[:,n] == col[n]
        '''
        return np.nan_to_num(self.processed_df[cols].as_matrix())


    def get_matrix_split(self, cols, row=800):
        '''
        Args:
            cols (list): list of dataframe colums to retrieve
            row: row to split dataframe on for training and test set
        returns:
            dictionary of np.arrays
        '''
        self.shuffle()
        return {
            "training": np.nan_to_num(self.processed_df[0:row][cols].as_matrix()),
            "test": np.nan_to_num(self.processed_df[row:][cols].as_matrix())
    }


    def get_matrix_scaled(self, cols, range=(0, 1)):
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
                                             feature_range=(0, 1),
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
	        _processed_df[field] = _processed_df[field].apply(self.preprocessing_map[field])
        return _processed_df

      
    # preprocessing functions (as class funcs meh...)
    def sex(sex):
        '''passenger.sex -> (int)'''
        if sex == 'female':
            return 1
        elif sex == 'male':
            return 0
        else:
            return -1

          
    def embarked(embarked):
        '''passenger.embarked -> (int)'''
        if embarked == 'S':
            return 0
        elif embarked == 'Q':
            return 1
        elif embarked == 'C':
            return 2
        else:
            return 3

          
#TODO: play around with change values for classes  -- investigate bias here
    def name(name):
        """passenger.name -> (int)"""
        if 'Sir.' in name:
            return 5
        elif 'Dr.' in name:
            return 4
        elif 'Mr.' in name:
            return 3
        elif 'Mrs.' in name:
            return 2
        elif 'Miss' in name:
            return 1
        else:
            return 0

          
    def ticket(ticket):
        '''passenger.ticket -> (int)'''
        try:
            return int(ticket)
        except:
            return 0

          
    def cabin(cabin):
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

        
    def zero(item):
        '''item -> 0 used for unknown fields'''
        return 0

      
    def sibsp(item):
        return item

      
    def parch(item):
        return item
    
    
    def fare(item):
        return item

      
    def shuffle(self):
        self.procesed_df = self.processed_df.sample(frac=1)
