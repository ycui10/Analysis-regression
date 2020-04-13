import numpy as np
import pandas as pd
import random
import csv
import statsmodels.api as sm
from statsmodels.tools import eval_measures

def split_data(data, prob):
    """input: 
     data: a list of pairs of x,y values
     prob: the fraction of the dataset that will be testing data, typically prob=0.2
     output:
     two lists with training data pairs and testing data pairs 
    """

    #TODO: Split data into fractions [prob, 1 - prob]
    length = len(data)
    rand = [random.random() for i in range(length)]
    test = []
    train = []

    for i in range(length):
        if rand[i] <= prob:
            test.append(data[i])
        else:
            train.append(data[i])

    return train, test
     
    

def train_test_split(x, y, test_pct):
    """input:
    x: list of x values, y: list of independent values, test_pct: percentage of the data that is testing data=0.2.

    output: x_train, x_test, y_train, y_test lists
    """
    
    #TODO: Split the features X and the labels y into x_train, x_test and y_train, y_test as specified by test_pct
    zipped = zip(x, y)
    train, test = split_data(list(zipped), test_pct)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for (x, y) in train:
        x_train.append(x)
        y_train.append(y)

    for (x, y) in test:
        x_test.append(x)
        y_test.append(y)

    return x_train, x_test, y_train, y_test



if __name__=='__main__':

    # DO not change this seed. It guarantees that all students perform the same train and test split
    random.seed(1)
    # Setting p to 0.2 allows for a 80% training and 20% test split
    p = 0.2

    #############################################
    # TODO: open csv and read data into X and y #
    #############################################
    def load_file(file_path):
        """input: file_path: the path to the data file
           output: X: array of independent variables values, y: array of the dependent variable values
        """
        #TODO: 
        #1. Use pandas to load data from the file. Here you can also re-use most of the code from part I.
        #2. Select which independent variables best predict the dependent variable count.
        data_df = pd.read_csv(file_path)
        X = data_df.drop(['row', 'is_winner'], axis=1)
        y = data_df['is_winner']
        return X , y

        


    X, y = load_file("tennis.csv")
    print(X)
    X = sm.add_constant(X)
    x_train, x_test, y_train, y_test = train_test_split(X.values, y, p)
    model = sm.OLS(y_train, x_train)
    results = model.fit()
    print(results.summary())

    train_y_cap = results.predict(x_train)
    y_cap = results.predict(x_test)
    training_MSE = eval_measures.mse(y_train, train_y_cap)
    testing_MSE = eval_measures.mse(y_test, y_cap)
    print('r-squared: '+ str(results.rsquared))
    print('training MSE: '+str(training_MSE))
    print('testing MSE: '+str(testing_MSE))

    ##################################################################################
    # TODO: use train test split to split data into x_train, x_test, y_train, y_test #
    #################################################################################


    ##################################################################################
    # TODO: Use StatsModels to create the Linear Model and Output R-squared
    #################################################################################


    # Prints out the Report
    # TODO: print R-squared, test MSE & train MSE