# svm.py
#
# Implementation of an SVM performing
# classification on hand landmarks
#
# Author: Ciara Sookarry
# Date: 30 November 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.stats import uniform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from skopt import BayesSearchCV

def prepareData():
    # import training and testing CSV data
    train = pd.read_csv("training_landmarks.csv")
    test = pd.read_csv("testing_landmarks.csv")

    # separate labels from landmarks
    X_train = train.drop('label', axis=1)
    X_test = test.drop('label', axis=1)
    
    y_train = train['label']
    y_test = test['label'] 
    
    return X_train, X_test, y_train, y_test
    

def main():
    X_train, X_test, y_train, y_test = prepareData()

    param_distributions = dict(
        kernel    = ['linear', 'rbf', 'sigmoid'],
        C         = uniform(loc=1, scale=99),
        gamma     = uniform(loc=1e-3, scale=1e3),
        coef0     = uniform(loc=-10, scale=20)
    )

    '''
    tuned_parameters = [
    {"kernel": ["linear"],  "C": [1, 10, 100]},
    {"kernel": ["rbf"],     "C": [1, 10, 100], "gamma": [1e-3, 1,  1e3]},
    {"kernel": ["sigmoid"], "C": [1, 10, 100], "gamma": [1e-3, 1,  1e3], "coef0": [-1.0, 0.0, 1.0]},
    ]   
    '''
    
    # svclassifier = RandomizedSearchCV(SVC(), param_distributions, random_state=0, n_iter=50, verbose=3)
    # svclassifier = GridSearchCV(SVC(), tuned_parameters, verbose=3, scoring="recall_weighted", cv=5)        
    svclassifier = SVC(kernel='rbf', C=53.98, gamma=40.18)
    svclassifier.fit(X_train, y_train)

    # Save model for later use with Pickle
    filename = 'svm_model.sav'
    pickle.dump(svclassifier, open(filename, 'wb'))

    # Predict
    y_pred = svclassifier.predict(X_test)

    # Evaluate
    # print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("Best parameters set found:")
    print(svclassifier.best_params_)

if __name__ == '__main__':
    main()
