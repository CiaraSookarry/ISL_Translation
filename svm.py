# svm.py
#
# Implementation of an SVM performing
# classification on hand landmarks
#
# Grid Search and Random Search implemented here as
# well was simply specifying hyperparameter values.
#
# Author: Ciara Sookarry
# Date: 30 November 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.stats import uniform
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

    # Hyperparameter distribution used for Random Search
    '''
    param_distributions = dict(
        kernel    = ['linear', 'rbf', 'sigmoid'],
        C         = uniform(loc=1, scale=99),
        gamma     = uniform(loc=1e-3, scale=1e3),
        coef0     = uniform(loc=-10, scale=20)
    )
    '''
    # Hyperparameter grid specifying values to try with Grid Search
    '''
    tuned_parameters = [
    {"kernel": ["linear"],  "C": [1, 10, 100]},
    {"kernel": ["rbf"],     "C": [1, 10, 100], "gamma": [1e-3, 1,  1e3]},
    {"kernel": ["sigmoid"], "C": [1, 10, 100], "gamma": [1e-3, 1,  1e3], "coef0": [-1.0, 0.0, 1.0]},
    ]   
    '''
    
    # Random Search
    # svclassifier = RandomizedSearchCV(SVC(), param_distributions, random_state=0, n_iter=50, verbose=3)

    # Grid Search
    # svclassifier = GridSearchCV(SVC(), tuned_parameters, verbose=3, scoring="recall_weighted", cv=5)

    # SVM with optimal parameters found via Bayes Search
    svclassifier = SVC(kernel='rbf', C=100, gamma=1)

    # Hyperparameters whcih give reasonable performance on user data
    # svclassifier = SVC(kernel='rbf', C=85.7, gamma=11.7)
 
    # Linear SVM (94% accuracy)
    # svclassifier = SVC(kernel='linear', C=2)
    
    svclassifier.fit(X_train, y_train)

    # Save model for later use with Pickle
    # filename = 'real_time_svm_model.sav'
    # pickle.dump(svclassifier, open(filename, 'wb'))

    # Predict
    y_pred = svclassifier.predict(X_test)

    # Evaluate
    # print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred, cmap='BuGn')
    plt.show()
    # Output best parameters after Grid/Random Search
    # print("Best parameters set found:")
    # print(svclassifier.best_params_)

if __name__ == '__main__':
    main()
