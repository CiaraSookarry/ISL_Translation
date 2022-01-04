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

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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

    tuned_parameters = [
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [10, 100, 1000]},
    ]   

    scores = ["precision", "recall"]

    for score in scores:
        print("# Tuning hyper-parameters for %s\n" % score)
        
        svclassifier = GridSearchCV(SVC(), tuned_parameters, verbose=3, scoring="%s_weighted" % score, cv=5)        

        #svclassifier = SVC(C=100, kernel='linear')
        svclassifier.fit(X_train, y_train)

        # Predict
        y_pred = svclassifier.predict(X_test)

        # Evaluate
        # print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print("Best parameters set found on development set:")
        print(svclassifier.best_params_)

if __name__ == '__main__':
    main()
