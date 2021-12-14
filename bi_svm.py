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
from sklearn.svm import SVC

def prepareData():
    # import training and testing CSV data
    train = pd.read_csv("training_landmarks.csv")
    test = pd.read_csv("testing_landmarks.csv")

    # drop all points that dont have label K or L
    train = train.drop(train[(train.label != 'K') & (train.label != 'L')].index)
    test = test.drop(test[(test.label != 'K') & (test.label != 'L')].index)

    # separate labels from landmarks
    X_train = train.drop('label', axis=1)
    X_test = test.drop('label', axis=1)
    
    y_train = train['label']
    y_test = test['label'] 

    return X_train, X_test, y_train, y_test
    

def main():
    X_train, X_test, y_train, y_test = prepareData()

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    # Predict
    y_pred = svclassifier.predict(X_test)

    # Evaluate
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    main()
