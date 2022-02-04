# bayes_svm.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

train = pd.read_csv("training_landmarks.csv")
X = train.drop('label', axis=1)
y = train['label']

train = pd.read_csv("testing_landmarks.csv")
X_test = train.drop('label', axis=1)
y_test = train['label']

def hyperopt_train_test(params):
    X_ = X[:]    
    '''
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = preprocessing.normalize(X_)
        del params['normalize']    
   
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = preprocessing.scale(X_)
        del params['scale']    
    '''
    clf = SVC(**params)
    
    return cross_val_score(clf, X_, y, verbose=3).mean()

space4svm = {
    'C': hp.uniform('C', 0, 100),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf']), #, 'poly']),
    'gamma': hp.uniform('gamma', 1e-3, 1e3),
    #'scale': hp.choice('scale', [0, 1]),
    #'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

start_time = time.time()
trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=5, trials=trials)
print('best:')
print(space_eval(space4svm, best))
print(classification_report(y_test,best))
print("--- %s seconds ---" % (time.time() - start_time))

parameters = ['C', 'kernel', 'gamma'] #, 'scale', 'normalize']
cols = len(parameters)

f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))

cmap = plt.cm.jet

for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, color=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.9, 1.0])

plt.show()
