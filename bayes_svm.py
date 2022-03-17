# bayes_svm.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# print("It's landmark time!!!")
train = pd.read_csv("training_landmarks.csv")
X = train.drop('label', axis=1)
y = train['label']


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
    
    return cross_val_score(clf, X_, y).mean()

space4svm = {
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf']),
    'C': hp.uniform('C', 1, 100),
    'gamma': hp.uniform('gamma', 1e-3, 1e3),
    'coef0': hp.uniform('coef0', -10, 10)
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=50, trials=trials)
print('best:')
print(space_eval(space4svm, best))

# Plot hyperparameter values
parameters = ['kernel', 'C', 'gamma', 'coef0'] 
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
