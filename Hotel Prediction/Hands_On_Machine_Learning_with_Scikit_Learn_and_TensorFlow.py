import os
import numpy as np
from numpy import genfromtxt
from numpy.random import rand
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cross_validation import *


import xgboost as xgb
#TODO: Worth explaining why we didnt use PCA (we wanted to be able to interpret results)

#X: 600000 x 3 np array
#Y: 600000 x 10 array (bubbles), 600000 x 5 array (stars)
def xgboost(X, y):
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    X_train = X[:17500,:]
    y_train = y[:17500]

    X_test = X[17500:,:]
    y_test = y[17500:]

    print("data ready?")
    xgb_model = xgb.XGBClassifier(eval_metric = 'auc', num_class = 10, nthread = 4,silent = 1, objective='multi:softmax', seed=4)

    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have much fun of fighting against overfit
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance

    print("starting parameter declaration")
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['multi:softmax'],
                  'learning_rate': [0.04], #so called `eta` value
                  'max_depth': [6,7,8],
                  'min_child_weight': [5, 10],
                  'silent': [1],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [100, 200, 300], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [17]}

    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, cv=5,
                       scoring='f1_micro', verbose=2, refit=True)


    clf.fit(X_train, y_train)

    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw F1 score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    print("best XGBOOST score: ", clf.best_score_)

    print("FEATURE IMPORTANCES: ", clf.best_estimator_.feature_importances_)


    joblib.dump(clf.best_estimator_, "bestxgb_model.pkl")

    #Evaluate prediction accuracy on test set
    preds = clf.predict(X_test)
    print(accuracy_score(y_test, preds))




def grad_boosting(X, y):
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    X_train = X[:17500,:]
    y_train = y[:17500]

    X_test = X[17500:,:]
    y_test = y[17500:]
    best_params = {'n_estimators':range(20,81,200)}
    clf = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
            min_samples_leaf=50, max_depth=20, max_features='sqrt',subsample=0.8,random_state=10),
            param_grid = best_params, n_jobs=4, iid=False, scoring='f1_micro', cv=5)

    clf.fit(X_train, y_train)
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw F1 score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    print("best XGBOOST score: ", clf.best_score_)

    print("FEATURE IMPORTANCES: ", clf.best_estimator_.feature_importances_)

    joblib.dump(clf.best_estimator_, "bestxgb_model.pkl")

    #Evaluate prediction accuracy on test set
    preds = clf.predict(X_test)
    print(accuracy_score(y_test, preds))




big_df = pd.read_csv(os.path.join('/home', 'adi', 'Desktop', 'Brown_Datathon', 'clean_data.csv'))

big_df.fillna(0)

X_brand_name = big_df['parent_brand_name_enc']
X_hotel_type = big_df['hotel_type_enc']
X_popularity = big_df['popularity']

X = np.stack((X_brand_name, X_hotel_type, X_popularity), axis=1)
y = np.array(big_df['star_rating'])

np.nan_to_num(y)

y *= 2
y -= 2



y = y.astype(np.uint8)
#y = np.eye(10)[y]

print("GRAD BOOST")
grad_boosting(X,y)

print("XG BOOST")
xgboost(X, y)
