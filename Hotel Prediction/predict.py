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
from sklearn.metrics import roc_auc_score
import sklearn.neural_network as neural_network

#TODO: Worth explaining why we didnt use PCA (we wanted to be able to interpret results)

#X is expected to be 10000 x 44 np array
#y is expected to be 10000 x 1 np array
def train_l1(X, y, savenm):

	p = np.random.permutation(len(X))
    X = X[p]
	y = y[p]

	X_train = X[:int(0.8*len(X)), :]
	y_train = y[:int(0.8*len(y))]

	X_test = X[int(0.8*len(X)):, :]
	y_test = y[int(0.8*len(y)):]



	clf = linear_model.LogisticRegression(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6), warm_start=True, multi_class='ovr')

	# prepare a uniform distribution to sample for the alpha parameter

	# create and fit a ridge regression model, testing random alpha values
	print("starting randomized search with lasso")


	print("X train shape: ", X_train.shape)
	print("Y train shape :", y_train.shape)
	#clf.fit(X_train, y_train)
	print(clf)


	scores = cross_val_score(clf,X, y, cv=5, scoring='roc_auc')

	joblib.dump(clf, savenm)


def reduce_features_and_train_model(X, y, l1_model, savenm):

	p = np.random.permutation(len(X))
        X = X[p]
	y = y[p]

	X_train = X[:int(0.8*len(X)), :]
	y_train = y[:int(0.8*len(y))]

	X_test = X[int(0.8*len(X)):, :]
	y_test = y[int(0.8*len(y)):]



        print("X TEST SHAPE: ", X_test.shape)


       # print("transformed X TEST SHAPE: ", transformed_X_test.shape)


        if(1 == 0):
		print("NO FEATURES R RELATED TO THIS VAR: ", savenm)
	else:


		#svr = GridSearchCV(LinearSVR(epsilon=0), cv=5, param_grid={"C": [1e0, 1e1, 1e2,1e3]})  # "gamma": np.logspace(-2, 2, 5)})



		param_test1 = {'n_estimators':range(20,81,200)}
		gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
		min_samples_split=500,min_samples_leaf=50, max_depth=20, max_features='sqrt',subsample=0.8,random_state=10),
		param_grid = param_test1, n_jobs=4, iid=False, scoring='roc_auc', cv=3)



		print("X_train: ", X_train.shape)

		print("y_train: ", y_train.shape)
		gsearch1.fit(X, y)



		print("ESTIMATOR: ", gsearch1.best_estimator_)

        #        svr.fit(X_train, y_train)
		#should spit out best score (who even knows what this score is??)
		print("best Gradient Boosted score: ", gsearch1.best_score_)

		print("FEATURE IMPORTANCES: ", gsearch1.best_estimator_.feature_importances_)
		joblib.dump(gsearch1.best_estimator_, savenm)





#X is expected to be 285619 x 44 np array
#all_y is expected to be 258619 x 14 np array
#do different savenms for each of the 14 features we predicting
def train_all_models(X, all_y):


  #      print("num diff models: ", all_y.shape[1])
#        for i in range (all_y.shape[1]):
                #train_l1(X, all_y[:,i], os.path.join('/home', 'adi', 'Dropbox', 'Datathon', 'models', 'l1_' + str(i) + '.pkl'))
		#train_nn(X, all_y[:,i], '/home/adi/Dropbox/Datathon/models/nn_' + str(i) + '.pkl')



	#train_l1(X, all_y, os.path.join('/home', 'adi', 'Dropbox', 'Datathon', 'models', 'l1.pkl'))
	reduce_features_and_train_model(X, all_y, os.path.join('/home', 'adi', 'Dropbox', 'Datathon', 'models', 'l1.pkl'),  os.path.join('/home', 'adi', 'Dropbox', 		'Datathon', 'models', 'gbm.pkl'))




X_a = genfromtxt(os.path.join('/home', 'adi', 'Downloads', 'feature_dict.csv'), delimiter=',', dtype=float, filling_values=0.0)
print("len X_a: ", len(X_a))
y_a = genfromtxt(os.path.join('/home', 'adi', 'Downloads', 'LABELS.csv'), delimiter=',', dtype=float, filling_values=0.0)


X_a = X_a[:len(y_a), :]





#print(y_a.shape)
y_new = np.argmax(y_a, axis=1)

print(y_new.shape)
#y_new = np.squeeze(np.eye(12)[y_new.reshape(-1)])

print(y_new.shape)



print("Y NEW SHAPE: ", y_new.shape)
train_all_models(X_a, y_new)
