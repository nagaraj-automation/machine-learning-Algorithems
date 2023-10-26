# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:17:51 2023

@author: nagar
"""
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
import pickle

from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score

df=pd.read_csv(r"C:\Users\nagar\OneDrive\Desktop\Ineuron\wine\WineQT.csv")
df
df=df.drop(columns='Id')
pf=ProfileReport(df)
pf.to_file("wine.html")
sns.boxenplot(df)

X=df.drop(columns='quality')
y=df['quality']
y=pd.DataFrame(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=40)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
y_predict
# Lets check the accuracy score without hyper parameters
accuracy=accuracy_score(y_test, y_predict)
accuracy

#Hyper parameter using grid search cv
from sklearn.model_selection import GridSearchCV
clf=DecisionTreeClassifier(random_state=42)
param_grid={'max_depth':[2,4,6,8],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2,4]}
grid_search=GridSearchCV(clf, param_grid,cv=5)
grid_search.fit(x_train,y_train)

#get the best hyper parametr
best_clf=grid_search.best_estimator_

y_pred_best=best_clf.predict(x_test)

accuracy_best=accuracy_score(y_test,y_pred_best)
accuracy_best
from sklearn import tree
plt.figure(figsize=(10,20))
tree.plot_tree(model)


#Using bagging classifier with Decision tree

from sklearn.ensemble import BaggingClassifier
bg_model=BaggingClassifier(DecisionTreeClassifier(),n_estimators=10)
bg_model.fit(x_train, y_train)
y_bgpredict=bg_model.predict(x_test)
y_bgpredict
bg_accuracy=accuracy_score(y_test, y_bgpredict)
bg_accuracy


#bagging classifier with using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
bg_model=BaggingClassifier(KNeighborsClassifier(),n_estimators=10)
bg_model.fit(x_train, y_train)
y_bgpredict=bg_model.predict(x_test)
y_bgpredict
bg_accuracy=accuracy_score(y_test, y_bgpredict)
bg_accuracy


#Bagging with Random forest
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_rfpredict=rf_model.predict(x_test)
y_rfpredict
rf_accuracy=accuracy_score(y_test, y_rfpredict)
rf_accuracy
