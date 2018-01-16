# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:16:04 2018

@author: freddie
"""

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as sm
import numpy
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap

data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z \
Template Folder/Part 3 - Classification/Section 14 -\
 Logistic Regression/Social_Network_Ads.csv"""
 
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([2,4], [4,5])
indep_vars = prep.feature_scaling(indep_vars, None)
sc_indep = StandardScaler()
indep_train, indep_test, dep_train, dep_test = prep.partition_training_test()

scale_X_train = sc_indep.fit_transform(indep_train)
scale_X_test = sc_indep.transform(indep_test)

classifier = GaussianNB()
classifier.fit(scale_X_train, dep_train[:,-1])

y_pred = classifier.predict(indep_test)
cm = confusion_matrix(dep_test, y_pred)

#plot
X_set, y_set = scale_X_train, dep_train.ravel()
X1, X2 = numpy.meshgrid(numpy.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),\
               numpy.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))
plot.contourf(X1,X2,classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),\
              alpha=0.75,cmap=ListedColormap(('red','green')))
plot.xlim(X1.min(),X1.max())
plot.ylim(X2.min(),X2.max())

for i,j in enumerate(numpy.unique(y_set)):
    plot.scatter(X_set[y_set == j,0],X_set[y_set == j,1], c = ListedColormap(('red','green'))(i), label=j)