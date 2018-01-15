#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:35:47 2017

@author: freddie
"""

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.formula.api as sm
import numpy
import matplotlib.pyplot as plot
from sklearn.svm import SVR


data_file="""/home/freddie/git/DataScience/data/\
Machine Learning A-Z Template Folder/Part 2 - \
Regression/Section 6 - Polynomial Regression/Position_Salaries.csv"""

prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([1,2], [2,3])

sc_indep = StandardScaler()
sc_dep = StandardScaler()

indep_vars = sc_indep.fit_transform(indep_vars)
dep_vars = sc_dep.fit_transform(dep_vars)

regressor = SVR(kernel = 'rbf')
regressor.fit(indep_vars, numpy.asarray(dep_vars).reshape(-1)) 

liar = sc_dep.inverse_transform(regressor.predict(sc_indep.transform(6.5)))

indep_grid = numpy.arange(min(indep_vars), max(indep_vars), 0.1)
indep_grid = indep_grid.reshape((len(indep_grid),1))
plot.plot(indep_grid, regressor.predict(indep_grid), color='blue')
plot.scatter(indep_vars, dep_vars, color='red')
