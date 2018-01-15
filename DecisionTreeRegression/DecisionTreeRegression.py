#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:02:07 2018

@author: freddie
"""

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as sm
import numpy
import matplotlib.pyplot as plot


data_file="""/home/freddie/git/DataScience/data/\
Machine Learning A-Z Template Folder/Part 2 -\
 Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv"""

prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([1,2], [2,3])
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(indep_vars, dep_vars)

y_pred = regressor.predict(6.5)

indep_grid = numpy.arange(min(indep_vars), max(indep_vars), 0.1)
indep_grid = indep_grid.reshape((len(indep_grid),1))

plot.plot(indep_grid, regressor.predict(indep_grid), color='blue')
plot.scatter(indep_vars, dep_vars, color='red')