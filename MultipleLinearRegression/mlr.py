#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:39:32 2017

@author: freddie
"""
import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import numpy

data_file = """/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv"""
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([0,4], [4,5])

#here one of the dummy variables need to be removed to avoid the Dummy variable trap.
#This is automatically done with the current libs but we will remove a dummy variable anyway.
indep_vars, dep_vars = prep.add_dummy_variables(encode_dep=False, columns=[3])
indep_vars = indep_vars[:,1:]

indep_train, indep_test, dep_train, dep_test = prep.partition_training_test()
#no need to scale feature. lr lib does that or us.

regressor = LinearRegression()
regressor.fit(indep_train, dep_train)
dep_pred = regressor.predict(indep_test)

error = dep_pred-dep_test

#backwoard elimination
#Notes: in the statsmodel library the constant mlr variable is not taken in to account.
#Need if you look at the matrix form of mlr y=b0+bX we can have y=b0x0+bX where x0=1 or
#y=bX where first row of X is 1s. here indep_vars=X

indep_vars = numpy.append(arr=numpy.ones((50,1)), values=indep_vars, axis=1)
X_opt = indep_vars[:,[0,3]]

#step1: select SL
#Step2: fit the model
regressor_OLS = sm.OLS(endog=dep_vars, exog = X_opt).fit()

#step3:consider predictor of highest p value

#Three tests should be conducted http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
#ANOVA test: Test for significance of regression: This test checks the significance of the whole regression model.
#t-test: This test checks the significance of individual regression coefficients.
#F-test This test can be used to simultaneously check the significance of a number of regression coefficients. It can also be used to test individual coefficients.

#step4: remove
regressor_OLS.summary()

#step5: retest