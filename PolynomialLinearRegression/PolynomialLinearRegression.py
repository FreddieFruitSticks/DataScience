# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm
import numpy
import matplotlib.pyplot as plot


data_file="""/home/freddie/git/DataScience/data/\
Machine Learning A-Z Template Folder/Part 2 - \
Regression/Section 6 - Polynomial Regression/Position_Salaries.csv"""

prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([1,2], [2,3])

lin_reg = LinearRegression()
lin_reg.fit(indep_vars, dep_vars)

#will automatically create columns on ones for const variable
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(indep_vars)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, dep_vars)

indep_grid = numpy.arange(min(indep_vars), max(indep_vars), 0.1)
indep_grid = indep_grid.reshape((len(indep_grid),1))

plot.plot(indep_grid, lin_reg_2.predict(poly_reg.fit_transform(indep_grid)), color='blue')
plot.scatter(indep_vars, dep_vars, color='red')

#predict if someone at level 6.5 really had a salary of 160 000
lin_reg_2.predict(poly_reg.fit_transform(6.5))