# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from untitled3 import DataPreparation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot


data_file = """/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv"""
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([0,1], [1,2])
indep_train, indep_test, dep_train, dep_test = prep.partition_training_test()

regressor = LinearRegression()
regressor.fit(indep_train, dep_train)
dep_predict = regressor.predict(indep_test)

error = dep_predict - dep_test
plot.scatter(indep_train, dep_train, color='red')
plot.scatter(indep_test, dep_test, color='green')
plot.plot(indep_train, regressor.predict(indep_train), color='blue')
plot.title('Salary vs Experience')
plot.ylabel('Salary')
plot.xlabel('Years')
plot.show()