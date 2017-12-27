# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from untitled3 import DataPreparation

data_file = """/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv"""
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([0,1], [1,2])
indep_train, indep_test, dep_train, dep_test = prep.partition_training_test()

