#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pandas
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


path="""../data/Machine Learning A-Z Template Folder/\
Part 1 - Data Preprocessing/Section 2 -------------------- \
Part 1 - Data Preprocessing --------------------/\
Data_Preprocessing/Data.csv"""
   
data_frame = pandas.read_csv(path)
independent_vars = data_frame.iloc[:,:-1].values
dependent_vars = data_frame.iloc[:,-1].values

# apply rule for missing data
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(independent_vars[:, 1:3])
independent_vars[:, 1:3] = imputer.transform(independent_vars[:, 1:3])

#convert categorical data in to numbers
label_encoder_indep_vars = LabelEncoder()
independent_vars[:, 0] = label_encoder_indep_vars.fit_transform(independent_vars[:, 0])
label_encoder_dep_vars = LabelEncoder()
dependent_vars = label_encoder_dep_vars.fit_transform(dependent_vars)

#convert categorical data in to three columns of 0 or 1 dummy variables
one_hot_encoder = OneHotEncoder(categorical_features = [0])
independent_vars = one_hot_encoder.fit_transform(independent_vars).toarray()

#partition randomly in to training and test data
indep_train, indep_test, dep_train, dep_test = \
 train_test_split(independent_vars, dependent_vars, test_size=0.25, random_state=43)


#feature scaling
sc_indep = StandardScaler()
indep_train[:,3:5] = sc_indep.fit_transform(indep_train[:,3:5])
indep_test[:,3:5] = sc_indep.fit_transform(indep_test[:,3:5])