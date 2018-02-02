#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 01:27:35 2017

@author: freddie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreparation:
    data_file="""../data/Machine Learning A-Z Template Folder/\
Part 1 - Data Preprocessing/Section 2 -------------------- \
Part 1 - Data Preprocessing --------------------/\
Data_Preprocessing/Data.csv"""
    
    def __init__(self, data_file=data_file):
        self.data_file = data_file
        
    def prepare_data_frame(self, indep_interval, dep_interval, header='infer'):        
        data_frame = pandas.read_csv(self.data_file, header=header)
        self.independent_vars = data_frame.iloc[:, indep_interval[0] : indep_interval[1]].values
        self.dependent_vars = data_frame.iloc[:, dep_interval[0] : dep_interval[1]].values
        return self.independent_vars, self.dependent_vars, data_frame
    
    def add_missing_data(self):
        # apply rule for missing data
        imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(self.independent_vars[:, 1:3])
        self.independent_vars[:, 1:3] = imputer.transform(self.independent_vars[:, 1:3])
        return self.independent_vars
    
    def add_dummy_variables(self, encode_dep=False, columns=[0]):    
        #convert categorical data in to numbers
        label_encoder_indep_vars = LabelEncoder()
        self.independent_vars[:, columns[0]] = label_encoder_indep_vars.fit_transform(self.independent_vars[:, columns[0]])

        if encode_dep:
            label_encoder_dep_vars = LabelEncoder()
            self.dependent_vars = label_encoder_dep_vars.fit_transform(self.dependent_vars)
        
        #convert categorical data in to three columns of 0 or 1 dummy variables
        one_hot_encoder = OneHotEncoder(categorical_features = [columns[0]])
        self.independent_vars = one_hot_encoder.fit_transform(self.independent_vars).toarray()
        return self.independent_vars, self.dependent_vars
    
    def partition_training_test(self):
        #partition randomly in to training and test data
        self.indep_train, self.indep_test, self.dep_train, self.dep_test = \
        train_test_split(self.independent_vars, self.dependent_vars, test_size=0.25, random_state=0)
        return self.indep_train, self.indep_test, self.dep_train, self.dep_test
        
    def feature_scaling(self, indep_vars, col_interval):    
        #feature scaling
        sc_indep = StandardScaler()
        self.independent_vars = sc_indep.fit_transform(indep_vars)
        return self.independent_vars
    
    def plot(self, indep_vars, dep_vars, regressor, X_poly_grid=None, indep_plot_grid=None):
        plot.scatter(indep_vars, dep_vars, color='red')
        if X_poly_grid is None or X_poly_grid.any() is None :
            plot.plot(indep_plot_grid, regressor.predict(indep_plot_grid),color='blue')
        else:
            plot.plot(X_poly_grid, regressor.predict(X_poly_grid), color='blue')



        