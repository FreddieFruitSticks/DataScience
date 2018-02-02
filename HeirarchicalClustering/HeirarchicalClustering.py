# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plot

data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv"""
 
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([2,4], [4,5])
X = data_frame.iloc[:,[3,4]].values

dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#visualise
plot.scatter(X[y_hc == 0, 0],X[y_hc == 0,1],s=100,c='red',label='Schnoep')
plot.scatter(X[y_hc == 1, 0],X[y_hc == 1,1],s=100,c='blue',label='moderate')
plot.scatter(X[y_hc == 2, 0],X[y_hc == 2,1],s=100,c='green',label='housewives')
plot.scatter(X[y_hc == 3, 0],X[y_hc == 3,1],s=100,c='magenta',label='reckless spenders')
plot.scatter(X[y_hc == 4, 0],X[y_hc == 4,1],s=100,c='cyan',label='poor low spenders')
plot.legend()
plot.xlabel('income ($k)')
plot.ylabel('spend score (1-100)')