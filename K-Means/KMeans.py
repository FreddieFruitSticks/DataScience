# -*- coding: utf-8 -*-

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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv"""
 
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([2,4], [4,5])
X = data_frame.iloc[:,[3,4]].values

#draw information loss graph to determine best number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plot.plot(range(1,11), wcss)
plot.show()

#cluster data
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10,max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualise
plot.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0,1],s=100,c='red',label='Schnoep')
plot.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1,1],s=100,c='blue',label='moderate')
plot.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2,1],s=100,c='green',label='housewives')
plot.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3,1],s=100,c='magenta',label='reckless spenders')
plot.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4,1],s=100,c='cyan',label='poor low spenders')
plot.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow', label='Centroids')
plot.legend()
plot.xlabel('income ($k)')
plot.ylabel('spend score (1-100)')