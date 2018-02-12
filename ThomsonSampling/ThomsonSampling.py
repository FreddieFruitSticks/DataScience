# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plot
import random, math

data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv"""
prep = DataPreparation(data_file)
indep_vars, dep_vars, data_frame = prep.prepare_data_frame([2,4], [4,5])

users = 10000
ads = 9
reward = 0