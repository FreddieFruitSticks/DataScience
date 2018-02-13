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

number_of_successes = [0]*(ads+1)
number_of_failures = [0]*(ads+1)
ads_played = []
total_ads_selected = [0]*(ads+1)

for i in range(1, 10000):
    sample = [0]*(ads + 1)
    for j in range(0, ads+1):
        sample[j] = random.betavariate(number_of_successes[j] + 1, number_of_failures[j] + 1)
    sample_to_play = sample.index(max(sample))
    
    if data_frame.values[i, sample_to_play] == 1:
        number_of_successes[sample_to_play] += 1
    else:
        number_of_failures[sample_to_play] += 1
    ads_played.append(sample_to_play)
    total_ads_selected[sample_to_play] += 1
        

plot.hist(ads_played)