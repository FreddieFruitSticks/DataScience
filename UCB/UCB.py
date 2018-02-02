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
#ads_chosen = []
#for i in range(0,10000):
#    ad = random.randrange(ads + 1)
#    value = data_frame.values[i,ad]
#    reward = reward + value
#    ads_chosen.append(ad)
#    
#plot.hist(ads_chosen)

ads_chosen = [1]*(ads+1)
sum_rewards = [0]*(ads+1)
UCB_score = [1e40]*(ads+1)
ads_selected = []

ad = UCB_score.index(max(UCB_score))
print(ads_chosen)
print(sum_rewards)
print(UCB_score)

for i in range(1, 10000):
    ads_chosen[ad] += 1
    ads_selected.append(ad)
    sum_rewards[ad] += data_frame.values[i-1, ad]
    for j in range(0, ads+1):
        UCB_score[j] = sum_rewards[j]/ads_chosen[j] + math.sqrt(2*math.log(i)/ads_chosen[j])
    ad = UCB_score.index(max(UCB_score))

print(ads_chosen)

print(UCB_score)

plot.hist(ads_selected)