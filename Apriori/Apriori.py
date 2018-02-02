import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plot
import pandas
from apyori import apriori

data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv"""
prep = DataPreparation(data_file)
data_frame = pandas.read_csv(data_file, header=None)
transactions = []

for i in range(0,7501):
    transactions.append([str(data_frame.values[i,j]) for j in range(0,20)])

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

results = list(rules)