# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/freddie/git/DataScience/dataPreProcess/')
from data_process import DataPreparation
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import pandas, re, nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


#cleaning text
data_file="""/home/freddie/git/DataScience/data/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv"""
dataset = pandas.read_csv(data_file, delimiter = '\t', quoting = 3)