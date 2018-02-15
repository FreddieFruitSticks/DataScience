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

# Steming is having only one version of the word, eg. Loved, loving -> love
porterStemmer = PorterStemmer()

corpus = []
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    
    #stop words are useless words like the, a, this, etc
    words = [porterStemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(words)
    corpus.append(review)
    
    
#Creating bag of words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

sc_indep = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#scale_X_train = sc_indep.fit_transform(X_train)
#scale_X_test = sc_indep.transform(X_test)

classifier = RandomForestClassifier(criterion = 'entropy', n_estimators=500)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)