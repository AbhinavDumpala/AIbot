import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import sys


def model():
    try:
        datainput = pd.read_csv("Emotions.csv", encoding='cp1252')
        
        y = datainput['Emotion']
        del datainput['Emotion']
    
        x_train, x_test, y_train, y_test = train_test_split(datainput, y, test_size=0.3)
       
        tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)  # TF-IDF
        
        print("Start Building Model NN Classifier")
        clf_nn = Pipeline([('NNTF_IDF', tfidf), ('nn_clf', MLPClassifier())])
        print(x_train, x_test)
        clf_nn.fit(x_train, y_train)
        predict = classify3.predict(x_test)
        acuracy = metrics.accuracy_score(y_test, predicted) * 100
    

    except Exception as e:
        print("Error=",  e)

if __name__ == '__main__':
    model()
