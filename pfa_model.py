from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

class pfa_model:
    
    def __init__(self, PATH='Crop_recommendation.csv'):
        self.PATH = PATH
        df = pd.read_csv(PATH)
        features = df[['temperature', 'humidity']]
        target = df['label']
        labels = df['label']
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
        self.DecisionTree = ""

    
    def Train( self, mcriterion="entropy", mdepth=5, mrandom_state=2):
        self.DecisionTree = DecisionTreeClassifier(criterion=mcriterion,random_state=mrandom_state,max_depth=mdepth)
        self.DecisionTree.fit(self.Xtrain, self.Ytrain)
        return self.DecisionTree
    
    def test_function(self , x_test ):
        return self.DecisionTree.predict(x_test)
    
    def metrics(self , x_test, y_test):
        predicted_values = self.test_function(x_test)
        print(classification_report(y_test,predicted_values))
        return metrics.accuracy_score(y_test, predicted_values)
    
    
    

