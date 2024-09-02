from prediction_model import config
import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler

# 1. log transformation ; since we are dealing with  a positively skewned
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,epsilon=1e-8):
        self.variables = variables
        self.epsilon = epsilon
    
    def fit(self,X,y=None):
        return self 
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col]+ self.epsilon)
        return X
    
# 2 Standardization

class Standardization(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
        self.scaler = StandardScaler() 

    def fit(self,X,y=None):
        self.scaler.fit(X[self.variables]) 
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.variables] = self.scaler.transform(X[self.variables])
        return X
    
# 3 encoding :

class CustomEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        """This is constructor"""
        self.variables = variables
        self.label_dict = {} # stores the mapping of categorical values to numerical
    
    # methods
    # a. fit method
    def fit(self,X,y=None):
        """This method learns unique values in each categorical variable and
        creates a mapping from these values to integers"""
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending = True).index
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    # b. transform method
    def transform(self,X):
        """This method applies the learned mappings from the fit method
         to convert the categorical variables into numerical format"""
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X
    
    # c. fit transform method 
    def fit_transform(self,X,y=None):
        """returns the transformed dataframe"""
        return self.fit(X,y).transform(X)