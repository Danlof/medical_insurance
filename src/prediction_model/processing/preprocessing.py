import config
import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler

# 1. log transformation ; since we are dealing with  a positively skewned
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self 
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X
    
# 2 Standardization

class Standardization(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
        self.scaler = Standardization()

    def fit(self,X,y=None):
        self.scaler.fit(X[self.variables]) 
        return self
    
    def standardize(self,X):
        X = X.copy()
        X[self.variables] = self.scaler.transform(X[self.variables])
        return X
