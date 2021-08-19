# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:55:07 2021

@author: dl7le
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Perceptron:
    
    def __init__(self, eta=0.001, epochs=50, is_verbose=False):
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        
        return np.where(self.get_activation(x_1), 1, -1)
    
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
        return activation
    
    def fit(self, X, y):
        
        ones = np.ones((X.shape[0],1))
        x_1 = np.append(X.copy(), ones, axis=1)
        self.w = np.random.rand(x_1.shape[1])
        self.list_of_errors = []
        
        for e in range(self.epochs):
            
            error = 0

            activation = self.get_activation(x_1)
            
            delta_w = self.eta * np.dot(y - activation, x_1)
            
            self.w += delta_w
            
            error = np.square(y - activation).sum() / 2.0
            
            self.list_of_errors.append(error)
            
            if(self.is_verbose):
                print(f'Epoch: {e}, weigths: {self.w}, error: {error}')
                
        
    
diag = pd.read_csv(r'c:/Python/datasetes/breast_cancer.csv')
X = diag[['area_mean', 'area_se', 'texture_mean', 'concavity_worst', 'concavity_mean']]
y = diag['diagnosis']

y = y.apply(lambda x: 1 if x =='M' else -1)


perc = Perceptron(eta=0.00001, epochs=100, is_verbose=(True))
perc.fit(X, y)    
plt.scatter(range(perc.epochs), perc.list_of_errors)
X.shape

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

perc = Perceptron(eta=0.001, epochs=100, is_verbose=(True))
perc.fit(X_std, y)    
plt.scatter(range(perc.epochs), perc.list_of_errors)


X_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

perc = Perceptron(eta=0.001, epochs=100, is_verbose=(True))
perc.fit(X_train, y_train)    
plt.scatter(range(perc.epochs), perc.list_of_errors)


y_pred = perc.predict(x_test)

good = np.count_nonzero(y_pred - y_test)
good
total = len(y_test)
total

result = good * 100 / total
result
