# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 15:29:18 2021

@author: dl7le
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self, eta=0.001, epochs=50, is_verbose=False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
        
    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        x_1 = np.append(x.copy(), ones, axis=1)
        
        return np.where(self.get_activation(x_1) > 0, 1, -1)
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
    
        return activation
    
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis = 1)
        
        self.w = np.random.rand(X_1.shape[1])

        for e in range(self.epochs):
            error = 0    
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w
            
            error = np.square(y - activation).sum() / 2
            self.list_of_errors.append(error)
            
            if(self.is_verbose):
                print(f'epoch: {e}, weights {self.w}, error: {error}')
            
            
            

            
data = pd.read_csv(r'c:/python/datasetes/iris.data', header=None)
data.head()

data = data.iloc[:100, :].copy()
data

data[4] = data[4].apply(lambda x: 1 if x =='Iris-setosa' else -1)

X = data.iloc[0:100, :-1].values
y = data[4].values

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


perceptron = Perceptron(eta=0.00001,epochs=500, is_verbose=(True))
perceptron.fit(X, y)


y_pred = perceptron.predict(x_test)

list(zip(y_pred, y_test))

plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)




























