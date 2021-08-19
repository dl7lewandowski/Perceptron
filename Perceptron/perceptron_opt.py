# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:47:29 2021

@author: dl7le
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Perceptron:
    
    def __init__(self, eta=0.01, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        X_1 = np.append(x.copy(), ones, axis =1)
        return self.__predict(X_1)
        
    
    def __predict(self, x):
        total_stimulation = np.dot(x, self.w)
        y_pred = np.where(total_stimulation > 0, 1, -1)
        return y_pred 
    
    
    def fit(self, X, y):
        
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis = 1)
        
        self.w = np.random.rand(X_1.shape[1])
        self.list_of_errors = []
        
        
        for e in range(self.epochs):
            
            error = 0 
            
            y_pred = self.__predict(X_1)
                
            delta_w = self.eta * np.dot((y - y_pred), X_1)
                
            self.w += delta_w
                
            error = np.count_nonzero(y - y_pred)
                
            self.list_of_errors.append(error)
                
            if(self.is_verbose):
                print('epoch: {}, weigths: {}, errors: {} '.format(e, self.w, error))
                
X = np.array([
    [2, 4,  20],  # 2*2 - 4*4 + 20 =   8 > 0
    [4, 3, -10],  # 2*4 - 4*3 - 10 = -14 < 0
    [5, 6,  13],  # 2*5 - 4*6 + 13 =  -1 < 0
    [5, 4,   8],  # 2*5 - 4*4 + 8 =    2 > 0
    [3, 4,   5],  # 2*3 - 4*4 + 5 =   -5 < 0 
])
 
y = np.array([1, -1, -1, 1, -1])

perceptron = Perceptron(is_verbose=True, eta= 0.01, epochs=100)
perceptron.fit(X, y)

print(perceptron.predict(np.array([[1, 2, 3]]))) 


plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)


# zastosowanie perceptronu w klasyfkikacji kwiatów 


data = pd.read_csv(r'c:/Python/datasetes/iris.data', header = None)
data = data.iloc[:100, :].copy()
data[4] = data[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)

X = data.iloc[0:100, :-1].values
y = data[4].values

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(x_test)
print(list(zip(y_pred, y_test)))

np.count_nonzero(y_pred - y_test)



plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)

