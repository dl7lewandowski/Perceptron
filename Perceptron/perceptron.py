# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:09:44 2021

@author: dl7le
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self, eta=0.01, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
    
    def predict(self, x):
        total_stimulation = np.dot(x, self.w)
        y_pred = 1 if total_stimulation > 0 else -1 
        return y_pred 
    
    
    def fit(self, X, y):
        
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis = 1)
        
        self.w = np.random.rand(X_1.shape[1])
        self.list_of_errors = []
        
        
        for e in range(self.epochs):
            error = 0 
            for x, y_target in zip(X_1, y):
                
                y_pred = self.predict(x)
                
                delta_w = self.eta * (y_target - y_pred) * x
                
                self.w += delta_w
                
                error += 1 if y_target != y_pred else 0 
                
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

perceptron = Perceptron(is_verbose=True, eta= 0.1)
perceptron.fit(X, y)

print(perceptron.predict(np.array([[1, 2, 3, 1]]))) 
print(perceptron.predict(np.array([[3, 3, 3, 1]]))) 



plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)



