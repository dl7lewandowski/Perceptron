# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:02:52 2021

@author: dl7le
"""

import numpy as np

X = np.arange(-25, 25, 1).reshape(10, 5)
X 

ones = np.ones((X.shape[0], 1))
ones

X_1 = np.append(X, ones, axis=1)
X_1


w = np.random.rand(X_1.shape[1])
w

def predict(x, w):
    total_stimulation = np.dot(x, w)
    
    y_pred = 1 if total_stimulation > 0 else -1 
    
    return y_pred
        
predict(X_1[0, ], w)

for x in X_1:
    y_pred = predict(x, w)
    print(y_pred)
    

