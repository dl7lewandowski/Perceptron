# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:09:43 2021

@author: dl7le
"""

import numpy as np

X = np.arange(1, 26, 1).reshape(5,5) 
X
Ones = np.ones(X.shape)
Ones

np.dot(X, Ones)


diag = np.zeros(X.shape)
diag

np.fill_diagonal(diag, 1)
diag

np.dot(X, diag)

np.where(X > 10, 1, 0)
np.where(X%2==0, 1, 0)
np.where(X%2==0, X, X + 1)

X_bis = np.where(X > 10 , X*2, 0)
X_bis

np.count_nonzero(X_bis)


x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100], [200]])

np.append(x, y, axis=1)


x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100, 200, 300]])
np.append(x, y, axis = 0)


np.append(x, x, axis=0)











