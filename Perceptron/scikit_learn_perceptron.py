# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:08:36 2021

@author: dl7le
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


data = pd.read_csv(r'c:/Python/datasetes/breast_cancer.csv')

X = data[['area_mean', 'area_se', 'texture_mean', 'concavity_worst', 'concavity_mean']]
X

y = data['diagnosis']
y = y.apply(lambda x: 1 if x == 'M' else 0)
y

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)
X_std

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

perc = Perceptron(eta0=0.01, max_iter=100)

perc.fit(X_train, y_train)

y_pred = perc.predict(X_test)

good = y_test[y_test == y_pred].count()
good
total = y_test.count()

result = good * 100 / total
result

perc.coef_
