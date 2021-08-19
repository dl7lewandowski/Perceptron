# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:47:17 2021

@author: dl7le
"""

import numpy as np
data = np.array([[10, 7, 4],
                 [3, 2, 1]])
data

data.mean()

data.mean(axis=0)
data.mean(axis=1)


np.mean(data)

np.average(data)

np.average(data, axis=0)
np.average(data, axis=1)

np.average(data, axis=1, weights=[0,1,1])

np.average(data, axis=1, weights=[2,3,5])

np.var(data)

np.var(data, axis=0)
np.var(data, axis=1)

np.std(data)

np.std(data, axis=0)
np.std(data, axis=1)
