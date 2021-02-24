
import numpy as np
import pandas as pd
from uu_ml import uu_ml
import matplotlib.pyplot as plt

#read the data
ex2data1 = pd.read_csv('samples/logistic regression/ex2data1.txt', header=None, delimiter=",")  # read samples


# create high order X data
def mapFeature(Xone, Xtwo):

    degree = 6
    m = len(Xone)
    out = pd.DataFrame(np.ones((m,1)))

    for i in range(1, degree+1):
        for j in range(i+1):
            out[out.shape[1]-1 +1] = (Xone ** (i-j)) * (Xtwo ** j)

    return out

X = mapFeature(ex2data1[0], ex2data1[1])
n = X.shape[1]
theta = pd.DataFrame(np.zeros((n,1)))

#assign y
y = ex2data1[2]

print(X.head())

