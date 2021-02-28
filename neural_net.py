import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from uu_ml import uu_ml




data1 = scipy.io.loadmat('samples/NN_ex3/ex3data1.mat')
X = np.array(data1["X"])          # 5000 samples with 400 parameters(20x20 gray scale) 
y = np.array(data1["y"])          # target as 0-9 digits

y[y==10] = 0            # convert 10s to 0s



# show some of the digits in sample
""" 
def showDigitRows(n, data):  #select n square samples to show

    grid = np.zeros((n*20, n*20))

    for r in range(n):
        for c in range(n):
            random_sample = random.randint(1, data.shape[0])

            grid[c*20: (c+1)*20, r*20: (r+1)*20] = data[random_sample].reshape(20,20).T


    return grid

n= 5

grid = showDigitRows(10, X)

fig, ax = plt.subplots()

ax.imshow(grid, extent=[0,20,0,20])
ax.axis("off")
plt.show()

 """
# prepare input paramters

X_copy = np.copy(X)

X = np.concatenate((np.ones((X.shape[0], 1)), X_copy), axis = 1)

theta_initial = np.zeros((X.shape[1], 1))
alpha = 0.1
lambdaa = 0.1
num_iter = 1500

ml = uu_ml(X, y, theta_initial, alpha, lambdaa, num_iter, "logistic regression")
all_theta = ml.oneVsAll()
[predictions, accuracy] = ml.predictOneVsAll(X, y, all_theta)
print(f" Samples are predicted with %{accuracy}accuracy")



# check for logistic regresiion cost and grad function for temporary created values
""" 
X_t = np.array([[1, 0.1, 0.6, 1.1],
               [1, 0.2, 0.7, 1.2],
               [1, 0.3, 0.8, 1.3],
               [1, 0.4, 0.9, 1.4],
               [1, 0.5, 1, 1.5]])

y_t = np.array([[1,0,1,0,1]]).T


theta_t = np.array([[-2], [-1], [1], [2]])

lambdaa_t = 3

ml = uu_ml(X_t , y_t, theta_t, 0, lambdaa_t, 1500, "logistic regression")

print(ml.computeCost())

 """