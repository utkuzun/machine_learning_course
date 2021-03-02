import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from uu_ml import uu_ml

data1 = scipy.io.loadmat('samples/NN_ex4/ex4data1.mat')
data2 = scipy.io.loadmat('samples/NN_ex4/ex4weights.mat')
X = np.array(data1["X"])          # 5000 samples with 400 parameters(20x20 gray scale) 
y = np.array(data1["y"])          # target as 0-9 digits
#y[y==10] = 0            # convert 10s to 0s


theta1 = np.array(data2["Theta1"])  # take first layer theta 
theta2 = np.array(data2["Theta2"])  #take second layer theta


# input_layer_size  = 400   # 20x20 Input Images of Digits
# hidden_layer_size = 25   # 25 hidden units
# num_labels = np.unique(y)          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)



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

grid = showDigitRows(n, X)

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
lambdaa = 1
num_iter = 1500

ml = uu_ml(X , y, theta1, alpha, lambdaa, num_iter, "logistic regression")
ml.nnComputeCost(X, y, theta1, theta2, lambdaa)

