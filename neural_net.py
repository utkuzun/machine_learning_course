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
y[y==10] = 0            # convert 10s to 0s

theta1 = np.array(data2["Theta1"])  # take first layer theta 

""" 

theta2 = np.array(data2["Theta2"])  #take second layer theta

temp2 = np.copy(theta2)

theta2[0, :] = temp2[9, :]

theta2[1:, :] = temp2[0: 9, :]

nn_params = np.zeros((np.concatenate((theta1.flatten(), theta2.flatten()), axis = 0).shape[0], 1))
nn_params[:, 0] = np.concatenate((theta1.flatten(), theta2.flatten()), axis = 0)

 """

# input_layer_size  = 400   # 20x20 Input Images of Digits
# hidden_layer_size = 25   # 25 hidden units
# num_labels = np.unique(y)          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)



# show some of the digits in sample
 
def showDigitRows(n, data):  #select n square samples to show

    grid = np.zeros((n*20, n*20))

    for r in range(n):
        for c in range(n):
            random_sample = random.randint(1, data.shape[0])

            grid[c*20: (c+1)*20, r*20: (r+1)*20] = data[random_sample].reshape(20,20).T


    return grid

n= 5

#grid = showDigitRows(n, theta1[:, 1:])  #visualize the hidden layer
grid = showDigitRows(n, X)

fig, ax = plt.subplots()

ax.imshow(grid, extent=[0,20,0,20])
ax.axis("off")
plt.show()


# prepare input paramters

X_copy = np.copy(X)

X = np.concatenate((np.ones((X.shape[0], 1)), X_copy), axis = 1)
# add X0 term 1s to X

# training parameters
alpha = 0.1
lambdaa = 1
num_iter = 1500
n_hidden_layers = 25
params = X.shape[1]

#initialize ml
ml = uu_ml(X , y, theta1, alpha, lambdaa, num_iter, "logistic regression")

# random initiliaze params

theta1_initial = ml.randInitializeWeights(n_hidden_layers , X.shape[1])
theta2_initial = ml.randInitializeWeights(len(np.unique(y)), n_hidden_layers + 1)


# concat the params
nn_initial = np.zeros((np.concatenate((theta1_initial.flatten(), theta2_initial.flatten()), axis = 0).shape[0], 1))
nn_initial[:, 0] = np.concatenate((theta1_initial.flatten(), theta2_initial.flatten()), axis = 0)


#training of the neural network
theta = ml.nn_optimize(X, y, nn_initial, lambdaa, params , n_hidden_layers , len(np.unique(y)))

thetacomp1 = theta[:(X.shape[1] * n_hidden_layers)].reshape(n_hidden_layers, X.shape[1])
thetacomp2 = theta[(X.shape[1]) * n_hidden_layers:].reshape(len(np.unique(y)), n_hidden_layers +1)

[pred, rating] = ml.predictNN(X, y, thetacomp1, thetacomp2)
print(f"NN predicted the first 10 samples as {pred[:10]}. \n All values predicted by %{rating}")


""" def numericalGradientCheck(X, y, nn_initial, lambdaa, params , n_hidden_layers, on):

    n = np.concatenate((theta1.flatten(), theta2.flatten()), axis = 0).shape[0]
    grad = np.zeros((n))

    for i in range(15):

        epsilon = 0.0001

        thetaPlus1 = np.copy(theta1)
        thetaPlus2 = np.copy(theta2)

        thetaPlus1 = thetaPlus1 + epsilon
        thetaPlus2 = thetaPlus2 + epsilon

        thetaMinus1 = thetaPlus1 -2 * epsilon
        thetaMinus2 = thetaPlus2 -2 * epsilon

        
        [J1, _] = ml.nnComputeCost(X, y, thetaPlus1, thetaPlus2, lambdaa)
        [J2, _] = ml.nnComputeCost(X, y, thetaMinus1, thetaMinus2, lambdaa)

        grad[i] = (J1 - J2) / 2 /epsilon

    return (grad[:15]) """

