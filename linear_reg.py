
import numpy as np
import pandas as pd
from uu_ml import uu_ml
import matplotlib.pyplot as plt



        
X = pd.read_csv('samples/linear gradient/X.csv', header=None, delimiter=",")  # read samples
y = pd.read_csv('samples/linear gradient/y.csv', header=None, delimiter=",")   # read targets 

theta = np.zeros((X.shape[1],1))                #initial parameters with zero values
alpha = 0.01                                    #learning rate
lambdaa = 0                                     # regulariation parameter
num_iter = 1500                                    #number of iterations for gradient descent algroithm
type_learn = "linear regression"
ml = uu_ml(X, y, theta, alpha,lambdaa, num_iter, type_learn)        #initialize ml tool

[thetares, J] = ml.gradientDescent()          #cost and parameter outputs


print(X, y)



## fig ops for gradient descent with linear regression

fig, ax1 = plt.subplots(figsize = (10,4))

ax1.plot(X[1], y, "x", markersize = 5)
ax1.plot(X[1], np.array(X @ thetares))

ax1.set_xlabel('Population of City in 10,000s')
ax1.set_ylabel('Profit in $10,000s')


plt.show()