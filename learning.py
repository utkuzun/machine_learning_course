
import numpy as np
import pandas as pd
from uu_ml import uu_ml


        
X = pd.read_csv('samples/linear gradient/X.csv', header=None, delimiter=",")  # read samples
y = pd.read_csv('samples/linear gradient/y.csv', header=None, delimiter=",")   # read targets 

theta = np.zeros((X.shape[1],1))                #initial parameters with zero values
alpha = 0.01                                    #learning rate
num_iter = 1500                                    #number of iterations for gradient descent algroithm

ml = uu_ml(X, y, theta, alpha, num_iter)        #initialize ml tool

[thetares, J] = ml.gradientDescent()

print(J, thetares)