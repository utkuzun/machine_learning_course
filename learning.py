
import numpy as np
import pandas as pd
from uu_ml import uu_ml


        
X = pd.read_csv('samples/linear gradient/X.csv', header=None, delimiter=",")
y = pd.read_csv('samples/linear gradient/y.csv', header=None, delimiter=",")
theta = np.zeros((X.shape[1],1))
alpha = 0.01
num_iter = 1500
type = "linear regression"

ml = uu_ml(X, y, theta, alpha, num_iter)

[thetares, J] = ml.gradientDescent()

print(J.shape)