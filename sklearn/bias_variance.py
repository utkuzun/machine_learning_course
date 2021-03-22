#################   Andrew NG Machine Learning course ex4 with sklearn #######################

# imports
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
import scipy.io
import joblib


# ########################################################################################
# Read and assign data

filename = "samples/ex5_biasvsvariance/ex5data1.mat"
data = scipy.io.loadmat(filename)

# All the train, validation and test splits are given. 
# Alternatively model_selection.train_test_split or ShuffleSplit can be used

X = np.array(data["X"])
X_test = np.array(data["Xtest"])
X_val = np.array(data["Xval"])

y = np.array(data["y"]).flatten()
y_test = np.array(data["ytest"]).flatten()
y_val = np.array(data["yval"]).flatten()




# write a saved model name in order to execute created model
# Leave it as it is and make new model below

model_name = ".sav"
new_model_name = ".sav"

# #############################################################################################
# # Plot the data

fig, ax = plt.subplots(figsize= (6, 4))

ax.plot(X, y, "rx", markersize=5)

ax.set_xlabel("Change in water level (x)") 
ax.set_ylabel("Water flowing out of the dam (y)")


# ############################################################################################
# # Fit linear regression with no reg

lin = Ridge(alpha= 0)
lin.fit(X, y)

ax.plot(X, lin.predict(X), "-b", linewidth = 2)


# #############################################################################################
# Learning curve



fig, ax2 = plt.subplots(1, 1, figsize= (6, 4))

names = ["Train", "Cross Validation"]



def learningCurve(X, y, X_val, y_val, mlp):

    error_train = []
    error_val = []

    for i in range(1, X.shape[0], 1):

        

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore" ,
                module= "sklearn")

            mlp.fit(X[:i, :], y[:i])

        pred = mlp.predict(X[:i, :])
        pred_val = mlp.predict(X_val)

        error_train.append(mean_squared_error(y[:i], pred))
        error_val.append(mean_squared_error(y_val, pred_val))

    return error_train, error_val

mlp = Ridge(alpha=0)

err_t, err_val = learningCurve(X, y, X_val, y_val, mlp)


ax2.plot(range(1, X.shape[0], 1), err_t, "-r", label=names[0])
ax2.plot(range(1, X.shape[0], 1), err_val, "-b", label=names[1])

ax2.set_xlabel("Number of training samples")
ax2.set_ylabel("Error")

ax2.legend(ax2.get_lines(), names, ncol =2, loc ="upper right")


plt.show()



