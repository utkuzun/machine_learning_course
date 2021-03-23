#################   Andrew NG Machine Learning course ex4 with sklearn #######################

# imports
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, normalize
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


# #####################################################################
# Polynomail Regression

poly = PolynomialFeatures(
        include_bias= True,
        interaction_only= False,
        degree = 8
    )


X_poly = poly.fit_transform(X)

def fetnormalize(data_to_norm):

    mu = []
    sigma = []

    for i in range(1, data_to_norm.shape[1], 1):

        mean = np.mean(data_to_norm[:, i])

        data_to_norm[:, i] = (data_to_norm[:, i] - mean)
        std = np.std(data_to_norm[:, i])

        data_to_norm[:, i] = data_to_norm[:, i] / std

        mu.append(mean)
        sigma.append(std)

    return data_to_norm, np.array(mu), np.array(sigma)

X_poly, mu, sigma = fetnormalize(X_poly)

poly_reg = Ridge(alpha= 0, fit_intercept= False, solver="lsqr")
poly_reg.fit(X_poly, y)

X_plot = np.arange(np.min(X) - 10, np.max(X) +5, 0.05)
X_plot_samp = poly.fit_transform(X_plot.reshape(-1, 1))

X_plot_samp[:,1:] = X_plot_samp[:,1:]  - mu
X_plot_samp[:,1:]  = X_plot_samp[:,1:]  / sigma

pred_plot = poly_reg.predict(X_plot_samp)

ax.plot(X_plot, pred_plot, "--y", label="Polynomial Fit")
ax.set_title(f"Polynomial and Linear Regression with no regularization")


# #############################################################################################
# Learning curve of polynomail regression

X_val_poly = poly.fit_transform(X_val.reshape(-1, 1))
X_val_poly[:,1:] = X_val_poly[:,1:]  - mu
X_val_poly[:,1:]  = X_val_poly[:,1:]  / sigma

fig, ax3 = plt.subplots(1, 1, figsize= (6, 4))

names = ["Train", "Cross Validation"]

pred_poly_val = poly_reg.predict(X_val_poly)
pred_poly = poly_reg.predict(X_poly)

err_t, err_val = learningCurve(X_poly, pred_poly, X_val_poly, pred_poly_val, poly_reg)

ax3.plot(range(1, X.shape[0], 1), err_t, "-r", label=names[0])
ax3.plot(range(1, X.shape[0], 1), err_val, "-b", label=names[1])

ax3.set_xlabel("Number of training samples")
ax3.set_ylabel("Error")

ax3.legend(ax3.get_lines(), names, ncol =2, loc ="upper right")

# ########################################################################################
# Cross Validation with lambdaas

lambdaa = np.logspace(-2, 1, 9)

def plot_CVlambda(X, y, X_val, y_val, lambdaas):

    err_t = []
    err_val = []

    for lambdaa in lambdaas:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn")

            poly_cv = Ridge(alpha = lambdaa)
            poly_cv.fit(X, y)

        pred = poly_cv.predict(X)
        pred_val = poly_cv.predict(X_val)

        err_t.append(mean_squared_error(y, pred))
        err_val.append(mean_squared_error(y_val, pred_val))

    return np.array(err_t), np.array(err_val)


err_t_cv, err_val_cv = plot_CVlambda(X_poly, y, X_val_poly, y_val, lambdaa) 

fig, ax4 = plt.subplots(figsize= (6, 4))

ax4.plot(lambdaa, err_t_cv, "-r", label = "Train")
ax4.plot(lambdaa, err_val_cv, "--b", label ="Cross Validation")

ax4.set_xlabel("Lambda values")
ax4.set_ylabel("Error")

ax4.legend(loc = "upper right", ncol=1)

print(err_t_cv, err_val_cv)

plt.show()



