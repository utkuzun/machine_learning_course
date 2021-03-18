#################   Andrew NG Machine Learning course ex1 with sklearn #######################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, scale
# ################################################################

# WarmUpExercise

def identical(number):

    return np.eye(number)

# print(identical(7))

# ##################################################################
# Read and plot the data

data1 = pd.read_csv("samples/linear gradient/ex1data1.txt", header=None)
X = np.zeros((data1.shape[0], 1))

X[:, 0] = data1[0].values
y = data1[1].values

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(X, y, "rx", markersize=5)

ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')


# ######################################################################
# Compute cost 
theta = np.array([0, 0]) # and intance of arrays

def computeCost(X, y, theta):

    regr = linear_model.Ridge(alpha = 0, max_iter=1)

    regr.coef_ = np.array([theta[1]])
    regr.intercept_ = theta[0]
    pred = regr.predict(X)


    cost = mean_squared_error(y, pred)
    R2 = r2_score(y, pred)

    print(f"Cost and R^2 are \n{cost}\n{R2} are for an instance of weigths: \n{regr.intercept_}\n{regr.coef_}")

computeCost(X, y, theta)

# ######################################################################
# Compute paramters (weights)

regr = linear_model.Ridge(alpha = 0)
regr.fit(X, y)
pred = regr.predict(X)


cost = mean_squared_error(y, pred)
R2 = r2_score(y, pred)

print(f"Computed gradient descent paramaters are: \n{regr.intercept_}\n{regr.coef_}")
print(f"Cost and R^2 are \n{cost}\n{R2}")

# #######################################################################
# plot fitted line

ax.plot(X, pred, "-b", linewidth=2, label="Linear Regression")
ax.legend()

# plt.show()


# #########################################################################
# Linear Regression for more than one variable variables

data2 = pd.read_csv("samples/linear gradient/ex1data2.txt", header = None)

X2 = data2.values[:, :2]
y2 = data2.values[:, 2]

# Feature normalization 

X2 = scale(X2)

# ######################################################################
# Compute parameters (weights)

regr2 = linear_model.Ridge(alpha = 0)
regr2.fit(X2, y2)
pred = regr2.predict(X2)


cost = mean_squared_error(y2, pred)
R2 = r2_score(y2, pred)

print(f"Computed gradient descent paramaters are: \n{regr2.intercept_}\n{regr2.coef_}")
print(f"Cost and R^2 are \n{cost}\n{R2}")



