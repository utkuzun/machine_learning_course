#################   Andrew NG Machine Learning course ex2 with sklearn #######################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# ##################################################################
# Read data 2

data = pd.read_csv("samples/logistic regression/ex2data2.txt", header=None)

X_raw = data.values[:, :2]
y = data[2].values


# ###################################################################
# Compute parameters with regularization

# Map high order features for X samples

poly = PolynomialFeatures(degree =6)

X = poly.fit_transform(X_raw)


# Execute with no regularizaton with a very high C value and lbfgs solver and ridge regularization

log_reg = LogisticRegression(C = 1, penalty= "l2", solver="lbfgs")

log_reg.fit(X, y)

pred = log_reg.predict(X)

print(f" Parameters are :\n{log_reg.intercept_}\n{np.ravel(log_reg.coef_)}")
print(f"Accuracy for predictions are {np.mean(pred == y) * 100}")


# # ####################################################################
# Plot actual data

fig, ax = plt.subplots(figsize = (8,6))

colors = "ry"

for i, color in zip(log_reg.classes_, colors):

    idx = np.where(y == i)

    ax.plot(X_raw[idx, 0], X_raw[idx, 1], f"{color}o")

ax.set_xlabel("Michrochip Test 1")
ax.set_ylabel("Michrochip Test 2")


# Plot a mesh of results

# Create grid
h = 0.02 # step size for mesh

xmin, xmax = np.min(X_raw[:, 0]) -1, np.max(X_raw[:, 0]) +1 
ymin, ymax = np.min(X_raw[:, 1]) -1, np.max(X_raw[:, 1]) +1
xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

# Make predictions

X_for_Z = np.c_[xx.ravel(), yy.ravel()]

Z = log_reg.predict(poly.fit_transform(X_for_Z))
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)


plt.show()
