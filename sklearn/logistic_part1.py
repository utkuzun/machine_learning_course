#################   Andrew NG Machine Learning course ex2 with sklearn #######################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, scale


# ##################################################################
# Read data 1

data = pd.read_csv("samples/logistic regression/ex2data1.txt", header=None)

X = data.values[:, :2]
y = data[2].values


# ###################################################################
# Compute parameters with no regularization

# Execute with no regularizaton with a very high C value and lbfgs solver and ridge regularization

log_reg = LogisticRegression(C = 100000, penalty= "l2", solver="lbfgs")

log_reg.fit(X, y)

pred = log_reg.predict(X)

print(f" Parameters are :\n{log_reg.intercept_}\n{np.ravel(log_reg.coef_)}")
print(f"Accuracy for predictions are {np.mean(pred == y) * 100}")

# ####################################################################
# Plot the actual data

fig, ax = plt.subplots(figsize=(8, 5))

colors ="br"

for i, color in zip(log_reg.classes_, colors):

    idx = np.where(y == i)

    ax.plot(X[idx, 0], X[idx, 1], f"{color}o", markersize=5)

# ax.legend()
ax.set_xlabel("Exam 1 score")
ax.set_ylabel("Exam 2 score")


# Create the mesh
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predicts
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Contour the results

ax.contourf(xx, yy, Z, cmap= plt.cm.Paired)
plt.show()



