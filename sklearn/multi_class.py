#################   Andrew NG Machine Learning course ex3 with sklearn #######################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
import scipy.io
import joblib

# ########################################################################################

#Read and assign data

filename = "samples/NN_ex3/ex3data1.mat"
data = scipy.io.loadmat(filename)

X = np.array(data["X"])
y = np.array(data["y"]).flatten()

y[y == 10] = 0


# labelize the targer values 

# lb = LabelBinarizer()
# lb.fit(y)
# y = lb.transform(y)


# Plot some selected binary values

def plot_numbers(squares, data):

    dim_size = 20
    grid = np.zeros((squares * dim_size, squares * dim_size))

    for row in range(squares):
        for col in range(squares):

            random_sample = random.randint(0, data.shape[0] -1)

            grid[row * 20: (row + 1) * 20, col * 20: (col + 1) * 20,] = X[random_sample, :].reshape(dim_size, dim_size).T

    return grid

def plot_one_sample(sample_place, samples):

    grid = samples[sample_place, :].reshape(20, 20).T

    _, ax = plt.subplots(figsize = (5, 5))

    ax.imshow(grid, extent = [0, 20, 0, 20])
    ax.axis("off")

    sample_pred = np.zeros((1, samples.shape[1]))
    sample_pred = samples[sample_place, :]

    ax.set_title(f"Sample {sample_place} is below predicted as {log_reg.predict(sample_pred.reshape(1,-1))}")

    plt.show()



n= 10

grid = plot_numbers(n, X)
fig, ax = plt.subplots()

ax.imshow(grid, extent=[0,20,0,20])
ax.axis("off")
# plt.show()

# ##############################################################################
# predict and evaluate for each label
model_saved = False

try:
    log_reg = joblib.load("multiclass.sav")
    model_saved = True
except:
    print("There is no file for this model !!!")

if not model_saved:
    log_reg = LogisticRegression(C= 1, multi_class= "multinomial", solver="lbfgs", penalty= "l2")

    log_reg.fit(X, y)

    joblib.dump(log_reg, "multiclass.sav")

for i in log_reg.classes_:

    idx = np.where(y == i)

    pred = log_reg.predict(X[idx])

    print(f"Class '{i}' predicted with accuracy of {np.mean(y[idx] == pred) * 100}%")


random_sample = random.randint(0, X.shape[0] -1)

plot_one_sample(random_sample, X)