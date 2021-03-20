#################   Andrew NG Machine Learning course ex4 with sklearn #######################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
import scipy.io
import joblib

# ########################################################################################

# Read and assign data

filename = "samples/NN_ex4/ex4data1.mat"
data = scipy.io.loadmat(filename)

X = np.array(data["X"])
y = np.array(data["y"]).flatten()

y[y == 10] = 0

# write a saved model name in order to execute created model
# Leave it as it is and make new model below

model_name = "neural_net_digit_reg.sav"
new_model_name = "neural_net_digit_reg_relu.sav"


# labelize the targer values 

# lb = LabelBinarizer()
# lb.fit(y)
# y = lb.transform(y)

# ########################################################################################
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

    ax.set_title(f"Sample {sample_place} is below predicted as {neural_net_digit_reg.predict(sample_pred.reshape(1,-1))}")




n= 10

grid = plot_numbers(n, X)
fig, ax = plt.subplots()

ax.imshow(grid, extent=[0,20,0,20])
ax.axis("off")


# ##############################################################################
# predict and evaluate for each label
model_saved = False

try:
    neural_net_digit_reg = joblib.load(new_model_name)
    model_saved = True
except:
    print("There is no file for this model !!!")

if not model_saved:
    neural_net_digit_reg = MLPClassifier(
        alpha= 1,
        solver="lbfgs",
        activation = "logistic",
        hidden_layer_sizes=(25,),
        warm_start= False,
        max_iter=50
        )

    neural_net_digit_reg.fit(X, y)

    joblib.dump(neural_net_digit_reg, new_model_name)

for i in range(neural_net_digit_reg.n_outputs_):

    idx = np.where(y == i)

    pred = neural_net_digit_reg.predict(X[idx])

    print(f"Class '{i}' predicted with accuracy of {np.mean(y[idx] == pred) * 100}%")


random_sample = random.randint(0, X.shape[0] -1)

plot_one_sample(random_sample, X)

# ##############################################################################
# display a random hidden layers

a1 = np.array(neural_net_digit_reg.coefs_)[0]
a2 = np.array(neural_net_digit_reg.coefs_)[1]

print(a1.shape)
fig, ax2 = plt.subplots()


grid_hidden = plot_numbers(5, a1.T)

ax2.imshow(grid_hidden, extent=[0,20,0,20])
ax2.axis("off")
plt.show()