
import numpy as np
import pandas as pd
from uu_ml import uu_ml
import matplotlib.pyplot as plt

#read the data
ex2data1 = pd.read_csv('samples/logistic regression/ex2data1.txt', header=None, delimiter=",")  # read samples
ex2data2 = pd.read_csv('samples/logistic regression/ex2data2.txt', header=None, delimiter=",")  # read samples


# create high order X data
def mapFeature(Xone, Xtwo):

    degree = 6
    m = len(Xone)
    out = np.ones((m,28))
    indice = 1

    for i in range(1, degree+1):
        for j in range(i+1):
            out[:, indice] = (Xone ** (i-j)) * (Xtwo ** j)
            indice+=1
    return out

X = mapFeature(ex2data2[0].values, ex2data2[1].values)


""" 
X = np.ones((ex2data1.shape[0],3))
 
X[: ,1] = ex2data1[0].values
X[:, 2] = ex2data1[1].values

 """
print(X.shape)
n = X.shape[1]
theta = np.ones((n, 1))
#theta = np.array([[-0.00132686887360888], [0.0104313829580538], [0.000559387028960758]]) #for manual array
#assign y
y = np.zeros((ex2data2.shape[0],1))
y[:, 0] = ex2data2[2].values


# init ml 
lambdaa = 10
alpha = 0,1
type_learn = "logistic regression"
num_iter = 1500

ml = uu_ml(X, y, theta, alpha, lambdaa, num_iter, type_learn)        #initialize ml tool


# visualize data
""" 
print(ex2data1.head())

fig, ax1 = plt.subplots(figsize = (6,6))

ax1.plot(ex2data1.loc[(ex2data1[2] == 1), [0]], ex2data1.loc[(ex2data1[2] == 1), [1]], "+", label="admitted" )
ax1.plot(ex2data1.loc[(ex2data1[2] == 0), [0]], ex2data1.loc[(ex2data1[2] == 0), [1]], "o", label="rejected" )

ax1.legend()
ax1.set_xlabel("Exam 1 score")
ax1.set_ylabel("Exam 2 score")

plt.show()

 """

[J, grad] = ml.computeCost()                # obtain for one theta grad and cost

print(f" Cost at initial try of theta\n J=\n {J},\n grad is {grad}")

