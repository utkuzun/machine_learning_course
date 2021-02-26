import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io




data1 = scipy.io.loadmat('samples/NN_ex3/ex3data1.mat')
X = np.array(data1["X"])          # 5000 samples with 400 parameters(20x20 gray scale) 
y = np.array(data1["y"])          # target as 0-9 digits

y[y==10] = 0            # convert 10s to 0s


# show some of the digits in sample

def showDigitRows(n, data):

    grid = np.zeros((n*20, n*20))
    

    for r in range(n):
        for c in range(n):
            random_sample = random.randint(1, data.shape[1])

            grid[r*20: (r+1)*20, c*20: (c+1)*20] = data[random_sample].reshape(20,20)


    return grid

n= 5

grid = showDigitRows(15, X)

fig, ax = plt.subplots()


ax.imshow(grid)
ax.axis("off")
plt.show()