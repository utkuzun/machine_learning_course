import numpy as np
import pandas as pd

class uu_ml:
    def __init__(self, X , y, theta, alpha, num_iter, type_learn):

        self.X= X                   # input data
        self.y= y                   # targets
        self.theta= theta           # initial parameters
        self.alpha= alpha           # learning rate
        self.num_iter= num_iter     # number iteration for gradient descent algorithm 
        self.type_learn = type_learn

    def computeCost(self):
        if self.type_learn == "linear regression":

            m = len(self.y)              # number of samples
            prediction = self.X @ self.theta    # hypothesis (prediction)
            errors = (prediction - self.y)**2   # sum of errors squares

            return 0.5/m*np.sum(errors)         # cost (J)

    def gradientDescent(self):
        m = len(self.y)                   # number of samples
        J = np.zeros((self.num_iter,1))     # J for each iteration

        for i in range(self.num_iter):
                  
            gradient = (self.X @ self.theta - self.y).T @ (self.X)      
            self.theta = self.theta - self.alpha/m*gradient.T
            J[i] = self.computeCost()
        
        return self.theta, J
    
    def featureNormalize(self):

        mu = self.X.mean()
        sigma = self.X.std()

        X_norm = (self.X - mu) / sigma
        X_norm[X_norm.isna()] = 1

        return X_norm 

    def normalEqn(self):

        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.theta


    @staticmethod
    def sigmiod(z):

        ## This functions calculates sigmoid of a given variable
        return (1 + np.exp(-1 * z)) ** -1