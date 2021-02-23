import numpy as np
import pandas as pd

class uu_ml:
    def __init__(self,X , y, theta, alpha, num_iter):

        self.X= X
        self.y= y
        self.theta= theta
        self.alpha= alpha
        self.num_iter= num_iter

    def computeCost(self, type):
        if type == "linear regression":

            m = len(self.y)
            prediction = self.X @ self.theta
            errors = (prediction - self.y)**2

            return 0.5/m*np.sum(errors)

    def gradientDescent(self):
        m = len(self.y)
        J = np.zeros((self.num_iter,1))

        for i in range(self.num_iter):
            gradient = (self.X @ self.theta - self.y).T @ (self.X)
            self.theta = self.theta - self.alpha/m*gradient.T
            J[i] = self.computeCost("linear regression")
        return self.theta,J
    
    def featureNormalize(self):

        mu = self.X.mean()
        sigma = self.X.std()

        X_norm = (self.X - mu) / sigma
        X_norm[X_norm.isna()] = 1

        return X_norm 

    def normalEqn(self):

        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.theta