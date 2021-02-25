import numpy as np
import pandas as pd
import math

class uu_ml:
    def __init__(self, X , y, theta, alpha, lambdaa, num_iter, type_learn):

        self.X= X                   # input data
        self.y= y                   # targets
        self.theta= theta           # initial parameters
        self.alpha= alpha           # learning rate
        self.num_iter= num_iter     # number iteration for gradient descent algorithm 
        self.type_learn = type_learn
        self.lambdaa = lambdaa   #regularization parameter

    def computeCost(self):
        if self.type_learn == "linear regression":

            m = len(self.y)              # number of samples
            prediction = self.X @ self.theta    # hypothesis (prediction)
            errors = (prediction - self.y)**2   # sum of errors squares

            J = 0.5/m*np.sum(errors)         # cost (J)
            grad = np.sum((prediction - self.y).values * self.X) / m

            return J, grad.T

        elif self.type_learn == "logistic regression":
            ## this function returns cost and gradient values of logistic regression

            m = len(self.y)              # number of samples
            #n = self.theta.shape[0]      # number of parameters

            # make prediciton 

            hypo = self.X @ self.theta
            prediction = self.sigmoid(hypo)
            # regularization term of cost
            J_reg_term = self.lambdaa/2/m * np.sum(self.theta[1:] ** 2)
            # cost 
            J = -1/m * np.sum(self.y.values * np.log(prediction) + (1 - self.y.values) * np.log(1-prediction)) + J_reg_term

            temp_theta = pd.DataFrame(np.copy(self.theta))
            temp_theta.loc[0] = 0  # parameters j = 1 to n (not included first term)
            grad = (1 / m * np.sum((prediction - self.y.values).values * self.X)) + self.lambdaa / m *temp_theta.T


            return J, grad.T

    def gradientDescent(self):
        m = len(self.y)                   # number of samples
        J = np.zeros((self.num_iter,1))     # J for each iteration

        for i in range(self.num_iter):
                  
            gradient = (self.X @ self.theta - self.y).T @ (self.X)      
            self.theta = self.theta - self.alpha/m*gradient.T
            [J[i], _] = self.computeCost()
        
        return self.theta, J
    


    def normalEqn(self):

        ## excplicit solution of the parameters

        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        return self.theta


    @staticmethod
    def sigmoid(z):

        ## This functions calculates sigmoid of a given variable
        return (1 + np.exp(-1 * z)) ** -1

    @staticmethod
    def featureNormalize(X):
        ## this function normalizes the given variable
        mu = X.mean()
        sigma = X.std()

        X_norm = (X - mu) / sigma
        X_norm[X_norm.isna()] = 1

        return X_norm 

    def predict(self):
        
        hypo = self.X @ self.theta
        print(self.X)
        res = self.sigmoid(hypo)
        print(hypo)

        res[(res >= 0.5)] = 1
        res[(res < 0.5)] = 0

        return res
