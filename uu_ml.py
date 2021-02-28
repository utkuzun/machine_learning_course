import numpy as np
import pandas as pd
import math
from scipy.optimize import fmin_bfgs 
from scipy.optimize import leastsq 

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
            grad = np.sum((prediction - self.y) * self.X) / m

            return J, grad.T

        elif self.type_learn == "logistic regression":
            ## this function returns cost and gradient values of logistic regression

            m = len(self.y)              # number of samples
            #n = self.theta.shape[0]      # number of parameters

            # make prediciton 

            hypo = self.X @ self.theta
            prediction = self.sigmoid(hypo)
            # regularization term of cost
            J_reg_term = self.lambdaa/2/m * np.sum(self.theta[1:, 0] ** 2)

            # cost 
            J = -1/m * np.sum(self.y * np.log(prediction) + (1 - self.y) * np.log(1-prediction), axis=0) + J_reg_term
            temp_theta = np.copy(self.theta)
            temp_theta[0] = 0  # parameters j = 1 to n (not included first term)
            grad = 1 / m * np.sum((prediction - self.y) * self.X, axis=0) + self.lambdaa / m * temp_theta.T


            return J, grad.T.flatten()


    def oneVsAll(self):

        _, n = self.X.shape
        labels = np.unique(self.y)
        all_theta = np.zeros((labels.shape[0], n))
        y = np.copy(self.y)

        def objectiveFunc(theta):
            self.theta[:, 0] = np.copy(theta)
            [J, _] = self.computeCost()
            

            return J

        def gradFunc(theta):
            self.theta[:, 0] = np.copy(theta)
            [_, grad] = self.computeCost()

            return grad

        for label in labels:
            self.y = np.array([[1 if y[i] == label else 0] for i in range(len(y))])
            theta = fmin_bfgs(objectiveFunc, self.theta, fprime= gradFunc)

            all_theta[label, :] = theta

        return all_theta


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
    def predictOneVsAll(X, y, theta):
        pred = np.argmax(X @ theta.T, axis = 1)
        rating = np.array([[1 if y[i] == pred[i] else 0] for i in range(len(y))])
        return pred, np.mean(rating) * 100
        


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
        res = self.sigmoid(hypo)

        res[(res >= 0.5)] = 1
        res[(res < 0.5)] = 0

        return res
