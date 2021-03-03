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


        # def resFunc(theta):
        #     self.theta[:, 0] = np.copy(theta)
        #     hypo = self.X @ self.theta
        #     prediction = self.sigmoid(hypo)
        #     err = (self.y - prediction)

        #     return err

        for label in labels:
            self.y = np.array([[1 if y[i] == label else 0] for i in range(len(y))])
            theta = fmin_bfgs(objectiveFunc, self.theta, fprime= gradFunc, maxiter= 100)
            # calculate theta with bfgs optimization function

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
    def nnComputeCost(X, y, theta1, theta2, lambdaa):
        
        def sigmoid(z):

        ## This functions calculates sigmoid of a given variable
            return (1 + np.exp(-1 * z)) ** -1

        def sigmoidGradient(z):
            return sigmoid(z) * (1 - sigmoid(z))

        m = len(y)
        
        z2 = theta1 @ X.T
        a2 = sigmoid(z2)
        a2 = np.concatenate((np.ones((a2.shape[1], 1)), a2.T), axis = 1)
        a3 = sigmoid(theta2 @ a2.T)

        labels = np.unique(y)
        
        J = 0
        theta1_grad = np.zeros_like(theta1)
        theta2_grad = np.zeros_like(theta2)
        pred = np.zeros((a3.shape[1], 1))

        for label in labels:
            pred[:, 0] = a3[label, :]
            c = np.array([[1 if y[i] == label else 0] for i in range(len(y))])
            J = J - 1/m * np.sum(c * np.log(pred) + (1 - c) * np.log(1 - pred), axis = 0)
            
        temp1 = np.copy(theta1)
        temp2 = np.copy(theta2)

        temp1[:,0] = 0
        temp2[:,0] = 0

        J_reg_term = lambdaa/2/m * (np.sum(temp1.flatten() ** 2) + np.sum(temp2.flatten() ** 2))

        J = J + J_reg_term
        am1 = np.zeros((1, X.shape[1]))
        for i in range(m):
            
            # feedforward for each sample
            am1[0, :] = np.copy(X[i, :])
            z2 = theta1 @ am1.T
            am2 = sigmoid(z2)
            am2 = np.concatenate(([[1]], am2), axis=0)
            z3 = theta2 @ am2
            am3 = sigmoid(z3)

            #backward for each sample

            delta3 = am3 - np.array([[1 if y[i] == label else 0] for label in labels])
            tempt2 = np.copy(theta2.T)
            delta2 = (tempt2[1:, :] @ delta3) * sigmoidGradient(z2)
            theta2_grad = theta2_grad + delta3 @ am2.T
            theta1_grad = theta1_grad + delta2 @ am1

        
        grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis = 0)

        grad = grad / m
        grad = grad + lambdaa / m * np.concatenate((temp1.flatten(), temp2.flatten()), axis = 0)
        print(theta2_grad[:, 1], theta1_grad[:, 1])


        return(J, grad[100:110])




        



    @staticmethod
    def predictOneVsAll(X, y, theta):

        # this function predicts multiclass labels from probailities

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
        # this functions predict logistic regression reults with probability
        
        hypo = self.X @ self.theta
        res = self.sigmoid(hypo)

        res[(res >= 0.5)] = 1
        res[(res < 0.5)] = 0

        return res
