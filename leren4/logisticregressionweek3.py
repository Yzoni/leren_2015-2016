#!/bin/env python3.4
import numpy as np

__author__ = 'Yorick de Boer (10786015)'

class LogisticRegression:
    def __init__(self, x, y):
        self.x = np.insert(np.asarray(x), 0, 1, axis=1)
        self.y = np.asarray(y)
        self.t = np.zeros(self.x.shape[1])

    # The variable for the hypothesis function
    def g_function(self, x, t):
        ttranspose = t.T
        g = -x.dot(ttranspose)
        return g

    # Sigmoid function as hypothesis
    def hypothesis(self, x, t):
        h0 = 1 / (1 + np.exp(self.g_function(x, t)))
        return h0

    # Calculates the cost
    def cost_function(self, t):
        h0 = self.hypothesis(t, self.x)
        total_cost = -(self.y * np.log(h0)) - ((1 - self.y) * np.log(1 - h0))
        mean_cost = np.mean(total_cost)
        return mean_cost

    # Does one gradient step
    def step_gradient(self, learnrate, t, regularization_parameter):
        h0 = self.hypothesis(self.x, t)
        regularization_term = regularization_parameter * t
        t -= (learnrate * (h0 - self.y).dot(self.x)) + regularization_term
        return t

    # Iterates set amount of times
    def train(self, iterations, learnrate, regularization_parameter=0):
        t = np.zeros(self.x.shape[1])
        for _ in range(iterations):
            t = self.step_gradient(learnrate, t, regularization_parameter)
        self.t = t
        return self.t

    def accuracy(self, xtest, ytest):
        reshaped_x = np.insert(np.asarray(xtest), 0, 1, axis=1)
        reshaped_y = np.asarray(ytest)
        probability = self.hypothesis(reshaped_x, self.t)
        predicted = np.where(probability >= 0.5, 1, 0)
        print(predicted, reshaped_y)
        return np.mean(reshaped_y == predicted)

    # Runs the hypothesis and returns the predicted values expressed as 1 or 0
    def test(self, xtest, ytest, t):
        probability = self.hypothesis(xtest, ytest, t)
        predicted = np.where(probability >= 0.5, 1, 0)
        return predicted
