#!/bin/env python3.4
import numpy as np

__author__ = 'Yorick de Boer (10786015)'

class LogisticRegression:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    # The variable for the hypothesis function
    def g_function(self, x, t):
        ttranspose = t.T
        g = -x.dot(ttranspose)
        return g

    # Sigmoid function as hypothesis
    def hypothesis(self, x, t):
        h0 = 1 / (1 + np.exp(self.g_function(x, y, t)))
        return h0

    # Calculates the cost
    def cost_function(self, t):
        h0 = self.hypothesis(t, self.x)
        total_cost = -(self.y * np.log(h0)) - ((1 - self.y) * np.log(1 - h0))
        mean_cost = np.mean(total_cost)
        return mean_cost

    # Does one gradient step
    def step_gradient(self, t, learnrate, regularization_parameter=0):
        h0 = self.hypothesis(self.x, t)
        regularization_term = regularization_parameter * t
        t -= (learnrate * (h0 - self.y).dot(self.x)) + regularization_term
        return t

    # Iterates set amount of times
    def train(self, iterations, learnrate, t, regularization_parameter):
        for _ in range(iterations):
            t = self.step_gradient(t, learnrate, regularization_parameter)
        return t

    def accuracy(self, xtest, ytest, t):
        probability = self.hypothesis(xtest, t)
        predicted = np.where(probability >= 0.5, 1, 0)
        return np.mean(ytest == predicted)

    # Runs the hypothesis and returns the predicted values expressed as 1 or 0
    def test(self, xtest, ytest, t):
        probability = self.hypothesis(xtest, ytest, t)
        predicted = np.where(probability >= 0.5, 1, 0)
        return predicted
