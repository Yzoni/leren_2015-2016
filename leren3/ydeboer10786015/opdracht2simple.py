#!/bin/env python3.4

import numpy as np

__author__ = 'Yorick de Boer (10786015)'

'''
Simple implementation of a neural network using only three nodes. An input node, one hidden
node and an output node.
'''

def sigmoid(a, w):
    return 1 / (1 + np.exp(-(float(a) * float(w))))

def sigmoid_derivative(a):
    return a * (1 - a)

def forward(w, x):
    a2 = sigmoid(x, w[0])
    a3 = sigmoid(a2, w[1])

    return [a2, a3]

def backward(w, a, y):
    error3 = y - a[1]
    error2 = error3 * w[1] * sigmoid_derivative(a[1])

    return [error2, error3]

def update_weight(w, a, error, lr):
    w[0] += lr * error[1] * a[1]
    w[1] += lr * error[0] * a[0]

    return w

def Main(w, x, y, lr, iterations):
    for _ in range(iterations):
        a = forward(w, x)
        error = backward(w, a, y)
        w = update_weight(w, a,  error, lr)
    return w

if __name__ == '__main__':
    # Using the variables from the written assignment
    w = [0.2, 0.1]
    x = 0.5
    y = 1
    lr = 0.001
    iterations = 10000

    print("After learning, the weights are: ")
    w = Main(w, x, y, lr, iterations)
    print("w1 = " + str(w[0]))
    print("w2 = " + str(w[1]))