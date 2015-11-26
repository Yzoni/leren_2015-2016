#!/bin/env python3.4

import numpy as np
import warnings

__author__ = 'Yorick de Boer (10786015)'

'''
Implementation of logistic regression for both linear and quadratic
'''

# Disable the warnings, they mess up the output
def fxn():
    warnings.warn("runtimewarning", RuntimeWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore")

# Example use: readFile('housesRegr.csv', [0,1,2], 3)
# Return contents of csv file in the form of two numpy matrixes. One for the the x values and one for the y values.
# It skips the first header line.
# Bias is set as 1 in the first column of vector x.
def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = np.insert(array[:,xcolumns], 0, 1, axis=1)
    y = array[:,ycolumn]
    return x, y

# The variable for the hypothesis function
def g_function(t, x):
    ttranspose = t.T
    g = -x.dot(ttranspose)
    return g

# Sigmoid function as hypothesis
def hypothesis(t, x):
    h0 = 1 / (1 + np.exp(g_function(t, x)))
    return h0

# Calculates the cost
def cost_function(t, x, y):
    h0 = hypothesis(t, x)
    total_cost = -(y * np.log(h0)) - ((1 - y) * np.log(1 - h0))
    mean_cost = np.mean(total_cost)
    return mean_cost

# Does one gradient step
def step_gradient(t, x, y, learnrate, regularization_parameter=0):
    h0 = hypothesis(t, x)
    regularization_term = regularization_parameter * t
    t -= (learnrate * (h0 - y).dot(x)) + regularization_term
    return t

# Iterates set amount of times
def do_iterations(iterations, learnrate, t, x, y, regularization_parameter):
    for _ in range(iterations):
        t = step_gradient(t, x, y, learnrate, regularization_parameter)
    return t

# Runs the hypothesis and returns the predicted values expressed as 1 or 0
def test(t, x):
    probability = hypothesis(t, x)
    predicted = np.where(probability >= 0.5, 1, 0)
    return predicted


# Main function with pretty prints the result of the calculations
# Example: Main('dataopdracht1a.csv', [1,2,3], 4, 0.001, 100000, 0.001, function_type='linear')
def Main(csvfile, x_columns, y_column, learnrate, iterations, regularization_parameter, function_type):
    x, y = read_file(csvfile, x_columns, y_column)
    shape = x.shape[1]
    initt = np.zeros(shape)

    if function_type == 'linear':
        initt = np.zeros(shape)
    elif function_type == 'quadratic_circular': # Function with the form h0=t0 + t1*x1 + t2*x2 + t1*x1^2 + t2*x2^2
        quadraticx = np.square(x)
        quadraticx = np.delete(quadraticx, 0, 1)
        x = np.concatenate((x, quadraticx), axis=1)
        shape = x.shape[1]
        initt = np.zeros(shape)
    elif function_type == 'quadratic_simple': # Parabolia function
        quadraticx = np.square(x)
        quadraticx = np.delete(quadraticx, 0, 1)
        x = np.concatenate((x, quadraticx), axis=1)
        shape = x.shape[1]
        initt = np.zeros(shape)

    print("Cost BEFORE gradient descent: " + str(cost_function(initt, x, y)))
    newt = do_iterations(iterations, learnrate, initt, x, y, regularization_parameter)
    print("New teta's: " + str(newt))
    new_cost = cost_function(newt, x, y)
    if np.isnan(new_cost):
        print("Cost AFTER gradient descent: Could not be calculated, probably RuntimeError: overflow")
    else:
        print("Cost AFTER gradient descent: " + str(new_cost))
    print("Check prediction: " + str(test(newt,x)))

if __name__ == '__main__':
    print("Runs linear logistic regression on the sample data from the exercise: ")
    Main('dataopdracht1a.csv', [1,2,3], 4, 0.001, 100000, 0.001, function_type='linear')
    print("\n")
    print("'Noise starting from line 20 in the csv files'")
    print("Runs logistic regression on data that can be visualized as circle the borderline should be (-10, 0) in \n"
          "every direction")
    Main('dataopdracht1b-1.csv', [0,1], 2, 0.001, 100000, 0.000001, function_type='quadratic_circular')
    print("\n")
    print("Runs logistic regression on data that can be visualized as a parabolia")
    Main('dataopdracht1b-2.csv', [0,1], 2, 0.001, 100000, 0.00001, function_type='quadratic_simple')

    print("\n")
    print("Discussion: \n It can be seen that the noise has a noticable effect on the performance of the \n"
          "prediction. This is most likely caused by the relatively small dataset. When the noise is removed the \n"
          "regression seems to work very well. The regularization parameter is tuned so that the output is \n"
          "the best achieveable given the used data and noise. This was done by simply trying different values \n"
          "for this term and looking at the result.")

