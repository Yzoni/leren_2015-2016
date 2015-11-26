#!/bin/env python3.4

# Implementation is limited to three layers, an input layer, hidden layer and output layer

import numpy as np

__author__ = 'Yorick de Boer (10786015)'

def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = np.insert(array[:,xcolumns], 0, 1, axis=1)
    y = array[:,ycolumn]
    return x, y

# input/activations are saved as list of vectors. Every item in the list is a input/activations vector.

# Computes the activation for one node in the next layer, z parameter is the dot product of x with the
# transpose of theta
def a_function(activation_layer, t_layer):
    return 1 / (1 + np.exp(-activation_layer.dot(t_layer.T)))

# Computes g with the derivative of the sigmoid function
def a_derivative_function(activation_layer):
    activation_layer * (1 - activation_layer)

# Computes the activations for all nodes in all layers (last activation should be h0, the output value)
def forward_function(activation_x_layer, t_all):
    activations = []
    # Add input variables to activation vector list
    activations.append(activation_x_layer)

    # Compute activaton in hidden layer
    a_layer_in_hidden = []
    for t_node_in_hidden in t_all[0]:
        a_node_in_hidden = a_function(np.asarray(activations[0]), np.asarray(t_node_in_hidden))
        a_layer_in_hidden.append(a_node_in_hidden)
    activations.append([1] + a_layer_in_hidden) # Add bias for next layer

    # Compute activation for output layer
    a_layer_in_output = []
    for t_node_in_output in t_all[1]:
        print(np.asarray(activations[1]), np.asarray(t_node_in_output))
        a_node_in_output = a_function(np.asarray(activations[1]), np.asarray(t_node_in_output))
        a_layer_in_output.append(a_node_in_output)
    activations.append(a_layer_in_output)

    return activations

def backward_function(activations, t_all, y):
    errors = []

    # Compute error between output and hidden layer
    error_output = activations[2] - y
    errors.append(error_output)

    # Compute errors for hidden layer
    t_hidden = np.asarray(t_all[0])
    error_hidden = t_hidden.T.dot(error_output) * a_derivative_function(activations[1])
    errors.append(error_hidden)

    return errors

def update(t_all, errors, activations):
    updated_t = []

    # Compute new weight between input and hidden
    for t_node_in_hidden in t_all[0]:


    # Compute new weight between hidden and output

    return updated_t

def MainCSV(csvfile, x_columns, y_column, init_t, iterations):
    x, y = read_file(csvfile, x_columns, y_column)
    for _ in range(iterations):
        for x_nodes in x:
            print(x_nodes, init_t)
            activations = forward_function(x_nodes, init_t)
    print(activations)

def mainSimple(data, init_t, iterations):
    weights = 0

    

    return weights

if __name__ == '__main__':
    MainCSV('xordata.csv', [0,1], 2, [[[1,1,2], [1,3,4]], [[1,1,1]]], 1)