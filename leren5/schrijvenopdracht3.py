#!/bin/env python3.4

import numpy as np

def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = array[:,xcolumns].flatten().tolist()
    y = array[:,ycolumn].flatten().tolist()
    return x, y

def get_partialdataset(x_data, y_data, y_class):
    index_list = []
    for index, class_in_y in enumerate(y_data):
        if class_in_y == y_class:
            index_list.append(index)
    newx = []
    newy = []
    for index in index_list:
        newx.append(x_data[index])
        newy.append(y_data[index])
    return newx, newy

def pdf(testvar, sigma, mean):
    return 1/(np.sqrt(2 * np.pi * sigma **2)) * np.exp((-(testvar -
                                                          mean) **2) / (2 * sigma **2))

x1, y = read_file('opdracht1schrijven.csv', [1], 3)
x2, _ = read_file('opdracht1schrijven.csv', [2], 3)

prior_probability_p = 4/6
prior_probability_q = 2/6

x1_p = get_partialdataset(x1, y, 1)
x2_p = get_partialdataset(x2, y, 1)

x1_q = get_partialdataset(x1, y, 0)
x2_q = get_partialdataset(x2, y, 0)

mean_x1_p = np.mean(x1_p)
mean_x2_p = np.mean(x2_p)

mean_x1_q = np.mean(x1_q)
mean_x2_q = np.mean(x2_q)

variance_x1_p = np.var(x1_p)
variance_x2_p = np.var(x2_p)

variance_x1_q = np.var(x1_q)
variance_x2_q = np.var(x2_q)

evidence = (prior_probability_p * pdf(10, variance_x1_p, mean_x1_p)
            * pdf(5, variance_x2_p, mean_x2_p)) + (prior_probability_q
                                                   * pdf(10, variance_x1_q, mean_x1_q)
                                                   * pdf(5, variance_x2_q, mean_x2_q))

posterior_p = (prior_probability_p * pdf(10, variance_x1_p, mean_x1_p)
               * pdf(5, variance_x2_p, mean_x2_p)) / evidence
print(posterior_p)

posterior_q = (prior_probability_q * pdf(10, variance_x1_q, mean_x1_q)
               * pdf(5, variance_x2_q, mean_x2_q)) / evidence
print(posterior_q)
