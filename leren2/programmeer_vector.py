#!/bin/env python3.4

import numpy as np

__author__ = 'Yorick de Boer (10786015)'

# Example use: readFile('housesRegr.csv', [0,1,2], 3)
# Return contents of csv file in the form of two numpy matrixes. One for the the x values and one for the y values
def readFile(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    print(array)
    x = np.insert(array[:,xcolumns], 0, 1, axis=1)
    y = array[:,ycolumn]
    return x, y

def hypothesis(t, x):
    h0 = 1./(1.+np.exp(-(np.dot(x,t))))
    return h0

def costFunction(t, x, y):
    h = hypothesis(t, x)
    m=np.shape(y)[0]
    #first term of cost function
    j_1=float(np.dot(y.T,np.log(h)))
    #second term of cost function
    j_2=float(np.dot(1-y.T,np.log(1-h)))
    return -((1./m)*(j_1+j_2))

stepGradient():
    return gradient

def doIterations():
    return t

if __name__ == '__main__':
    x, y = readFile('housesRegr.csv', [1,2,3], 4)
    xt = x.transpose()
    teta = ((np.linalg.inv(x * xt)) * xt)* y
    print(teta)