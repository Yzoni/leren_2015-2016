#!/bin/env python3.4

import csv
import plotly.tools as pt
import plotly.plotly as py
import plotly.graph_objs as go
from enum import Enum
import math

__author__ = 'yorick'

# Enum to identify column index by name
class VarType(Enum):
    x1 = 0
    x2 = 1
    y = 2

# Return the csvfile as a list of lists. A list for every row.
def readFile(csvfilename):
    list = []
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader) # Skip first header line
        for row in reader:
            list.append(row)
    return list

# Return from two rows of the datafile
def createdatalists(csvlist, typex1, typex2, typey):
    x1 = []
    x2 = []
    y = []

    for entry in csvlist:
        x1.append(int(entry[typex1.value]))
        x2.append(int(entry[typex2.value]))
        y.append(int(entry[typey.value]))
    return x1, x2, y

# Execute hypothesis function with t0 and t1
def generatehypopoints(t0, t1, t2, x1, x2):
    y = t0 + t1 * int(x1) + t2 * int(x2)
    return y

# Returns the cost
def costFunction(t0, t1, t2, listx1, listx2, listy):
    totalCost = 0
    for x1, x2, y in zip(listx1, listx2, listy):
        totalCost += (generatehypopoints(t0, t1, t2, x1, x2) - int(y)) ** 2
    listlength = len(listx1)
    return (1 / (2 * listlength)) * totalCost

# Returns t0 and t1 for one gradient step
def gradient(t0, t1, t2, listx1, listx2, listy, learnrate):
    gradt0 = 0
    gradt1 = 0
    gradt2 = 0
    n = len(listx1)
    for x1, x2, y in zip(listx1, listx2, listy):
        h0 = generatehypopoints(t0, t1, t2, x1, x2)
        gradt0 += (1/n) * (h0 - int(y))
        gradt1 += (1/n) * (h0 - int(y)) * int(x1)
        gradt2 += (1/n) * (h0 - int(y)) * int(x2)
    t0 -= (learnrate * gradt0)
    t1 -= (learnrate * gradt1)
    t2 -= (learnrate * gradt2)
    return t0, t1, t2

# Returns t0 and t1 for set iterations and learnrate
def univLinReg(initt0, initt1, initt2, listx1, listx2, listy, iterations, learnrate):
    t0 = initt0
    t1 = initt1
    t2 = initt2
    for _ in range(iterations):
        t0, t1, t2 = gradient(t0, t1, t2, listx1, listx2, listy, learnrate)
    return t0, t1, t2

# Main function with pretty print
def Main(csvfile, typex1, typex2, typey, learnrate, iterations):
    print("Learnrate: " + str(learnrate) + "\t Iterations: " + str(iterations))
    print("Startvalues: t0=0.2 \t t1=0.2 \t t2=0.2")
    csvlist = readFile(csvfile)
    listx1, listx2, listy = createdatalists(csvlist, typex1, typex2, typey)
    t0, t1, t2 = univLinReg(0.2, 0.2, 0.2, listx1, listx2, listy, iterations, learnrate)
    if not math.isnan(t0) or not math.isnan(t1):
        print("Finalvalues: t0=" + str(t0) + "\t t1=" + str(t1) + "\t t2=" + str(t2))
        print("Startcost: " + str(costFunction(0.2, 0.2, 0.2, listx1, listx2, listy)) + "\t Finalcost: " + str(costFunction(t0, t1, t2, listx1, listx2, listy)))
        #print('Url to the plot ' + typex1.x1 + ' vs ' + typey.name + ": " + plot(listx, listy, t0, t1, typex.name, typey.name))
    else:
        print("t0 or t1 is NaN, try to decrease the learning rate with this dataset")
    print("\n")

if __name__ == '__main__':
    Main('opdracht1.csv', VarType.x1, VarType.x2, VarType.y, 0.01, 1)
