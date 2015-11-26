#!/bin/env python3.4

import csv
import plotly.tools as pt
import plotly.plotly as py
import plotly.graph_objs as go
from enum import Enum
import math

__author__ = 'Yorick de Boer (1786015)'

# Enum to identify column index by name
class VarType(Enum):
    bedrooms = 1
    bathrooms = 2
    size = 3
    price = 4

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
def createdatalists(csvlist, typex, typey):
    x = []
    y = []

    for entry in csvlist:
        x.append(int(entry[typex.value]))
        y.append(int(entry[typey.value]))
    return x, y

# Execute hypothesis function with t0 and t1
def generatehypopoints(t0, t1, x):
    y = t0 + t1 * int(x)
    return y

# Plot the scatter and hypothesis function in plotly
def plot(scatterlistx, scatterlisty, t0, t1, xtitle, ytitle):
    # Sign in to plotly
    pt.set_credentials_file(username='Yzoni', api_key='7iqvws7lp3')

    # For hypotheses line
    minxhypotheses = min(scatterlistx)
    maxxhypotheses = max(scatterlistx)
    xl = [minxhypotheses, maxxhypotheses]
    yl = [generatehypopoints(t0, t1, minxhypotheses), generatehypopoints(t0, t1, maxxhypotheses)]

    tracescatter = go.Scatter(
        x=scatterlistx,
        y=scatterlisty,
        mode='markers',
        name='scatter'
    )

    traceline = go.Scatter(
        x=xl,
        y=yl,
        mode='lines',
        name='gradientdescent'
    )

    data = [tracescatter, traceline]
    layout = go.Layout(
        title=xtitle + " vs " + ytitle,
        xaxis=dict(
            title=xtitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=ytitle,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)

    url = py.plot(fig, filename=xtitle + ytitle)
    return url

# Returns the cost
def costFunction(t0, t1, listx, listy):
    totalCost = 0
    for x, y in zip(listx, listy):
        h0 = generatehypopoints(t0, t1, x)
        totalCost += (h0 - int(y)) ** 2
    listlength = len(listx)
    return 1 / (2 * listlength) * totalCost

# Returns t0 and t1 for one gradient step
def gradient(t0, t1, listx, listy, learnrate):
    gradt0 = 0
    gradt1 = 0
    n = len(listx)
    for x, y in zip(listx, listy):
        h0 = generatehypopoints(t0, t1, x)
        gradt0 += (1/n) * (h0 - int(y))
        gradt1 += (1/n) * (h0 - int(y)) * int(x)
    t0 -= (learnrate * gradt0)
    t1 -= (learnrate * gradt1)
    return t0, t1

# Returns t0 and t1 for set iterations and learnrate
def univLinReg(initt0, initt1, listx, listy, iterations, learnrate):
    t0 = initt0
    t1 = initt1
    for _ in range(iterations):
        t0, t1 = gradient(t0, t1, listx, listy, learnrate)
    return t0, t1

# Main function with pretty print
def Main(csvfile, typex, typey, learnrate, iterations):
    print("Beginning gradient descent on " + csvfile + " ...  With types: " + typex.name + " as x and " + typey.name + " as y")
    print("Learnrate: " + str(learnrate) + "\t Iterations: " + str(iterations))
    print("Startvalues: t0=0 \t t1=0.")
    csvlist = readFile(csvfile)
    listx, listy = createdatalists(csvlist, typex, typey)
    t0, t1 = univLinReg(0, 0, listx, listy, iterations, learnrate)
    if not math.isnan(t0) or not math.isnan(t1):
        print("Finalvalues: t0=" + str(t0) + "\t t1=" + str(t1))
        print("Startcost: " + str(costFunction(0, 0, listx, listy)) + "\t Finalcost: " + str(costFunction(t0, t1, listx, listy)))
        print('Url to the plot ' + typex.name + ' vs ' + typey.name + ": " + plot(listx, listy, t0, t1, typex.name, typey.name))
    else:
        print("t0 or t1 is NaN, try to decrease the learning rate with this dataset")
    print("\n")

# Learningrate needs to be extremely low to avoid skipping over the the minimum
# Price vs size
Main('housesRegr.csv', VarType.price, VarType.size, 0.000000000001, 1000)

# Price vs bedrooms
Main('housesRegr.csv', VarType.price, VarType.bedrooms, 0.000000000001, 1000)

# Price vs bathrooms
Main('housesRegr.csv', VarType.price, VarType.bathrooms, 0.000000000001, 1000)
