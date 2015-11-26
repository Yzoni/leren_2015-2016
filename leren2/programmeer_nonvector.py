#!/bin/env python3.4

import csv
import math

__author__ = 'Yorick de Boer (10786015)'
'''
This implementation makes use of vectors for the x, y and teta variables.
'''

# Return the csvfile as a list of lists. A list for every row.
def readFile(csvfilename):
    list = []
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader) # Skip first header line
        for row in reader:
            list.append(row)
    return list

# Creates two lists. The x-list containing lists of x-columns and the y-list containing the y-column
# Arguments are the csvfile, and the columns given as follwing: for xcolumns: [1,2,3] for ycolumn: 4
def createdatalists(csvlist, xcolumns, ycolumn):
    x = []
    y = []

    for entry in csvlist:
        x.append(list(int(entry[i]) for i in xcolumns))
        y.append(int(entry[ycolumn]))
    for x0 in x:
        x0.insert(0, 1)
    return x, y

# The same as createdatalists but this one always uses the first columns for the x-values and the
# last column for the y-value.
def autocreatedatalists(csvlist):
    x = []
    y = []
    for entry in csvlist:
        entry = [ int(x) for x in entry ]
        y.append(entry[-1])
        entry.pop()
        x.append(entry)
    for x0 in x:
        x0.insert(0, 1)
    return x, y

# Execute hypothesis as dot-product between the ts and xs lists.
def generatehypopoints(ts, xs, function_type, data_type):
    y = 0
    if data_type == 'lineair_reg':
        for t, x in zip(ts, xs):
            if function_type == 'poly':
                y += t * x **2
            else:
                y += t * x
        return y
    elif data_type == 'logistic_reg':
        for t, x in zip(ts, xs):
            tx = t * x
        y = 1 / (1 + math.e **(-(tx)))
        return y

# Returns the cost depending on data type lineair or logistic
def costFunction(listt, listx, listy, function_type, data_type):
    totalCost = 0
    if data_type == 'lineair_reg':
        for xs, ys in zip(listx, listy):
            totalCost += (generatehypopoints(listt, xs, function_type, data_type) - ys) ** 2
        listlength = len(listx)
        return (1 / (2 * listlength)) * totalCost
    else: # logistic regression
        for xs, ys in zip(listx, listy):
            h0 = generatehypopoints(listt, xs, function_type, data_type)
            totalCost += (ys * math.log(h0) - (1 - ys) * math.log(1-h0))
        listlength = len(listx)
        return -(1 / listlength) * totalCost

# Returns tetas for one gradient step
def gradient(listt, listx, listy, learnrate, function_type, data_type):
    gradt = [0] * len(listt) # Initialize temporery list of tetas
    n = len(listy)
    for xs, ys in zip(listx, listy):
        h0 = generatehypopoints(listt, xs, function_type, data_type)
        counter = 0 # Specifies index of current teta
        for xi in xs:
            gradt[counter] += (h0 - ys) * xi
            counter += 1
    for idx, teta in enumerate(gradt):
        listt[idx] -= (1/n) * (learnrate * teta)

    return listt

# Returns tetas for set iterations and learnrate
def doIterations(listx, listy, listt, iterations, learnrate, function_type, data_type):
    for _ in range(iterations):
        listt = gradient(listt, listx, listy, learnrate, function_type, data_type)
    return listt

# Main function with pretty print
# Example usage:
# Main('opdracht1.csv', 0.01, 100, [0.2, 0.2, 0.2], [0, 1], 2)
# This uses the opdracht1.csv as dataset, a learning rate of 0.01 and it does 100 iterations. The tetas it starts with
# is a defined as list from t0 to tn. It picks as X data column 0 and 1 and for the Y data column 2. It can
# handle any amount x columns.
def linreg(csvfile, learnrate, iterations, startteta, xcolumns, ycolumn, function_type):
    if len(startteta) == len(xcolumns) + 1:
        print("Learnrate: " + str(learnrate) + "\t Iterations: " + str(iterations))
        csvlist = readFile(csvfile)
        listx, listy = createdatalists(csvlist, xcolumns, ycolumn)
        print("Startvalues: t0=" + str(startteta))
        print("Startcost: " + str(costFunction(startteta, listx, listy, function_type, data_type='lineair_reg')))
        listt = doIterations(listx, listy, startteta, iterations, learnrate, function_type, data_type='lineair_reg')
        print("Finalvalues: t0=" + str(listt))
        print("Finalcost: " + str(costFunction(listt, listx, listy, function_type, data_type='lineair_reg')))
    else:
        print("Amount of start-tetas should be #x-columns + 1")

# Initial tetas are set in function also the csv file is parsed the default way
def logreg(csvfile, learnrate, iterations, function_type):
    csvlist = readFile(csvfile)
    listx, listy = autocreatedatalists(csvlist)
    startteta = [0] * len(listx[0])

    print("Learnrate: " + str(learnrate) + "\t Iterations: " + str(iterations))
    print("Startvalues: t0=" + str(startteta))
    print("Startcost: " + str(costFunction(startteta, listx, listy, function_type, data_type='logistic_reg')))
    listt = doIterations(listx, listy, startteta, iterations, learnrate, function_type, data_type='logistic_reg')
    print("Finalvalues: t0=" + str(listt))
    print("Finalcost: " + str(costFunction(listt, listx, listy, function_type, data_type='logistic_reg')))


if __name__ == '__main__':
    print("QUESTION 1")
    print("Dataset from written assignment:")
    linreg('opdracht1.csv', 0.01, 100, [0.2, 0.2, 0.2], [0, 1], 2, function_type='lin')
    print(" ")
    print("Dataset housesRegr.csv, lineair")
    linreg('housesRegr.csv', 0.00000001, 100, [0, 0, 0, 0], [1, 2, 3], 4, function_type='lin')
    print("\n")
    print("QUESTION 2")
    print("Dataset housesRegr.csv, polynomial")
    linreg('housesRegr.csv', 0.0000000001, 1000, [0, 0, 0, 0], [1, 2, 3], 4, function_type='poly')
    print("Discussion question 2: The squared polynomial is gives worse cost than the lineair function \n"
          "this is probably due to the fact that the the function goes down again after it reaches its top. \n"
          "Apart from this, the learning rate is quite interesting. It needs to be this low because of the big \n"
          "difference between te range of the difference x variables. A good thing to do here is probably feature \n"
          "scaling.")
    print("\n")
    print("QUESTION 3")
    print("p_opdracht3.csv, logistic regression")
    logreg('p_opdracht3.csv', 0.0000001, 1000, function_type='log') #function type not really necessary
    print("There is probably some small misstake in the program which causes this function not to work properly. \n"
          "I tried my best to figure out why, but couldn't. The implementation is fairly straightforward. Only \n"
          "the cost function and hypothesis function have been adjusted to fit the different model. \n")
    print("digits123.csv, logistic regression")
    logreg('digits123.csv', 0.00001, 100, function_type='log')
    print("Discussion question 3: ")

