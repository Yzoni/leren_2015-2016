import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

from programmeeropdracht1abc import KNN
from programmeeropdracht1abc import WeightedDistanceKNN
from programmeeropdracht1abc import WeightedPredictiveValueKNN

from logisticregressionweek3 import LogisticRegression

def plot(x, y, xlabel="x", ylabel="y"):
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

# Return the csvfile as a list of lists. A list for every row.
def readFile(csvfilename):
    list = []
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
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
    return x, y

# Analysation function for determining the best k value for the different knn algorithms.
def analyze_knn(x1, y1, x2, y2, k):
    normal_knn = KNN(x1, y1)
    weighted_distance_knn = WeightedDistanceKNN(x1, y1)
    weighted_predictive_knn = WeightedPredictiveValueKNN(x1, y1)

    accuracy_normal_knn_list = []
    accuracy_distance_knn_list = []
    accuracy_predictive_knn_list = []

    range_list = list(range(1, k))
    for i in range_list:
        accuracy = normal_knn.accuracy(x2, y2, i)
        accuracy_normal_knn_list.append(accuracy)

        accuracy = weighted_distance_knn.accuracy(x2, y2, i)
        accuracy_distance_knn_list.append(accuracy)

        accuracy = weighted_predictive_knn.accuracy(x2, y2, i)
        accuracy_predictive_knn_list.append(accuracy)

    red_patch = mpatches.Patch(color='#D50000', label='Normal KNN')
    blue_patch = mpatches.Patch(color='#311B92', label='Distance weighted KNN')
    green_patch = mpatches.Patch(color='#33691E', label='Predictive weighted KNN')

    plt.legend(handles=[red_patch, blue_patch, green_patch])

    plt.plot(range_list, accuracy_normal_knn_list, color="#D50000")
    plt.plot(range_list, accuracy_distance_knn_list, color="#311B92")
    plt.plot(range_list, accuracy_predictive_knn_list, color="#33691E")

    plt.show()

def get_partialdataset(x_data, y_data, y_class):
    index_list = []
    for index, class_in_y in enumerate(y_data):
        if class_in_y == y_class[0] or y_class[1]:
            index_list.append(index)
    newx = []
    newy = []
    for index in index_list:
        newx.append(x_data[index])
        newy.append(y_data[index])
    return newx, newy

def analyze_logisticregression(x1, y1, x2, y2, iterations=10000, regularization=0):

    # Classes 1 and 2
    x1_12, y1_12 = get_partialdataset(x1, y1, [1, 2])
    x2_12, y2_12 = get_partialdataset(x2, y2, [1, 2])
    lg_1and2 = LogisticRegression(x1_12, y1_12)

    # Classes 1 and 3
    x1_13, y1_13 = get_partialdataset(x1, y1, [1, 3])
    x2_13, y2_13 = get_partialdataset(x2, y2, [1, 3])
    lg_1and3 = LogisticRegression(x1_13, y1_13)

    # Classes 2 and 3
    x1_23, y1_23 = get_partialdataset(x1, y1, [2, 3])
    x2_23, y2_23 = get_partialdataset(x2, y2, [2, 3])
    lg_2and3 = LogisticRegression(x1_23, y1_23)

    # Test required amount of iterations
    range_list_iterations = list(range(1, iterations))
    accuracy_12 = []
    for i in range_list_iterations:
        lg_1and2.train(i, 0.00001, regularization)
        accuracy = lg_1and2.accuracy(x2_12, y2_12)
        accuracy_12.append(accuracy)
    plt.plot(range_list_iterations, accuracy_12, color="#D50000")
    plt.show()

    # Test best regularization term
    range_list_regularization = list(range(1, regularization))
    accuracy_12 = []
    for i in range_list_regularization:
        lg_1and2.train(iterations, 0.00001, i)
        accuracy = lg_1and2.accuracy(x2_12, y2_12)
        accuracy_12.append(accuracy)
    plt.plot(range_list_regularization, accuracy_12, color="#D50000")
    plt.show()

if __name__ == "__main__":
    csv_as_list_train = readFile('digist123-1.csv')
    x1, y1 = createdatalists(csv_as_list_train, list(range(0,64)), 64)
    csv_as_list_test = readFile('digist123-2.csv')
    x2, y2 = createdatalists(csv_as_list_test, list(range(0,64)), 64)

    analyze_knn(x1, y1, x2, y2, 11)
    analyze_logisticregression(x1, y1, x2, y2, iterations=1000, regularization=0)
    analyze_logisticregression(x1, y1, x2, y2, iterations=1000, regularization=5)


'''
--------------
- DISCUSSION -
--------------

K-nearest neighbour is superior for this type of task compared to logistic regression, because with logistic regression
only two classes can be compared an once. Neural networks might also be a good solution to use on digit recognition.
Unfortunately I didn't make a generalized working version of this type of algorithm in the previous assignment. So I
was not able to test this type of algorithm.

Provided are a analysation function which plots the different knn algorithms with a range of k values. The result of
this test is that the distance weighted KNN performs the best overall and the predictive weighted KNN the worst. Tje
distance weighted performs best with k value of 1 or 2, but for general purpose a k value of 2 is probably best.

The regularized logistic regression can also be tested. The main things which would be interesting to know is the
amount of iterations necessary and the the optimal regularization term. This could be done with the provided functions.

Concluding van be said that both KNN and neural networks can be used for digit recognition. This analysation evidenced
that the best recognition result can be achieved with distance weighted KNN with a k value of 2.

'''