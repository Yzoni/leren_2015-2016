import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from programmeeropdracht1abc import KNN
from programmeeropdracht1abc import WeightedDistanceKNN
from programmeeropdracht1abc import WeightedPredictiveValueKNN

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

def analyze_knn(x1, y1, x2, y2, iterations):
    normal_knn = KNN(x1, y1)
    weighted_distance_knn = WeightedDistanceKNN(x1, y1)
    weighted_predictive_knn = WeightedPredictiveValueKNN(x1, y1)

    accuracy_normal_knn_list = []
    accuracy_distance_knn_list = []
    accuracy_predictive_knn_list = []

    range_list = list(range(1, iterations))
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

if __name__ == "__main__":
    csv_as_list_train = readFile('digist123-1.csv')
    x1, y1 = createdatalists(csv_as_list_train, list(range(0,64)), 64)
    csv_as_list_test = readFile('digist123-2.csv')
    x2, y2 = createdatalists(csv_as_list_test, list(range(0,64)), 64)

    analyze_knn(x1, y1, x2, y2, 11)
