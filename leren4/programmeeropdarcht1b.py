import numpy as np
import operator

import csv

__author__ = "Yorick de Boer (10786015)"

# EXAMPLE:
# knn = KNearestNeighbour([[1,2],[[3,2],[1,2],[5,3]], [1,2,1,3], 3)
# knn.get_class([[3,2]])
# >> 2
# knn.accuracy([[3,2]], 2)
# >> 100%
class KNearestNeighbour:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Returns the euclidean distance between two points
    def euclidean_distance(self, pn, qn):
        pq_square = 0
        for p,q in zip(pn, qn):
            pq_square += np.square(p - q)
        distance = np.sqrt(pq_square)
        return distance

    def _find_all_distances_(self, newx):
        distance_list = []
        for xrow in self.x:
            distance = self.euclidean_distance(xrow, newx)
            distance_list.append(distance)
        return distance_list

    def _check_dups_(self, list, k):
        if list[-k-1][0] == list[-k][0]:
            k += 1
            return self._check_dups_(list, k)
        return list[-k:]

    # Returns a list of the closest classes
    def _find_closest_kclasses_(self, newx, k):
        distance_list = self._find_all_distances_(newx)
        combined_list = []
        for d, y in zip(distance_list, self.y):
            combined_list.append([d, y])
        sorted_combined_list = sorted(combined_list, key=operator.itemgetter(0))
        # If there are more nodes with the same distance add them aswell
        closest_k = self._check_dups_(sorted_combined_list, k)
        return closest_k

    def _weight_closest_kclasses(self, newx, k):
        closest_list = self._find_closest_kclasses_(newx, k)
        weigted_closes_list = []
        for e in closest_list:
            calc_weight = 1 / e[0]
            e.append(calc_weight)
            weigted_closes_list.append(e)
        return weigted_closes_list

    # Get most frequent element in k-range list
    def get_class(self, xtest, k):
        weighted_closest_list = self._weight_closest_kclasses(xtest, k)
        weight_by_class = []
        for e in weighted_closest_list:
            y_class = e[1]
            weight_item = e[2]
            if not weight_by_class:
                weight_by_class.append([y_class, weight_item])
            for x in weight_by_class:
                if y_class == x[0]:
                    x[1] += weight_item
                else:
                    weight_by_class.append([y_class, weight_item])
        max_list = max(weight_by_class, key=operator.itemgetter(1))
        print(max_list)
        return max_list[0]

    def accuracy(self, xtest, ytest, k):
        totalrows = len(ytest)
        correct_counter = 0
        for x_row, y_row in zip(xtest, ytest):
            eval = self.get_class(x_row, k)
            if eval == y_row:
                correct_counter += 1
        return (correct_counter / totalrows) * 100


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
    for x0 in x:
        x0.insert(0, 1)
    return x, y

if __name__ == "__main__":
    csv_as_list_train = readFile('digist123-1.csv')
    x1, y1 = createdatalists(csv_as_list_train, list(range(0,63)), 64)
    csv_as_list_test = readFile('digist123-2.csv')
    x2, y2 = createdatalists(csv_as_list_test, list(range(0,63)), 64)

    a = KNearestNeighbour(x1, y1)
    print(str(a.accuracy(x2, y2, 2)) + "%")

