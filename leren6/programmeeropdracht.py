import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pprint

class KMeans():
    def __init__(self, x_training, k):
        self.x = x_training
        self.k = k

        self.xrows, self.xcolumns = self.x.shape
        self.r = random.Random()
        self.r.seed(1)
        self.centroids = self._init_random_centroid()

    # Returns a list of k initialized random variable in the range of the data
    def _init_random_centroid(self):
        centroids = []
        max = np.amax(self.x)
        min = np.amin(self.x)
        for _ in range(self.k):
            rarray = np.random.randint(min, max, (self.xcolumns))
            centroids.append(rarray)
        return centroids

    def _assign_dpoint_to_centroid(self):
        example_by_centroid = {}
        # initialize centroids with empty list. The lists will contain the examples
        # which are closest to the centroid
        for id, centroid in enumerate(self.centroids):
            example_by_centroid[id] = []
        
        for lineid, example in enumerate(self.x):
            # dict containing distance between example x and all centroids
            distances = {}
            
            for id, centroid in enumerate(self.centroids):
                dist = self.euclidDist(example, centroid)
                distances[id] = dist
            closest_centroid = min(distances.items(), key=operator.itemgetter(1))[0]
            
            example_by_centroid[closest_centroid].append([lineid, example])
        return example_by_centroid
                
    def _find_new_centroid(self, array):
        return np.mean(array, axis=0)

    def _remove_every_first_element_nested_list(selfs, list):
        for i in list:
            i.pop(0)
        return list

    def learn(self, iterations, plot_error=0):
        errors = []
        for i in range(iterations):
            old_centroids = self.centroids
            dict_example_by_centroid = self._assign_dpoint_to_centroid()
            self.centroids = [] # clear centroids
            for c in dict_example_by_centroid:
                popped = self._remove_every_first_element_nested_list(dict_example_by_centroid[c])
                fnparray = np.vstack(popped)
                new_mean = self._find_new_centroid(fnparray)
                self.centroids.append(new_mean)

            if plot_error == 1:
                euc = 0
                for c in dict_example_by_centroid:
                    for array in dict_example_by_centroid[c]:
                        euc += self.euclidDist(array, old_centroids[c])
                euc = euc/self.xrows
                errors.append(euc)
                
        if plot_error == 1:
            self.plot(list(range(iterations)), errors)

        return self._assign_dpoint_to_centroid()
            
    def means(self):
        return self.centroids

    # returns the euclidian distance between vectors p and q
    def euclidDist(self, p, q):
        pminq = np.subtract(p, q)
        pminqSqr = np.square(pminq)
        dist = np.sqrt(np.sum(pminqSqr))
        return dist

    # plot cost vs iterations
    def plot(self, x, y):
        plt.plot(x, y)
        plt.show()

    def find_anomalies(self, dict_example_by_centroid, threshold):
        anomalies = []
        for c in dict_example_by_centroid:
            for array in dict_example_by_centroid[c]:
                euc = self.euclidDist(array, self.centroids[c])
                if euc > threshold:
                    anomalies.append(array)
        return anomalies

def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = array[:,xcolumns]
    y = array[:,ycolumn]
    return x, y

def true_mean_dataset(xdata, ydata):
    dicts = dict.fromkeys(list(set(ydata)), [])

    for x, y in zip(xdata, ydata):
        dicts[y].append(x)

    means = {}
    for d in dicts:
        means[d] = np.mean(dicts[d], axis=0)

    return means

digits1_x, digits1_y = read_file("digist123-1.csv", list(range(0, 64)), 64)

# QUESTION 1: k-means clusterin implementation
kmeans = KMeans(digits1_x, 3)
kmeans.learn(5)
pprint.pprint(kmeans.means())

# QUESTION 2: k optimization
kmeans_optimal = KMeans(digits1_x, 3)
example_dict = kmeans_optimal.learn(15, plot_error=0)
# It can be seen that the error is constant after 6 iterations, so k is optimal after 6 iterations


# QUESTION 3: comparing found clusters with true clusters
# Compares the found cluster class with the given example cluster class. This is done line by line.

count_class1 = []
count_class2 = []
count_class3 = []
for lineid, class_element in enumerate(digits1_y):
    if class_element == 1:
        count_class1.append(lineid)
    elif class_element == 2:
        count_class2.append(lineid)
    elif class_element == 3:
        count_class3.append(lineid)
real_classes = [count_class1, count_class2, count_class3]

found_classes = []
for class_element in example_dict:
    class_list = example_dict[class_element]
    list_rows_one_class = []
    for i in class_list: # every element in the class
        list_rows_one_class.append(i[0])
    found_classes.append(list_rows_one_class)

print(found_classes)

# Manually setting class to found clusters:
found_class1 = [0, 1, 3, 4, 5, 6, 7, 8, 9, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 53, 55, 56, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 237, 276, 277, 278, 289, 291]
found_class2 = [2, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 101, 102, 103, 104, 105, 106, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 154, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 177, 179, 181, 182, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
found_class3 = [13, 49, 51, 54, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 107, 109, 129, 149, 150, 151, 152, 153, 155, 157, 174, 176, 178, 180, 183, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 292, 293, 294, 295, 296, 297, 298]

count1 = 0
for c in found_class1:
    if c in real_classes[0]:
        count1 += 1
print("Correct class 1: " + str(count1))

count2 = 0
for c in found_class2:
    if c in real_classes[1]:
        count2 += 1
print("Correct digit 2: " + str(count2))

count3 = 0
for c in found_class3:
    if c in real_classes[2]:
        count3 += 1
print("Correct digit 3: " + str(count3))

# ACCURACY:
print("Accuracy of: " + str((count1 + count2 + count3) / len(digits1_y) * 100))

# QUESTION 4:
# find_anomalies returns the examples of which the distance to the mean is
# greater than the threshold distance. These examples are the anomalies

#anomalies = kmeans_optimal.find_anomalies(example_dict, 40)

# QUESTION 5:
# We have examined the anomalies by printing the 8 x 8 matrix of each training
# example which was classified as an anomaly. The shapes of the grey values
# all resembled the number 4. Since our algorithm uses only 3 clusters, and
# the number 4 does not occur frequently in the data, this number was classified
# as an anomaly. Using more clusters might result in having a better classification.
'''
print("Number of anomalies: ", len(anomalies))

# prints 8 x 8 grey-value array
for a in anomalies:
    print(a.reshape(8, 8))
    print("\n")
'''

# QUESTINO 6:
