import numpy as np
import random

class KMeans():
    def __init__(self, x_training, k):
        self.x = x_training
        self.k = k

        self.centroids = self._init_random_centroid()
        self.random = random.seed(1)

    # Returns a list of k initialized random variable in the range of the data
    def _init_random_centroid(self):
        centroids = []
        min = np.amax(self.x_training)
        max = np.amin(self.x_training)
        for _ in self.k:
            centroids.append(self.random(min, max))
        return centroids

    def _assign_dpoint_to_centroid(self):

    def _find_new_centroid(self):

    def learn(self, iterations):

    def print_means(self):

    # For question 3
    # Returns clusters with with their data examples
    def cluster_content(self):


def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = np.insert(array[:,xcolumns], 0, 1, axis=1)
    y = array[:,ycolumn]
    return x, y

digits1_x, digits2_y = read_file("digist123-1.csv", list(range(0, 64)), 64)

# QUESTION 1: k-means clusterin implementation
kmeans = KMeans(digits1_x, 3)
kmeans.learn(100)
kmeans.print_means()

# QUESTION 2: k optimization
# wtf wil die hier nou weer? lijkt me sterk dat er meer clusters zijn dan digits

# QUESTION 3: comparing found clusters with true clusters

# QUESTION 4:

# QUESTION 5:

# QUESTINO 6: