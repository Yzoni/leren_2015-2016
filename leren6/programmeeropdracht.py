import numpy as np
import random
import operator
import matplotlib.pyplot as plt

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
        
        for example in self.x:
            # dict containing distance between example x and all centroids
            distances = {}
            
            for id, centroid in enumerate(self.centroids):
                dist = self.euclidDist(example, centroid)
                distances[id] = dist
            closest_centroid = min(distances.items(), key=operator.itemgetter(1))[0]
            
            example_by_centroid[closest_centroid].append(example)
        return example_by_centroid            
                
    def _find_new_centroid(self, array):
        return np.mean(array, axis=0)
        
    def learn(self, iterations, plot_error=0):
        errors = []
        for i in range(iterations):
            old_centroids = self.centroids
            dict_example_by_centroid = self._assign_dpoint_to_centroid()
            self.centroids = [] # clear centroids
            for c in dict_example_by_centroid:
                fnparray = np.vstack(dict_example_by_centroid[c])
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

        return dict_example_by_centroid
            
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


        
    # For question 3
    # Returns clusters with with their data examples
#    def cluster_content(self):


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
kmeans.learn(5, plot_error=0)
#print(kmeans.means())

# QUESTION 2: k optimization
kmeans_optimal = KMeans(digits1_x, 3)
example_dict = kmeans_optimal.learn(15, plot_error=1)

# It can be seen that the error is constant after 6 iterations, so k is optimal after 6 iterations
#print(kmeans.means())

# QUESTION 3: comparing found clusters with true clusters
print(true_mean_dataset(digits1_x, digits1_y))

# QUESTION 4:
# find_anomalies returns the examples of which the distance to the mean is
# greater than the threshold distance. These examples are the anomalies

anomalies = kmeans_optimal.find_anomalies(example_dict, 40)
# QUESTION 5:
# We have examined the anomalies by printing the 8 x 8 matrix of each training
# example which was classified as an anomaly. The shapes of the grey values
# all resembled the number 4. Since our algorithm uses only 3 clusters, and
# the number 4 does not occur frequently in the data, this number was classified
# as an anomaly. Using more clusters might result in having a better classification.

print("Number of anomalies: ", len(anomalies))

# prints 8 x 8 grey-value array
for a in anomalies:
    print(a.reshape(8, 8))
    print("\n")


# QUESTINO 6:
