import numpy as np

class GausianNaiveBayes():
    def __init__(self, training_data, class_values):
        self.training_data = training_data
        self.class_values = class_values
        self.training_data_size, _ = training_data.shape

    def _remove_zero_features(self, training_data):
        indexes = np.where(~training_data.any(axis=0))[0]
        new_training_data = np.delete(training_data, indexes, 1)
        return new_training_data

    def _splitdata(self):
        splitted_data_list = []
        for y in self.class_values:
            split = self.training_data[self.training_data[:, -1] == y]
            splitted_data_list.append(split)
        return splitted_data_list

    def _mean_and_var(self, array):
        array = np.delete(array, -1, 1) # remove class column
        means = np.mean(array, axis=0)
        variances = np.var(array, axis=0)
        return means, variances

    def _prior(self, array):
        size_array, _ = array.shape
        prior = size_array / self.training_data_size
        return prior

    def pdf(self, mean, variance, test_value):
        exps = (( -(test_value - mean) ** 2) / (2 * variance **2))
        return ( exps / np.sqrt(2 * np.pi * variance ** 2) )

    def posteriors(self, test_data_row):
        test_data_row = np.delete(test_data_row, -1) # remove class column
        split_by_class = self._splitdata()
        pdf_values_all_classes = []
        prior_values_all_classes = []

        for class_data in split_by_class:
            class_means, class_variances = self._mean_and_var(class_data)

            variance_zero_list = []
            for i, variance in enumerate(class_variances):
                if variance == 0:
                    variance_zero_list.append(i)

            pdf_values_class = 1
            for mean, variance, test_value in zip(class_means, class_variances, test_data_row):
                pdf_value = self.pdf(mean, variance, test_value)
                #if np.isnan(pdf_value) or pdf_value == 0: # FUCK THIS SHIT
                #    pdf_value = 1
                pdf_values_class *= pdf_value
            pdf_values_all_classes.append(pdf_values_class)

            prior_values_all_classes.append(self._prior(class_data))

        teller_values = []
        for pdf_value_class, prior_value in zip(pdf_values_all_classes, prior_values_all_classes):
            teller_values.append( pdf_value_class * prior_value )

        posteriors = []
        evidence = sum(teller_values)
        for value in teller_values:
            posteriors.append( value / evidence )
        return posteriors

    def accuracy(self, test_data):
        counter = 0
        totalrows, _ = test_data.shape
        for i, test_data_row in enumerate(test_data):
            posteriors = self.posteriors(test_data_row)
            max_class = posteriors.index(max(posteriors))
            if test_data[i][-1] == (max_class + 1):
                counter += 1
            print(test_data[i][-1], (max_class + 1))
        return (counter / totalrows) * 100

def read_file(csvfilename):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=0, dtype=np.float64)
    return array

if __name__ == "__main__":
    train_array = read_file("digist123-1.csv")
    test_array = read_file("digist123-2.csv")

    def _remove_zero_features(training_data, test_data):
        indexes = np.where(~training_data.any(axis=0))[0]
        new_training_data = np.delete(training_data, [0, 7, 8, 15, 16, 23, 24, 31, 32, 33, 39, 40, 47, 48, 56, 63], axis=1)
        new_test_data = np.delete(test_data, [0, 7, 8, 15, 16, 23, 24, 31, 32, 33, 39, 40, 47, 48, 56, 63], axis=1)
        return new_training_data, new_test_data

    new_train_array,  new_test_data = _remove_zero_features(train_array, test_array)
    gnb = GausianNaiveBayes(new_train_array, [1, 2, 3])
    print(gnb.accuracy(new_test_data))


    ##############
    # SKLEARN
    ##############
    def read_file(csvfilename, xcolumns, ycolumn):
        array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
        x = np.insert(array[:,xcolumns], 0, 1, axis=1)
        y = array[:,ycolumn]
        return x, y

    from sklearn.naive_bayes import GaussianNB
    x, y = read_file("digist123-1.csv", list(range(0, 64)), 64)
    x1, y2 = read_file("digist123-2.csv", list(range(0, 64)), 64)
    gnb = GaussianNB()
    y_pred = gnb.fit(x, y).predict(x1)
    print("Number of mislabeled points out of a total %d points : %d"
          % (x.shape[0],(y2 != y_pred).sum()))
