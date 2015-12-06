import numpy as np

# Initialize with training data and the possible class values
# EXAMPLE:
# gnn = GausianNaiveBayes([[1 2 1] [5 2 0] [0 4 1]], [0, 1])
# gnn.posteriors([1 2])
# gnn.accuracy
##
# NOTE: possible classes list needs to be a serie of consecutive numbers
##
class GausianNaiveBayes():
    def __init__(self, training_data, class_values):
        self.training_data = training_data
        self.class_values = class_values
        self.training_data_size, _ = training_data.shape

    # Splits data in sublists by class
    def _splitdata(self):
        splitted_data_list = []
        for y in self.class_values:
            split = self.training_data[self.training_data[:, -1] == y]
            splitted_data_list.append(split)
        return splitted_data_list

    # Computes the mean and var columnwise of an vector
    def _mean_and_var(self, array):
        array = np.delete(array, -1, 1) # remove class column
        means = np.mean(array, axis=0)
        variances = np.var(array, axis=0)
        return means, variances

    # Computes the prior by dividing the size of the class vector by the size of the vector
    # of all classes
    def _prior(self, array):
        size_array, _ = array.shape
        prior = size_array / self.training_data_size
        return prior

    # Computes the probability density function given sigma, mu and an test example value
    def pdf(self, mean, variance, test_value):
        exps = np.exp(( -(test_value - mean) ** 2) / (2 * variance **2))
        return ( exps / np.sqrt(2 * np.pi * variance ** 2) )

    def posteriors(self, test_data_row):
        test_data_row = np.delete(test_data_row, -1) # remove class column (TODO: should be done before this function)
        split_by_class = self._splitdata() # Get a list of class vectors.
        pdf_values_all_classes = []
        prior_values_all_classes = []

        # Loop over the class vectors separately
        for class_data in split_by_class:
            class_means, class_variances = self._mean_and_var(class_data) # Get mean and variance of an class vector

            pdf_values_class = 1
            for mean, variance, test_value in zip(class_means, class_variances, test_data_row):
                # Avoid division by zero, if variance is 1 make pdf_value 1
                if variance == 0:
                    pdf_value = 1
                else:
                    pdf_value = self.pdf(mean, variance, test_value)

                pdf_values_class *= pdf_value

            # Stores the pdf and prior values as an element in a list
            pdf_values_all_classes.append(pdf_values_class)
            prior_values_all_classes.append(self._prior(class_data))

        # Store the values above the dividing sign as element by class
        teller_values = []
        for pdf_value_class, prior_value in zip(pdf_values_all_classes, prior_values_all_classes):
            teller_values.append(pdf_value_class * prior_value)

        # Sum the evidence
        evidence = sum(teller_values)

        # Store all posteriors by class as element in list
        posteriors = []
        for value in teller_values:
            posteriors.append(value / evidence)
        return posteriors

    # Computes the accuracy percentage by dividing the amount of correctly guessed classes by the total amount
    # of classes in the test-set.
    def accuracy(self, test_data):
        counter = 0
        totalrows, _ = test_data.shape
        for i, test_data_row in enumerate(test_data):
            posteriors = self.posteriors(test_data_row)
            max_class = posteriors.index(max(posteriors))
            if test_data[i][-1] == (max_class + 1):
                counter += 1
        return (counter / totalrows) * 100

    def max_prob_misclassified(self, test_data):
        l = []

        for i, test_data_row in enumerate(test_data):
            posteriors = self.posteriors(test_data_row)
            max_class = posteriors.index(max(posteriors))
            if test_data[i][-1] != (max_class + 1):
                l.append([i, test_data[i][-1], max_class + 1])

        r = []
        p = []
        for e in self.class_values:
            count1 = 0
            count2 = 0
            for li in l:
                if e == li[1]:
                    count1 += 1
                if e == li[2]:
                    count2 += 1
            r.append(count1)
            p.append(count2)
        return r, p

# Reads a csv-file into a vector given a file name.
def read_file(csvfilename):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=0, dtype=np.float64)
    return array

if __name__ == "__main__":
    train_array = read_file("digist123-1.csv")
    test_array = read_file("digist123-2.csv")

    gnb = GausianNaiveBayes(train_array, [1, 2, 3])
    print("Percentage correct: " + str(gnb.accuracy(test_array)) + "%")

    most_often_wrong, most_often_wronged_by = gnb.max_prob_misclassified(test_array)
    print("Count of wrong guessed classes by test class: " + str(most_often_wrong))
    print("Count of guessed classes but were wrong: by class: " + str(most_often_wronged_by))

    '''
    DISCUSSION 1b

    It can be seen that the most wrongly guessed digit is digit 2, which is guessed wrong
    68 times. The digit most often guessed wrong is digit 3, this digit was wrongly substituted
    76 in classes 1 or 2. There seems to be a systematic problem.

    A possible solution to this problem without changing the dataset is to add a bias so
    more 2s classified as 3s classify to 2s.

    '''