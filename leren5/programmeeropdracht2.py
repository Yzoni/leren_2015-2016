'''

DISCUSSION:

Scipy is used to avoid dependence on the quality of my own implementations.

'''

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_rel
import numpy as np

def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = np.insert(array[:,xcolumns], 0, 1, axis=1)
    y = array[:,ycolumn]
    return x, y


if __name__ == "__main__":

    x, y = read_file("digist123-1.csv", list(range(0, 64)), 64)
    x1, y2 = read_file("digist123-2.csv", list(range(0, 64)), 64)

    # LOGISTIC REGRESSION CLASSIFICATION
    lg = LogisticRegression()
    y_pred_lg = lg.fit(x, y).predict(x1)


    # NAIVE BAYES CLASSIFICIATION
    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(x, y).predict(x1)


    # Dependendent T test because the the two algorithms are executed ont the same dataset
    tvalue, pvalue = ttest_rel(y_pred_lg, y_pred_gnb)
    if pvalue > 0.05:
        print("Null hypothesis can not be rejected, pvalue is " + str(pvalue))
    else:
        print("Null hypothesis must be rejected, pvalue is " + str(pvalue))
