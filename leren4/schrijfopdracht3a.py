#!/bin/env python3.4

import numpy as np
from scipy.stats.stats import pearsonr

def read_file(csvfilename, xcolumns, ycolumn):
    array = np.loadtxt(csvfilename, delimiter=';', skiprows=1, dtype=int)
    x = array[:,xcolumns].flatten().tolist()
    y = array[:,ycolumn].flatten().tolist()
    return x, y

x1, y = read_file('housesRegr.csv', [1], 4)
x2, _ = read_file('housesRegr.csv', [2], 4)
x3, _ = read_file('housesRegr.csv', [3], 4)

# Bedrooms:
correlation_bedrooms = print(pearsonr(x1, y))

# Bathrooms:
correlation_bathrooms = print(pearsonr(x2, y))

# Size:
correlation_size = print(pearsonr(x3, y))