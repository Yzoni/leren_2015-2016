import numpy as np

def likelihood(std, mean, x):
    exp = np.exp(-(x-mean)**2 / (2 * std**2))
    return exp / np.sqrt(2 * np.pi * std**2)

def printlikelihoodlatex(std, mean, x):
    print("$ \\frac{1}{\sqrt{2 \pi " + str(std) + "^2}} e^{\\frac{-(" + str(x) + "- " + str(mean) + ")^2}{2 \cdot" + str(std) + "^2})} = " + str(likelihood(std, mean, x)) + " $ \\\ ")


x_list = [1, 2, 3, 3, 4, 5, 5, 7, 10, 11, 13, 14, 15, 17, 20, 21]

prior1 = 1/8
prior3 = 5/16
prior8 = 9/16

likelihood_list_1 = []
likelihood_list_3 = []
likelihood_list_8 = []
for x in x_list:
    likelihood_list_1.append(likelihood(2, 1, x))
    likelihood_list_3.append(likelihood(2, 3, x))
    likelihood_list_8.append(likelihood(2, 8, x))

# bereken teller breuk
teller_list_1 = []
for likelihoodz in likelihood_list_1:
    teller_list_1.append(likelihoodz * prior1)

teller_list_3 = []
for likelihoodz in likelihood_list_3:
    teller_list_3.append(likelihoodz * prior3)

teller_list_8 = []
for likelihoodz in likelihood_list_8:
    teller_list_8.append(likelihoodz * prior8)


# bereken evidence (noemer breuk) per example so 16 evidences
evidences = []
for teller1, teller3, teller8 in zip(teller_list_1, teller_list_3, teller_list_8):
    evidences.append(teller1 + teller3 + teller8)


# calculate bayes
# for cluster 1
cluster1_weights = []
for teller1, evidence in zip(teller_list_1, evidences):
    cluster1_weights.append(teller1 / evidence)
print(cluster1_weights)

# cluster 3
cluster3_weights = []
for teller3, evidence in zip(teller_list_3, evidences):
    cluster3_weights.append(teller3 / evidence)
print(cluster3_weights)

cluster8_weights = []
for teller8, evidence in zip(teller_list_8, evidences):
    cluster8_weights.append(teller8 / evidence)
print(cluster8_weights)

print("\n")
print("NEW MEANS")
# calculate new mean
# new mean cluster 1
summation = 0
for cluster1_weight, x in zip(cluster1_weights, x_list):
    summation = cluster1_weight * x
newmean1 = (summation / sum(cluster1_weights))
print(newmean1)

# new mean cluster 3
summation = 0
for cluster3_weight, x in zip(cluster3_weights, x_list):
    summation = cluster3_weight * x
newmean3 = summation / sum(cluster3_weights)
print(newmean3)

# new mean cluster 8
summation = 0
for cluster8_weight, x in zip(cluster8_weights, x_list):
    summation = cluster8_weight * x
newmean8 = summation / sum(cluster8_weights)
print(newmean8)

print("\n")
print("NEW VARIANCES")
# new variance of clusters
# new variance cluster 1
squared_differences = []
for x in x_list:
    squared_differences.append( (x - newmean1)**2 )
newvariance1 = sum(squared_differences) / len(x_list)
print(newvariance1)


# new variance cluster 3
squared_differences = []
for x in x_list:
    squared_differences.append( (x - newmean3)**2 )
newvariance3 = sum(squared_differences) / len(x_list)
print(newvariance3)

# new variance cluster 8
squared_differences = []
for x in x_list:
    squared_differences.append( (x - newmean8)**2 )
newvariance8 = sum(squared_differences) / len(x_list)
print(newvariance8)


# New Standard deviations
print("\n")
print("NEW STD")
newstd1 = np.sqrt(newvariance1)
print(newstd1)
newstd3 = np.sqrt(newvariance3)
print(newstd3)
newstd8 = np.sqrt(newvariance8)
print(newstd8)


# New likelihood
print("\n")
print("NEW LIKELIHOOD")
for x in x_list:
    printlikelihoodlatex(newstd1, newmean1, x)
    printlikelihoodlatex(newstd3, newmean3, x)
    printlikelihoodlatex(newstd8, newmean8, x)