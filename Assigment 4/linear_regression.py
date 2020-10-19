# Yash Naik, Na Li

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

data= pd.read_csv("linear-regression.txt", header = None, delimiter = ',')
data.head()

label = data[2]
label.head()

data = data.drop(columns=[2])
data.head()

warnings.filterwarnings("ignore")
Z = label[:,np.newaxis]
#print(Z)
padding = np.ones((data.shape[0], 1))
X = np.array(data)
X = np.concatenate((X, padding), axis = 1)
#print(X)

weights = (np.mat(X.T)* np.mat(X)).I * np.mat(X.T) * np.mat(Z)
weights = weights.T
print("After the Final iteration, the weights are: ",weights)