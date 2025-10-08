from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

path = "../artificial/"
dataframe, meta = arff.loadarff(path + '2d-3c-no123.arff')
x = np.array(dataframe['a0'],dtype=float)
y = np.array(dataframe['a1'],dtype=float)
plt.scatter(x,y)
plt.xlabel('a0')
plt.ylabel('a1')
plt.show()