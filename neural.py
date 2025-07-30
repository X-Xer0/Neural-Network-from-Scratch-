import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("/home/xer0/Neuron/train.csv")

#print(data.head())

data = np.array(data)
m, n = data.shape
#print(data.shape)
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:]/255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:]/255.
_, m_train = X_train.shape

'''print(X_train)
print(Y_train)
print(X_train.shape)
print(Y_train.shape)'''



