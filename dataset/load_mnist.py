import pandas as pd
import mat73
import scipy.io
import numpy as np

data_dict = scipy.io.loadmat('MNIST_dataset.mat')

data = np.zeros((60000,785))
print(data_dict["Xtrain"].shape)
print(data_dict["ytrain"].shape)
data_dict["ytrain"] -= 1
data[:,0] = data_dict["ytrain"].reshape(60000,)
data[:,1:] = data_dict["Xtrain"].T
print(data[0,data[0,1:].nonzero()])

#data_array = data_dict["optimal_solution"]
#data_value = data_dict["optimal_value"]

df = pd.DataFrame(data)
#df.to_csv("optimal_lr.csv", header=False)
df.to_csv("mnist0.csv",index=False,header=False)