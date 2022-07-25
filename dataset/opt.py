import scipy.io
import scipy.io
import numpy as np
import pandas as pd
import mat73
data_dict = np.array(mat73.loadmat('MNIST_LR_opt.mat'))
print(data_dict)
data = mat73.loadmat('MNIST_LR_opt.mat')
print(data) 

optimal = data_dict.item()['optimal_solution']
print(optimal.shape)

(pd.DataFrame(optimal)).to_csv("optimal_lr.csv", index = False, header = False, encoding='gbk',float_format='%.16f') 