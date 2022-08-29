import pandas as pd
import mat73

data_dict = mat73.loadmat('MNIST_LR_opt.mat')
print(data_dict)
data_array = data_dict["optimal_solution"]
data_value = data_dict["optimal_value"]

df = pd.DataFrame(data_array)
df.to_csv("optimal_lr.csv",index=False ,header=False)
print(data_value)
