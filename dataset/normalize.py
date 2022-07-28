import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


data = pd.read_csv("mnist0.csv", header = None).to_numpy()

data[:,1:] = data[:,1:]/255.0

pd.DataFrame(data).to_csv("mnist0.csv", index=False, header = False)
