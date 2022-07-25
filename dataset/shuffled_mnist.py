import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 80000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("mnist0.csv", header = None)

for i in range(1, 11):
    shuffled = df.sample(frac=1)
    filename = "mnist"+ str(i) + ".csv"
    shuffled.to_csv(filename, index=False, header = False)
    print(filename+ " done")
