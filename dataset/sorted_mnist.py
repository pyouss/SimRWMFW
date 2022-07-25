import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv("mnist0.csv", header = None)
df_sort = df.sort_values(df.columns[0], ascending = True)
df_sort.to_csv("sorted_mnist.csv", index=False, header = False)