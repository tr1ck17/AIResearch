import pandas as pd

df = pd.read_csv('pima.csv', header=None)
print(df.shape)
print(df.head())