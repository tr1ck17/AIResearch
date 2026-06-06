import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data
df = pd.read_csv('pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_val = scaler.transform(X)

