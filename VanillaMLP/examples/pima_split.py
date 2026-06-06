import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load
df = pd.read_csv('pima.csv', header=None)

# split features and labels
X = df.iloc[:, :8].values # first 8 columns - features
y = df.iloc[:, 8].values # last column - labels

# normalize features (mean 0, std 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 80/20 split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train_shape:", X_train.shape)
print("X_val_shape:", X_val.shape)
print("y_train_shape:", y_train.shape)
print("y_val_shape:", y_val.shape)