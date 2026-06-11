import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

# data
df = pd.read_csv('../data/pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

def relu(A):
    return np.maximum(0, A)

def softmax(A):
    e = np.exp(A - np.max(A, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def wave_forward(X, W, b):
    return relu(X @ W + b)

def cross_entropy(probs, y_true):
    batch_size = probs.shape[0]
    correct_probs = probs[np.arange(batch_size), y_true]
    return -np.mean(np.log(correct_probs + 1e-8))

def accuracy(probs, y_true):    
    return np.mean(np.argmax(probs, axis=1) == y_true)

# Weights
W1 = he_init(8, 3)
b1 = np.zeros(3)
W_out = he_init(3, 2)
b_out = np.zeros(2)

lr = 0.01
epochs = 1000

for epoch in range(epochs):
    # forward pass
    A1 = wave_forward(X_train, W1, b1)
    probs = softmax(A1 @ W_out + b_out)

    # backward
    batch_size = X_train.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_train] -= 1
    dlogits /= batch_size

    dW_out = A1.T @ dlogits
    db_out = dlogits.sum(axis=0)

    dA1 = dlogits @ W_out.T
    dZ1 = dA1 * (A1 > 0)
    
    dW1 = X_train.T @ dZ1
    db1 = dZ1.sum(axis=0)

    # update
    W1 -= lr * dW1
    b1 -= lr * db1
    W_out -= lr * dW_out
    b_out -= lr * db_out

    if epoch % 100 == 0 or epoch == 999:
        val_A1 = wave_forward(X_val, W1, b1)
        val_probs = softmax(val_A1 @ W_out + b_out)
        loss = cross_entropy(probs, y_train)
        val_accuracy = accuracy(val_probs, y_val)
        train_accuracy = accuracy(probs, y_train)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")