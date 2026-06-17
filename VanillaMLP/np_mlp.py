import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

# Data
df = pd.read_csv('../data/pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit and transform training
X_val = scaler.transform(X_val) # ONLY transform validation

# Network
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(fan_in, fan_out): # better initialization for ReLU
    std = np.sqrt(2.0/fan_in)
    return np.random.randn(fan_in, fan_out) * std

n_inputs = 8
n_hidden = 14
n_classes = 2
    
W1 = he_init(n_inputs, n_hidden)
b1 = np.zeros(n_hidden)
W2 = he_init(n_hidden, n_classes)
b2 = np.zeros(n_classes)

# Forward
def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return A1, A2

# Loss
def cross_entropy_loss(probs, y_true):
    batch_size = probs.shape[0] # takes the number of rows in the probs matrix, stores it as a num in batch_size
    correct_probs = probs[np.arange(batch_size), y_true]
    # creates a range for the indexes of batch_size, and selects the y_true column for each one, resulting in an array the same size
    # as y_true 
    return -np.mean(np.log(correct_probs + 1e-8)) # gives us a resulting array

# Backward
def backward(X, y_true, A1, A2, W2):
    batch_size = X.shape[0]

    dZ2 = A2.copy()     # copy the probabilities into dZ2
    dZ2[np.arange(batch_size), y_true] -= 1     # grab the probabilities that align with the correct answers, find error amounts
    dZ2 /= batch_size   # average the loss

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (A1 > 0)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return dW1, db1, dW2, db2

# Update
def update_weights(dW1, db1, dW2, db2, lr=0.01):
    global W1, b1, W2, b2
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# Accuracy
def accuracy(X, y_true):
    _, probs = forward(X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)

# Training loop
epochs = 1000
lr = 0.01

for epoch in range(epochs):
    # forward
    A1, A2 = forward(X_train)

    # loss
    loss = cross_entropy_loss(A2, y_train)

    # backward
    dW1, db1, dW2, db2 = backward(X_train, y_train, A1, A2, W2)

    # update
    update_weights(dW1, db1, dW2, db2, lr)

    # log every 100 epochs
    if epoch % 100 == 0:
        train_acc = accuracy(X_train, y_train)
        val_acc = accuracy(X_val, y_val)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    if epoch == 999:
        train_acc = accuracy(X_train, y_train)
        val_acc = accuracy(X_val, y_val)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")