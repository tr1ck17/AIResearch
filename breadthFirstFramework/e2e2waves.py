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
    return np.random.randn(fan_in, fan_out) * std # randn rather than rand

def wave_forward(X, W, b):
    return relu(X @ W + b)

def cross_entropy(probs, y_true):
    batch_size = probs.shape[0]
    correct_probs = probs[np.arange(batch_size), y_true]
    return -np.mean(np.log(correct_probs + 1e-8))

def accuracy(probs, y_true):
    return np.mean(np.argmax(probs, axis=1) == y_true)

# Wave 1
W1 = he_init(8, 3)
b1 = np.zeros(3)
W_out = he_init(3, 2)
b_out = np.zeros(2)

lr = 0.01
epochs = 1000

print("Training Wave 1:")
for epoch in range(epochs):
    A1 = wave_forward(X_train, W1, b1)
    probs = softmax(A1 @ W_out + b_out) # get the actual probabilities from forward passing

    batch_size = X_train.shape[0]   # ready to divide results by the batch size
    dlogits = probs.copy()          
    dlogits[np.arange(batch_size), y_train] -= 1
    dlogits /= batch_size       # effectively have 

    dW_out = A1.T @ dlogits
    db_out = dlogits.sum(axis=0)

    dA1 = dlogits @ W_out.T
    dZ1 = dA1 * (A1 > 0)
    
    dW1 = X_train.T @ dZ1
    db1 = dZ1.sum(axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W_out -= lr * dW_out
    b_out -= lr * db_out

    if epoch % 100 == 0 or epoch == 999:
        val_A1 = wave_forward(X_val, W1, b1)
        val_probs = softmax(val_A1 @ W_out + b_out)
        loss = cross_entropy(probs, y_train)
        val_accuracy = accuracy(val_probs, y_val)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Val Acc: {val_accuracy:.4f}")

# now wave 1 frozen

# Wave 2
W2 = he_init(8, 3)
b2 = np.zeros(3)
# init wave 2 output weights to a tiny variance to break the cold-start
W_out = np.vstack([W_out, np.zeros((3, 2))])    # (6, 2)
b_out = b_out

print("\n-- Training Wave 2 (Wave 1 Frozen) --")
for epoch in range(epochs):
    # forward - both waves see raw input
    A1 = wave_forward(X_train, W1, b1)  # frozen, no grad
    A2 = wave_forward(X_train, W2, b2)  # trainable
    A_cat = np.hstack([A1, A2])         # (batch, 6)
    probs = softmax(A_cat @ W_out + b_out)

    # backward (only wave 2 + W_out update)
    batch_size = X_train.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_train] -= 1
    dlogits /= batch_size
    
    dW_out = A_cat.T @ dlogits
    db_out = dlogits.sum(axis=0)

    dA_cat = dlogits @ W_out.T          # (batch, 6)
    dA2 = dA_cat[:, 3:]                 # only Wave 2's slice
    dZ2 = dA2 * (A2 > 0)

    dW2 = X_train.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dW_out[:3, :] = 0.0

    W2 -= lr * dW2
    b2 -= lr * db2
    W_out -= lr * dW_out
    b_out -= lr * db_out

    if epoch % 100 == 0 or epoch == 999:
        A1_v = wave_forward(X_val, W1, b1)
        A2_v = wave_forward(X_val, W2, b2)
        A_cat_v = np.hstack([A1_v, A2_v])
        val_probs = softmax(A_cat_v @ W_out + b_out)
        loss = cross_entropy(probs, y_train)
        val_accuracy = accuracy(val_probs, y_val)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Val Acc: {val_accuracy:.4f}")


# XAI Feature Audit Visualization

import matplotlib.pyplot as plt

# 1. Calculate feature importance (mean absolute weight magnitude across hidden units)
# W1 and W2 both have shapes (8, 3) -> mapping 8 inputs to 3 hidden units
importance_wave1 = np.mean(np.abs(W1), axis=1)
importance_wave2 = np.mean(np.abs(W2), axis=1)

# 2. Define standard Pima Indian dataset feature labels
feature_names = [
    "Pregnancies", "Glucose", "Blood Press.", "Skin Thick.", 
    "Insulin", "BMI", "Pedigree Func.", "Age"
]

# 3. Set up the bar chart positioning
x = np.arange(len(feature_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for Wave 1 and Wave 2 side-by-side
rects1 = ax.bar(x - width/2, importance_wave1, width, label='Wave 1 (Base Knowledge)', color='#1f77b4')
rects2 = ax.bar(x + width/2, importance_wave2, width, label='Wave 2 (No Diversity Penalty)', color='#ff7f0e')

# 4. Add styling, labels, and legends
ax.set_ylabel('Mean Absolute Weight Magnitude', fontsize=12)
ax.set_title('Feature Importance Audit: Wave 1 vs Wave 2 (Redundancy Check)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=15, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the visual artifact for your check-in
plt.savefig('xai_feature_audit__manual_2waves.png', dpi=300)
print("\n[SUCCESS] Visual feature audit saved as 'xai_feature_audit_2waves.png'")
plt.show()