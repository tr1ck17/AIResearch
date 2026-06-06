import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # subtract max for numerical stability
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

# network dimensions
n_inputs = 8 # pima has 8 features
n_hidden = 16 # hidden layer size, hyperparameter
n_classes = 2 # diabetic or not

# initialize weights and biases
W1 = xavier_init(n_inputs, n_hidden) # (8, 16)
b1 = np.zeros(n_hidden) # (16,)

W2 = xavier_init(n_hidden, n_classes) # (16, 2)
b2 = np.zeros(n_classes) # (2,)

def forward(X):
    # hidden layer
    Z1 = X @ W1 + b1    # (batch, 16)
    A1 = relu(Z1)   # (batch, 16)

    # output layer
    Z2 = A1 @ W2 + b2   # (batch, 2)
    A2 = softmax(Z2)    # (batch, 2) - probabilities

    return A1, A2   # return hidden activations too (needed for backprop)

# fake batch of 4 samples with 8 features
X_fake = np.random.randn(4, 8)
A1, probs = forward(X_fake)

print("Hidden activations shape:", A1.shape) # (4, 16)
print("Output probabilities shape:", probs.shape) # (4, 2)
print("Sample output (should sum to 1):", probs[0])
print("Sum:", probs[0].sum())