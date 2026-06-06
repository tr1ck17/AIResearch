import numpy as np

def relu(x):
    return np.maximum(0, x)

# 4 samples, 3 features each (e.g., 4 patients, 3 measurements)
X = np.array([
    [0.5,  1.2, -0.3],
    [1.1, -0.4,  0.8],
    [-0.2, 0.9,  0.5],
    [0.3,  0.1, -0.9],
])

# layer with 5 neurons
n_inputs = 3
n_neurons = 5

# random init for now (Xavier comes next)
W = np.random.randn(n_inputs, n_neurons) * 0.1  # shape: (3, 5)
b = np.zeros(n_neurons)                         # shape: (5,)

Z = X @ W + b       # shape: (4, 5) - 4 samples, 5 neuron outputs each
A = relu(Z)         # same shape as Z but with nonlinearity applied via ReLU

print(f"Z shape: {Z.shape}")
print(f"A shape: {A.shape}")
print(A)