import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.array([0.5, 1.2, -0.3])

w = np.array([0.4, -0.1, 0.7])
b = 0.1

z = np.dot(w, x) + b
output = relu(z)

print(f"z = {z:.4f}")
print(f"output = {output:.4f}")