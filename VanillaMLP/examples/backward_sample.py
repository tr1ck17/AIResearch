import numpy as np

dZ2 = np.array([
    [0.12, 0.88],
    [0.76, 0.24],
    [0.33, 0.67],
    [0.91, 0.09]
])

y_true = np.array([1, 0, 1, 0])
batch_size = 4

print("dZ2:\n", dZ2)
print()
print("np.arange(batch_size):", np.arange(batch_size))
print()
print("y_true:", y_true)
print()
print("dZ2[np.arange(batch_size), y_true]:", dZ2[np.arange(batch_size), y_true])

dZ2[np.arange(batch_size), y_true] -= 1
print("dZ2 after -= 1:\n", dZ2)