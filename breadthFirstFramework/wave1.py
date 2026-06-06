import numpy as np

def relu(x):
    return np.maximum(0, x)

def wave_forward(X, W, b):  # just wrapping the math Z1 = X @ W1 + b1 in a function since we're calling it for as
    # many waves as we're using
    return relu(X @ W + b)

def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in))
    return np.random.randn(fan_in, fan_out) * std

# wave 1
W1 = he_init(8, 3)
b1 = np.zeros(3)

# quick sanity check
X_fake = np.random.randn(5, 8) # 5 samples, 8 features
out = wave_forward(X_fake, W1, b1)
print(out.shape)    # should be (5, 3)
print(out)  # all values >= 0 (ReLU)