import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -- Data --
df = pd.read_csv('../../data/pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# base functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(i, o):
    return np.random.randn(i, o) * np.sqrt(2.0 / i)

def cross_entropy(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-8))

def accuracy(p, y):
    return np.mean(np.argmax(p, axis=1) == y)

# config hyperparams
N_WAVES   = 5
WAVE_SIZE = 3   # neurons per wave
N_INPUTS  = 8
N_CLASSES = 2
LR        = 0.01
EPOCHS    = 1000

np.random.seed(42)

# constructive wave architecture
# each wave is a small group of ReLU neurons that reads the raw input directly
# waves are added one at a time; once a wave is trained, it is frozen
# all waves feed into a single shared output layer that grows with each wave

waves = []                              # list of (W, b) for each trained wave
W_out = np.empty((0, N_CLASSES))       # output layer that grows as waves are added (init to 0)
b_out = np.zeros(N_CLASSES)

for wave_idx in range(N_WAVES):
    W_new = he_init(N_INPUTS, WAVE_SIZE)
    b_new = np.zeros(WAVE_SIZE)
    W_out = np.vstack([W_out, np.random.randn(WAVE_SIZE, N_CLASSES) * 1e-3])

    print(f"\n── Training Wave {wave_idx + 1} ──")

    for epoch in range(EPOCHS):
        # forward pass: frozen waves produce fixed activations; only the new wave trains
        frozen_acts = np.hstack([relu(X_train @ W + b) for W, b in waves]) if waves else np.empty((len(X_train), 0))
        new_acts    = relu(X_train @ W_new + b_new)
        A           = np.hstack([frozen_acts, new_acts]) if frozen_acts.shape[1] else new_acts
        probs       = softmax(A @ W_out + b_out)

        n = len(X_train)
        dlogits = probs.copy()
        dlogits[np.arange(n), y_train] -= 1
        dlogits /= n

        dW_out = A.T @ dlogits
        db_out = dlogits.sum(0)

        # zero out gradients for frozen waves' output rows (we don't want them moving/changing)
        if wave_idx > 0:
            dW_out[:wave_idx * WAVE_SIZE] = 0.0

        dA_new = (dlogits @ W_out.T)[:, wave_idx * WAVE_SIZE:]
        dZ_new = dA_new * (new_acts > 0)
        dW_new = X_train.T @ dZ_new
        db_new = dZ_new.sum(0)

        W_new -= LR * dW_new
        b_new -= LR * db_new
        W_out -= LR * dW_out
        b_out -= LR * db_out

    waves.append((W_new, b_new))

    # show how accuracy improves as each wave is added
    val_acts  = np.hstack([relu(X_val @ W + b) for W, b in waves])
    val_probs = softmax(val_acts @ W_out + b_out)
    print(f"After Wave {wave_idx + 1}: val accuracy = {accuracy(val_probs, y_val):.4f}")
