import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wave_core import relu, softmax, he_init, cross_entropy, acc

# -- Data --
df = pd.read_csv('../../data/pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# -- Config --
N_WAVES   = 5
WAVE_SIZE = 3
N_INPUTS  = 8
N_CLASSES = 2
LR        = 0.01
EPOCHS    = 1000
SEED      = 42

# -- Vanilla Breadth-First Wave Net (no diversity penalty) --
rng    = np.random.default_rng(SEED)
waves  = []
W_out  = np.empty((0, N_CLASSES))
b_out  = np.zeros(N_CLASSES)
n      = len(X_train)

for wave_idx in range(N_WAVES):
    W_new = he_init(rng, N_INPUTS, WAVE_SIZE)
    b_new = np.zeros(WAVE_SIZE)
    W_out = np.vstack([W_out, rng.standard_normal((WAVE_SIZE, N_CLASSES)) * 1e-3])

    print(f"\n── Training Wave {wave_idx + 1} ──")

    for _ in range(EPOCHS):
        fo = np.hstack([relu(X_train @ W + b) for W, b in waves]) if waves else np.empty((n, 0))
        no = relu(X_train @ W_new + b_new)
        A  = np.hstack([fo, no]) if fo.shape[1] else no
        p  = softmax(A @ W_out + b_out)

        grad_logits = p.copy()
        grad_logits[np.arange(n), y_train] -= 1
        grad_logits /= n

        grad_W_out = A.T @ grad_logits
        grad_b_out = grad_logits.sum(0)

        if wave_idx > 0:
            grad_W_out[:wave_idx * WAVE_SIZE] = 0.0

        grad_new_acts   = (grad_logits @ W_out.T)[:, wave_idx * WAVE_SIZE:]
        grad_new_preact = grad_new_acts * (no > 0)

        W_new -= LR * (X_train.T @ grad_new_preact)
        b_new -= LR * grad_new_preact.sum(0)
        W_out -= LR * grad_W_out
        b_out -= LR * grad_b_out

    waves.append((W_new, b_new))

    val_fo    = np.hstack([relu(X_val @ W + b) for W, b in waves])
    val_probs = softmax(val_fo @ W_out + b_out)
    print(f"After Wave {wave_idx + 1}: val accuracy = {acc(val_probs, y_val):.4f}")
