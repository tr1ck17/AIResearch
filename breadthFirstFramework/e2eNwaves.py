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

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(i, o):
    return np.random.randn(i, o) * np.sqrt(2.0 / i)

def wave_forward(X, W, b):
    return relu(X @ W + b)

def cross_entropy(probs, y_true):
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-8))

def accuracy(probs, y_true):
    return np.mean(np.argmax(probs, axis=1) == y_true)

def all_waves_forward(X, waves):
    if len(waves) == 0:
        return np.empty((X.shape[0], 0))
    return np.hstack([wave_forward(X, W, b) for W, b in waves])

def apply_penalty(mode, lam, fo, no, Wn, frozen_W, dA, n):
    dW_pen = 0.0
    if mode == "uncentered":
        dA = dA + lam * 2 * (fo @ (fo.T @ no)) / (n**2)
    elif mode == "centered":
        Fc, Nc = fo - fo.mean(0), no - no.mean(0)
        g = 2 * (Fc @ (Fc.T @ Nc)) / (n**2)
        dA = dA + lam * (g - g.mean(0))
    elif mode == "cosine_act":
        Fc, Nc = fo - fo.mean(0), no - no.mean(0)
        fn = np.linalg.norm(Fc, axis=0) + 1e-12
        U  = Fc / fn
        ng2 = (Nc * Nc).sum(0) + 1e-12
        MG  = U @ (U.T @ Nc)
        Sb  = (Nc * MG).sum(0) / ng2
        gg  = 2 * (MG - Nc * Sb) / ng2
        dA = dA + lam * (gg - gg.mean(0))
    elif mode == "weight_dec":
        Wf = np.hstack(frozen_W)
        fn = np.linalg.norm(Wf, axis=0) + 1e-12
        Uf = Wf / fn
        wn2 = (Wn * Wn).sum(0) + 1e-12
        MW  = Uf @ (Uf.T @ Wn)
        Sj  = (Wn * MW).sum(0) / wn2
        dW_pen = lam * 2 * (MW - Wn * Sj) / wn2
    return dA, dW_pen

# Config 
n_waves   = 2
wave_size = 3
n_inputs  = 8
n_classes = 2
lr        = 0.01
epochs    = 1000

# Matrix toggles
freeze_old_paths = True # true means lock frozen wave rows in W_out, False = let them drift
penalty_mode = "uncentered"     # "off" | "uncentered" | "centered" | "cosine_act" | "weight_dec"
lambda_ = 0.001

# State 
waves  = []
W_out  = np.empty((0, n_classes))
b_out  = np.zeros(n_classes)

# Training loop
for wave_idx in range(n_waves):
    W_new = he_init(n_inputs, wave_size)
    b_new = np.zeros(wave_size) 

    new_slots = np.random.randn(wave_size, n_classes) * 1e-3
    W_out = np.vstack([W_out, new_slots])

    print(f"\n── Training Wave {wave_idx + 1} ──")

    for epoch in range(epochs):
        # Forward
        frozen_out = all_waves_forward(X_train, waves)
        new_out    = wave_forward(X_train, W_new, b_new)
        A_cat      = np.hstack([frozen_out, new_out]) if frozen_out.shape[1] > 0 else new_out
        probs      = softmax(A_cat @ W_out + b_out)

        batch_size = X_train.shape[0]

        # Backward
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y_train] -= 1
        dlogits /= batch_size

        dW_out  = A_cat.T @ dlogits
        db_out  = dlogits.sum(axis=0)

        # structural freezing
        if freeze_old_paths and wave_idx > 0:
            frozen_rows = wave_idx * wave_size
            dW_out[:frozen_rows, :] = 0.0

        dA_cat = dlogits @ W_out.T
        dA_new = dA_cat[:, wave_idx * wave_size:]

        # conceptual diversity (mode selected in config; shared with cv_evaluation.py)
        dW_pen = 0.0
        if penalty_mode != "off" and frozen_out.shape[1] > 0:
            frozen_W = [W for W, b in waves]
            dA_new, dW_pen = apply_penalty(penalty_mode, lambda_, frozen_out, new_out,
                                           W_new, frozen_W, dA_new, batch_size)

        dZ_new = dA_new * (new_out > 0)
        dW_new = X_train.T @ dZ_new + dW_pen
        db_new = dZ_new.sum(axis=0)

        # updating only new wave + W_out
        W_new -= lr * dW_new
        b_new -= lr * db_new
        W_out -= lr * dW_out
        b_out -= lr * db_out

        if epoch % 100 == 0 or epoch == 999:
            frozen_val = all_waves_forward(X_val, waves)
            new_val    = wave_forward(X_val, W_new, b_new)
            A_cat_val  = np.hstack([frozen_val, new_val]) if frozen_val.shape[1] > 0 else new_val
            val_probs  = softmax(A_cat_val @ W_out + b_out)
            val_accuracy = accuracy(val_probs, y_val)
            loss       = cross_entropy(probs, y_train)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Val Acc: {val_accuracy:.4f}")

    waves.append((W_new, b_new))

# XAI feature audit
import matplotlib.pyplot as plt

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"
]

print("\n Running XAI Feature Audit ")
# note: weight magnitudes correlate with feature importance here specifically because inputs are normalized via StandardScaler

plt.figure(figsize=(12, 8))
x_indices = np.arange(len(feature_names))
bar_width = 0.18

# extracting and plotting feature importances for each wave
for idx, (W_wave, b_wave) in enumerate(waves):
    # calculating absolute magnitude averaged across the wave's hidden neurons
    feature_importance = np.mean(np.abs(W_wave), axis=1)

    # center group bars
    plt.bar(
        x_indices + (idx - (n_waves - 1) / 2) * bar_width,
        feature_importance,
        width=bar_width,
        label=f"Wave {idx+1}"
    )

    # to print feature profile (in full) to console
    print(f"\nWave {idx+1} Full Feature Profile:")
    sorted_pairs = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
    for feat, val in sorted_pairs:
        print(f" {feat}: {val:.4f}")

# formatting chart
plt.title("XAI Feature Audit: How Each Wave Prioritizes Raw Patient Biometrics", fontsize=14, fontweight='bold')
plt.xlabel("Physical Features", fontsize=12)
plt.ylabel("Mean Absolute Weight Magnitude (Feature Importance)", fontsize=12)
plt.xticks(x_indices, feature_names, rotation=15)
plt.legend()
plt.tight_layout()

# save the plot directly to research folder
out_name = f"xai_feature_audit_{n_waves}waves_{penalty_mode}.png"
plt.savefig(out_name, dpi=300)
print(f"\n[SUCCESS] Feature audit saved as '{out_name}'.")
plt.show()