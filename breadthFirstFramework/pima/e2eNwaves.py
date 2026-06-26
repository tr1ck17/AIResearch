import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wave_core import (softmax, cross_entropy, acc as accuracy,
                       apply_penalty, wave_forward, all_waves_forward)
np.random.seed(42)

def he_init(i, o):
    return np.random.randn(i, o) * np.sqrt(2.0 / i)

# Data
df = pd.read_csv('../../data/pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Config 
n_waves   = 5
wave_size = 3
n_inputs  = 8
n_classes = 2
lr        = 0.01
epochs    = 1000

# Matrix toggles
freeze_old_paths = True # true means lock frozen wave rows in W_out, False = let them drift
penalty_mode = "weight_dec"     # "off" | "uncentered" | "centered" | "cosine_act" | "weight_dec"
#lambda_ = 0.001
lambda_ = 1.00

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

# Faithfulness Test (wave-level): does the chart's prominence track real contribution?
# the bar chart ranks waves by INPUT-weight magnitude. It never checks W_out
# This test removes each wave's OUTPUT(its W_out rows) and measures the accuracy drop
# if prominent waves cause bigger drops, the chart is faithful. if not, it's cosmetic

print("\n FAITHFULNESS TEST: wave prominence vs. actual contribution ")

# baseline: full trained net on validation set
frozen_val = all_waves_forward(X_val, waves)
full_probs = softmax(frozen_val @ W_out + b_out)
baseline_acc = accuracy(full_probs, y_val)

prominences = []
drops = []
for i in range(n_waves):
    #prominence = total bar height for this wave (sum of its 8 feature bars)
    W_wave = waves[i][0]
    prominence = np.sum(np.mean(np.abs(W_wave), axis=1))

    # ablate: zero this wave's 3 output rows, leave everything else intact
    W_out_ablated = W_out.copy()
    W_out_ablated[i*wave_size:(i+1)*wave_size, :]= 0.0

    ablated_probs = softmax(frozen_val @ W_out_ablated + b_out)
    ablated_acc = accuracy(ablated_probs, y_val)
    drop = baseline_acc - ablated_acc

    prominences.append(prominence)
    drops.append(drop)

prominences = np.array(prominences)
drops = np.array(drops)

print(f"\n baseline val acc: {baseline_acc:.4f}")
print(f"    raw drops per wave: {np.round(drops, 4)}")
    
if n_waves < 3:
    print(f" Only {n_waves} waves - need >= 3 points for correlation.")
elif drops.std() < 1e-9:
    print("\n Removing any wave changed nothing - net is fully redundant on this data ")
    print(" Inconclusive: can't distinguish faithful-but-redundant from cosmetic here ")
else:
    corr = np.corrcoef(prominences, drops)[0, 1]
    print(f"\n baseline val acc: {baseline_acc:.4f}")
    print(f" > 0: prominence-vs-drop correlation (5 waves): {corr:+.3f}")
    print(" > 0: prominent waves contribute more -> profile FAITHFUL (directional, n=5)")
    print(" ~ 0 or < 0: bars don't track contribution -> profile may be COSMETIC")
    print(" NOTE: n = 5 waves -> directional sanity check, not a statistical result.")