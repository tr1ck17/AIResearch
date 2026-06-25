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

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(i, o):
    return np.random.randn(i, o) * np.sqrt(2.0 / i)

def wave_forward(X, W, b):
    A = X @ W + b
    return relu(A)

def cross_entropy(probs, y_true):
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-8))

def accuracy(probs, y_true):
    return np.mean(np.argmax(probs, axis=1) == y_true)

def all_waves_forward(X, waves):
    if len(waves) == 0:
        return np.empty((X.shape[0], 0))
    return np.hstack([wave_forward(X, W, b) for W, b in waves])

# config
n_waves = 5
wave_size = 3
n_inputs = 8
n_classes = 2
lr = 0.01
epochs = 1000

# state
waves = []
W_out = np.empty((0, n_classes))
b_out = np.zeros(n_classes)

# training loop -- residual scheme
# wave 1 trains normally on the full label
# each later wave trains on the FROZEN net's remaining error,
# with the frozen logits held as a fixed baseline (detached): the
# new wave must reduce what the frozen waves left wrong, and it alone
# carries the gradient

for wave_idx in range(n_waves):
    W_new = he_init(n_inputs, wave_size)
    b_new = np.zeros(wave_size)

    new_slots = np.random.randn(wave_size, n_classes) * 1e-3
    W_out = np.vstack([W_out, new_slots])

    print(f"\n Training Wave {wave_idx + 1} (residual) ")

    for epoch in range(epochs):
        frozen_out = all_waves_forward(X_train, waves)
        # frozen activations
        new_out = wave_forward(X_train, W_new, b_new)
        # new wave activations

        # frozen baseline logits (DETACHED: treated as a fixed constant, no grad)
        if frozen_out.shape[1] > 0:
            frozen_W_out = W_out[:wave_idx * wave_size, :]
            base_logits = frozen_out @ frozen_W_out # fixed contribution

        else:
            base_logits = np.zeros((X_train.shape[0], n_classes))

        # new wave's own logit contribution (this is what we optimize)
        new_W_out = W_out[wave_idx * wave_size:, :]
        new_logits = new_out @ new_W_out
        logits = base_logits + new_logits + b_out
        probs = softmax(logits)

        batch_size = X_train.shape[0]

        # gradient of CE w.r.t. logits = the remaining error after the frozen baseline

        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y_train] -= 1
        dlogits /= batch_size
        
        # only the NEW wave's output rows + bias get updated; frozen rows are fixed
        dW_out_new = new_out.T @ dlogits
        db_out = dlogits.sum(axis=0)

        # backprop into the new wave (frozen baseline is detached -> no grad through it)
        dA_new = dlogits @ new_W_out.T
        dZ_new = dA_new * (new_out > 0)
        dW_new = X_train.T @ dZ_new
        db_new = dZ_new.sum(axis=0)

        # updates
        W_new -= lr * dW_new
        b_new -= lr * db_new
        W_out[wave_idx * wave_size:, :] -= lr * dW_out_new
        b_out -= lr * db_out

        if epoch % 100 == 0 or epoch == 999:
            frozen_val = all_waves_forward(X_val, waves)
            new_val = wave_forward(X_val, W_new, b_new)
            if frozen_val.shape[1] > 0:
                base_val = frozen_val @ W_out[:wave_idx*wave_size, :]
            else:
                base_val = np.zeros((X_val.shape[0], n_classes))
            new_val_logits = new_val @ W_out[wave_idx*wave_size:, :]
            val_probs = softmax(base_val)
            val_accuracy = accuracy(val_probs, y_val)
            loss = cross_entropy(probs, y_train)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Val Acc: {val_accuracy:.4f}")

    waves.append((W_new, b_new))

# feature profiles
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"
]

print("\n Running XAI Feature Audit ")
for idx, (W_wave, b_wave) in enumerate(waves):
    feature_importance = np.mean(np.abs(W_wave), axis=1)
    print(f"\nWave {idx+1} Full Feature Profile:")
    for feat, val in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
        print(f" {feat}: {val:.4f}")

# faithfulness test (same as shared-target file, for direct comparison)
print("\n Faithfulness Test: wave prominence vs actual contribution ")
frozen_val = all_waves_forward(X_val, waves)
baseline_acc = accuracy(softmax(frozen_val @ W_out + b_out), y_val)

prominences, drops = [], []
for i in range(n_waves):
    prominence = np.sum(np.mean(np.abs(waves[i][0]), axis=1))
    W_out_ablated = W_out.copy()
    W_out_ablated[i*wave_size:(i+1)*wave_size, :] = 0.0
    ablated_acc = accuracy(softmax(frozen_val @ W_out_ablated + b_out), y_val)
    prominences.append(prominence)
    drops.append(baseline_acc - ablated_acc)

prominences = np.array(prominences)
drops = np.array(drops)
print(f"\n baseline val acc: {baseline_acc:.4f}")
print(f" raw drops per wave: {np.round(drops, 4)}")
if n_waves < 3:
    print(f" Only {n_waves} waves -- need >= 3 points for correlation.")
elif drops.std() < 1e-9:
    print(" All waves removed -> ~0 change. Fully redundant.")
    print(" INCONCLUSIVE on this data.")
else:
    corr = np.corrcoef(prominences, drops)[0][1]
    print(f"    prominence-vs-drop correlation ({n_waves} waves): {corr:+.3f}")
    print(" > 0: prominent waves contribute more -> FAITHFUL (direction)")
    print(" ~0 or <0: bars don't track contribution -> possibly COSMETIC")