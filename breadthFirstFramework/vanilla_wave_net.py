import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo
from wave_core import relu, softmax, he_init, acc

# config
SEEDS     = [42, 7, 123, 2024, 88]
N_FOLDS   = 5
N_WAVES   = 5
WAVE_SIZE = 3
LR        = 0.01
EPOCHS    = 1000

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, '..', 'data')
CACHE_DIR = os.path.join(ROOT, 'dataset_cache')

# helpers

def drop_zero_variance(X):
    return X[:, X.std(axis=0) > 1e-12]

def encode_labels(y):
    _, encoded = np.unique(y, return_index=False, return_inverse=True, return_counts=False)
    return encoded.astype(int)

def load_cache(name):
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d['X'], d['y']
    return None

def save_cache(name, X, y):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, f"{name}.npz"), X=X, y=y)

def fetch_and_cache(name, fetch_fn):
    cached = load_cache(name)
    if cached:
        print(f"  [OK] {name} (cached)")
        return cached
    X, y = fetch_fn()
    save_cache(name, X, y)
    print(f"  [OK] {name}")
    return X, y

# dataset loaders

def load_datasets():
    datasets = []
    print("Loading datasets...")

    # Pima: local CSV, binary, 8 features, 768 samples
    df = pd.read_csv(os.path.join(DATA_DIR, 'pima.csv'), header=None)
    X  = drop_zero_variance(df.iloc[:, :8].values.astype(float))
    y  = df.iloc[:, 8].values.astype(int)
    datasets.append(("Pima", X, y))
    print("  [OK] Pima")

    # Ionosphere: local CSV, binary (g=1, b=0), 33 features after zero-variance drop
    df = pd.read_csv(os.path.join(DATA_DIR, 'ionosphere.csv'), header=None)
    X  = drop_zero_variance(df.iloc[:, :-1].values.astype(float))
    y  = np.where(df.iloc[:, -1].values == 'g', 1, 0)
    datasets.append(("Ionosphere", X, y))
    print("  [OK] Ionosphere")

    # Breast Cancer: sklearn built-in, binary, 30 features, 569 samples
    def fetch_breast_cancer():
        X, y = load_breast_cancer(return_X_y=True)
        return drop_zero_variance(X), y
    X, y = fetch_and_cache("breast_cancer", fetch_breast_cancer)
    datasets.append(("Breast Cancer", X, y))

    # Banknote Authentication: UCI id=267, binary, 4 features, 1372 samples
    def fetch_banknote():
        repo = fetch_ucirepo(id=267)
        X = drop_zero_variance(repo.data.features.values.astype(float))  # type: ignore[union-attr]
        y = repo.data.targets.values.ravel().astype(int)                 # type: ignore[union-attr]
        return X, y
    X, y = fetch_and_cache("banknote", fetch_banknote)
    datasets.append(("Banknote Auth", X, y))

    # Heart Disease: UCI id=45, 13 features, 303 samples (6 rows dropped for missing values)
    # binarized as 0 = no disease, >0 = disease present — standard in the literature
    def fetch_heart():
        repo = fetch_ucirepo(id=45)
        X = repo.data.features.values.astype(float)           # type: ignore[union-attr]
        y = (repo.data.targets.values.ravel() > 0).astype(int)  # type: ignore[union-attr]
        mask = ~np.isnan(X).any(axis=1)
        return drop_zero_variance(X[mask]), y[mask]
    X, y = fetch_and_cache("heart_disease", fetch_heart)
    datasets.append(("Heart Disease", X, y))

    # Wine Quality (Red): UCI id=186, 11 features, 1599 samples
    # binarized at quality >= 6: the standard cut in the wine quality literature;
    # gives a ~55/45 split on red wine, which is the most defensible threshold
    def fetch_wine():
        repo = fetch_ucirepo(id=186)
        X = drop_zero_variance(repo.data.features.values.astype(float))  # type: ignore[union-attr]
        y = (repo.data.targets.values.ravel() >= 6).astype(int)          # type: ignore[union-attr]
        return X, y
    X, y = fetch_and_cache("wine_quality", fetch_wine)
    datasets.append(("Wine Quality", X, y))

    # Glass: UCI id=42, 6-class, 9 features, 214 samples
    # labels are {1,2,3,5,6,7} (no class 4) -- encode_labels remaps to 0-indexed
    # one class has n<5 samples so it's dropped before evaluation (can't stratify across 5 folds)
    def fetch_glass():
        repo = fetch_ucirepo(id=42)
        X = drop_zero_variance(repo.data.features.values.astype(float))  # type: ignore[union-attr]
        y = encode_labels(repo.data.targets.values.ravel())              # type: ignore[union-attr]
        unique, counts = np.unique(y, return_counts=True)
        valid = unique[counts >= N_FOLDS]
        mask  = np.isin(y, valid)
        return X[mask], encode_labels(y[mask])
    X, y = fetch_and_cache("glass", fetch_glass)
    datasets.append(("Glass", X, y))

    # Vehicle: UCI id=149, 4-class (bus/van/saab/opel), 18 features, 846 samples
    # ucimlrepo includes 1 stray sample with a 5th label -- drop classes with n < N_FOLDS
    def fetch_vehicle():
        repo = fetch_ucirepo(id=149)
        X = drop_zero_variance(repo.data.features.values.astype(float))  # type: ignore[union-attr]
        y = encode_labels(repo.data.targets.values.ravel())              # type: ignore[union-attr]
        unique, counts = np.unique(y, return_counts=True)
        valid = unique[counts >= N_FOLDS]
        mask  = np.isin(y, valid)
        return X[mask], encode_labels(y[mask])
    X, y = fetch_and_cache("vehicle", fetch_vehicle)
    datasets.append(("Vehicle", X, y))

    # Phoneme: OpenML id=1489, binary (nasal vs oral), 5 features, 5404 samples
    def fetch_phoneme():
        ds = fetch_openml(data_id=1489, as_frame=False, parser='auto')
        X = drop_zero_variance(ds.data.astype(float))
        y = encode_labels(ds.target)
        return X, y
    X, y = fetch_and_cache("phoneme", fetch_phoneme)
    datasets.append(("Phoneme", X, y))

    print()
    return datasets

# model

def train_vanilla_wave(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed):
    rng   = np.random.default_rng(seed)
    waves = []
    W_out = np.empty((0, n_classes))
    b_out = np.zeros(n_classes)
    n     = len(Xtr)

    for wave_idx in range(N_WAVES):
        W_new = he_init(rng, n_inputs, WAVE_SIZE)
        b_new = np.zeros(WAVE_SIZE)
        W_out = np.vstack([W_out, rng.standard_normal((WAVE_SIZE, n_classes)) * 1e-3])

        for _ in range(EPOCHS):
            fo = np.hstack([relu(Xtr @ W + b) for W, b in waves]) if waves else np.empty((n, 0))
            no = relu(Xtr @ W_new + b_new)
            A  = np.hstack([fo, no]) if fo.shape[1] else no
            p  = softmax(A @ W_out + b_out)

            gl = p.copy()
            gl[np.arange(n), ytr] -= 1
            gl /= n

            gW_out = A.T @ gl
            gb_out = gl.sum(0)

            if wave_idx > 0:
                gW_out[:wave_idx * WAVE_SIZE] = 0.0

            gA_new = (gl @ W_out.T)[:, wave_idx * WAVE_SIZE:]
            gZ_new = gA_new * (no > 0)

            W_new -= LR * (Xtr.T @ gZ_new)
            b_new -= LR * gZ_new.sum(0)
            W_out -= LR * gW_out
            b_out -= LR * gb_out

        waves.append((W_new, b_new))

    val_fo = np.hstack([relu(Xva @ W + b) for W, b in waves])
    return acc(softmax(val_fo @ W_out + b_out), yva)

# evaluation

def evaluate(name, X, y):
    n_inputs  = X.shape[1]
    n_classes = len(np.unique(y))
    majority  = max(np.mean(y == c) for c in np.unique(y))
    accs = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(X, y)):
            mu, sd   = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
            ytr, yva = y[tr], y[va]
            accs.append(train_vanilla_wave(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed=seed * 1000 + k))

    m, s = np.mean(accs), np.std(accs)
    print(f"  {name:<22} {m:.4f} +/- {s:.4f}   maj: {majority:.1%}")
    return m, s

# main

def main():
    datasets = load_datasets()

    print("=" * 65)
    print("VANILLA WAVE NET -- BASELINE ACROSS DATASETS")
    print(f"  {N_WAVES} waves x {WAVE_SIZE} neurons | {len(SEEDS)} seeds x {N_FOLDS} folds = {len(SEEDS)*N_FOLDS} runs each")
    print("=" * 65)
    print(f"  {'Dataset':<22} {'Mean Acc':>8}   {'Std':>6}   {'Majority':>8}")
    print("-" * 65)

    for name, X, y in datasets:
        evaluate(name, X, y)

    print("=" * 65)

if __name__ == "__main__":
    main()