import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from wave_core import relu, softmax, he_init, acc
from vanilla_wave_net import load_datasets

# config -- matched to wave net for fair comparison
SEEDS   = [42, 7, 123, 2024, 88]
N_FOLDS = 5
HIDDEN  = 15    # 5 waves x 3 neurons = 15 total; exact capacity match
LR      = 0.01
EPOCHS  = 1000

def train_vanilla_mlp(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed):
    rng = np.random.default_rng(seed)
    W1  = he_init(rng, n_inputs, HIDDEN)
    b1  = np.zeros(HIDDEN)
    W2  = he_init(rng, HIDDEN, n_classes)
    b2  = np.zeros(n_classes)
    n   = len(Xtr)

    for _ in range(EPOCHS):
        A1 = relu(Xtr @ W1 + b1)
        p  = softmax(A1 @ W2 + b2)

        gl = p.copy()
        gl[np.arange(n), ytr] -= 1
        gl /= n

        gW2 = A1.T @ gl
        gb2 = gl.sum(0)
        gA1 = gl @ W2.T
        gZ1 = gA1 * (A1 > 0)
        gW1 = Xtr.T @ gZ1
        gb1 = gZ1.sum(0)

        W1 -= LR * gW1
        b1 -= LR * gb1
        W2 -= LR * gW2
        b2 -= LR * gb2

    A1v = relu(Xva @ W1 + b1)
    return acc(softmax(A1v @ W2 + b2), yva)

def evaluate(name, X, y):
    n_inputs  = X.shape[1]
    n_classes = len(np.unique(y))
    majority  = max(np.mean(y == c) for c in np.unique(y))
    accs = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(X, y)):
            mu, sd      = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Xtr, Xva   = (X[tr] - mu) / sd, (X[va] - mu) / sd
            ytr, yva    = y[tr], y[va]
            accs.append(train_vanilla_mlp(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed=seed * 1000 + k))

    m, s = np.mean(accs), np.std(accs)
    print(f"  {name:<22} {m:.4f} +/- {s:.4f}   maj: {majority:.1%}")
    return m, s

def main():
    datasets = load_datasets()

    print("=" * 65)
    print("VANILLA MLP -- BASELINE ACROSS DATASETS")
    print(f"  1 hidden x {HIDDEN} neurons | {len(SEEDS)} seeds x {N_FOLDS} folds = {len(SEEDS)*N_FOLDS} runs each")
    print("=" * 65)
    print(f"  {'Dataset':<22} {'Mean Acc':>8}   {'Std':>6}   {'Majority':>8}")
    print("-" * 65)

    for name, X, y in datasets:
        evaluate(name, X, y)

    print("=" * 65)

if __name__ == "__main__":
    main()
