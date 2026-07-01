import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from vanilla_wave_net import load_datasets, train_vanilla_wave, SEEDS, N_FOLDS
from vanilla_mlp import train_vanilla_mlp

T_CRIT = 2.064  # t, 24 dof (n=25 paired runs), 95% two-tailed

def compare(name, X, y):
    n_inputs  = X.shape[1]
    n_classes = len(np.unique(y))
    wave_accs = []
    mlp_accs  = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(X, y)):
            mu, sd      = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Xtr, Xva   = (X[tr] - mu) / sd, (X[va] - mu) / sd
            ytr, yva    = y[tr], y[va]
            init        = seed * 1000 + k

            wave_accs.append(train_vanilla_wave(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed=init))
            mlp_accs.append(train_vanilla_mlp(Xtr, ytr, Xva, yva, n_inputs, n_classes, seed=init))

    wave_accs = np.array(wave_accs)
    mlp_accs  = np.array(mlp_accs)

    wm, ws    = wave_accs.mean(), wave_accs.std()
    mm, ms    = mlp_accs.mean(), mlp_accs.std()

    diff      = wave_accs - mlp_accs
    mean_diff = diff.mean()
    std_err   = diff.std(ddof=1) / np.sqrt(len(diff))
    ci_lo     = mean_diff - T_CRIT * std_err
    ci_hi     = mean_diff + T_CRIT * std_err

    if ci_lo <= 0 <= ci_hi:
        verdict = "~"
    elif ci_hi < 0:
        verdict = "MLP"
    else:
        verdict = "WAVE"

    print(f"  {name:<18} {wm:.4f}+/-{ws:.4f}  {mm:.4f}+/-{ms:.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  {verdict}")

def main():
    datasets = load_datasets()

    print("=" * 85)
    print("WAVE NET vs VANILLA MLP -- PAIRED COMPARISON ACROSS DATASETS")
    print(f"  wave: 5x3=15 neurons  |  mlp: 1 hidden x 15 neurons  |  25 runs  |  95% CI (t, 24 dof)")
    print("=" * 85)
    print(f"  {'Dataset':<18} {'Wave Net':^16} {'MLP':^16} {'95% CI (wave-MLP)':^20}  verdict")
    print("-" * 85)

    for name, X, y in datasets:
        compare(name, X, y)

    print("=" * 85)
    print("  ~ = indistinguishable   MLP = MLP significantly better   WAVE = wave net significantly better")

if __name__ == "__main__":
    main()
