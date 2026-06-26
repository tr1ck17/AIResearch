# what this file is for:
# scientific rigor file
# this is to answer the two important questions regarding research so far:
#   1. Is the constructive wave architecture more or less accurate than a vanilla MLP of matched capacity?
#   2. Does any diversity penalty actually decorellate the waves (i.e., make Wave 2 specialize differently from Wave 1)
# how to validate:
#   1. cross-validation: every result is averaged over 5 folds of 5 different fixed seeds (reproducibility purposes)
    # so a number is never just a lucky split
#   2. reported as mean +/- std. the std is the result; if an effect is smaller than its std, it's indistinguishable
    # (or statistically irrelevant) from noise 
#   3. fair comparison between vanilla and wave net (identical hyperparameters)
#   4. no leakage; scaling stats are computed on the train fold only
#   5. verified math; the two non-trivial penalty gradients are checked against finite differences at startup(Section 5)
    # if a check fails, the run aborts

# file map
# section 1: settings, tunable parameters
# section 3: the two models (vanilla mlp, constructive wave net)
# section 7: experiment runner + printed results
# (sections 2, 4, 5, 6 live in wave_core.py)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from wave_core import (relu, softmax, he_init, cross_entropy, acc,
                       apply_penalty, run_gradient_checks,
                       mean_pairwise_sim, activation_decorrelation, subspace_alignment)

# SECTION 1
DATA_PATH = "../data/pima.csv"      # path to data from current dir

# fixed seed list with enough randomness for meaningful results + reproducibility
SEEDS = [42, 7, 123, 2024, 88]
N_FOLDS = 5

# frozen hyperparams (will show that the architecture is the difference maker, not how the models are tuned)
EPOCHS = 1000
LR = 0.01

# matched capacity (16 for vanilla, 15 for constructive wave net (5 waves, 3 nuerons each = 15 total neurons))
VANILLA_HIDDEN = 16
N_WAVES = 5
WAVE_SIZE = 3

# different levels of magnitude for penalty strength, checking for effect
LAMBDAS = [0.1, 1.0, 5.0]
N_INPUTS = 8
N_CLASSES = 2
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"]

# SECTION 3 (the two models)
def train_vanilla(Xtr, ytr, Xva, yva, seed):
    # 8 inputs features -> VANILLA_HIDDEN -> 2 classes
    rng = np.random.default_rng(seed)
    W1 = he_init(rng, N_INPUTS, VANILLA_HIDDEN)
    b1 = np.zeros(VANILLA_HIDDEN)
    W2 = he_init(rng, VANILLA_HIDDEN, N_CLASSES)
    b2 = np.zeros(N_CLASSES)
    n = len(Xtr)
    for _ in range(EPOCHS):
        A1 = relu(Xtr @ W1 + b1)
        A2 = softmax(A1 @ W2 + b2)
        dZ2 = A2.copy()
        dZ2[np.arange(n), ytr] -= 1
        dZ2 /= n

        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (A1 > 0)

        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)

        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

    A1v = relu(Xva @ W1 + b1)
    A2v = softmax(A1v @ W2 + b2)
    return acc(A2v, yva)

def train_waves(Xtr, ytr, Xva, yva, penalty_mode, lam, seed):
    # constructive wave net
    # each wave is reading raw input, outputs WAVE_SIZE ReLU neurons into a shared growing output layer
    # when a new wave trains, the previous wave is frozen (their output rows get zero gradient)

    # penalty_mode selects the SECTION 4 diversity penalty:
        # "off" | "uncentered" | "cosine_act" | "weight_dec"
    
    # returns (val_accuracy, waves) so callers can audit each wave's weights

    rng = np.random.default_rng(seed)
    waves = []
    W_out = np.empty((0, N_CLASSES))
    b_out = np.zeros(N_CLASSES)
    frozen_W = []       # frozen weight columns, for weight_dec
    n = len(Xtr)

    def forward_frozen(X):
        if not waves:
            return np.empty((len(X), 0))
        return np.hstack([relu(X @ W + b) for W, b in waves])
    
    for wave_idx in range(N_WAVES):
        W_new = he_init(rng, N_INPUTS, WAVE_SIZE)
        b_new = np.zeros(WAVE_SIZE)
        # tiny random init: exact zeros deadlock the new wave (zero cols in W_out -> zero gradient into it)
        W_out = np.vstack([W_out, rng.standard_normal((WAVE_SIZE, N_CLASSES)) * 1e-3])

        for _ in range(EPOCHS):
            fo = forward_frozen(Xtr)        # frozen waves' activations
            no = relu(Xtr @ W_new + b_new)        # new wave's activations

            A = np.hstack([fo, no]) if fo.shape[1] else no
            p = softmax(A @ W_out + b_out)

            grad_logits = p.copy()
            grad_logits[np.arange(n), ytr] -= 1
            grad_logits /= n

            grad_W_out = A.T @ grad_logits
            grad_b_out = grad_logits.sum(axis=0)

            if wave_idx > 0:                      # freeze old output rows
                grad_W_out[:wave_idx*WAVE_SIZE, :] = 0.0

            grad_new_acts = (grad_logits @ W_out.T)[:, wave_idx*WAVE_SIZE:]       # gradient into the new wave
            penalty_grad_W = 0.0                                # weight-space penalties add here

            # SECTION 4 penalties, applied
            if penalty_mode != "off" and fo.shape[1]:
                grad_new_acts, penalty_grad_W = apply_penalty(penalty_mode, lam, fo, no, W_new, frozen_W, grad_new_acts, n)

            grad_new_preact = grad_new_acts * (no > 0)
            W_new -= LR * (Xtr.T @ grad_new_preact + penalty_grad_W)
            b_new -= LR * grad_new_preact.sum(0)
            W_out -= LR * grad_W_out
            b_out -= LR * grad_b_out

        waves.append((W_new, b_new))
        frozen_W.append(W_new.copy())

    Av = forward_frozen(Xva)
    return acc(softmax(Av @ W_out + b_out), yva), waves

# SECTION 7 -- EXPERIMENT RUNNER
def main():
    run_gradient_checks()

    df = pd.read_csv(DATA_PATH, header=None)
    X = np.asarray(df.iloc[:,:8].values, dtype=float)
    y = np.asarray(df.iloc[:,8].values, dtype=int)

    van_acc, wav_acc = [], []
    # penalty modes for CLAIM B: baseline + each formulation at each lambda
    pen_specs = {"OFF (baseline)": ("off", 0.0)}
    for lam in LAMBDAS:
        for mode in ["uncentered", "centered", "cosine_act", "weight_dec"]:
            pen_specs[f"{mode:<11} l={lam}"] = (mode, lam)
    sim_mean = {k: [] for k in pen_specs}
    sim_max = {k: [] for k in pen_specs}
    sim_sub = {k: [] for k in pen_specs}
    sim_act = {k: [] for k in pen_specs}
    acc_w = {k: [] for k in pen_specs}

    print(f"Running {len(SEEDS)} seeds x {N_FOLDS} folds = {len(SEEDS)*N_FOLDS} "
          f"runs per configuration...\n")
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
            Xtr, Xva = (X[tr]-mu)/sd, (X[va]-mu)/sd
            ytr, yva = y[tr], y[va]
            init = seed * 1000 + k          # reproducible, unique per (seed, fold)

            # Claim A: accuracy, identical settings
            van_acc.append(train_vanilla(Xtr, ytr, Xva, yva, seed=init))
            a, _ = train_waves(Xtr, ytr, Xva, yva, "off", 0.0, seed=init)
            wav_acc.append(a)

            # Claim B: decorrelation per penalty (same init => only penalty differs)
            for name, (mode, lam) in pen_specs.items():
                a_w, w = train_waves(Xtr, ytr, Xva, yva, mode, lam, seed=init)
                acc_w[name].append(a_w)
                m_, mx_ = mean_pairwise_sim(w)
                sim_mean[name].append(m_)
                sim_max[name].append(mx_)
                sim_sub[name].append(subspace_alignment(w))
                sim_act[name].append(activation_decorrelation(w, Xva))

    mean_std = lambda values: (np.mean(values), np.std(values))

    print("="*70)
    print("CLAIM A -- ACCURACY")
    print("="*70)

    vanilla_accs = np.array(van_acc)
    wave_accs    = np.array(wav_acc)

    vanilla_mean, vanilla_std = mean_std(vanilla_accs)
    wave_mean, wave_std       = mean_std(wave_accs)
    print(f"    vanilla MLP (16 hidden) : {vanilla_mean:.4f} +/- {vanilla_std:.4f}")
    print(f"    wave net    (5x3 = 15)  : {wave_mean:.4f} +/- {wave_std:.4f}")

    # paired test: same fold ran both nets, so the difference cancels fold-difficulty
    per_fold_diff = wave_accs - vanilla_accs
    num_folds     = len(per_fold_diff)
    mean_diff     = per_fold_diff.mean()
    std_error     = per_fold_diff.std(ddof=1) / np.sqrt(num_folds)
    t_critical    = 2.064                   # t, 24 dof, 95% (n=25)
    ci_low        = mean_diff - t_critical * std_error
    ci_high       = mean_diff + t_critical * std_error

    print(f"\n  paired difference (wave - vanilla), n={num_folds}")
    print(f"    mean diff = {mean_diff:+.4f}    std error = {std_error:.4f}")
    print(f"    95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]")

    if ci_low <= 0 <= ci_high:
        verdict = "INDISTINGUISHABLE (CI contains 0)"
    elif ci_high < 0:
        verdict = "WAVE NET WORSE (CI entirely below 0)"
    else:
        verdict = "WAVE NET BETTER (CI entirely above 0)"
    print(f"    --> {verdict}")
    
    print("\n" + "="*70)
    print("CLAIM B -- DECORRELATION (Wave1 vs Wave2 similarity; lower = better)")
    print("="*70)
    base_m, base_s = mean_std(sim_mean["OFF (baseline)"])
    print(f"    noise bar (std of baseline mean-sim) = {base_s:.3f}\n")
    print(f"    {'configuration':<20} {'mean-sim':>16} {'max-sim':>10} {'subspace':>10} {'act-corr':>10} {'acc':>8}")
    bm, bs = mean_std(sim_mean["OFF (baseline)"])
    xm, _  = mean_std(sim_max["OFF (baseline)"])
    sm, _  = mean_std(sim_sub["OFF (baseline)"])
    ac, _  = mean_std(sim_act["OFF (baseline)"])
    am, _  = mean_std(acc_w["OFF (baseline)"])
    print(f"    {'OFF (baseline)':<20} {bm:>8.3f} +/- {bs:.3f} {xm:>10.3f} {sm:>10.3f} {ac:>10.3f} {am:>8.3f}")
    for name in pen_specs:
        if name == "OFF (baseline)": continue
        m, s   = mean_std(sim_mean[name])
        x, _   = mean_std(sim_max[name])
        sub, _ = mean_std(sim_sub[name])
        act, _ = mean_std(sim_act[name])
        sim_gap = m - base_m
        flag = " <--" if abs(sim_gap) > base_s else ""
        a, _ = mean_std(acc_w[name])
        print(f"    {name:<20} {m:>8.3f} +/- {s:.3f} {x:>10.3f} {sub:>10.3f} {act:>10.3f} {a:>8.3f}{flag}")

if __name__ == "__main__":
    main()