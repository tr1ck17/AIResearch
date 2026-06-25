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
# section 2: base functions (relu, softmax)
# section 3: the two models (vanilla mlp, constructive wave net)
# section 4: the four penalty formulations (specifically for claim B)
# section 5: gradient checks (proof that the penalty math is correct)
# section 6: measurement helpers (how to quantify if "two waves are different")
# section 7: experiment runner + printed results

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# SECTION 1
DATA_PATH = "../../data/ionosphere.csv"      # path to data from current dir

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
N_INPUTS = 33
N_CLASSES = 2
FEATURES = [f"f{i}" for i in range(33)]

# SECTION 2
def relu(z):
    return np.maximum(0, z)
def he_init(rng, i, o):
    return rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
def softmax(z):
    e = np.exp(z  - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
def cross_entropy(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-8))
def acc(p, y):
    return np.mean(np.argmax(p, 1) == y)

# SECTION 3 (vanilla standard 1 hidden, vanilla standard 2 hidden, vanilla breadth)
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

def train_deep(Xtr, ytr, Xva, yva, seed):
    # 33 inputs -> 8 -> 8 -> 2 (two hidden layers, 16 total units to match vanilla 1 hidden layer's units)
    rng = np.random.default_rng(seed)
    H1, H2 = 8, 8
    W1 = he_init(rng, N_INPUTS, H1)
    b1 = np.zeros(H1)
    W2 = he_init(rng, H1, H2)
    b2 = np.zeros(H2)
    W3 = he_init(rng, H2, N_CLASSES)
    b3 = np.zeros(N_CLASSES)
    n = len(Xtr)
    for _ in range(EPOCHS):
        A1 = relu(Xtr @ W1 + b1)
        A2 = relu(A1 @ W2 + b2)
        A3 = softmax(A2 @ W3 + b3)
        dZ3 = A3.copy()
        dZ3[np.arange(n), ytr] -= 1
        dZ3 /= n
        dW3 = A2.T @ dZ3
        db3 = dZ3.sum(axis=0)
        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * (A2 > 0)
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
        W3 -= LR * dW3
        b3 -= LR * db3
    
    A1v = relu(Xva @ W1 + b1)
    A2v = relu(A1v @ W2 + b2)
    A3v = softmax(A2v @ W3 + b3)
    return acc(A3v, yva)

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
        # new output rows: tiny RANDOM, not zero
        # exact zeros would give the new wave zero gradient (dA = dlogits @ W_out.T would be 0 on its columns)
        # thereby deadlocking it. Small random breaks that

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

# SECTION 4 (CORE OF CLAIM B - the 4 diversity penalties)

# goal of every penalty: push Wave 2 to be different from Wave 1
# they differ in WHAT they measure as similarity
# 2 act on activations (what the waves output)
# 1 acts on weights directly

# uncentered : sum( (frozen.T @ new)^2 )
    # the naive first instinct
    # FLAW: ReLU outputs are non-negative, so the cheapest way to shrink the product is to shrink the new wave (magnitude)
    # not rotate it (correlation). Penalizes size, not similarity

# centered : same, but mean-center activations first (covariance). Removes  the magnitude shortcut
    # this is the Cascade-Correlation form

# cosine_act : correlation normalized by magnitude (pure angle). Fully magnitude-invariant. A Rayleigh quotient -> clean
    # gradient

# weight_dec : decorrelate the WEIGHT vectors directly, not the activations
    # targets the exact quantity the XAI audit reads. Also Rayleigh

# the activation gradients chain through column-centering; weight_dec adds  straight to the weight gradient
# gradients for cosine_act and weight_dec are verified in section 5

def apply_penalty(mode, lam, fo, no, W_new, frozen_W, grad_new_acts, n):
    penalty_grad_W = 0.0
    if mode == "uncentered":
        grad_new_acts = grad_new_acts + lam * 2 * (fo @ (fo.T @ no)) / (n**2)

    elif mode == "centered":
        frozen_centered = fo - fo.mean(0)
        new_centered = no - no.mean(0)
        g = 2 * (frozen_centered @ (frozen_centered.T @ new_centered)) / (n**2)
        grad_new_acts = grad_new_acts + lam * (g - g.mean(0))

    elif mode == "cosine_act":              # Option 2
        frozen_centered = fo - fo.mean(0)
        new_centered = no - no.mean(0)
        fn = np.linalg.norm(frozen_centered, axis=0) + 1e-12
        frozen_dirs = frozen_centered / fn
        ng2 = (new_centered * new_centered).sum(0) + 1e-12
        proj = frozen_dirs @ (frozen_dirs.T @ new_centered)
        proj_coef = (new_centered * proj).sum(0) / ng2
        gg = 2 * (proj - new_centered * proj_coef) / ng2
        grad_new_acts = grad_new_acts + lam * (gg-gg.mean(0))

    elif mode == "weight_dec":              # Option 3
        Wf = np.hstack(frozen_W)
        fn = np.linalg.norm(Wf, axis=0) + 1e-12
        frozen_weight_dirs = Wf / fn
        wn2 = (W_new * W_new).sum(0) + 1e-12
        proj = frozen_weight_dirs @ (frozen_weight_dirs.T @ W_new)
        proj_coef = (W_new * proj).sum(0) / wn2
        penalty_grad_W = lam * 2 * (proj - W_new * proj_coef) / wn2
    return grad_new_acts, penalty_grad_W

# SECTION 5
# Gradient Checks (to prove section 4 math is correct)
# each penalty is a Rayleigh quotient; we verify the analytic gradient against central finite differences
# Relative error ~1e-9 = correct to ~9 digits
# runs automatically at startup; aborts the program if anything fails

def _pen2_val(fo, no):
    frozen_centered, new_centered = fo - fo.mean(0), no - no.mean(0)
    fn = np.linalg.norm(frozen_centered, axis=0) + 1e-12
    frozen_dirs = frozen_centered / fn
    ng2 = (new_centered * new_centered).sum(0) + 1e-12
    return ((new_centered * (frozen_dirs @ (frozen_dirs.T @ new_centered))).sum(0) / ng2).sum()

def _pen2_grad(fo, no):
    frozen_centered, new_centered = fo - fo.mean(0), no - no.mean(0)
    fn = np.linalg.norm(frozen_centered, axis=0) + 1e-12
    frozen_dirs = frozen_centered / fn
    ng2 = (new_centered*new_centered).sum(0) + 1e-12
    proj = frozen_dirs @ (frozen_dirs.T @ new_centered)
    proj_coef = (new_centered*proj).sum(0)/ng2
    gg = 2*(proj - new_centered*proj_coef)/ng2
    return gg - gg.mean(0)

def _pen3_val(Wf, W_new):
    fn = np.linalg.norm(Wf, axis=0) + 1e-12
    frozen_weight_dirs = Wf / fn
    wn2 = (W_new*W_new).sum(0) + 1e-12
    return ((W_new * (frozen_weight_dirs @ (frozen_weight_dirs.T @ W_new))).sum(0) / wn2).sum()

def _pen3_grad(Wf, W_new):
    fn = np.linalg.norm(Wf, axis=0) + 1e-12
    frozen_weight_dirs = Wf / fn
    wn2 = (W_new*W_new).sum(0) + 1e-12
    proj = frozen_weight_dirs @ (frozen_weight_dirs.T @ W_new)
    proj_coef = (W_new*proj).sum(0)/wn2
    return 2*(proj - W_new*proj_coef)/wn2

def _gradcheck(val, grad, fixed, var, label, eps=1e-6):
    ga = grad(*fixed, var)
    gn = np.zeros_like(var)
    it = np.nditer(var, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        v = var[i]
        var[i] = v + eps
        fp = val(*fixed, var)
        var[i] = v - eps
        fm = val(*fixed, var)
        var[i] = v
        gn[i] = (fp - fm) / (2*eps)
        it.iternext()
    err = np.max(np.abs(ga - gn)) / (np.max(np.abs(gn)) + 1e-12)
    status = "PASS" if err < 1e-5 else "FAIL"
    print(f" gradient check {label:<28}: rel error {err:.1e} {status}")
    return err < 1e-5

def run_gradient_checks():
    print("SECTION 5 -- verifying penalty gradients before trusting results")
    rng = np.random.default_rng(0)
    fo = np.maximum(0, rng.standard_normal((40, 6)))
    no = np.maximum(0, rng.standard_normal((40, 3)))
    Wf = rng.standard_normal((8, 6))
    W_new = rng.standard_normal((8, 3))
    ok1 = _gradcheck(_pen2_val, _pen2_grad, (fo,), no, "cosine_act (d/d activations)")
    ok2 = _gradcheck(_pen3_val, _pen3_grad, (Wf,), W_new, "weight_dec (d/d weights)")
    if not (ok1 and ok2):
        raise SystemExit("Gradient check FAILED -- results would be invalid. Aborting.")
    print()


# SECTION 6 -- Measurement Helpers
# each wave -> an 8-number feature profile (mean |weight| per input feature).
# cosine similarity of Wave1 vs Wave2 profiles: 1.0 = identical specialization
# lower = more decorrelated. This is the yardstick for CLAIM B

def profile(W):
    return np.mean(np.abs(W), axis=1)       # (8, 3) -> (8,)

def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_pairwise_sim(waves):
    profs = [profile(W) for W, _ in waves]
    sims = [cosine(profs[i], profs[j])
            for i in range(len(profs)) for j in range(i+1, len(profs))]
    return float(np.mean(sims)), float(np.max(sims))

def subspace_alignment(waves):
    Ws = [W for W, _ in waves]
    vals = []
    for i in range(len(Ws)):
        others = np.hstack([Ws[j] for j in range(len(Ws)) if j != i])
        fn = np.linalg.norm(others, axis=0) + 1e-12
        frozen_dirs = others / fn
        Wi = Ws[i]
        wn2 = (Wi * Wi).sum(0)+ 1e-12
        vals.append(((Wi * (frozen_dirs @ (frozen_dirs.T @ Wi))).sum(0) / wn2).mean())
    return float(np.mean(vals))

# SECTION 7 -- EXPERIMENT RUNNER
def main():
    run_gradient_checks()

    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    #y = (df.iloc[:, -1].values == 'g').astype(int)
    y = np.where(df.iloc[:, -1].values == 'g', 1, 0)
    keep = X.std(axis=0) > 1e-12
    X = X[:, keep]

    van_acc, wav_acc, deep_acc = [], [], []
    # penalty modes for CLAIM B: baseline + each formulation at each lambda
    pen_specs = {"OFF (baseline)": ("off", 0.0)}
    for lam in LAMBDAS:
        for mode in ["uncentered", "centered", "cosine_act", "weight_dec"]:
            pen_specs[f"{mode:<11} l={lam}"] = (mode, lam)
    sim_mean = {k: [] for k in pen_specs}
    sim_max = {k: [] for k in pen_specs}
    sim_sub = {k: [] for k in pen_specs}
    acc_w = {k: [] for k in pen_specs}

    print(f"Running {len(SEEDS)} seeds x {N_FOLDS} folds = {len(SEEDS)*N_FOLDS} "
          f"runs per configuration...\n")
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(X, y)):
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
            Xtr, Xva = (X[tr]-mu)/sd, (X[va]-mu)/sd
            ytr, yva = y[tr].astype(int), y[va].astype(int)
            init = seed * 1000 + k          # reproducible, unique per (seed, fold)

            # Claim A: accuracy, identical settings
            van_acc.append(train_vanilla(Xtr, ytr, Xva, yva, seed=init))
            deep_acc.append(train_deep(Xtr, ytr, Xva, yva, seed=init))
            a, _ = train_waves(Xtr, ytr, Xva, yva, "off", 0.0, seed=init)
            wav_acc.append(a)

            # Claim B: decorrelation per penalty (same init => only penalty differs)
            for name, (mode, lam) in pen_specs.items():
                #_, w = train_waves(Xtr, ytr, Xva, yva, mode, lam, seed=init)
                a_w, w = train_waves(Xtr, ytr, Xva, yva, mode, lam, seed=init)
                acc_w[name].append(a_w)
                m_, mx_ = mean_pairwise_sim(w)
                sim_mean[name].append(m_)
                sim_max[name].append(mx_)
                sim_sub[name].append(subspace_alignment(w))

    mean_std = lambda values: (np.mean(values), np.std(values))

    print("="*70)
    print("CLAIM A -- ACCURACY")
    print("="*70)

    vanilla_accs = np.array(van_acc)
    wave_accs = np.array(wav_acc)

    # descriptive: each net in its own (shared fold-noise still inside these stds)
    deep_accs = np.array(deep_acc)
    vanilla_mean, vanilla_std = mean_std(vanilla_accs)
    deep_mean, deep_std       = mean_std(deep_accs)
    wave_mean, wave_std       = mean_std(wave_accs)
    print(f"    vanilla MLP (16 hidden) : {vanilla_mean:.4f} +/- {vanilla_std:.4f}")
    print(f"    deep MLP    (8+8=16)    : {deep_mean:.4f} +/- {deep_std:.4f}")
    print(f"    wave net    (5x3 = 15)  : {wave_mean:.4f} +/- {wave_std:.4f}")

    # paired test: same fold ran both nets, so the difference cancels fold-difficulty
    per_fold_diff   = wave_accs - vanilla_accs        # 25 head-to-head margins
    num_folds       = len(per_fold_diff)
    mean_diff       = per_fold_diff.mean()
    std_error       = per_fold_diff.std(ddof=1) / np.sqrt(num_folds)
    t_critical      = 2.064         # t, 24 dof, 95% (n=25)
    ci_low          = mean_diff - t_critical * std_error
    ci_high         = mean_diff + t_critical * std_error

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

    diff_wd = wave_accs - deep_accs
    mean_wd = diff_wd.mean()
    se_wd = diff_wd.std(ddof=1) / np.sqrt(len(diff_wd))
    lo_wd = mean_wd - t_critical * se_wd
    hi_wd = mean_wd + t_critical * se_wd
    print(f"\n  paired difference (wave - deep), n={len(diff_wd)}")
    print(f"    mean diff = {mean_wd:+.4f}  std error = {se_wd:.4f}")
    print(f"    95% CI = [{lo_wd:+.4f}, {hi_wd:+.4f}]")
    if lo_wd <= 0 <= hi_wd:
        print("     --> wave vs deep: INDISTINGUISHABLE (CI contains 0)")
    elif hi_wd < 0:
        print("     --> wave vs deep: WAVE WORSE")
    else:
        print("     --> wave vs deep: WAVE BETTER")
    
    print("\n" + "="*70)
    print("CLAIM B -- DECORRELATION (Wave1 vs Wave2 similarity; lower = better)")
    print("="*70)
    base_m, base_s = mean_std(sim_mean["OFF (baseline)"])
    print(f"    noise bar (std of baseline mean-sim) = {base_s:.3f}\n")
    print(f"    {'configuration':<20} {'mean-sim':>16} {'max-sim':>10} {'subspace':>10} {'acc':>8}")
    bm, bs = mean_std(sim_mean["OFF (baseline)"])
    xm, _  = mean_std(sim_max["OFF (baseline)"])
    sm, _  = mean_std(sim_sub["OFF (baseline)"])
    am, _ = mean_std(acc_w["OFF (baseline)"])
    print(f"    {'OFF (baseline)':<20} {bm:>8.3f} +/- {bs:.3f} {xm:>10.3f} {sm:>10.3f} {am:>8.3f}")
    for name in pen_specs:
        if name == "OFF (baseline)": continue
        m, s = mean_std(sim_mean[name])
        x, _ = mean_std(sim_max[name])
        sub, _ = mean_std(sim_sub[name])
        sim_gap = m - base_m
        flag = " <--" if abs(sim_gap) > base_s else ""
        a, _ = mean_std(acc_w[name])
        print(f"    {name:<20} {m:>8.3f} +/- {s:.3f} {x:>10.3f} {sub:>10.3f} {a:>8.3f}{flag}")

if __name__ == "__main__":
    main()