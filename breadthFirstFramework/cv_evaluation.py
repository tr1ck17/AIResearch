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

# SECTION 2
def relu(z):
    return np.maximum(0, z)
def he(rng, i, o):
    return rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
def softmax(z):
    e = np.exp(z  - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
def cross_entropy(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-8))
def acc(p, y):
    return np.mean(np.argmax(p, 1) == y)

# SECTION 3 (the two models)
def train_vanilla(Xtr, ytr, Xva, yva, seed):
    # 8 inputs features -> VANILLA_HIDDEN -> 2 classes
    rng = np.random.default_rng(seed)
    W1 = he(rng, N_INPUTS, VANILLA_HIDDEN)
    b1 = np.zeros(VANILLA_HIDDEN)
    W2 = he(rng, VANILLA_HIDDEN, N_CLASSES)
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
    
    for wi in range(N_WAVES):
        Wn = he(rng, N_INPUTS, WAVE_SIZE)
        bn = np.zeros(WAVE_SIZE)
        # new output rows: tiny RANDOM, not zero
        # exact zeros would give the new wave zero gradient (dA = dlogits @ W_out.T would be 0 on its columns)
        # thereby deadlocking it. Small random breaks that

        W_out = np.vstack([W_out, rng.standard_normal((WAVE_SIZE, N_CLASSES)) * 1e-3])

        for _ in range(EPOCHS):
            fo = forward_frozen(Xtr)        # frozen waves' activations
            no = relu(Xtr @ Wn + bn)        # new wave's activations

            A = np.hstack([fo, no]) if fo.shape[1] else no
            p = softmax(A @ W_out + b_out)

            dl = p.copy()
            dl[np.arange(n), ytr] -= 1
            dl /= n

            dW_out = A.T @ dl
            db_out = dl.sum(axis=0)

            if wi > 0:                      # freeze old output rows
                dW_out[:wi*WAVE_SIZE, :] = 0.0

            dA = (dl @ W_out.T)[:, wi*WAVE_SIZE:]       # gradient into the new wave
            dW_pen = 0.0                                # weight-space penalties add here

            # SECTION 4 penalties, applied
            if penalty_mode != "off" and fo.shape[1]:
                dA, dW_pen = apply_penalty(penalty_mode, lam, fo, no, Wn, frozen_W, dA, n)

            dZ = dA * (no > 0)
            Wn -= LR * (Xtr.T @ dZ + dW_pen)
            bn -= LR * dZ.sum(0)
            W_out -= LR * dW_out
            b_out -= LR * db_out

        waves.append((Wn, bn))
        frozen_W.append(Wn.copy())

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

def apply_penalty(mode, lam, fo, no, Wn, frozen_W, dA, n):
    dW_pen = 0.0
    if mode == "uncentered":
        dA = dA + lam * 2 * (fo @ (fo.T @ no)) / (n**2)

    elif mode == "centered":
        Fc = fo - fo.mean(0)
        Nc = no - no.mean(0)
        g = 2 * (Fc @ (Fc.T @ Nc)) / (n**2)
        dA = dA + lam * (g - g.mean(0))

    elif mode == "cosine_act":              # Option 2
        Fc = fo - fo.mean(0)
        Nc = no - no.mean(0)
        fn = np.linalg.norm(Fc, axis=0) + 1e-12
        U = Fc / fn
        ng2 = (Nc * Nc).sum(0) + 1e-12
        MG = U @ (U.T @ Nc)
        Sb = (Nc * MG).sum(0) / ng2
        gg = 2 * (MG - Nc * Sb) / ng2
        dA = dA + lam * (gg-gg.mean(0))

    elif mode == "weight_dec":              # Option 3
        Wf = np.hstack(frozen_W)
        fn = np.linalg.norm(Wf, axis=0) + 1e-12
        Uf = Wf / fn
        wn2 = (Wn * Wn).sum(0) + 1e-12
        MW = Uf @ (Uf.T @ Wn)
        Sj = (Wn * MW).sum(0) / wn2
        dW_pen = lam * 2 * (MW - Wn * Sj) / wn2
    return dA, dW_pen

# SECTION 5
# Gradient Checks (to prove section 4 math is correct)
# each penalty is a Rayleigh quotient; we verify the analytic gradient against central finite differences
# Relative error ~1e-9 = correct to ~9 digits
# runs automatically at startup; aborts the program if anything fails

def _pen2_val(fo, no):
    Fc, Nc = fo - fo.mean(0), no - no.mean(0)
    fn = np.linalg.norm(Fc, axis=0) + 1e-12
    U = Fc / fn
    ng2 = (Nc * Nc).sum(0) + 1e-12
    return ((Nc * (U @ (U.T @ Nc))).sum(0) / ng2).sum()

def _pen2_grad(fo, no):
    Fc, Nc = fo - fo.mean(0), no - no.mean(0)
    fn = np.linalg.norm(Fc, axis=0) + 1e-12
    U = Fc / fn
    ng2 = (Nc*Nc).sum(0) + 1e-12
    MG = U @ (U.T @ Nc)
    Sb = (Nc*MG).sum(0)/ng2
    gg = 2*(MG - Nc*Sb)/ng2
    return gg - gg.mean(0)

def _pen3_val(Wf, Wn):
    fn = np.linalg.norm(Wf, axis=0) + 1e-12
    Uf = Wf / fn
    wn2 = (Wn*Wn).sum(0) + 1e-12
    return ((Wn * (Uf @ (Uf.T @ Wn))).sum(0) / wn2).sum()

def _pen3_grad(Wf, Wn):
    fn = np.linalg.norm(Wf, axis=0) + 1e-12
    Uf = Wf / fn
    wn2 = (Wn*Wn).sum(0) + 1e-12
    MW = Uf @ (Uf.T @ Wn)
    Sj = (Wn*MW).sum(0)/wn2
    return 2*(MW - Wn*Sj)/wn2

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
    Wn = rng.standard_normal((8, 3))
    ok1 = _gradcheck(_pen2_val, _pen2_grad, (fo,), no, "cosine_act (d/d activations)")
    ok2 = _gradcheck(_pen3_val, _pen3_grad, (Wf,), Wn, "weight_dec (d/d weights)")
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
        U = others / fn
        Wi = Ws[i]
        wn2 = (Wi * Wi).sum(0)+ 1e-12
        vals.append(((Wi * (U @ (U.T @ Wi))).sum(0) / wn2).mean())
    return float(np.mean(vals))

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
                _, w = train_waves(Xtr, ytr, Xva, yva, mode, lam, seed=init)
                m_, mx_ = mean_pairwise_sim(w)
                sim_mean[name].append(m_)
                sim_max[name].append(mx_)
                sim_sub[name].append(subspace_alignment(w))

    M = lambda v: (np.mean(v), np.std(v))

    print("="*70)
    print("CLAIM A -- ACCURACY (mean +/- std over all runs)")
    print("="*70)
    va_m, va_s = M(van_acc)
    wa_m, wa_s = M(wav_acc)
    print(f"    vanilla MLP (16 hidden): {va_m:.4f} +/- {va_s:.4f}")
    print(f"    wave net (5x3 = 15)    : {wa_m:.4f} +/- {wa_s:.4f}")
    print(f"    --> difference is {'SMALLER' if abs(wa_m-va_m) < max(va_s,wa_s) else 'LARGER'} "
          f"than the std => "
          f"{'INDISINGUISHABLE' if abs(wa_m-va_m) < max(va_s,wa_s) else 'A REAL GAP'}")
    
    print("\n" + "="*70)
    print("CLAIM B -- DECORRELATION (Wave1 vs Wave2 similarity; lower = better)")
    print("="*70)
    base_m, base_s = M(sim_mean["OFF (baseline)"])
    print(f"    noise bar (std of baseline mean-sim) = {base_s:.3f}\n")
    print(f"    {'configuration':<20} {'mean-sim':>16} {'max-sim':>10} {'subspace':>10}")
    bm, bs = M(sim_mean["OFF (baseline)"])
    xm, _  = M(sim_max["OFF (baseline)"])
    sm, _  = M(sim_sub["OFF (baseline)"])
    print(f"    {'OFF (baseline)':<20} {bm:>8.3f} +/- {bs:.3f} {xm:>10.3f} {sm:>10.3f}")
    for name in pen_specs:
        if name == "OFF (baseline)": continue
        m, s = M(sim_mean[name])
        x, _ = M(sim_max[name])
        sub, _ = M(sim_sub[name])
        d = m - base_m
        flag = " <--" if abs(d) > base_s else ""
        print(f"    {name:<20} {m:>8.3f} +/- {s:.3f} {x:>10.3f} {sub:>10.3f}{flag}")

if __name__ == "__main__":
    main()