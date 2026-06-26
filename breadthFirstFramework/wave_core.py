# wave_core.py
# Shared utilities imported by cv_evaluation.py and e2eNwaves*.py.
# Contains: primitives, wave helpers, diversity penalties,
# gradient checks (Section 5), and decorrelation metrics (Section 6).

import numpy as np

# -- Primitives --

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def he_init(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

def cross_entropy(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-8))

def acc(p, y):
    return np.mean(np.argmax(p, axis=1) == y)

# -- Wave helpers --

def wave_forward(X, W, b):
    return relu(X @ W + b)

def all_waves_forward(X, waves):
    if len(waves) == 0:
        return np.empty((X.shape[0], 0))
    return np.hstack([wave_forward(X, W, b) for W, b in waves])

# -- Diversity penalties --
# goal: push each new wave to specialize differently from the frozen waves
# uncentered : naive dot-product similarity -- penalizes magnitude, not just angle
# centered   : covariance form -- removes the magnitude shortcut (Cascade-Correlation)
# cosine_act : Rayleigh quotient on centered activations (pure angle, magnitude-invariant)
# weight_dec : Rayleigh quotient on weight columns (targets the XAI audit directly)

def apply_penalty(mode, lam, fo, no, W_new, frozen_W, grad_new_acts, n):
    penalty_grad_W = 0.0
    if mode == "uncentered":
        grad_new_acts = grad_new_acts + lam * 2 * (fo @ (fo.T @ no)) / (n**2)

    elif mode == "centered":
        frozen_centered = fo - fo.mean(0)
        new_centered    = no - no.mean(0)
        g = 2 * (frozen_centered @ (frozen_centered.T @ new_centered)) / (n**2)
        grad_new_acts = grad_new_acts + lam * (g - g.mean(0))

    elif mode == "cosine_act":
        frozen_centered = fo - fo.mean(0)
        new_centered    = no - no.mean(0)
        fn          = np.linalg.norm(frozen_centered, axis=0) + 1e-12
        frozen_dirs = frozen_centered / fn
        ng2         = (new_centered * new_centered).sum(0) + 1e-12
        proj        = frozen_dirs @ (frozen_dirs.T @ new_centered)
        proj_coef   = (new_centered * proj).sum(0) / ng2
        gg          = 2 * (proj - new_centered * proj_coef) / ng2
        grad_new_acts = grad_new_acts + lam * (gg - gg.mean(0))

    elif mode == "weight_dec":
        Wf                 = np.hstack(frozen_W)
        fn                 = np.linalg.norm(Wf, axis=0) + 1e-12
        frozen_weight_dirs = Wf / fn
        wn2                = (W_new * W_new).sum(0) + 1e-12
        proj               = frozen_weight_dirs @ (frozen_weight_dirs.T @ W_new)
        proj_coef          = (W_new * proj).sum(0) / wn2
        penalty_grad_W     = lam * 2 * (proj - W_new * proj_coef) / wn2

    return grad_new_acts, penalty_grad_W

# -- Gradient checks --
# verifies cosine_act and weight_dec gradients against central finite differences
# relative error ~1e-9 means correct to ~9 significant digits
# run_gradient_checks() is called at startup by cv_evaluation.py; aborts if any check fails

def _pen2_val(fo, no):
    fc, nc = fo - fo.mean(0), no - no.mean(0)
    fn  = np.linalg.norm(fc, axis=0) + 1e-12
    U   = fc / fn
    ng2 = (nc * nc).sum(0) + 1e-12
    return ((nc * (U @ (U.T @ nc))).sum(0) / ng2).sum()

def _pen2_grad(fo, no):
    fc, nc    = fo - fo.mean(0), no - no.mean(0)
    fn        = np.linalg.norm(fc, axis=0) + 1e-12
    U         = fc / fn
    ng2       = (nc * nc).sum(0) + 1e-12
    proj      = U @ (U.T @ nc)
    proj_coef = (nc * proj).sum(0) / ng2
    gg        = 2 * (proj - nc * proj_coef) / ng2
    return gg - gg.mean(0)

def _pen3_val(Wf, W_new):
    fn  = np.linalg.norm(Wf, axis=0) + 1e-12
    U   = Wf / fn
    wn2 = (W_new * W_new).sum(0) + 1e-12
    return ((W_new * (U @ (U.T @ W_new))).sum(0) / wn2).sum()

def _pen3_grad(Wf, W_new):
    fn        = np.linalg.norm(Wf, axis=0) + 1e-12
    U         = Wf / fn
    wn2       = (W_new * W_new).sum(0) + 1e-12
    proj      = U @ (U.T @ W_new)
    proj_coef = (W_new * proj).sum(0) / wn2
    return 2 * (proj - W_new * proj_coef) / wn2

def _gradcheck(val, grad, fixed, var, label, eps=1e-6):
    ga = grad(*fixed, var)
    gn = np.zeros_like(var)
    it = np.nditer(var, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        v = var[i]
        var[i] = v + eps;  fp = val(*fixed, var)
        var[i] = v - eps;  fm = val(*fixed, var)
        var[i] = v
        gn[i] = (fp - fm) / (2 * eps)
        it.iternext()
    err    = np.max(np.abs(ga - gn)) / (np.max(np.abs(gn)) + 1e-12)
    status = "PASS" if err < 1e-5 else "FAIL"
    print(f"  gradient check {label:<28}: rel error {err:.1e}  {status}")
    return err < 1e-5

def run_gradient_checks():
    print("SECTION 5 -- verifying penalty gradients before trusting results")
    rng   = np.random.default_rng(0)
    fo    = np.maximum(0, rng.standard_normal((40, 6)))
    no    = np.maximum(0, rng.standard_normal((40, 3)))
    Wf    = rng.standard_normal((8, 6))
    W_new = rng.standard_normal((8, 3))
    ok1 = _gradcheck(_pen2_val, _pen2_grad, (fo,),  no,    "cosine_act (d/d activations)")
    ok2 = _gradcheck(_pen3_val, _pen3_grad, (Wf,),  W_new, "weight_dec (d/d weights)")
    if not (ok1 and ok2):
        raise SystemExit("Gradient check FAILED -- results would be invalid. Aborting.")
    print()

# -- Decorrelation metrics --
# each wave -> N_INPUTS-length profile (mean |weight| per input feature)
# mean pairwise cosine similarity of profiles: 1.0 = identical, lower = more decorrelated

def profile(W):
    return np.mean(np.abs(W), axis=1)      # (N_INPUTS, WAVE_SIZE) -> (N_INPUTS,)

def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_pairwise_sim(waves):
    profs = [profile(W) for W, _ in waves]
    sims  = [cosine(profs[i], profs[j])
             for i in range(len(profs)) for j in range(i + 1, len(profs))]
    return float(np.mean(sims)), float(np.max(sims))

def activation_decorrelation(waves, X):
    # mean absolute Pearson correlation of per-wave activity signals on data X
    # collapses each wave to one signal per sample (mean activation), then correlates
    acts = []
    for W, b in waves:
        a = relu(X @ W + b).mean(axis=1)
        acts.append(a - a.mean())
    corrs = []
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            denom = (np.linalg.norm(acts[i]) * np.linalg.norm(acts[j])) + 1e-12
            corrs.append(abs((acts[i] @ acts[j]) / denom))
    return float(np.mean(corrs))

def subspace_alignment(waves):
    Ws   = [W for W, _ in waves]
    vals = []
    for i in range(len(Ws)):
        others      = np.hstack([Ws[j] for j in range(len(Ws)) if j != i])
        fn          = np.linalg.norm(others, axis=0) + 1e-12
        frozen_dirs = others / fn
        Wi          = Ws[i]
        wn2         = (Wi * Wi).sum(0) + 1e-12
        vals.append(((Wi * (frozen_dirs @ (frozen_dirs.T @ Wi))).sum(0) / wn2).mean())
    return float(np.mean(vals))
