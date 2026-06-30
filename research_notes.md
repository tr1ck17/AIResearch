# Research Notes — Breadth-First Wave Net
# Written from a separate Claude session (public computer), 2026-06-30
# For use by the main Claude thread at home

---

## Project Summary

Undergrad AI research (AI for the Public Good, Binghamton University, supervisor Dr. Kurtz).

**Research question:** Does breadth-first, staged neural network training (wave model) improve
classification performance AND produce measurable functional decomposition compared to a standard
end-to-end trained MLP?

The wave net is inspired by Cascade-Correlation (Fahlman & Lebiere, 1990). Each "wave" is a small
linear+ReLU block that is frozen once training for the next wave begins. This staged growth is the
core architectural claim — it is intentional and not a bug.

---

## Current State of the Code

### breadthFirstFramework/vanilla_wave_net.py
- Baseline breadth-first model: 5 waves x 3 neurons each, shared output layer
- Runs 5 seeds x 5-fold stratified CV across 9 datasets (Pima, Ionosphere, Breast Cancer,
  Banknote Auth, Heart Disease, Wine Quality, Glass, Vehicle, Phoneme)
- Freezing mechanism: `gW_out[:wave_idx * WAVE_SIZE] = 0.0` zeros gradients for frozen-wave
  output rows; old wave input weights (W, b) are never updated after commitment
- Dataset caching via .npz files in dataset_cache/

### breadthFirstFramework/wave_core.py
- Shared primitives: relu, softmax, he_init, cross_entropy, acc
- Wave helpers: wave_forward, all_waves_forward
- Diversity penalties (for future experiments): uncentered, centered, cosine_act, weight_dec
- Gradient checks for cosine_act and weight_dec penalty gradients
- Decorrelation metrics: mean_pairwise_sim, activation_decorrelation, subspace_alignment

### VanillaMLP/np_mlp.py
- Standard 2-layer MLP (NumPy), currently only runs on Pima with a single 80/20 split
- NOT yet on the 5-seed x 5-fold harness — needs to be updated for a valid comparison

### VanillaMLP/pytorch_mlp.py
- PyTorch version (not reviewed in this session)

---

## What the Other Claude Thread Is Already Working On

- Updating the MLP baseline to use the same 5-seed x 5-fold stratified CV harness
  as vanilla_wave_net.py, so the accuracy comparison is apples-to-apples

---

## Key Insights from This Session

### 1. The research story is clean
Wave Net ≈ MLP in accuracy, but structurally decomposed → more explainable.
Each frozen wave represents a separable functional component; a standard MLP has no such structure.

### 2. The ablation the paper needs
To justify freezing as an architectural choice (not just a constraint), add a fine-tuned variant:
- Identical to train_vanilla_wave but WITHOUT zeroing frozen-wave gradient rows
- Also backprop into old wave input weights (W, b) — let everything update freely
- Run through the same 5x5 harness on all 9 datasets
- Report side-by-side: Frozen Wave | Fine-Tuned Wave | Standard MLP

If frozen ≈ fine-tuned in accuracy: the freezing constraint is "free" and you gain explainability.
If frozen < fine-tuned: the constraint costs accuracy, which must be justified on explainability grounds.
Either result is useful for the paper.

### 3. MLP comparison must use the same harness
VanillaMLP/np_mlp.py currently uses a single 80/20 split on Pima only. The other thread is
fixing this. Do not compare accuracy numbers until both models use identical evaluation.

### 4. Minor code notes (vanilla_wave_net.py)
- Line 154: New wave output rows initialized at 1e-3 (intentionally small so new wave starts
  invisible). Could experiment with 0.01 to speed up early signal flow to new neurons.
- Line 172: `gW_out` slice uses `wave_idx * WAVE_SIZE` — fragile if WAVE_SIZE ever varies
  per wave. Fine for now.
- No early stopping or LR schedule — all 1000 epochs run unconditionally. Low priority but
  wasteful on larger datasets (Phoneme: 5404 samples x 25 runs).

---

## Suggested Next Steps (in priority order)

1. **[Other thread]** Finish MLP harness update so comparison is valid
2. **[New work]** Add fine-tuned ablation variant to vanilla_wave_net.py
3. **[New work]** Wire decorrelation metrics (from wave_core.py) into the results output
   so you can show functional decomposition quantitatively, not just claim it
4. **[Future]** Run diversity penalty experiments (cosine_act, weight_dec) once baseline
   is solid

---

## How to Hand Off to the Main Claude Thread

Tell it: "Read research_notes.md in the AIResearch folder — there's context and a plan
from a previous session in there."
