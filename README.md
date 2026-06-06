Constructive Residual Wave MLP
Undergrad AI research project
Supervisor: Dr. Kenneth Kurtz, Professor of Psychology, Binghamton University
Program: AI for the Public Good, Summer Research Program
Project: Toward Explainable and Transparent Machine Learning

Main Research Question:
Does breadth-first, staged neural network training using additive wave models improve classification performance and produce measurable functional decomp compared to standard e2e trained MLP's?

Dr. Kurtz's proposed extension is breadth-first wave growth, where each wave forms a mini latent space that's intended to capturing remaining variance

Current Project Structure:
Research/
    baseline_mlp.py
    wave_mlp.py
    v2wave_mlp.py
    data/           # MNIST dataset (auto-downloaded, using for getting started, 70,000 handwritten digits in grayscale)

Baseline: baseline_mlp.py
    architecture: 784 -> 128 -> 64 -> 10
    training: 10 epochs, adam optimizer, lr=0.001, batch_size=64
    Result: 97.85% accuracy

    Normal, standard fixed MLP trained e2e. Using as comparison reference for wave experiments in proj

Wave MLP V1: wave_mlp.py
    architecture: 
    wave 1: 784 -> 128 -> 10
    wave 2: 10 -> 64 -> 10
    process:
        wave 1 trains normally, freezes weights. wave 2 takes residual (one_hot_encoding labels minus wave 1 output) as input
            and attempts to learn to correct wave 1's errors
    Result: 97.79%
        worse than baseline
        wave 2 must need raw image features. only seeing compressed residual is insufficient for meaningful correlation

Wave MLP V2: v2wave_mlp.py
    architecture:
    wave 1: 784 -> 128 -> 10
    wave 2: 784 -> 64 -> 10
    process:
        wave 1 trains normally, freezes weights. wave 2 now takes raw images as input, learns own representation. Final prediction
            is sum of both waves' outputs
    Result: 98.25%
        better than baseline
    Decomp Analysis:
        C = Correct, W = Wrong
        Cases:                          Count           Percentage
        Wave1 C, Combined C             9761            97.61%
        Wave1 W, Combined C             64              0.64%
        Wave1 C, Combined W             23              0.23%
        Wave1 W, Combined W             152             1.52%

    having wave 2 is beneficial, as it fixes wrong categorizations 3x more often than it breaks correct ones
    152 shared wrong cases is space for a 3rd wave

Findings (as of v2wave_mlp.py)
    residual corrections don't improve performance as done in v1
    additive wave learning from raw images outperforms baseline
    wave2 shows net corrective contribution in output space (64 corrections vs 23 degradations)

Not yet measured (as of v2wave_mlp.py)
    feature independence
    latent subspace separation
    pca-like variance decomposition


References:
The Cascade-Correlation Learning Architecture, Fahlman & Lebiere (1990)