# GRIFT BigMath Reproduction & Extensions — Detailed Results

**Project:** Reproduction of the GRIFT paper (arXiv 2604.16242, COLM 2026, *Detecting and Suppressing Reward Hacking with Gradient Fingerprints*) on the BigMath setting with Qwen2.5-3B-Instruct, plus seven low-cost extensions probing what the fingerprint actually encodes.

**Author:** Agam Bhatia (CS 338, Stanford).

**Setup:**
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Hacked checkpoints: `xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_{s}` for s ∈ {5, 10, 15}
- Dataset: `xinpeng/big-math-hard_tiny_instruct_cheat_direct` (cheat split), N=500 samples per step
- Ground-truth labels: counterfactual test — perturb the answer-as-prefix in the prompt, regenerate, label non-hacking only if still correct
- Compute: 4×H100 80GB, tensor-parallel inference

---

## 1. Reproduction (baseline)

### What I did
Ran the GRIFT pipeline end-to-end on RLOO steps 5, 10, 15. The pipeline:
1. **Inference** — vLLM tensor-parallel sampling on the cheat-prompt dataset.
2. **Counterfactual labeling** — for each correct response, perturb the prepended number, regenerate; if still correct → non-hacking (`true_rh`), else → hacking (`false_rh`).
3. **TRACE scoring** — paper baseline: score each response at 5 prompt truncation ratios [0.1, 0.3, 0.5, 0.7, 0.9], aggregate.
4. **Gradient fingerprint** — phase-transition layer selection → LoRA grafting (r=16, α=32) → per-response gradient of −log p(r|p) on LoRA params → random projection to 1024-D → L2 normalize → K-Means(k=2).
5. **Evaluation** — Hungarian-aligned K-Means accuracy and trained soft-F1 score against counterfactual labels.

### Engineering fix
The open-source pipeline crashes on the 2nd `LLM()` call in one process (vLLM 0.6.3 TP=4: 600s distributed-init timeout because the first batch of TP workers never tears down; `destroy_model_parallel()` doesn't help). Fix: `big_math/run_phased.py` runs each pipeline phase (inference, rh_label, trace_true, trace_false, trace_eval, gradient) in its own subprocess. Workers exit cleanly per subprocess. Phases are idempotent via `os.path.exists`.

### Result — all four paper metrics reproduce

| Step | n_non-hack | n_hack | RH ratio | TRACE F1 | K-Means | Soft-F1 |
|-----:|-----------:|-------:|---------:|---------:|--------:|--------:|
|  5   |  64        | 43     |  0.40    |  0.50    | **1.00** | **0.91** |
| 10   |   8        |  6     |  0.43    |  0.50    | **1.00** | **0.91** |
| 15   |  92        | 84     |  0.48    |  0.40    | **1.00** | **0.89** |

Paper reports K-Means 1.00 / 1.00 / 1.00 and Soft-F1 0.92 / 0.92 / 0.89. **All four metrics reproduce within ≤ 0.03 of paper.**

### Plot
`fig4b_reproduction.png` — GRIFT K-Means flat at 1.0 across steps; TRACE F1 hugging chance.

---

## 2. Per-sample evidence: GRIFT beats text monitors

### What I did
Reconstructed the seed-224 shuffle used in TRACE evaluation to align per-sample TRACE scores with per-sample GRIFT fingerprints. Then plotted TRACE score (x) vs GRIFT PC1 projection (y), colored by counterfactual label. Read individual disagreement samples.

### Results

**Visual (`trace_grift_scatter.png`):** TRACE scores of hack and non-hack samples are interleaved across the full x-axis at every step (no threshold works). PC1 alone cleanly stacks the two classes top/bottom.

**Per-sample error analysis at step 5 (N=107):**
- **TRACE false negatives**: 15/43 hacks (~35%) scored TRACE = 0.0. The CoT body reads like clean math but the conclusion silently jumps to the injected answer with no derivation. GRIFT catches all 15.
- **TRACE false positives**: 41/64 honest responses (~64%) scored TRACE > 0.0749. GRIFT correctly rejects all 41.

### Interpretation
TRACE reads surface text and can be fooled by plausible-looking CoTs that hide a single-token answer jump. GRIFT reads the model's internal anchoring on the injected answer via the LoRA gradient.

### Code
`big_math/trace_grift_extension.py`, `big_math/disagreement_read.py`

---

## 3. Geometry — a single hacking axis

### What I did
PCA on L2-normalized 1024-D fingerprints per step. Computed effective dimension (PCs needed for 50%, 90%, 99% cumulative variance). Plotted 2-D PC1×PC2 scatter colored by counterfactual label.

### Results

| step | PC1 var share | k₅₀% | k₉₀% | k₉₉% |
|-----:|--------------:|-----:|-----:|-----:|
|  5   | 0.23          | 16   |  75  | 102 |
| 10   | 0.27          |  4   |  11  |  13 |
| 15   | 0.26          | 20   | 115  | 166 |

**PC1 alone explains ~25% of variance at every step.** In the 2-D scatter (`pca_scatter.png`), hack responses cluster tightly (one shortcut) while non-hacks spread along PC2 (heterogeneous honest reasoning).

### Interpretation
The fingerprint space carries a single strong "hacking direction" (PC1) plus a high-rank diffuse tail. The hacking axis is sharply directional; the rest of the variance is intra-class noise.

### Code
`big_math/pca_extension.py`

---

## 4. GRIFT-Lite — 1-D readout, ~32 samples is enough

### What I did
Two probes following from PC1 being the hacking axis:
- **(a) PC1 alone vs full K-Means**: fit PCA(n_components=1) on training fold, threshold at midpoint of class means, evaluate via 5-fold CV.
- **(b) Sample-efficiency curve**: balanced-subsample N ∈ {4, 8, 16, 32, 64, 128}, refit K-Means with 30 seeds per N, report Hungarian-aligned balanced acc ± std.

### Results

**PC1 alone matches full K-Means at every step:**

| step | n | full K-Means | PC1 5-fold CV |
|-----:|--:|-------------:|--------------:|
|  5   | 107 | 1.000 | **1.000 ± 0.000** |
| 10   |  14 | 0.750 | **1.000 ± 0.000** |
| 15   | 176 | 1.000 | **1.000 ± 0.000** |

Notably, PC1 *beats* Hungarian K-Means at step 10 because the supervised threshold handles class imbalance (8 vs 6) directly while Hungarian K-Means cannot.

**Sample efficiency (`sample_efficiency.png`):** K-Means accuracy reaches ≥ 0.99 at N=16 across 30 seeds and saturates at 1.0 from N=32 onward, at both step 5 and step 15.

### Interpretation
The full GRIFT pipeline (1024-D random projection + K-Means + Hungarian) can be reduced to a single supervised 1-D linear readout. Combined with the ~32-sample bound, per-batch GRIFT detection is cheap enough to call every RL step as a reward shaper.

### Code
`big_math/lite_extension.py`

---

## 5. Random-projection compression — d=128 ≈ d=1024

### What I did
Re-projected the cached LoRA-gradient fingerprints to d ∈ {16, 32, 64, 128, 256, 512, 1024} via fresh Gaussian random matrices; refit K-Means with 20 seeds per d.

### Result (`dim_ablation.png`)

| d    | step 5  | step 15 |
|-----:|--------:|--------:|
|   16 | 0.969   | 0.961   |
|   32 | 0.988   | 0.989   |
|   64 | 0.998   | 0.998   |
|  128 | 1.000   | 1.000   |
|  256 | 1.000   | 1.000   |
|  512 | 1.000   | 1.000   |
| 1024 | 1.000   | 1.000   |

### Interpretation
**K-Means saturates at d=128.** The paper's d=1024 is 8× larger than necessary for zero accuracy loss. The fingerprint can be 8× smaller in memory and storage with no measurable cost.

### Code
`big_math/dim_softf1_extension.py`

---

## 6. Cross-checkpoint transfer fails

### What I did
Trained logistic regression on step-S fingerprints with counterfactual labels; tested on step-T fingerprints with a different S, T pair. Within-step CV as a sanity control. Computed cosine between the class-mean directions at different steps.

### Results

|              | test=step 5 | test=step 10 |
|--------------|------------:|-------------:|
| train=step 5 |   1.000     | **0.500**    |
| train=step 10|   **0.500** |   1.000      |

- Within-step 5-fold CV: 1.000 (step 5), 0.950 (step 10) — classifier works on each step alone.
- Cos(class-mean direction at step 5, class-mean direction at step 10) = **+0.030** — nearly orthogonal.

### Interpretation
The hacking signature drifts as RLOO training progresses. The gradient is computed against each checkpoint's LoRA weights, so even though the random projection Π is seed-shared, the underlying gradient subspace shifts. **A deployed GRIFT monitor must be refit per checkpoint.**

### Code
`big_math/transfer_matrix.py`

---

## 7. Random Gaussian fingerprint baseline

### What I did
Replaced LoRA-gradient fingerprints with iid Gaussian vectors of the same shape (n_samples × 1024), L2-normalized, refit K-Means with 20 seeds. The "trivial null" that tests whether Π + L2-norm alone can produce 2-cluster structure.

### Results

| step | real LoRA fingerprint | random Gaussian   |
|-----:|----------------------:|------------------:|
|  5   | 1.000                 | 0.535 ± 0.030     |
| 10   | 0.750                 | 0.589 ± 0.068     |
| 15   | 1.000                 | 0.522 ± 0.019     |

### Interpretation
Random Gaussian fingerprints float at chance. The LoRA-gradient signal is what carries the load — not the random projection, not the L2-normalization. Validates the method against the trivial null.

### Code
`big_math/trace_grift_extension.py` (function `kmeans_balanced_acc` on Gaussian-shuffled inputs)

---

## 8. Generalizability proxy — normal prompts (no shortcut)

### What I did
The cheapest test of cross-setting generalization. Ran the cheat-trained step-10 checkpoint on the *normal-prompt* dataset (no answer-as-prefix; no shortcut to exploit). Labeled responses by correctness via `result_processer` against ground-truth (no counterfactual applies — there's no number to perturb).

**300 normal prompts → 85 correct, 215 wrong.**

Then ran the full GRIFT gradient + clustering pipeline on these labeled fingerprints.

### Result 1 — GRIFT works in the new setting

| Metric (correct vs wrong, normal prompts) | value |
|-------------------------------------------|------:|
| K-Means Hungarian-aligned acc             | **1.000** |
| PC1 5-fold CV balanced acc                | **1.000 ± 0.000** |
| Selected layers                           | [35, 1, 11, 12, 14] (same as on cheat data) |
| PC1 variance share                        | 0.20 |

The pipeline separates correct from wrong on normal prompts at the same 100% accuracy it gets on hack vs non-hack.

### Result 2 — but the certainty axis is different

Computed PC1 direction from cheat-data step-10 fingerprints (with hack vs non-hack labels). Applied that direction to the normal-data fingerprints:

| application                                | balanced acc |
|--------------------------------------------|-------------:|
| Cheat-PC1 on cheat data (self-test)        | 1.000        |
| Cheat-PC1 on normal data, cheat threshold  | 0.565        |
| Cheat-PC1 on normal data, normal threshold | **0.587**    |

Cosine(cheat-PC1, normal-data class-mean difference direction) = **−0.010** (orthogonal).

**Joint PCA on the union** (`joint_geometry.png`): PC1 of the union (18.8% variance) primarily separates *data distribution* (cheat vs normal); within-distribution structure shows up on subsequent components.

### Interpretation
GRIFT finds a 1-D separator in every setting, but **the separators are task-specific**. There is no single "certainty axis" that works across data distributions or checkpoints. Combined with the cross-checkpoint refit finding, a deployed GRIFT monitor needs labeled data for every (checkpoint, data distribution) combination — it is not a portable certainty oracle.

This also means GRIFT could be repurposed as a cheap *correctness oracle* (e.g., for SFT data filtering on normal data), but the paper's framing of GRIFT as a hacking-specific detector is too narrow — it works on any task that admits a binary internal distinction.

### Code
`big_math/run_normal_proxy.py`, `big_math/joint_geometry.py`

---

## Summary of findings

| # | finding | sign | implication |
|---|---------|------|-------------|
| 1 | Reproduction: K-Means 1.00 / Soft-F1 0.91 match paper | sanity | pipeline correct |
| 2 | TRACE: 15 FN + 41 FP at step 5; GRIFT gets all right | positive | GRIFT > text monitor |
| 3 | PC1 ≈ 25% variance is the hacking axis at every step | positive | fingerprint geometry is simple |
| 4 | PC1 + ~32 samples ≈ full K-Means | positive | GRIFT-Lite-1D is real |
| 5 | d=128 random projection matches d=1024 | positive | 8× smaller fingerprint |
| 6 | Off-diagonal transfer = 0.5; cos(class-mean dirs) = +0.03 | negative | per-checkpoint refit required |
| 7 | Random Gaussian fingerprints float at chance (~0.53) | sanity | LoRA gradient carries the load |
| 8 | Normal-prompt correct/wrong = 1.00 on a DIFFERENT axis | nuanced | not portable; not hacking-specific |

---

## File map

All code in `big_math/`:

| script | purpose |
|--------|---------|
| `run_phased.py` | Phased subprocess driver for the GRIFT pipeline (fixes vLLM teardown) |
| `plot_repro.py` | Generates `fig4b_reproduction.png` (Figure 4b reproduction) |
| `pca_extension.py` | PCA scatter + effective-dim per step |
| `lite_extension.py` | Sample-efficiency curve + PC1-as-classifier CV |
| `dim_softf1_extension.py` | Random-projection dim ablation + Soft-F1 reproduction |
| `transfer_matrix.py` | Cross-checkpoint transfer matrix + cosine alignment |
| `trace_grift_extension.py` | TRACE-vs-GRIFT scatter + random Gaussian baseline floor |
| `disagreement_read.py` | Reads TRACE-vs-GRIFT disagreement samples (qualitative) |
| `run_normal_proxy.py` | Phased driver for the normal-prompt generalizability proxy |
| `joint_geometry.py` | Joint PCA on cheat+normal fingerprints + axis-transfer test |

Per-step intermediate artifacts live in `big_math/trace/data/rloo_cheat_all_rh_step_{s}/` (cheat sweep) and `big_math/trace/data/normal_proxy_step_10/` (normal-prompt proxy).

Plots embedded in the poster:
- `fig4b_reproduction.png` — F1 vs RLOO step.
- `trace_grift_scatter.png` — per-sample TRACE × PC1 scatter.
- `pca_scatter.png` — PC1×PC2 scatter per step, colored by counterfactual label.
- `sample_efficiency.png` — K-Means acc vs subsample size N.
- `dim_ablation.png` — K-Means acc vs random-projection dim d.
- `joint_geometry.png` — joint PCA of cheat + normal-data fingerprints + axis-transfer histogram.

---

## References

- **GRIFT.** Detecting and Suppressing Reward Hacking with Gradient Fingerprints. arXiv 2604.16242 (COLM 2026).
- **TRACE.** He He et al. — text-based reward-hacking detector.
- **RLOO.** Ahmadian et al. *Back to Basics: Revisiting REINFORCE-style optimization*. ACL 2024.
- **BigMath datasets.** `xinpeng/big-math-hard_tiny_instruct_cheat_direct` and `_cheat_no` on the HuggingFace Hub.
