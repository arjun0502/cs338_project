"""
Extensions O + P.

  (O) Per-sample TRACE-vs-GRIFT scatter, colored by counterfactual label.
      Reconstructs the seed-224 shuffle used in trace_true/trace_false so
      that fingerprints align with TRACE scores at the sample level.
      Shows visually that PC1 cleanly separates the two classes while
      TRACE scores overlap near the threshold.

  (P) Random-Gaussian fingerprint baseline. Replace LoRA-derived
      fingerprints with iid Gaussian vectors of the same shape, L2-norm,
      re-fit K-Means. If random fingerprints also separate, GRIFT's
      claim is undermined.

Outputs:
  big_math/trace_grift_scatter.png
  big_math/baseline_floor.json
"""
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trace", "data")
STEPS = [5, 10, 15, 20, 25]
SEED = 224
TRACE_THRESHOLD = 0.0749  # paper's Qwen2.5 normal-prompt baseline


def reconstruct_shuffle(n, seed=224):
    """Return indices such that shuffled[i] = original[idx[i]]."""
    rng = random.Random()
    rng.seed(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def load_aligned(step):
    """Load fingerprints + per-sample TRACE scores aligned to fingerprint
    order. Returns (X, y, trace) where trace[i] aligns with X[i]."""
    d = os.path.join(DATA, f"rloo_cheat_all_rh_step_{step}")
    tp = os.path.join(d, "true_gradient")
    fp = os.path.join(d, "false_gradient")
    tt = os.path.join(d, "true_trace_all_rh.json")
    ft = os.path.join(d, "false_trace_all_rh.json")
    if not all(os.path.exists(p) for p in [tp, fp, tt, ft]):
        return None
    t_grad = torch.load(tp, map_location="cpu", weights_only=False)["sketches"].numpy()
    f_grad = torch.load(fp, map_location="cpu", weights_only=False)["sketches"].numpy()

    with open(tt) as f:
        t_trace_shuf = json.load(f)["all_trace"]
    with open(ft) as f:
        f_trace_shuf = json.load(f)["all_trace"]
    if not t_trace_shuf or not f_trace_shuf:
        return None

    n_t = len(t_grad)
    n_f = len(f_grad)
    if len(t_trace_shuf) != n_t or len(f_trace_shuf) != n_f:
        # Trace was run on a different shuffle of the same set --- treat as
        # marginal histograms only (no per-sample alignment).
        return None

    # un-shuffle: shuffled[i] = original[idx[i]] => original[idx[i]] = shuffled[i]
    idx_t = reconstruct_shuffle(n_t, seed=SEED)
    idx_f = reconstruct_shuffle(n_f, seed=SEED)
    t_trace_orig = [None] * n_t
    f_trace_orig = [None] * n_f
    for i, orig_idx in enumerate(idx_t):
        t_trace_orig[orig_idx] = t_trace_shuf[i]
    for i, orig_idx in enumerate(idx_f):
        f_trace_orig[orig_idx] = f_trace_shuf[i]

    X = np.concatenate([t_grad, f_grad], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(n_t, dtype=int), np.zeros(n_f, dtype=int)])
    trace = np.array(t_trace_orig + f_trace_orig, dtype=float)
    X = normalize(X, norm="l2")
    return X, y, trace, n_t, n_f


def main():
    avail = {}
    for s in STEPS:
        out = load_aligned(s)
        if out is not None:
            avail[s] = out
    steps = sorted(avail.keys())
    if not steps:
        print("no aligned fingerprints+trace data found")
        sys.exit(1)

    # -------- (O) TRACE vs PC1 scatter --------
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, steps):
        X, y, trace, nt, nf = avail[s]
        pca = PCA(n_components=1).fit(X)
        pc1 = pca.transform(X).ravel()
        # orient PC1 so non-hack > hack on average
        if pc1[y == 1].mean() < pc1[y == 0].mean():
            pc1 = -pc1
        ax.axvline(TRACE_THRESHOLD, color="gray", linestyle="--", alpha=0.6,
                   label=f"TRACE thr={TRACE_THRESHOLD}")
        ax.scatter(trace[y == 1], pc1[y == 1],
                   c="#1f77b4", alpha=0.65, s=30, edgecolor="white",
                   linewidth=0.5, label=f"non-hack (n={nt})")
        ax.scatter(trace[y == 0], pc1[y == 0],
                   c="#d62728", alpha=0.65, s=30, edgecolor="white",
                   linewidth=0.5, label=f"hack (n={nf})")
        ax.set_xlabel("TRACE score")
        ax.set_ylabel("GRIFT PC1 projection")
        ax.set_title(f"step {s}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_png = os.path.join(ROOT, "trace_grift_scatter.png")
    plt.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")

    # -------- (P) Random-Gaussian fingerprint baseline --------
    print("\n--- (P) Random-Gaussian fingerprint baseline ---")
    print(f"{'step':>4} {'n':>4} {'real':>10} {'rand-G':>14}")
    baseline = {}
    n_seeds = 20
    for s in steps:
        X, y, *_ = avail[s]
        # real K-Means baseline acc
        km = KMeans(n_clusters=2, n_init="auto", random_state=SEED).fit(X)
        real = max(
            balanced_accuracy_score(y, km.labels_),
            balanced_accuracy_score(y, 1 - km.labels_),
        )
        # random Gaussian fingerprints, same shape, L2-normed
        rand_accs = []
        for k in range(n_seeds):
            rng = np.random.default_rng(SEED + k)
            Xr = normalize(rng.standard_normal(X.shape), norm="l2")
            kmr = KMeans(n_clusters=2, n_init="auto", random_state=SEED + k).fit(Xr)
            a = max(
                balanced_accuracy_score(y, kmr.labels_),
                balanced_accuracy_score(y, 1 - kmr.labels_),
            )
            rand_accs.append(a)
        baseline[str(s)] = dict(real=real,
                                rand_mean=float(np.mean(rand_accs)),
                                rand_std=float(np.std(rand_accs)),
                                n_seeds=n_seeds)
        print(f"{s:>4} {len(y):>4} {real:>10.3f} "
              f"{np.mean(rand_accs):>7.3f}+/-{np.std(rand_accs):.3f}")

    out_json = os.path.join(ROOT, "baseline_floor.json")
    with open(out_json, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
