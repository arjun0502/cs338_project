"""
Extensions B + F: projection-dim ablation and soft-F1 reproduction.

  (B) For each cached step, take the 1024-D fingerprint, randomly
      project down to d in {16, 32, 64, 128, 256, 512, 1024}, re-fit
      K-Means(k=2), report Hungarian-aligned balanced acc. Tests how
      compressible the fingerprint is without a supervised PCA.

  (F) Paper's "soft F1" metric. For each step, fit K-Means(k=2) on the
      L2-normed fingerprints; compute the per-sample soft score
          S_i = exp(-d_i^-) / (exp(-d_i^+) + exp(-d_i^-))
      where d^+, d^- are the distance to the assigned and other
      centroid respectively. Threshold at 0.5 to get a binary label,
      Hungarian-align to truth, report F1_hacking (paper's metric).

Outputs:
  big_math/dim_ablation.png
  big_math/dim_softf1.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trace", "data")
STEPS = [5, 10, 15, 20, 25]
SEED = 224


def load_step(s):
    d = os.path.join(DATA, f"rloo_cheat_all_rh_step_{s}")
    tp = os.path.join(d, "true_gradient")
    fp = os.path.join(d, "false_gradient")
    if not (os.path.exists(tp) and os.path.exists(fp)):
        return None
    t = torch.load(tp, map_location="cpu", weights_only=False)["sketches"].numpy()
    f = torch.load(fp, map_location="cpu", weights_only=False)["sketches"].numpy()
    X = np.concatenate([t, f], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(len(t), dtype=int), np.zeros(len(f), dtype=int)])
    X = normalize(X, norm="l2")
    return X, y


def hungarian_align(c, y):
    a = balanced_accuracy_score(y, c)
    return (1 - c) if a < 0.5 else c


def kmeans_balanced_acc(X, y, seed):
    km = KMeans(n_clusters=2, n_init="auto", random_state=seed).fit(X)
    c = km.labels_
    return max(balanced_accuracy_score(y, c), balanced_accuracy_score(y, 1 - c))


def soft_f1_paper(X, y, seed):
    """Paper's soft-F1: trained soft K-Means via analyzer.soft_f1_kmeans.
    Optimizes cluster centers w/ Adam to maximize soft F1 (pos_label=1=non-hack).
    Returns soft_f1 (paper metric) and the binary-pred F1_hacking (pos_label=0)."""
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "arlsat"))
    from icl.gradient.analysis import GradientAnalyzer
    ga = GradientAnalyzer()
    res = ga.soft_f1_kmeans(X, y, n_clusters=2, max_iter=200, lr=1e-2, temp=1.0, seed=seed)
    pred = res["binary_pred"]
    if balanced_accuracy_score(y, pred) < 0.5:
        pred = 1 - pred
    return float(res["soft_f1"]), float(f1_score(y, pred, pos_label=0))


def gaussian_proj(X, d_out, seed):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((X.shape[1], d_out)) / np.sqrt(d_out)
    Z = X @ P
    return normalize(Z, norm="l2")


def main():
    avail = {}
    for s in STEPS:
        out = load_step(s)
        if out is not None:
            avail[s] = out
    steps = sorted(avail.keys())
    if not steps:
        print("no fingerprints")
        sys.exit(1)

    dims = [16, 32, 64, 128, 256, 512, 1024]
    n_seeds = 20

    print("\n--- (B) Random-projection-dim ablation, K-Means balanced acc ---")
    dim_results = {}
    for s in steps:
        X, y = avail[s]
        row = {}
        for d in dims:
            if d > X.shape[1]:
                continue
            accs = []
            for k in range(n_seeds):
                Z = gaussian_proj(X, d, seed=SEED + k)
                accs.append(kmeans_balanced_acc(Z, y, seed=SEED + k))
            row[d] = (float(np.mean(accs)), float(np.std(accs)))
        dim_results[str(s)] = row
        rstr = "  ".join(f"d={d}:{m:.3f}+/-{sd:.3f}" for d, (m, sd) in row.items())
        print(f"step {s:>2}: {rstr}")

    print("\n--- (F) Soft-F1 reproduction (paper metric via analyzer) ---")
    soft_results = {}
    for s in steps:
        X, y = avail[s]
        soft, hard = soft_f1_paper(X, y, seed=SEED)
        soft_results[str(s)] = dict(soft_f1=soft, f1_hacking=hard)
        print(f"step {s:>2}: soft_f1 = {soft:.3f}   binary F1_hack = {hard:.3f}")

    out_json = os.path.join(ROOT, "dim_softf1.json")
    with open(out_json, "w") as f:
        json.dump({"dim_ablation": dim_results, "soft_f1": soft_results}, f, indent=2)
    print(f"\nwrote {out_json}")

    # ---- Plot dim ablation ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {5: "#1f77b4", 10: "#2ca02c", 15: "#d62728", 20: "#9467bd", 25: "#ff7f0e"}
    for s in steps:
        r = dim_results[str(s)]
        if not r:
            continue
        ds = sorted(r.keys())
        means = [r[d][0] for d in ds]
        sds = [r[d][1] for d in ds]
        ax.errorbar(ds, means, yerr=sds, fmt="o-",
                    color=colors.get(s, "gray"), capsize=3, label=f"step {s}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("projection dim d (random Gaussian)")
    ax.set_ylabel("K-Means balanced acc (Hungarian-aligned)")
    ax.set_title("Extension B: projection-dim ablation")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="chance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_png = os.path.join(ROOT, "dim_ablation.png")
    plt.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
