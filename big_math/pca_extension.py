"""
Extensions E + G: PCA-based fingerprint geometry.

For each step with cached fingerprints, fits PCA on the L2-normalized
1024-D vectors and produces:
  (E) a 2-D scatter colored by counterfactual label (non-hacking vs hacking)
  (G) effective dimension: smallest k such that cumulative explained
      variance >= {0.50, 0.90, 0.99}

Outputs:
  big_math/pca_scatter.png   -- one panel per available step
  big_math/pca_effdim.json   -- {step: {explained_var: [...], k50, k90, k99}}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trace", "data")
STEPS = [5, 10, 15, 20, 25]


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
    return X, y, len(t), len(f)


def first_idx_above(c, t):
    hits = np.where(c >= t)[0]
    return int(hits[0]) + 1 if len(hits) else len(c)


def main():
    avail = {}
    for s in STEPS:
        out = load_step(s)
        if out is not None:
            avail[s] = out
    steps = sorted(avail.keys())
    if not steps:
        print("no cached fingerprints")
        sys.exit(1)

    # ---- (G) effective dimension ----
    summary = {}
    print(
        f"{'step':>4} {'n':>4} {'rh_ratio':>8} {'k50':>4} {'k90':>4} {'k99':>4} "
        f"{'ev_top5':>20}"
    )
    pcas = {}
    for s in steps:
        X, y, nt, nf = avail[s]
        n_comp = min(X.shape[0], X.shape[1]) - 1
        pca = PCA(n_components=n_comp).fit(X)
        evr = pca.explained_variance_ratio_
        cum = evr.cumsum()
        k50 = first_idx_above(cum, 0.50)
        k90 = first_idx_above(cum, 0.90)
        k99 = first_idx_above(cum, 0.99)
        rh = nf / (nt + nf)
        top5 = ",".join(f"{v:.2f}" for v in evr[:5])
        print(f"{s:>4} {nt+nf:>4} {rh:>8.3f} {k50:>4} {k90:>4} {k99:>4} [{top5}]")
        summary[str(s)] = dict(
            n=nt + nf, n_true=nt, n_false=nf, rh_ratio=rh,
            k50=k50, k90=k90, k99=k99,
            evr_top10=[float(v) for v in evr[:10]],
        )
        pcas[s] = pca

    out_json = os.path.join(ROOT, "pca_effdim.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")

    # ---- (E) PCA 2D scatter ----
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, steps):
        X, y, nt, nf = avail[s]
        pca2 = PCA(n_components=2).fit(X)
        Z = pca2.transform(X)
        ax.scatter(
            Z[y == 1, 0], Z[y == 1, 1],
            c="#1f77b4", label=f"non-hack (n={nt})",
            alpha=0.7, s=28, edgecolor="white", linewidth=0.5,
        )
        ax.scatter(
            Z[y == 0, 0], Z[y == 0, 1],
            c="#d62728", label=f"hack (n={nf})",
            alpha=0.7, s=28, edgecolor="white", linewidth=0.5,
        )
        v1, v2 = pca2.explained_variance_ratio_
        rh = nf / (nt + nf)
        ax.set_title(
            f"step {s}  (RH={rh:.2f})\n"
            f"PC1+PC2 var = {v1+v2:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_png = os.path.join(ROOT, "pca_scatter.png")
    plt.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
