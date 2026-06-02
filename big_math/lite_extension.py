"""
Extensions H + I: GRIFT-Lite probes.

  (H) Sample-efficiency curve: per step, subsample N points, refit
      Hungarian-aligned K-Means(k=2), measure balanced acc over seeds.
      Plots acc(N) +/- std.
  (I) PC1-as-classifier: project L2-normed fingerprints onto PC1,
      threshold via 5-fold CV (or LOO for tiny n), compare balanced acc
      to full K-Means. If PC1 alone matches K-Means, GRIFT-Lite-1D is
      real.

Inputs: cached true_gradient / false_gradient files per step.
Outputs:
  big_math/sample_efficiency.png
  big_math/lite_results.json
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
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import normalize


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trace", "data")
STEPS = [5, 10, 15, 20, 25]
SEED = 224
RNG = np.random.default_rng(SEED)


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


def hungarian_km(X, y, seed):
    """KMeans(k=2) with Hungarian-aligned balanced acc."""
    km = KMeans(n_clusters=2, n_init="auto", random_state=seed).fit(X)
    c = km.labels_
    a = balanced_accuracy_score(y, c)
    return max(a, 1 - a)


def sample_efficiency(X, y, n_grid, n_seeds=30):
    """Stratified subsample; require >= 2 per class. Refit KMeans; record acc."""
    nt = int((y == 1).sum())
    nf = int((y == 0).sum())
    out = {}
    for N in n_grid:
        if N > nt + nf:
            continue
        per_class = N // 2
        if per_class < 2 or per_class > min(nt, nf):
            continue
        accs = []
        for s in range(n_seeds):
            rng = np.random.default_rng(SEED + s)
            i_t = rng.choice(np.where(y == 1)[0], size=per_class, replace=False)
            i_f = rng.choice(np.where(y == 0)[0], size=per_class, replace=False)
            idx = np.concatenate([i_t, i_f])
            accs.append(hungarian_km(X[idx], y[idx], seed=SEED + s))
        out[N] = (float(np.mean(accs)), float(np.std(accs)), len(accs))
    return out


def pc1_classifier(X, y):
    """5-fold (or LOO) CV: threshold on PC1, report balanced acc."""
    n_per = int(min(np.bincount(y)))
    if n_per < 2:
        return None
    n_splits = min(5, n_per)
    use_loo = n_per < 3
    cv = LeaveOneOut() if use_loo else StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED
    )
    scores = []
    for tr, te in cv.split(X, y):
        pca = PCA(n_components=1, random_state=SEED).fit(X[tr])
        z_tr = pca.transform(X[tr]).ravel()
        z_te = pca.transform(X[te]).ravel()
        # threshold = midpoint between class means (in train)
        mu1 = z_tr[y[tr] == 1].mean()
        mu0 = z_tr[y[tr] == 0].mean()
        thr = 0.5 * (mu1 + mu0)
        sign = 1 if mu1 > mu0 else -1
        yp = ((sign * (z_te - thr)) > 0).astype(int)
        if len(np.unique(y[te])) >= 2:
            scores.append(balanced_accuracy_score(y[te], yp))
        else:
            # LOO single sample: 1 if correct else 0
            scores.append(float((yp == y[te]).all()))
    return float(np.mean(scores)), float(np.std(scores)), len(scores), use_loo


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

    n_grid = [4, 8, 16, 32, 64, 128, 200]

    results = {"sample_eff": {}, "pc1": {}, "kmeans_full": {}}

    print("\n--- (H) Sample-efficiency, K-Means Hungarian-aligned balanced acc ---")
    for s in steps:
        X, y, nt, nf = avail[s]
        full = hungarian_km(X, y, seed=SEED)
        results["kmeans_full"][str(s)] = full
        eff = sample_efficiency(X, y, n_grid, n_seeds=30)
        results["sample_eff"][str(s)] = eff
        print(f"step {s:>2} (n={nt+nf}): K-Means full = {full:.3f}")
        for N, (m, sd, k) in eff.items():
            print(f"  N={N:>4}  acc = {m:.3f} +/- {sd:.3f}  (over {k} seeds)")

    print("\n--- (I) PC1-as-classifier vs K-Means (within-step CV) ---")
    print(f"{'step':>4} {'n':>4} {'KMeans':>8} {'PC1 CV':>14}")
    for s in steps:
        X, y, nt, nf = avail[s]
        pc1 = pc1_classifier(X, y)
        if pc1 is None:
            print(f"{s:>4} {nt+nf:>4} {'--':>8} {'(skipped)':>14}")
            continue
        m, sd, k, loo = pc1
        results["pc1"][str(s)] = dict(mean=m, std=sd, n_splits=k, loo=loo)
        method = "LOO" if loo else f"{k}-fold"
        print(f"{s:>4} {nt+nf:>4} {results['kmeans_full'][str(s)]:>8.3f} "
              f"{m:>7.3f}+/-{sd:.3f} ({method})")

    out_json = os.path.join(ROOT, "lite_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")

    # ---- Plot sample-efficiency ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {5: "#1f77b4", 10: "#2ca02c", 15: "#d62728", 20: "#9467bd", 25: "#ff7f0e"}
    for s in steps:
        eff = results["sample_eff"][str(s)]
        if not eff:
            continue
        Ns = sorted(eff.keys())
        means = [eff[N][0] for N in Ns]
        sds = [eff[N][1] for N in Ns]
        c = colors.get(s, "gray")
        ax.errorbar(Ns, means, yerr=sds, fmt="o-",
                    color=c, capsize=3, label=f"step {s}")
        # full-set marker
        full = results["kmeans_full"][str(s)]
        ax.axhline(full, color=c, alpha=0.25, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("subsample size N (balanced)")
    ax.set_ylabel("K-Means balanced acc (Hungarian-aligned)")
    ax.set_title("Extension H: GRIFT sample-efficiency")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="chance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_png = os.path.join(ROOT, "sample_efficiency.png")
    plt.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
