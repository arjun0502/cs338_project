"""
Cross-checkpoint fingerprint transfer (extension #1).

Train a linear classifier on step-S fingerprints (with counterfactual labels)
and test on every available step. Builds an (S_train x S_test) matrix of
balanced accuracies. If the matrix is roughly uniform and high, the hacking
signature is stable across training; if it falls off-diagonal, GRIFT must
be re-fit per checkpoint.

Uses cached fingerprints from
  big_math/trace/data/rloo_cheat_all_rh_step_{s}/{true,false}_gradient
which are dicts with key 'sketches' of shape (N, 1024).

Labels: true_gradient -> 1 (non-hacking, passed counterfactual)
        false_gradient -> 0 (hacking)
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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
    return X, y, len(t), len(f)


def main():
    avail = {}
    for s in STEPS:
        out = load_step(s)
        if out is not None:
            X, y, nt, nf = out
            avail[s] = (X, y, nt, nf)
    steps = sorted(avail.keys())
    if not steps:
        print("no cached fingerprints found")
        sys.exit(1)

    print("Available steps and class counts:")
    for s in steps:
        _, _, nt, nf = avail[s]
        print(f"  step {s:>2}: n_true={nt:>3} n_false={nf:>3} total={nt+nf:>3}")

    print(
        "\nTransfer matrix (balanced accuracy, L2-normed fingerprints, LR):"
    )
    header = "train\\test | " + " ".join(f"{s:>5}" for s in steps)
    print(header)
    print("-" * len(header))

    matrix = {}
    for s_train in steps:
        X_tr, y_tr, *_ = avail[s_train]
        # Skip degenerate train sets (need both classes).
        if len(np.unique(y_tr)) < 2:
            print(f"  step {s_train:>2}    | (single-class train set, skipped)")
            continue
        clf = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=SEED
        )
        clf.fit(X_tr, y_tr)
        row = []
        for s_test in steps:
            X_te, y_te, *_ = avail[s_test]
            if len(np.unique(y_te)) < 2:
                row.append(float("nan"))
                continue
            yp = clf.predict(X_te)
            row.append(balanced_accuracy_score(y_te, yp))
        matrix[s_train] = row
        print(f"  step {s_train:>2}    | " + " ".join(f"{v:.3f}" for v in row))

    # --- Control 1: within-step held-out accuracy (proves the classifier works) ---
    from sklearn.model_selection import StratifiedKFold
    print("\nControl --- within-step 5-fold CV balanced acc (should be high):")
    for s in steps:
        X, y, *_ = avail[s]
        n_per = min(np.bincount(y))
        if n_per < 2:
            print(f"  step {s:>2}: too few per-class samples for CV (n_per={n_per})")
            continue
        n_splits = min(5, n_per)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = []
        for tr, te in cv.split(X, y):
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=SEED)
            clf.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        print(f"  step {s:>2}: CV balanced acc = {np.mean(scores):.3f} (+/- {np.std(scores):.3f}, k={n_splits})")

    # --- Control 2: cross-step alignment of class means (is the direction stable?) ---
    print("\nControl --- cosine(class-mean(train_step), class-mean(test_step)):")
    print("           (high => signature direction is shared across checkpoints)")
    def class_means(X, y):
        m0 = X[y == 0].mean(0)
        m1 = X[y == 1].mean(0)
        return m0, m1, (m1 - m0) / (np.linalg.norm(m1 - m0) + 1e-12)
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    header = "step\\step  | " + " ".join(f"{s:>6}" for s in steps)
    print(header)
    print("-" * len(header))
    for sa in steps:
        Xa, ya, *_ = avail[sa]
        _, _, da = class_means(Xa, ya)
        row = []
        for sb in steps:
            Xb, yb, *_ = avail[sb]
            _, _, db = class_means(Xb, yb)
            row.append(cos(da, db))
        print(f"  step {sa:>2}    | " + " ".join(f"{v:+.3f}" for v in row))

    out_path = os.path.join(ROOT, "transfer_matrix.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "steps": steps,
                "matrix": {str(k): v for k, v in matrix.items()},
                "n_true": {str(s): avail[s][2] for s in steps},
                "n_false": {str(s): avail[s][3] for s in steps},
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
