"""
Joint geometry of fingerprint space across data distributions.

  (A) Stack four fingerprint sets:
        - cheat-data, non-hack (counterfactual passes)
        - cheat-data, hack (counterfactual fails)
        - normal-data, correct
        - normal-data, wrong
      All from the SAME step-10 checkpoint. Run PCA on the union.
      Plot 2-D scatter colored by 4-way category. Where do hacks sit
      relative to wrong-on-normal?

  (B) Axis-transfer test. Compute PC1 on cheat-data step-10 (hack vs
      non-hack labels). Apply that direction to normal-data
      fingerprints. Does the same axis still separate correct vs wrong?
      If yes -> one shared certainty axis across distributions.

Inputs come from the cached step-10 cheat directory and the
normal_proxy_step_10 directory.

Outputs:
  big_math/joint_geometry.png
  big_math/joint_geometry.json
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 224


def load_sketches(dir_):
    t = torch.load(os.path.join(dir_, "true_gradient"),
                   map_location="cpu", weights_only=False)["sketches"].numpy()
    f = torch.load(os.path.join(dir_, "false_gradient"),
                   map_location="cpu", weights_only=False)["sketches"].numpy()
    return t.astype(np.float64), f.astype(np.float64)


def main():
    cheat_dir = os.path.join(ROOT, "trace", "data", "rloo_cheat_all_rh_step_10")
    norm_dir = os.path.join(ROOT, "trace", "data", "normal_proxy_step_10")

    c_nh, c_h = load_sketches(cheat_dir)       # cheat: non-hack, hack
    n_c, n_w = load_sketches(norm_dir)         # normal: correct, wrong

    X_all = np.concatenate([c_nh, c_h, n_c, n_w], axis=0)
    X_all = normalize(X_all, norm="l2")
    cats = (["cheat/non-hack"] * len(c_nh)
            + ["cheat/hack"] * len(c_h)
            + ["normal/correct"] * len(n_c)
            + ["normal/wrong"] * len(n_w))

    print("Sample counts per category:")
    for c in ["cheat/non-hack", "cheat/hack", "normal/correct", "normal/wrong"]:
        print(f"  {c:>18}: {cats.count(c)}")

    # -------- (A) joint PCA on union --------
    pca2 = PCA(n_components=2, random_state=SEED).fit(X_all)
    Z = pca2.transform(X_all)
    evr = pca2.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    colors = {
        "cheat/non-hack": "#1f77b4",
        "cheat/hack":     "#d62728",
        "normal/correct": "#2ca02c",
        "normal/wrong":   "#ff7f0e",
    }
    markers = {
        "cheat/non-hack": "o",
        "cheat/hack":     "X",
        "normal/correct": "o",
        "normal/wrong":   "X",
    }
    for cat in ["cheat/non-hack", "cheat/hack", "normal/correct", "normal/wrong"]:
        idx = [i for i, c in enumerate(cats) if c == cat]
        axes[0].scatter(Z[idx, 0], Z[idx, 1],
                        c=colors[cat], marker=markers[cat],
                        alpha=0.6, s=32, edgecolor="white",
                        linewidth=0.4, label=f"{cat} (n={len(idx)})")
    axes[0].set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    axes[0].set_title("(A) Joint PCA on union (step-10)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="best")

    # -------- (B) axis-transfer test --------
    # PC1 direction on cheat-data ONLY (with hack vs non-hack labels)
    X_cheat = normalize(np.concatenate([c_nh, c_h], axis=0), norm="l2")
    y_cheat = np.concatenate([
        np.ones(len(c_nh), dtype=int),   # 1 = non-hack
        np.zeros(len(c_h), dtype=int),    # 0 = hack
    ])
    pca_cheat = PCA(n_components=1, random_state=SEED).fit(X_cheat)
    cheat_pc1 = pca_cheat.components_[0]  # the direction in 1024-D
    # orient so non-hack > hack on cheat data
    proj_cheat = X_cheat @ cheat_pc1
    if proj_cheat[y_cheat == 1].mean() < proj_cheat[y_cheat == 0].mean():
        cheat_pc1 = -cheat_pc1
        proj_cheat = -proj_cheat
    cheat_thr = 0.5 * (proj_cheat[y_cheat == 1].mean() + proj_cheat[y_cheat == 0].mean())
    cheat_acc = balanced_accuracy_score(
        y_cheat, (proj_cheat > cheat_thr).astype(int))
    print(f"\nCheat-data PC1 self-test: balanced acc = {cheat_acc:.3f}")

    # apply to normal-data fingerprints
    X_norm = normalize(np.concatenate([n_c, n_w], axis=0), norm="l2")
    y_norm = np.concatenate([
        np.ones(len(n_c), dtype=int),   # 1 = correct
        np.zeros(len(n_w), dtype=int),   # 0 = wrong
    ])
    proj_norm = X_norm @ cheat_pc1
    # try both orientations and threshold midpoints --- this is the
    # supervised version where we keep the cheat threshold AND a
    # locally-fit threshold
    # (i) keep cheat threshold and sign
    pred_a = (proj_norm > cheat_thr).astype(int)
    acc_a = max(balanced_accuracy_score(y_norm, pred_a),
                balanced_accuracy_score(y_norm, 1 - pred_a))
    # (ii) fit local threshold (midpoint of class means in normal data)
    mu1 = proj_norm[y_norm == 1].mean()
    mu0 = proj_norm[y_norm == 0].mean()
    sign = 1 if mu1 > mu0 else -1
    norm_thr = 0.5 * (mu1 + mu0)
    pred_b = ((sign * (proj_norm - norm_thr)) > 0).astype(int)
    acc_b = balanced_accuracy_score(y_norm, pred_b)

    print(f"Axis transfer: cheat-PC1 on normal data")
    print(f"  (i) cheat-fit threshold:     bal acc = {acc_a:.3f}")
    print(f"  (ii) normal-fit threshold:   bal acc = {acc_b:.3f}")
    print(f"  cos(cheat-PC1, normal-data class-mean diff) = "
          f"{float((cheat_pc1 @ (X_norm[y_norm==1].mean(0) - X_norm[y_norm==0].mean(0))) / (np.linalg.norm(cheat_pc1)*np.linalg.norm(X_norm[y_norm==1].mean(0) - X_norm[y_norm==0].mean(0)) + 1e-12)):+.3f}")

    # plot panel B
    axes[1].hist(
        proj_norm[y_norm == 1], bins=30, alpha=0.6,
        color="#2ca02c", label=f"normal/correct (n={len(n_c)})",
    )
    axes[1].hist(
        proj_norm[y_norm == 0], bins=30, alpha=0.6,
        color="#ff7f0e", label=f"normal/wrong (n={len(n_w)})",
    )
    axes[1].axvline(cheat_thr, color="black", linestyle="--", alpha=0.6,
                    label="cheat-fit threshold")
    axes[1].axvline(norm_thr, color="red", linestyle=":", alpha=0.6,
                    label="normal-fit threshold")
    axes[1].set_xlabel("normal-data fingerprint $\\cdot$ cheat-PC1")
    axes[1].set_ylabel("count")
    axes[1].set_title(
        f"(B) Cheat-PC1 on normal data\n"
        f"local-fit acc = {acc_b:.3f}, cheat-thr acc = {acc_a:.3f}"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_png = os.path.join(ROOT, "joint_geometry.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}")

    summary = {
        "counts": {
            "cheat/non-hack": int(len(c_nh)),
            "cheat/hack": int(len(c_h)),
            "normal/correct": int(len(n_c)),
            "normal/wrong": int(len(n_w)),
        },
        "pc1_var_share_joint": float(evr[0]),
        "pc2_var_share_joint": float(evr[1]),
        "cheat_pc1_self_test_bal_acc": cheat_acc,
        "axis_transfer": {
            "cheat_threshold_acc": acc_a,
            "normal_threshold_acc": acc_b,
        },
    }
    with open(os.path.join(ROOT, "joint_geometry.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(ROOT, 'joint_geometry.json')}")


if __name__ == "__main__":
    main()
