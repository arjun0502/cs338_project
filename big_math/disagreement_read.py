"""
Find samples where TRACE and GRIFT (PC1) disagree at step 5, and dump
the prompts/generations so we can read what's actually there.

Two interesting buckets:
  A. counterfactual=hack, but TRACE says non-hack (TRACE false negative;
     GRIFT correct) --- shows what GRIFT catches that TRACE misses.
  B. counterfactual=non-hack, but TRACE says hack (TRACE false positive)
     --- shows what TRACE flags spuriously.

Only step 5 because n is largest and PC1 is cleanest there.
"""
import json
import os
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
STEP = 5
SEED = 224
TRACE_THRESHOLD = 0.0749

D = os.path.join(ROOT, "trace", "data", f"rloo_cheat_all_rh_step_{STEP}")


def reconstruct_shuffle(n, seed=224):
    rng = random.Random()
    rng.seed(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def main():
    with open(os.path.join(D, "true_rh_all_rh.json")) as f:
        true_set = json.load(f)
    with open(os.path.join(D, "false_rh_all_rh.json")) as f:
        false_set = json.load(f)
    t_grad = torch.load(os.path.join(D, "true_gradient"),
                        map_location="cpu", weights_only=False)["sketches"].numpy()
    f_grad = torch.load(os.path.join(D, "false_gradient"),
                        map_location="cpu", weights_only=False)["sketches"].numpy()
    with open(os.path.join(D, "true_trace_all_rh.json")) as f:
        t_trace_shuf = json.load(f)["all_trace"]
    with open(os.path.join(D, "false_trace_all_rh.json")) as f:
        f_trace_shuf = json.load(f)["all_trace"]

    # un-shuffle trace scores back to original order
    idx_t = reconstruct_shuffle(len(true_set))
    idx_f = reconstruct_shuffle(len(false_set))
    t_trace = [None] * len(true_set)
    f_trace = [None] * len(false_set)
    for i, j in enumerate(idx_t):
        t_trace[j] = t_trace_shuf[i]
    for i, j in enumerate(idx_f):
        f_trace[j] = f_trace_shuf[i]

    X = np.concatenate([t_grad, f_grad], axis=0).astype(np.float64)
    y = np.concatenate([
        np.ones(len(true_set), dtype=int),
        np.zeros(len(false_set), dtype=int),
    ])
    X = normalize(X, norm="l2")
    pca = PCA(n_components=1, random_state=SEED).fit(X)
    pc1 = pca.transform(X).ravel()
    # orient so non-hack > hack
    if pc1[y == 1].mean() < pc1[y == 0].mean():
        pc1 = -pc1
    midpoint = 0.5 * (pc1[y == 1].mean() + pc1[y == 0].mean())

    samples = []
    for i, item in enumerate(true_set):
        samples.append(dict(
            label="non-hack", trace=t_trace[i],
            pc1=float(pc1[i]),
            grift="hack" if pc1[i] < midpoint else "non-hack",
            trace_pred="non-hack" if t_trace[i] < TRACE_THRESHOLD else "hack",
            prompt=item.get("prompt", ""),
            gen=item.get("gen", ""),
        ))
    for j, item in enumerate(false_set):
        i = len(true_set) + j
        samples.append(dict(
            label="hack", trace=f_trace[j],
            pc1=float(pc1[i]),
            grift="hack" if pc1[i] < midpoint else "non-hack",
            trace_pred="non-hack" if f_trace[j] < TRACE_THRESHOLD else "hack",
            prompt=item.get("prompt", ""),
            gen=item.get("gen", ""),
        ))

    # bucket A: label=hack, trace_pred=non-hack, grift=hack (TRACE FN, GRIFT TP)
    A = [s for s in samples if s["label"] == "hack" and s["trace_pred"] == "non-hack"
         and s["grift"] == "hack"]
    # bucket B: label=non-hack, trace_pred=hack, grift=non-hack (TRACE FP, GRIFT TN)
    B = [s for s in samples if s["label"] == "non-hack" and s["trace_pred"] == "hack"
         and s["grift"] == "non-hack"]

    print(f"\n=== step {STEP} | true={len(true_set)} false={len(false_set)} ===")
    print(f"TRACE FN (caught only by GRIFT): {len(A)}")
    print(f"TRACE FP (rejected by GRIFT):    {len(B)}")
    print(f"TRACE-PC1 agreement on hacks:    "
          f"{sum(1 for s in samples if s['label']=='hack' and s['trace_pred']=='hack')}/{len(false_set)}")
    print(f"TRACE-PC1 agreement on non-hacks:"
          f"{sum(1 for s in samples if s['label']=='non-hack' and s['trace_pred']=='non-hack')}/{len(true_set)}")

    print("\n--- 3 samples from bucket A (TRACE missed, GRIFT caught) ---")
    A.sort(key=lambda s: s["trace"])  # most confidently-non-hack by TRACE
    for k, s in enumerate(A[:3]):
        print(f"\n### A.{k+1}  trace={s['trace']:.3f}  pc1={s['pc1']:+.3f}")
        print(f"PROMPT: {s['prompt'][:300]}")
        print(f"GEN  : {s['gen'][:500]}...")

    print("\n--- 3 samples from bucket B (TRACE flagged, GRIFT rejected) ---")
    B.sort(key=lambda s: -s["trace"])  # most confidently-hack by TRACE
    for k, s in enumerate(B[:3]):
        print(f"\n### B.{k+1}  trace={s['trace']:.3f}  pc1={s['pc1']:+.3f}")
        print(f"PROMPT: {s['prompt'][:300]}")
        print(f"GEN  : {s['gen'][:500]}...")


if __name__ == "__main__":
    main()
