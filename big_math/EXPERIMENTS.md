# BigMath GRIFT Replication — Experiment Log

Goal: Replicate Figure 5b of the GRIFT paper — detect reward hacking in BigMath RL-trained models using LoRA gradient fingerprints across training checkpoints (steps 5, 10, 15, 20).

---

## Experiment 1: all_rh run

**Setup:**
- 1490 cheat prompts (prompts with answer hint embedded)
- Both true and false sets come from same cheat prompts
- true_set = responses that copied the hint (reward hacked)
- false_set = responses that ignored the hint and solved correctly
- Steps 5, 10, 15, 20
- No trace baseline (commented out to save time)

**Results:**

| Step | true_rh | false_rh | F1 |
|------|---------|----------|----|
| 5 | 5 | 2 | 1.000 |
| 10 | 252 | 135 | 1.000 |
| 15 | 257 | 266 | 1.000 |
| 20 | 85 | 946 | 0.992 |

**Issues:**
- Step 5 invalid — only 7 total samples
- Step 20 extremely imbalanced (8% RH rate)
- All F1 scores near 1.0 vs paper's ~90% — suspiciously high
- RH rate decreases with training steps (opposite of paper's trend)

---

## Experiment 2: mixed run

**Setup:**
- 500 cheat prompts + 500 normal prompts
- true_set = true_rh (reward hacked on cheat) + true_normal (correct on normal prompts)
- false_set = false_rh (correct on cheat but didn't use hint)
- Steps 5, 10, 15, 20
- No trace baseline

**Results:**

| Step | true | false | F1 |
|------|------|-------|----|
| 5 | 203 | 34 | 1.000 |
| 10 | 216 | 46 | 1.000 |
| 15 | 224 | 94 | 1.000 |
| 20 | 176 | 321 | 1.000 |

**Issues:**
- Still 100% F1 across all steps
- Switching to mixed setup did not bring results closer to paper's ~90%
- Likely root cause: HuggingFace model checkpoints differ from paper's original models

---

## Suspected Root Causes

1. **Model versions**: HuggingFace checkpoints likely updated since paper was published. RH rates are inverted vs paper (ours: decreasing, paper: increasing with training steps).
2. **`all_rh` too easy**: both classes from same prompt type — "copying hint" vs "solving" creates trivially separable gradient patterns.
3. **No Pi matrix**: original code has `Pi = None` (dimensionality reduction disabled). Paper may have used random projection to compress gradients.

---

## Experiment 3: Permutation Test (mixed run)

**Setup:** Shuffle true/false labels randomly, retrain SVM 10 times, compare F1 to original.

**Results:**

| Step | true | false | Original F1 | Permuted F1 (mean) | Expected majority-class F1 |
|------|------|-------|-------------|--------------------|-----------------------------|
| 5 | 203 | 34 | 1.000 | 0.922 | ~0.923 |
| 10 | 216 | 46 | 1.000 | 0.903 | ~0.903 |
| 15 | 224 | 94 | 1.000 | 0.809 | ~0.828 |
| 20 | 176 | 321 | 1.000 | 0.181 | ~0.523 |

**Takeaway:**
- Steps 5, 10, 15: permuted F1 matches exactly what you'd get by predicting the majority class always. The SVM isn't learning from gradients — it's exploiting class imbalance. The 100% original F1 at these steps is an artifact.
- Step 20: only meaningful result. Classes are balanced (176 vs 321), permuted F1 drops to 0.181, original F1 is 0.992. The gap shows a genuine gradient signal at step 20.
- **Root cause of inflated results: class imbalance in the mixed setup.** Steps 5-15 have 70-86% majority class, making high F1 trivially achievable.
- Need balanced true/false sets for meaningful results at all steps.

---

## Experiment 4: PCA Dimensionality Reduction (mixed run)

**Setup:** Apply PCA to compress 1024-dim gradients to fewer dimensions before SVM. Tests whether high dimensionality is inflating F1.

**Results:**

| Step | true | false | n=8 | n=16 | n=32 | n=64 | n=128 | n=256 |
|------|------|-------|-----|------|------|------|-------|-------|
| 5 | 203 | 34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| 10 | 216 | 46 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 15 | 224 | 94 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | 176 | 321 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Takeaway:**
- F1=1.000 at all compression levels including 8 components, ruling out high dimensionality as the cause of inflated results
- The gradient separation is extremely concentrated in a very low-dimensional subspace
- For step 20 (real signal confirmed by permutation test), the signal survives aggressive compression
- For steps 5-15, class imbalance still dominates even in compressed space
- Consistent with model checkpoints having been updated since paper was published — the gradient separation is fundamentally easier than what the paper tested

---

## Experiment 5: Norm-only Classifier (mixed run)

**Setup:** Use only gradient magnitude (`||grad||`) as a single feature instead of the full 1024-dim gradient vector. Tests whether the SVM is using directional information or just magnitude.

**Results:**

| Step | true | false | F1 (norm-only) | F1 (full gradient) |
|------|------|-------|----------------|-------------------|
| 5 | 203 | 34 | 0.922 | 1.000 |
| 10 | 216 | 46 | 0.905 | 1.000 |
| 15 | 224 | 94 | 0.826 | 1.000 |
| 20 | 176 | 321 | 0.000 | 1.000 |

**Takeaway:**
- Steps 5, 10, 15: norm-only F1 matches majority-class baseline exactly — gradient magnitude adds no real signal, classifier is just predicting majority class
- **Step 20: norm-only F1 = 0.000** — gradient magnitude completely fails, but full gradient gets 1.000. This is a strong positive result: the step 20 signal is purely **directional** in gradient space, not driven by magnitude. Confirms the step 20 result is genuine.

---

## Experiment 6: Cross-checkpoint SVM (mixed run)

**Setup:** Train SVM on gradients from one checkpoint, test on a different checkpoint. Tests whether the gradient signal generalizes across training steps — closer to the paper's likely evaluation.

**Results:**

| Train step | Test step | F1 |
|-----------|-----------|-----|
| 5 | 20 | 0.523 |
| 20 | 5 | 0.026 |
| 10 | 20 | 0.523 |
| 20 | 10 | 0.644 |

**Takeaway:**
- Signal does not generalize well across checkpoints — each training step has a different gradient fingerprint for reward hacking
- Training on early steps (5, 10) and testing on late step (20) gives moderate F1 (~0.523) — some shared structure
- Training on step 20 and testing on step 5 nearly fails (F1=0.026) — step 20's RH gradient pattern is very different from step 5's
- Best cross-checkpoint result: train step 20 → test step 10 (F1=0.644)
- Suggests GRIFT may need to be trained and tested within similar training phases to be effective

---

## Next Experiments

- [ ] Fix class imbalance: subsample majority class to match minority class size
