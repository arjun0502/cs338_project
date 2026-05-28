# Plan: BigMath Detection (GRIFT Replication)

## Context

Replicate BigMath reward hacking detection (Figure 5b of GRIFT paper) on a GCP instance, then extend with projection-dimension ablation, cross-checkpoint transfer, and layer selection ablation experiments. All models and datasets are public on HuggingFace — no training required. The code needs 3 small fixes to run on a single-GPU machine.

---

## Setup

### 1. GCP instance

Spin up `a2-highgpu-1g` (A100 40GB, ~$1.50/hr spot) or `g2-standard-8` (L4 24GB, ~$0.70/hr spot). Attach a 50GB disk.

### 2. Setup script (re-run after every spot instance restart)

Create `setup.sh` at the repo root. Since code fixes are committed to GitHub, the script only needs to handle environment setup — clone/pull the repo, install dependencies, and set PYTHONPATH.

```bash
#!/bin/bash
# setup.sh — re-run after every spot instance restart
set -e

REPO_URL="https://github.com/arjun0502/cs338_project"
REPO_DIR="$HOME/cs338_project"

# Clone repo if not present, otherwise pull latest (picks up committed code fixes)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Pulling latest..."
    git -C "$REPO_DIR" pull
else
    echo "Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

# Install dependencies
pip install torch transformers peft vllm trl datasets \
    scikit-learn joblib openai tiktoken pandas tqdm matplotlib

# Set PYTHONPATH for this session and future shells
export PYTHONPATH="$REPO_DIR:$REPO_DIR/arlsat"
grep -qxF "export PYTHONPATH=$REPO_DIR:$REPO_DIR/arlsat" ~/.bashrc \
    || echo "export PYTHONPATH=$REPO_DIR:$REPO_DIR/arlsat" >> ~/.bashrc

# Verify imports
cd "$REPO_DIR"
python -c "from big_math.trace.load_data import load_data; from icl.gradient.analysis import GradientAnalyzer; print('imports ok')"

echo ""
echo "Setup complete. Next: cd $REPO_DIR/big_math && python smoke_test.py"
```

Run it with:
```bash
chmod +x setup.sh && bash setup.sh
```

---

## Code fixes (apply before running anything, then commit so setup.sh picks them up)

### Fix 1: `tensor_parallel_size=4` — `big_math/trace/rh_model_setting.py`

The code was written for the authors' 4-GPU cluster. We have 1 GPU.

**Problem:** Two functions both hardcode 4 GPUs:

```python
# inference_on_ds — n_gpu parameter exists but default is 4
llm = LLM(model=model_name, tensor_parallel_size=n_gpu)  # n_gpu default is 4

# RH_labeling — completely hardcoded, no parameter at all
llm = LLM(model=model_name, tensor_parallel_size=4)
```

**Fix:** Change the `n_gpu` default in `inference_on_ds` from 4 to 1, and change the hardcoded `tensor_parallel_size=4` in `RH_labeling` to 1:

```python
# inference_on_ds
def inference_on_ds(ds, model_name, save_path, max_token=2048, n_gpu=1, k=100):
    ...
    llm = LLM(model=model_name, tensor_parallel_size=n_gpu)

# RH_labeling
llm = LLM(model=model_name, tensor_parallel_size=1)
```

### Fix 2: PYTHONPATH — two directories needed

The codebase has two separate import roots:
- `big_math/trace/main_bigmath.py` imports `from icl.gradient.analysis import GradientAnalyzer` and `from utils_ import result_processer` — these live inside `arlsat/`, not the repo root
- The `big_math.*` package itself lives at the repo root

So Python needs both roots on its path. Set before running:

```bash
export PYTHONPATH=/path/to/cs338_project:/path/to/cs338_project/arlsat
```

Without this, you'll get `ModuleNotFoundError: No module named 'icl'` and `ModuleNotFoundError: No module named 'utils_'`.

### Fix 3: Create stub `arlsat/env.py`

Some files import `from arlsat.env import KEY` for the OpenAI API key. BigMath uses automatic counterfactual labeling so we never call OpenAI — but the import still runs at module load time and will crash without the file.

```python
# arlsat/env.py  (create this file)
KEY = ""
```

---

## Smoke test

Run a 50-sample test on one checkpoint (step 10) to verify three things before spending GPU budget:

1. **The pipeline runs end-to-end** — no import errors, no path errors, inference and counterfactual test complete without crashing.
2. **The labeling produces a real true/false split** — the counterfactual test must label some samples as reward-hacked ("false set"). If `false_set` is empty, the cheat mechanism isn't working or the model at step 10 hasn't learned to cheat yet (try step 20).
3. **Compute timing** — measure seconds/sample for gradient extraction and project to full run cost.

**What to check in the output:**

| Check | Expected | If wrong |
|-------|----------|----------|
| `true_set` size | >0 (model gets some right via real reasoning) | Pipeline failed or dataset issue |
| `false_set` size | >0, ideally >5 out of 50 | Try a later checkpoint (step 20/30) where hacking is stronger |
| Cheat accuracy vs. normal accuracy | Cheat >> normal (model exploits hint) | Model may not have learned to cheat yet |
| Gradient norms (true vs. false) | Visibly different mean norms | Gradient extraction may be wrong |
| Timing | Calibrates full-run estimate | Decide 1 vs. 4 checkpoints |

```python
# smoke_test.py — run from cs338_project/big_math/
import time, numpy as np
from trace.load_data import load_data
from trace.rh_model_setting import inference_on_ds, RH_labeling
from icl.gradient.gradient_h import get_gradients_over_dataset, layer_selection
from icl.gradient.gradient import load_model_and_tokenizer

model_name = "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-og-rloo-implicit-cheat-direct-global_step_10"

ds = load_data(cheat=True)[:50]
gen = inference_on_ds(ds, model_name, save_path="/tmp/test_inference.json")
true_set, false_set = RH_labeling(gen, model_name, cheat=True)
print(f"true: {len(true_set)}, false: {len(false_set)}")  # both should be >0

# Quick gradient norm sanity check
model, tokenizer = load_model_and_tokenizer(model_name)
selected = layer_selection(true_set + false_set, model, tokenizer)
print(f"Selected layers: {selected}")  # should be 5 layer indices

start = time.time()
true_grads = get_gradients_over_dataset(true_set[:10], model, tokenizer, selected, "/tmp/true_g")
false_grads = get_gradients_over_dataset(false_set[:min(10,len(false_set))], model, tokenizer, selected, "/tmp/false_g")
elapsed = time.time() - start

# Sanity: norms should differ
true_norms = [np.linalg.norm(g) for g in true_grads]
false_norms = [np.linalg.norm(g) for g in false_grads]
print(f"True grad norm mean: {np.mean(true_norms):.3f}")
print(f"False grad norm mean: {np.mean(false_norms):.3f}")  # should be visibly different

print(f"10 samples: {elapsed:.1f}s → {elapsed/10:.1f}s/sample")
print(f"Projected full run (200 samples × 4 checkpoints): {200*4*elapsed/10/3600:.1f} GPU-hours")
```

Based on this, decide whether to run 1 checkpoint or all 4.

---

## Full pipeline

```bash
export PYTHONPATH=/path/to/cs338_project:/path/to/cs338_project/arlsat
cd cs338_project/big_math

# Full pipeline for steps 10, 15, 20, 25, 30, 35, 40
python -m trace.main_bigmath

# If gradients already saved, skip to analysis only
python -m trace.main_bigmath --gradient_only
```

Per checkpoint this:
1. Loads cheat + normal datasets from HuggingFace
2. Runs vllm inference (500 prompts)
3. Counterfactual test (re-run correct responses with wrong hint)
4. Saves `true_set.json` / `false_set.json`
5. Extracts gradients → saves `true_gradient/` / `false_gradient/`
6. Clusters + SVM → saves `gradient_svm_t.json`

---

## Comparing results to paper (Figure 5b)

### What the paper reports

Figure 5b shows **detection F1** at each checkpoint step (10–40) for BigMath, comparing three methods:

| Method | What it is |
|--------|-----------|
| **GRIFT (gradient)** | Our method — KMeans on LoRA gradient fingerprints |
| **TRACE** | Truncates chain-of-thought at 20/40/60/80% and forces answer; AUROC over truncation levels |
| **CoT-Monitor** | Qwen2.5-72B-Instruct as LLM judge on the full CoT |

The paper reports GRIFT outperforming both baselines, with F1 peaking around step 20–30 and declining slightly at later steps (as reward hacking stabilizes).

### Where results live in the code

After running `python -m trace.main_bigmath`, each checkpoint's results are written to:

```
big_math/trace/data/rloo_cheat_step_{s}/gradient_svm_t.json
```

This JSON contains the output of `GradientAnalyzer` — look for:
- `f1` (or `soft_f1`) — the key metric to compare against Figure 5b
- `svm_acc` — SVM classification accuracy (secondary metric)
- `g_vendi` — G-Vendi diversity score (measures separation between true/false sets)

### How to compare

1. Collect `f1` values from each `gradient_svm_t.json` (steps 10, 15, 20, 25, 30, 35, 40)
2. Plot F1 vs. step — should reproduce the trend in Figure 5b
3. TRACE runs automatically as part of the pipeline; CoT-Monitor requires a 72B model — skip it and use paper's reported numbers as reference line

### Quick comparison script (run after full pipeline)

```python
import json
import matplotlib.pyplot as plt

steps, f1s = [], []
for s in range(10, 45, 5):
    path = f"trace/data/rloo_cheat_step_{s}/gradient_svm_t.json"
    with open(path) as f:
        data = json.load(f)
    steps.append(s)
    f1s.append(data["f1"])  # adjust key name if needed

plt.plot(steps, f1s, marker="o", label="GRIFT (ours)")
paper_f1 = [...]  # read off from Figure 5b
plt.plot(steps, paper_f1, linestyle="--", label="Paper Figure 5b")
plt.xlabel("Training step"); plt.ylabel("F1"); plt.legend()
plt.savefig("comparison_fig5b.png")
```

### What counts as a successful replication

- F1 trend shape matches Figure 5b (peaks around step 20–30, gradient method beats TRACE)
- Absolute F1 values within ~5 percentage points (small deviations expected: different random seed for KMeans, slight dataset version differences)

---

## Extensions

### Extension A: Projection dimension ablation

**Why it's meaningful:** The paper fixes the random projection dimension at d=1024 and never justifies or ablates this choice. In practice, larger d means more memory and slower dot products — if the method works equally well at d=256, you could deploy it at 4× lower cost with no accuracy loss. Conversely, if F1 collapses below d=512, that tells you the gradient signal is genuinely high-dimensional and the 1024 choice is load-bearing. Either outcome is a concrete, actionable finding the paper leaves on the table.

**What to actually do (3 steps):**

1. **One code change in `arlsat/icl/gradient/gradient_h.py`** — in `get_gradients_over_dataset`, before the line `y = (Pi @ g).detach().to("cpu")`, also save `g` (the full un-projected gradient vector) to disk in a parallel file (e.g., `{save_path}_raw/sample_{i}.pt`). This is the only GPU-requiring step; it happens automatically during the main pipeline run.

2. **Write `big_math/ablation_dim.py`** — a CPU-only script that:
   - Loads the saved raw `g` vectors for each checkpoint step
   - Loops over d ∈ [128, 256, 512, 1024]
   - For each d: calls `make_dense_pi(d)` from `arlsat/icl/gradient/gradient.py` to generate a fresh random projection matrix, projects all gradients (`Pi @ g`), L2-normalizes, runs `GradientAnalyzer.cluster_analysis` and `svm_analysis`, records F1 and SVM accuracy
   - Plots F1 vs. d (one curve per checkpoint step)

3. **Verify:** At d=1024 the ablation F1 must match the main pipeline F1 exactly (same gradients, same projection dimension). If it doesn't, there's a bug in the re-projection logic.

### Extension B: Cross-checkpoint transfer

**Why it's meaningful:** The paper trains a fresh detector at every checkpoint — it never asks whether an early detector generalizes forward in time. This matters practically: if you're monitoring a live training run, you don't want to re-label data and re-fit a detector at every step. A detector fit at step 10 that still works at step 30 means you can alarm early and stop checking. If it *doesn't* transfer, that's also informative — it means the gradient signature of reward hacking shifts substantially as training progresses, which is a novel characterization of how hacking evolves over RL training.

**What to actually do (3 steps):**

1. **Run the main pipeline for all checkpoints (steps 10–40)** — during the step-10 run, pass `save_model="step10_km"` to `cluster_analysis` so the fitted KMeans model is saved to disk via joblib.

2. **Write `big_math/transfer_analysis.py`** — for each later step (15, 20, 25, 30, 35, 40):
   - Load the already-extracted gradients for that step (no GPU needed)
   - Call `GradientAnalyzer.cluster_analysis(baseline_km_centers="step10_km_km", ...)` — this loads the step-10 KMeans, predicts cluster labels on the new step's gradients without refitting, and computes F1
   - Also record the fresh per-step F1 from the main pipeline's `gradient_svm_t.json`

3. **Plot** — two lines: "fresh detector" (F1 per step from main pipeline) vs. "step-10 detector transferred" (F1 from transfer script). If the lines track closely, transfer works. If transferred F1 degrades while fresh F1 holds, gradient signatures are shifting.

**No code change needed** — `cluster_analysis` in `arlsat/icl/gradient/analysis.py` already has the `baseline_km_centers` argument. Just use it.

```python
# Step 10: fit and save
analyzer.cluster_analysis(true_grads, false_grads, save_model="step10_km", ...)

# Steps 15, 20, ...: load step-10 model, predict on new gradients
analyzer.cluster_analysis(true_grads_s15, false_grads_s15, baseline_km_centers="step10_km_km", ...)
```

**Verify:** Run the transferred detector on step 10's own gradients — it must give the same F1 as the fresh detector (it *is* the step-10 model). If not, there's a label-alignment bug in the `baseline_km_centers` path.

### Extension C: Layer selection ablation

**Why it's meaningful:** The paper's core design choice for *which* layers to extract gradients from is the phase-transition heuristic — select the K=5 layers with lowest adjacent-layer cosine similarity, on the theory that these are the most semantically active. This is intuitive but never validated. If random layer selection gives the same F1, the heuristic is doing nothing and the method works because LoRA gradients are informative everywhere. If phase-transition layers are genuinely better, it confirms the layer-selection step is load-bearing. Either way, this directly tests a central claim of the paper.

**What to actually do (3 steps):**

1. **No code change needed** — `get_gradients_over_dataset` in `arlsat/icl/gradient/gradient_h.py` already accepts `selected_layers` as an explicit argument. You can pass any layer indices you want.

2. **Write `big_math/ablation_layers.py`** — for one checkpoint (step 20, where hacking signal is strongest), extract gradients under 4 layer selection strategies, then run `GradientAnalyzer.cluster_analysis` and `svm_analysis` for each:

   | Strategy | How to get layer indices |
   |----------|--------------------------|
   | Phase-transition (paper) | `layer_selection(ds, model, tokenizer, K=5)` |
   | Random-K | `random.sample(range(num_layers), 5)` — repeat 3× and average F1 |
   | Last-K | last 5 transformer layers |
   | First-K | first 5 transformer layers |

   Each strategy requires a separate gradient extraction pass (GPU), but only for one checkpoint.

3. **Plot** — bar chart of F1 by strategy. If phase-transition >> random, the heuristic is validated. If they're similar, layer selection doesn't matter much.

**Verify:** Phase-transition strategy at step 20 must match the main pipeline F1 for that step exactly (same layers, same gradients). If it doesn't, there's a mismatch in how layer indices are being passed.

**Note on compute:** This is the only extension that requires additional GPU time — 4 gradient extraction passes for one checkpoint instead of 1. Budget roughly 4× the per-checkpoint gradient cost from the smoke test timing.

---

## Files to change

| File | What changes |
|------|-------------|
| `big_math/trace/rh_model_setting.py` | `tensor_parallel_size=4` → `1` in 2 places; `n_gpu` default 4 → 1 |
| `arlsat/env.py` | Create new stub file |
| `arlsat/icl/gradient/gradient_h.py` | Save raw gradient `g` alongside projected sketch (for Extension A) |

## Files unchanged

| File | Why unchanged |
|------|--------------|
| `big_math/trace/main_bigmath.py` | Entry point works as-is (non-`all_rh` mode has no private paths) |
| `big_math/trace/counterfact.py` | BigMath path has no hardcoded file paths |
| `big_math/trace/load_data.py` | Loads from HuggingFace automatically |
| `arlsat/icl/gradient/analysis.py` | `GradientAnalyzer` self-contained, no path dependencies |
