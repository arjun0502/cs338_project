# BigMath GRIFT Replication — Runbook

Goal: replicate Figure 5b of the GRIFT paper (detection F1 vs. training step for BigMath) on a single-GPU GCP instance. All models and datasets are public on HuggingFace — no credentials needed.

---

## 1. Instance setup

Spin up one of:
- `a2-highgpu-1g` — A100 40GB, ~$1.50/hr spot
- `g2-standard-8` — L4 24GB, ~$0.70/hr spot (tighter on memory, should still work)

Attach a 50GB disk. After SSH-ing in, clone the repo and run setup:

```bash
git clone https://github.com/arjun0502/cs338_project
cd cs338_project
bash setup.sh
export PYTHONPATH=~/cs338_project:~/cs338_project/arlsat
```

`setup.sh` installs pip dependencies and writes the `PYTHONPATH` to `~/.bashrc`. Re-run it after every spot instance restart (pip installs are lost; `~/.bashrc` persists if disk is attached).

---

## 2. Smoke test

Run this first. It takes ~10–20 minutes and validates three things before committing GPU budget to the full run.

```bash
cd ~/cs338_project/big_math
python smoke_test.py
```

**What it does:**
1. Loads 50 samples from the BigMath cheat-prompt dataset (HuggingFace)
2. Runs vLLM inference with the step-10 checkpoint
3. Runs the **counterfactual test** to split into `true_set` / `false_set` (see below)
4. Loads the model in PyTorch, selects informative layers, extracts gradients on 10 samples from each set
5. Prints gradient norm means, timing, and GPU memory at each stage

**What to check:**

| Output | Expected | If wrong |
|--------|----------|----------|
| `true: N, false: M` | Both > 0 | If `false_set` empty: model hasn't learned to cheat at step 10 yet — edit `smoke_test.py` and change `global_step_10` to `global_step_20` |
| `True grad norm mean` vs `False grad norm mean` | Visibly different | Gradient signal is weak — detection may not work well |
| `[mem] after vllm inference` | < GPU capacity | If OOM: use A100 instance |
| `[mem] after loading model for gradients` | < GPU capacity | Same |
| Timing projection | Use to estimate full run cost | If > 8 GPU-hours, consider running fewer steps |

---

## 3. What is the counterfactual test?

The BigMath "cheat" dataset embeds the correct answer as a hint in the prompt (e.g. the prompt starts with `42.` before the math question). A reward-hacking model learns to copy that number rather than solve the problem.

The test:
1. Takes a response the model got **correct**
2. Replaces the hint with a **wrong number** (e.g. `42.` → `79.`)
3. Re-runs the model on the modified prompt

- Model answers `79` → it was following the hint → **reward hacking** → goes to `false_set`
- Model answers `42` → it ignored the hint and solved it → **genuine reasoning** → goes to `true_set`

The gradient detector is then trained to distinguish `true_set` gradients from `false_set` gradients.

---

## 4. Full pipeline

Once the smoke test passes:

```bash
cd ~/cs338_project/big_math
python -m trace.main_bigmath
```

This runs over checkpoints at steps 5, 10, 15, ..., 45 and for each one:

1. **Inference** — runs the model on 500 cheat-prompt samples and 500 normal-prompt samples, saves to `trace/data/rloo_cheat_step_{s}/`
2. **Labeling** — counterfactual test splits cheat-prompt responses into `true_set` / `false_set`
3. **Trace score** — computes TRACE baseline (truncates chain-of-thought at increasing percentages, measures answer consistency)
4. **Gradient extraction** — extracts LoRA gradients on selected layers, projects to 1024-dim, saves to `true_gradient/` / `false_gradient/`
5. **Detection** — KMeans clustering + SVM on gradient vectors, saves F1 and accuracy to `gradient_svm_t.json`

If inference is already done and you only want to re-run gradient analysis:

```bash
python -m trace.main_bigmath --gradient_only
```

**Runtime:** expect several hours depending on instance. Monitor with `watch -n5 nvidia-smi`.

---

## 5. Results

After the full pipeline, results are in:

```
big_math/trace/data/rloo_cheat_step_{s}/gradient_svm_t.json
```

Key fields in each JSON:
- `f1` — detection F1 (main metric; compare to Figure 5b of paper)
- `svm_acc` — SVM classification accuracy
- `g_vendi` — G-Vendi diversity score (how separated true/false gradient sets are)

**What to expect:** F1 should rise from ~0.5 at step 5 (model barely cheating), peak around steps 20–30 (hacking in full swing), then plateau or slightly decline at later steps. The gradient method (GRIFT) should outperform the TRACE baseline.

To collect and plot all results:

```python
import json, matplotlib.pyplot as plt

steps, f1s = [], []
for s in range(5, 50, 5):
    path = f"trace/data/rloo_cheat_step_{s}/gradient_svm_t.json"
    with open(path) as f:
        data = json.load(f)
    steps.append(s)
    f1s.append(data["f1"])

plt.plot(steps, f1s, marker="o", label="GRIFT (ours)")
plt.xlabel("Training step")
plt.ylabel("Detection F1")
plt.legend()
plt.savefig("results_fig5b.png")
```

A successful replication matches the trend shape from Figure 5b within ~5 percentage points.
