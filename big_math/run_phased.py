"""
Phased BigMath reproduction driver.

The original pipeline instantiates vLLM `LLM(...)` multiple times in a single
process (inference, RH_labeling, trace_true, trace_false). The second
instantiation fails with a 600s distributed-init timeout because the first
batch of workers has not been torn down. Each phase here runs in its own
subprocess so workers are reaped between LLM() calls.

Phases per checkpoint:
  1. inference   -> cheat_inference_500.json
  2. rh_label    -> true_rh_all_rh.json, false_rh_all_rh.json (counterfactual)
  3. trace_true  -> true_trace_all_rh.json
  4. trace_false -> false_trace_all_rh.json
  5. trace_eval  -> trace_eval_all_rh.json   (no vLLM, sklearn-only)
  6. gradient    -> true_gradient, false_gradient, gradient_svm_t_all_rh.json
                    (no vLLM, transformers+peft)

Env overrides:
  BIGMATH_STEPS  - comma-separated, e.g. "10" or "5,10,15,20,25,30"
                   default = "5,10,15,20,25,30"
  SMOKE_LIMIT    - truncate cheat dataset to N samples for smoke tests
                   default = 0 (full 500)
  SMOKE_PHASES   - comma-separated subset of phase names to run
                   default = all six

Run from big_math/:
    cd big_math
    source ../.venv/bin/activate
    python run_phased.py
"""

import os
import sys
import subprocess


REPO_BIGMATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_BIGMATH)
ARLSAT = os.path.join(REPO_ROOT, "arlsat")

STEPS = [int(x) for x in os.environ.get("BIGMATH_STEPS", "5,10,15,20,25,30").split(",")]
SMOKE_LIMIT = int(os.environ.get("SMOKE_LIMIT", "0"))
ALL_PHASES = ["inference", "rh_label", "trace_true", "trace_false", "trace_eval", "gradient"]
PHASES = os.environ.get("SMOKE_PHASES", ",".join(ALL_PHASES)).split(",")

MODEL_TEMPLATE = (
    "xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-"
    "og-rloo-implicit-cheat-direct-global_step_{s}"
)


def run_subprocess(label, code):
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    extra = f"{REPO_BIGMATH}:{ARLSAT}:{REPO_ROOT}"
    env["PYTHONPATH"] = extra + ":" + env.get("PYTHONPATH", "")
    print(f"\n========== {label} ==========", flush=True)
    r = subprocess.run([sys.executable, "-u", "-c", code], env=env, cwd=REPO_BIGMATH)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {r.returncode})")


def phase_inference(step, save_dir):
    out = os.path.join(save_dir, "cheat_inference_500.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    model = MODEL_TEMPLATE.format(s=step)
    slice_line = f"ds = ds[:{SMOKE_LIMIT}]" if SMOKE_LIMIT > 0 else ""
    code = f"""
import os, json
os.makedirs({save_dir!r}, exist_ok=True)
from trace.rh_model_setting import inference_on_ds, load_data
ds = load_data(cheat=True)
{slice_line}
print(f"inference samples: {{len(ds)}}", flush=True)
inference_on_ds(ds, {model!r}, save_path={out!r})
"""
    run_subprocess(f"inference step={step}", code)


def phase_rh_label(step, save_dir):
    out = os.path.join(save_dir, "false_rh_all_rh.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    inf_path = os.path.join(save_dir, "cheat_inference_500.json")
    model = MODEL_TEMPLATE.format(s=step)
    code = f"""
import os, json
from trace.rh_model_setting import RH_labeling
with open({inf_path!r}) as f:
    inference = json.load(f)
print(f"loaded {{len(inference)}} inference samples", flush=True)
true_rh, false_rh = RH_labeling(inference=inference, model_name={model!r})
print(f"RH true={{len(true_rh)}}, false={{len(false_rh)}}", flush=True)
with open(os.path.join({save_dir!r}, "true_rh_all_rh.json"), "w") as f:
    json.dump(true_rh, f, indent=4)
with open(os.path.join({save_dir!r}, "false_rh_all_rh.json"), "w") as f:
    json.dump(false_rh, f, indent=4)
"""
    run_subprocess(f"rh_label step={step}", code)


def phase_trace_set(step, save_dir, which):
    """which: 'true' or 'false'"""
    out = os.path.join(save_dir, f"{which}_trace_all_rh.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    in_path = os.path.join(save_dir, f"{which}_rh_all_rh.json")
    model = MODEL_TEMPLATE.format(s=step)
    code = f"""
import os, json, random
random.seed(224)
from trace.trace import get_trace_on_ds
with open({in_path!r}) as f:
    ds = json.load(f)
random.shuffle(ds)
print(f"trace_{which} set len: {{len(ds)}}", flush=True)
if len(ds) == 0:
    # write empty result so downstream eval doesn't crash
    with open({out!r}, "w") as f:
        json.dump({{"model_name": {model!r}, "trace_score": 0.0, "set_name": "{which}_all_rh", "all_trace": []}}, f, indent=4)
else:
    get_trace_on_ds(ds, output_path={out!r}, model_name={model!r},
                    set_name="{which}_all_rh", max_token=3072, n_gpu=4, K=3)
"""
    run_subprocess(f"trace_{which} step={step}", code)


def phase_trace_eval(step, save_dir):
    code = f"""
from trace.rh_model_setting import get_trace_f1
get_trace_f1(save_dir={save_dir!r}, all_rh=True)
"""
    run_subprocess(f"trace_eval step={step}", code)


def phase_gradient(step, save_dir):
    out = os.path.join(save_dir, "gradient_svm_t_all_rh.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    model = MODEL_TEMPLATE.format(s=step)
    code = f"""
from icl.gradient.analysis import GradientAnalyzer
from trace.gradient import big_math_gradient
big_math_gradient(GradientAnalyzer(), model_name={model!r},
                  save_dir={save_dir!r}, all_rh=True, get_gradient=True)
"""
    run_subprocess(f"gradient step={step}", code)


PHASE_FNS = {
    "inference": phase_inference,
    "rh_label": phase_rh_label,
    "trace_true": lambda s, d: phase_trace_set(s, d, "true"),
    "trace_false": lambda s, d: phase_trace_set(s, d, "false"),
    "trace_eval": phase_trace_eval,
    "gradient": phase_gradient,
}


def main():
    print(f"BigMath phased driver. STEPS={STEPS} SMOKE_LIMIT={SMOKE_LIMIT} PHASES={PHASES}",
          flush=True)
    for s in STEPS:
        save_dir = f"trace/data/rloo_cheat_all_rh_step_{s}"
        os.makedirs(save_dir, exist_ok=True)
        for ph in PHASES:
            if ph not in PHASE_FNS:
                raise SystemExit(f"unknown phase: {ph}")
            PHASE_FNS[ph](s, save_dir)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
