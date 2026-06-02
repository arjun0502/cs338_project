"""
Generalizability proxy --- run GRIFT on NORMAL prompts (no cheat number)
using the cheat-trained model. With no shortcut available, we label by
correctness (correct vs wrong) instead of counterfactual hack/non-hack.

Three possible outcomes for fingerprint geometry on normal prompts:
  (a) Diffuse single blob --- GRIFT correctly says "no hack" when none.
  (b) Two clean clusters split by correctness --- GRIFT encodes general
      correctness, not hacking specifically.
  (c) Two clean clusters not aligned with correctness --- GRIFT picks up
      a spurious internal axis.

Phases per checkpoint:
  1. inference (normal prompts)
  2. correctness label (no counterfactual; just result_processer)
  3. gradient

Env:
  PROXY_STEP    -- which RLOO checkpoint (default 10)
  PROXY_LIMIT   -- truncate normal-prompt dataset (default 300)
"""
import os
import sys
import subprocess

REPO_BIGMATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_BIGMATH)
ARLSAT = os.path.join(REPO_ROOT, "arlsat")

STEP = int(os.environ.get("PROXY_STEP", "10"))
LIMIT = int(os.environ.get("PROXY_LIMIT", "300"))

MODEL = (
    f"xinpeng/big-math-hard-tiny-qwen2.5-3b-instruct-"
    f"og-rloo-implicit-cheat-direct-global_step_{STEP}"
)
SAVE_DIR = os.path.join(REPO_BIGMATH, "trace", "data",
                        f"normal_proxy_step_{STEP}")


def run_subprocess(label, code):
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PYTHONPATH"] = (f"{REPO_BIGMATH}:{ARLSAT}:{REPO_ROOT}:"
                         + env.get("PYTHONPATH", ""))
    print(f"\n========== {label} ==========", flush=True)
    r = subprocess.run([sys.executable, "-u", "-c", code],
                       env=env, cwd=REPO_BIGMATH)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} ({r.returncode})")


def phase_inference():
    out = os.path.join(SAVE_DIR, "normal_inference.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    code = f"""
import os
os.makedirs({SAVE_DIR!r}, exist_ok=True)
from trace.rh_model_setting import inference_on_ds, load_data
ds = load_data(cheat=False)[:{LIMIT}]
print(f"inference samples: {{len(ds)}}", flush=True)
inference_on_ds(ds, {MODEL!r}, save_path={out!r})
"""
    run_subprocess(f"normal-proxy inference (step={STEP})", code)


def phase_label():
    out = os.path.join(SAVE_DIR, "false_rh_all_rh.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    inf_path = os.path.join(SAVE_DIR, "normal_inference.json")
    code = f"""
import json, os
from utils_ import result_processer
with open({inf_path!r}) as f:
    inference = json.load(f)
true_set, false_set = [], []
for inf in inference:
    ans, res = result_processer(response=inf['gen'], label=inf['label'])
    if res:
        true_set.append(inf)
    else:
        false_set.append(inf)
print(f"correct={{len(true_set)}}  wrong={{len(false_set)}}", flush=True)
os.makedirs({SAVE_DIR!r}, exist_ok=True)
with open(os.path.join({SAVE_DIR!r}, 'true_rh_all_rh.json'), 'w') as f:
    json.dump(true_set, f, indent=4)
with open({out!r}, 'w') as f:
    json.dump(false_set, f, indent=4)
"""
    run_subprocess(f"normal-proxy label (step={STEP})", code)


def phase_gradient():
    out = os.path.join(SAVE_DIR, "gradient_svm_t_all_rh.json")
    if os.path.exists(out):
        print(f"  [skip] {out} exists")
        return
    code = f"""
from icl.gradient.analysis import GradientAnalyzer
from trace.gradient import big_math_gradient
big_math_gradient(GradientAnalyzer(), model_name={MODEL!r},
                  save_dir={SAVE_DIR!r}, all_rh=True, get_gradient=True)
"""
    run_subprocess(f"normal-proxy gradient (step={STEP})", code)


def main():
    print(f"Normal-prompt proxy. STEP={STEP} LIMIT={LIMIT}", flush=True)
    print(f"Save dir: {SAVE_DIR}", flush=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    phase_inference()
    phase_label()
    phase_gradient()
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
