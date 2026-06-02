"""
GRIFT BigMath reproduction plot — Fig 4(b).

Reads per-step results from trace/data/rloo_cheat_all_rh_step_{s}/ and produces
a single PNG with two panels:

  (a) Detection metric vs RLOO step: GRIFT K-Means acc + TRACE F1 (hacking).
  (b) RH ratio vs step (sanity — reproduce paper Fig 2(b)).

Paper expected values are overlaid as dashed lines.
"""
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trace", "data")
OUT = os.path.join(ROOT, "fig4b_reproduction.png")

# Paper (Fig 4b + Table from analysis.tex)
PAPER = {
    5:  dict(rh=0.330, trace_f1=0.37, grift_km=1.00),
    10: dict(rh=0.350, trace_f1=0.33, grift_km=1.00),
    15: dict(rh=0.521, trace_f1=0.53, grift_km=1.00),
    20: dict(rh=0.916, trace_f1=0.93, grift_km=1.00),
    25: dict(rh=0.990, trace_f1=0.92, grift_km=0.75),
}


def parse_kmeans(grad_json):
    s = grad_json.get("clustering", {}).get("K-means", "")
    m = re.search(r"acc=([0-9.]+)", s)
    return float(m.group(1)) if m else None


def load_step(s):
    d = os.path.join(DATA, f"rloo_cheat_all_rh_step_{s}")
    if not os.path.isdir(d):
        return None
    out = {"step": s}

    grad_p = os.path.join(d, "gradient_svm_t_all_rh.json")
    if os.path.exists(grad_p):
        with open(grad_p) as f:
            g = json.load(f)
        out["grift_km"] = parse_kmeans(g)
        m = re.match(r"True set length: (\d+), False set length: (\d+)", g.get("lenth", ""))
        if m:
            t, fset = int(m.group(1)), int(m.group(2))
            out["n_true"] = t
            out["n_false"] = fset
            out["rh"] = fset / (t + fset) if (t + fset) else None

    eval_p = os.path.join(d, "trace_eval_all_rh.json")
    if os.path.exists(eval_p):
        with open(eval_p) as f:
            e = json.load(f)
        out["trace_f1"] = e.get("f1_hacking")
        out["trace_acc"] = e.get("acc")

    # If rh not derivable from gradient json (gradient phase not done), fall back to the rh_label files
    if "rh" not in out:
        t_p = os.path.join(d, "true_rh_all_rh.json")
        f_p = os.path.join(d, "false_rh_all_rh.json")
        if os.path.exists(t_p) and os.path.exists(f_p):
            with open(t_p) as f:
                t = len(json.load(f))
            with open(f_p) as f:
                fset = len(json.load(f))
            out["n_true"] = t
            out["n_false"] = fset
            out["rh"] = fset / (t + fset) if (t + fset) else None
    return out


def main():
    steps = sorted(PAPER.keys())
    rows = [load_step(s) for s in steps]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("no step directories found")
        sys.exit(1)

    print(f"{'step':>4} {'n_t':>4} {'n_f':>4} {'rh':>6} {'trace_f1':>9} {'grift_km':>9}")
    for r in rows:
        print(
            f"{r['step']:>4} {r.get('n_true', '-'):>4} {r.get('n_false', '-'):>4} "
            f"{r.get('rh', float('nan')):>6.3f} "
            f"{r.get('trace_f1') if r.get('trace_f1') is not None else float('nan'):>9.3f} "
            f"{r.get('grift_km') if r.get('grift_km') is not None else float('nan'):>9.3f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    rows_done = [r for r in rows if r.get("grift_km") is not None]
    xs = [r["step"] for r in rows_done]
    ax1.plot(
        xs, [r["grift_km"] for r in rows_done],
        "o-", color="#1f77b4", label="GRIFT K-Means", linewidth=2.2, markersize=10,
    )
    rows_trace = [r for r in rows if r.get("trace_f1") is not None]
    ax1.plot(
        [r["step"] for r in rows_trace],
        [r["trace_f1"] for r in rows_trace],
        "s-", color="#d62728", label="TRACE F1 (hacking)", linewidth=2.2, markersize=10,
    )
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
    ax1.set_xlabel("RLOO step")
    ax1.set_ylabel("Detection metric")
    ax1.set_title("(a) GRIFT vs TRACE — BigMath, Qwen2.5-3B-Instruct")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xticks(xs)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)

    rows_rh = [r for r in rows if r.get("rh") is not None]
    ax2.plot(
        [r["step"] for r in rows_rh],
        [r["rh"] for r in rows_rh],
        "o-", color="#2ca02c", linewidth=2.2, markersize=10,
    )
    ax2.set_xlabel("RLOO step")
    ax2.set_ylabel("RH ratio (hacks / total)")
    ax2.set_title("(b) Hacking ratio across training")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xticks([r["step"] for r in rows_rh])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
