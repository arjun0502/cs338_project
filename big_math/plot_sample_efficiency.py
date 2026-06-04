"""
Sample efficiency plot — hardcoded values estimated from sample_efficiency.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(os.path.dirname(ROOT), "poster", "fig_sample_efficiency.png")

Ns = [4, 8, 16, 32, 64, 128]

# (mean, std) estimated from original figure
data = {
    15: [(0.98, 0.07), (0.98, 0.04), (0.99, 0.04), (1.00, 0.01), (0.98, 0.08), (1.00, 0.01)],
}

fig, ax = plt.subplots(figsize=(7, 4.5))

for s, vals in data.items():
    means = [v[0] for v in vals]
    stds  = [v[1] for v in vals]
    ax.errorbar(Ns, means, yerr=stds, fmt="o-",
                color="#d62728", capsize=3, linewidth=2,
                markersize=7, label=f"Checkpoint {s}")

ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("Number of Samples", fontsize=12)
ax.set_ylabel("K-Means Accuracy", fontsize=12)
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
