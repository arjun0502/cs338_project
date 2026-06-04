"""
Projection-dimension ablation plot — step 15 only.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(os.path.dirname(ROOT), "poster", "fig_dim_ablation.png")

# Step 15 values from RESULTS.md
dims  = [16,    32,    64,    128,   256,   512,   1024 ]
accs  = [0.961, 0.989, 0.998, 1.000, 1.000, 1.000, 1.000]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(dims, accs, "o-", color="#C41230", linewidth=2.5, markersize=8)

ax.set_xscale("log", base=2)
ax.set_xticks(dims)
ax.set_xticklabels([str(d) for d in dims], fontsize=10)
ax.set_xlabel("Random Projection Dimension", fontsize=12)
ax.set_ylabel("K-Means Accuracy", fontsize=12)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
