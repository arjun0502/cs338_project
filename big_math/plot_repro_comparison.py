"""
Reproduction comparison figure — paper vs. ours.

Grouped bar chart: Soft-F1 (paper) vs Soft-F1 (ours) at steps 5, 10, 15.
K-Means = 1.00 for both at all steps; annotated as text.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "fig_repro_comparison.png")

STEPS = [5, 10, 15]
SOFT_F1_PAPER = [0.92, 0.92, 0.89]
SOFT_F1_OURS  = [0.91, 0.91, 0.89]

BAR_W = 0.32
COLOR_PAPER = "#1f77b4"
COLOR_OURS  = "#ff7f0e"

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(STEPS))
bars_paper = ax.bar(x - BAR_W / 2, SOFT_F1_PAPER, BAR_W,
                    label="Paper", color=COLOR_PAPER, alpha=0.85)
bars_ours  = ax.bar(x + BAR_W / 2, SOFT_F1_OURS,  BAR_W,
                    label="Ours",  color=COLOR_OURS,  alpha=0.85)

# value labels on bars
for bar in list(bars_paper) + list(bars_ours):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
            f"{h:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"Step {s}" for s in STEPS])
ax.set_ylabel("Soft-F1")
ax.set_ylim(0.80, 1.00)
ax.set_title("Reproduction: Paper vs. Ours\n(K-Means = 1.00 at all steps for both)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
