"""
Reproduction bar chart: Paper vs Ours for GRIFT F1 (left) and TRACE F1 (right).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(os.path.dirname(ROOT), "poster", "fig_repro_f1.png")

STEPS = [5, 10, 15]
LABELS = ["Step 5", "Step 10", "Step 15"]

PAPER_GRIFT = [0.92, 0.92, 0.89]
OURS_GRIFT  = [0.91, 0.91, 0.89]

PAPER_TRACE = [0.37, 0.33, 0.53]
OURS_TRACE  = [0.50, 0.50, 0.40]

PAPER_COLOR = "#4472C4"
OURS_COLOR  = "#ED7D31"
BAR_W = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.3)

x = np.arange(len(STEPS))

for ax, paper_vals, ours_vals, title in [
    (ax1, PAPER_GRIFT, OURS_GRIFT, "GRIFT F1"),
    (ax2, PAPER_TRACE, OURS_TRACE, "TRACE F1"),
]:
    b1 = ax.bar(x - BAR_W/2, paper_vals, BAR_W, label="Paper",
                color=PAPER_COLOR, alpha=0.88)
    b2 = ax.bar(x + BAR_W/2, ours_vals,  BAR_W, label="Ours",
                color=OURS_COLOR,  alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")

plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
