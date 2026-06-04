"""
Generates a clean table figure for the reproduction results.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(os.path.dirname(ROOT), "poster", "fig_repro_table.png")

columns = ["Step", "RH Ratio", "Clustering Acc.", "Soft-F1"]
col_widths = [0.12, 0.22, 0.34, 0.22]
rows = [
    ["5",  "0.40", "1.00", "0.91"],
    ["10", "0.43", "1.00", "0.91"],
    ["15", "0.48", "1.00", "0.89"],
]

fig, ax = plt.subplots(figsize=(5.5, 1.8))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=columns,
    loc="center",
    cellLoc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.0)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#2E5090")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
