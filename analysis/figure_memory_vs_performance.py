"""Memory kept at inference against detection performance, for every method.

Usage:
  python analysis/figure_memory_vs_performance.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
OURS = "#2a78d6"

#   name, memory (xp), Spearman rho with the per-image MSE, AUROC for the worst 5%,
#   both over FST12/FST56/HUSC/HCUV.  Transcribed from tables/final/{corr,auroc}_mse.tex.
METHODS = [
    ("Encoder std.",  0.004, [-0.776, -0.771, -0.823, -0.645], [0.075, 0.063, 0.137, 0.455], False),
    ("MC-dropout",    0.004, [0.183, 0.342, -0.061, 0.366],    [0.716, 0.833, 0.728, 0.677], False),
    ("Low-rank LLA",  1.0,   [0.356, 0.568, 0.580, 0.550],     [0.911, 0.953, 0.846, 0.737], False),
    ("SWAG",          3.0,   [0.818, 0.844, 0.835, 0.622],     [0.970, 0.971, 0.854, 0.682], False),
    ("Deep ensemble", 5.0,   [0.896, 0.935, 0.905, 0.793],     [0.990, 0.985, 0.890, 0.851], False),
    ("SLUG (ours)",   0.018, [0.889, 0.872, 0.876, 0.766],     [0.979, 0.981, 0.886, 0.838], True),
]
# Labels are anchored to the ENDS of each spread bar, not to the mean: a label offset from
# the mean lands on top of the bar it belongs to.
#   above / below -> clear of the bar's top or bottom;  left / right -> beside the marker
PLACE = {
    "Encoder std.":  ("right", 2.0),
    "MC-dropout":    ("right", 2.0),
    "SLUG (ours)":   ("above", 1.0),
    "Low-rank LLA":  ("left",  0.70),
    "SWAG":          ("below", 1.0),
    "Deep ensemble": ("above", 1.0),
}

PLACE_RHO = {                   # the rho panel needs its own anchors: encoder std is negative
    "Encoder std.":  ("right", 2.0),
    "MC-dropout":    ("right", 2.0),
    "SLUG (ours)":   ("above", 1.0),
    "Low-rank LLA":  ("left",  0.70),
    "SWAG":          ("below", 1.0),
    "Deep ensemble": ("above", 1.0),
}
WARN = "#c0806a"


def draw(ax, col, place, ylim, hline, hlabel):
    for name, mem, rho, auroc, ours in METHODS:
        v = np.array(rho if col == "rho" else auroc)
        ax.plot([mem, mem], [v.min(), v.max()], color=OURS if ours else INK2, lw=1.0,
                alpha=0.45, zorder=3, solid_capstyle="round")
        ax.scatter([mem], [v.mean()], s=200 if ours else 55, c=OURS if ours else INK2,
                   marker="*" if ours else "o", zorder=4, linewidths=0)
        mode, fx = place[name]
        if mode == "above":
            xy, ha, va = (mem * fx, v.max() + 0.045), "center", "bottom"
        elif mode == "below":
            xy, ha, va = (mem * fx, v.min() - 0.045), "center", "top"
        elif mode == "left":
            xy, ha, va = (mem * fx, v.mean()), "right", "center"
        else:
            xy, ha, va = (mem * fx, v.mean()), "left", "center"
        ax.annotate(name, xy, fontsize=8.5, ha=ha, va=va,
                    color=OURS if ours else INK2, fontname="cmb10" if ours else None)
    ax.set_xscale("log")
    ax.set_xlim(0.0022, 20)
    ax.set_ylim(*ylim)
    ax.axhline(hline, color=GRID, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.0026, hline + 0.02, hlabel, fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("memory kept at inference (multiples of $p$)", fontsize=9.5, color=INK,
                  labelpad=4)
    ax.grid(color=GRID, lw=0.5, alpha=0.9); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6))
draw(axs[0], "rho", PLACE_RHO, (-1.0, 1.12), 0.0, "no relation")
axs[0].set_ylabel(r"Spearman $\rho$ with the per-image MSE", fontsize=9.5, color=INK, labelpad=4)
axs[0].annotate("ranks images backwards", (0.0048, -0.70), xytext=(0.014, -0.95), fontsize=8,
                color=WARN, va="center",
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=0.9, alpha=0.8,
                                shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=-0.25"))
draw(axs[1], "auroc", PLACE, (0.0, 1.14), 0.5, "chance")
axs[1].set_ylabel("AUROC, worst 5% of images", fontsize=9.5, color=INK, labelpad=4)
axs[1].annotate("ranks images backwards", (0.0048, 0.175), xytext=(0.013, 0.055), fontsize=8,
                color=WARN, va="center",
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=0.9, alpha=0.8,
                                shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=-0.25"))
fig.tight_layout(w_pad=2.0)

os.makedirs("tables/final", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"tables/final/fig_memory_vs_performance.{ext}", dpi=400, bbox_inches="tight")
print("-> tables/final/fig_memory_vs_performance.{pdf,png}")
for name, mem, rho, auroc, _ in METHODS:
    r_, a_ = np.array(rho), np.array(auroc)
    print(f"  {name:<15s} {mem:>7.3f}p   rho {r_.mean():+.3f}   AUROC {a_.mean():.3f}")
