"""Ablation figures: correlation and tail detection against the stored memory ks/p.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated all-pairs
GROUPS = ["FST12", "FST56", "HUSC", "HCUV"]
TITLES = {"FST12": "FST 1-2", "FST56": "FST 5-6", "HUSC": "HUSC", "HCUV": "HCUV"}
REF_MEM = 90 * 10_000 / 50_636_355


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tables/final/ablation_rank_sketch.csv")
    ap.add_argument("--stat", default="pearson",
                    choices=["pearson", "calib", "aurc", "auroc"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    sd = a.stat + "_sd"
    YLAB = {"pearson": "Pearson $r$ with per-image MSE",
            "calib":   "calibration gap  (lower is better)",
            "aurc":    "normalised AURC  (lower is better)",
            "auroc":   "AUROC, worst 5\\% by MSE"}
    ylab = YLAB[a.stat]

    # Shared y-limits across the four panels. Per-panel autoscaling zooms onto a 0.003 band
    # and makes noise look like signal -- the classic misleading-axis error. One honest range
    # for the whole figure keeps the panels comparable and the differences correctly sized.
    lo = float((d[a.stat] - d[sd].fillna(0)).min())
    hi = float((d[a.stat] + d[sd].fillna(0)).max())
    pad = 0.42 * (hi - lo)   # two label rows above and below the data
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    flat = axes.ravel()
    for k, (ax, g) in enumerate(zip(flat, GROUPS)):
        ax.grid(True, color=GRID, lw=0.6, alpha=0.9); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=16, length=5, width=1.0)

        sub = d[d.group == g]
        rank = sub[sub.sketch == 10_000].sort_values("mem")        # vary rank
        sk = sub[(sub.iters == 10) & (sub.sketch > 10_000)].sort_values("mem")  # vary sketch
        # the two sweeps share their origin (rank 9, sketch 1e4); draw it once, on the rank
        # series, and start the sketch series from it so the line still connects
        sk = pd.concat([sub[(sub.iters == 10) & (sub.sketch == 10_000)], sk])

        ax.axvline(REF_MEM, color=INK2, lw=0.8, ls=(0, (2, 2)), alpha=0.7, zorder=1)
        means = {"rank": rank[a.stat].mean(), "sketch": sk[a.stat].mean()}
        for df, col, mk, lab, other_mean in [
                (rank, BLUE, "o", "vary rank $k$   (sketch $s=10^4$)", means["sketch"]),
                (sk, ORANGE, "s", "vary sketch $s$   (rank $k=9$)", means["rank"])]:
            ax.errorbar(df.mem, df[a.stat], yerr=df[sd], color=col, lw=2.6, marker=mk,
                        ms=8.5, mfc="white", mew=2.2, capsize=4, elinewidth=1.4,
                        label=lab if k == 0 else None, zorder=3)
            # Memory has to be the x-axis -- the claim is about equal budgets differently
            # allocated, which is only visible on a shared axis. But memory alone hides WHICH
            # parameter produced each point, so every marker carries its own value.
            # Labels go in two fixed ROWS pinned to the axes, not at offsets from the
            # markers: the rank curve descends by roughly the stagger amount between
            # consecutive points, so marker-relative offsets cancel out and adjacent labels
            # end up flush against each other. Axes-anchored rows make the separation
            # independent of the curve. Each series sits on whichever side is empty, decided
            # from the data (the rank sweep runs above the sketch sweep for Pearson and below
            # it for calibration). x stays at the data point so the association is clear.
            above = df[a.stat].mean() >= other_mean
            rows = (0.955, 0.865) if above else (0.045, 0.135)
            va = "top" if above else "bottom"
            for i, (_, r) in enumerate(df.iterrows()):
                txt = (f"$k{{=}}{int(r['rank'])}$" if col is BLUE
                       else f"$s{{=}}10^{{{int(round(np.log10(r['sketch'])))}}}$")
                ax.annotate(txt, xy=(r["mem"], rows[i % 2]),
                            xycoords=("data", "axes fraction"),
                            ha="center", va=va, fontsize=16, color=col, zorder=4)

        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(d.mem.min() / 3.0, d.mem.max() * 2.5)
        ax.set_title(TITLES[g], fontsize=20, color=INK, pad=11, fontname="cmb10")   # centred, bold
        if k >= 2:
            ax.set_xlabel("posterior memory  $ks/p$", fontsize=18, color=INK2)
        if k % 2 == 0:
            ax.set_ylabel(ylab, fontsize=18, color=INK2)
        # The dashed vertical marks equal memory / different allocation. It is explained in
        # the caption rather than annotated in-axes: an in-panel note collides with whichever
        # curve happens to run high, and the collision moves with the statistic being plotted.

    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=17,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.012), columnspacing=2.6,
               handlelength=3.0, handletextpad=0.8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = a.out or f"tables/final/fig_ablation_{a.stat}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
