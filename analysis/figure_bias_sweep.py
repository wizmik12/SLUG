"""Bias-sweep figure in the style of the manuscript's slug_fitzpatrick.pdf.

Layout follows the existing figures: one panel per measure, the three training
configurations on the x-axis, and one box per subgroup within each configuration. The
background is a light-to-dark ramp across the configurations, which is how the submitted
figures encode the training composition (A: Light pale, C: Dark grey); it is decorative
shading, not data, so it stays far lighter than any mark.

Subgroup colour is a two-step SEQUENTIAL ramp, because the subgroups are ordered (lighter to
darker skin, HUSC to HCUV). The manuscript's own pale cream sits at 1.57 contrast against
white, below the 2.0 floor for a lightest step, so the light end is darkened slightly to
#dda15e (2.25) while keeping the warm palette; the two steps differ by dL = 0.31.

Usage:
  python analysis/figure_bias_sweep.py --dataset fitz --metric MSE \
      --methods mse encoderstd dropout slug --out tables/final/fig_bias_fitz_mse.pdf
"""
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
RUNS = list(range(10))

# Palette is chosen per dataset because the two subgroup variables are of different KINDS,
# and using one scheme for both would misrepresent one of them.
#
#   Fitzpatrick17k -- skin tone is ORDERED (FST 1-2 is lighter than FST 5-6), so a sequential
#   tan-to-brown ramp is the honest encoding and also reads as the quantity itself.
#
#   AI4SkIN -- the imaging centre is NOMINAL. HUSC is not "more" than HCUV, so a sequential
#   ramp would invent an ordering that does not exist. These are the first two colours of
#   Colorbrewer Dark2, a qualitative palette, and they are the exact values used in the
#   submitted figures (sampled from them), so the revision stays visually consistent.
#
# Bands shade each training configuration by its dominant group, which is why they differ
# per dataset too: warm at A, neutral at B (50/50), and the other group's hue at C.
PALETTES = {
    "fitz":    dict(pair=("#ffdbb6", "#855028"),
                    bands=["#fff7ef", "#fae9da", "#e6dcd4"]),
    "passion": dict(pair=("#e0a970", "#6b3d1e"),
                    bands=["#fff7ef", "#fae9da", "#e6dcd4"]),
    "ai4skin": dict(pair=("#d95f02", "#1b9e77"),
                    bands=["#f7dfcc", "#e4e5d8", "#cedcd8"]),
}
CONFIG_LABELS = {
    "fitz":    [("0", "A: Light"), ("0.5", "B: Mixed"), ("1", "C: Dark")],
    "passion": [("0", "A: Light"), ("0.5", "B: Mixed"), ("1", "C: Dark")],
    "ai4skin": [("0", "A: HUSC"), ("0.5", "B: Mixed"), ("1", "C: HCUV")],
}
DATASETS = {
    "fitz":    dict(groups=[("light_skin", "FST 1-2"), ("dark_skin", "FST 5-6")],
                    root="fitz"),
    "ai4skin": dict(groups=[("HUSC", "HUSC"), ("HCUV", "HCUV")], root="ai4skin"),
    # PASSION is the EXTERNAL test set, so the configurations are still the Fitzpatrick17k
    # training compositions; only the images being scored change. It spans FST 3-6 only, so
    # the subgroups are the lighter and darker halves OF PASSION, not FST 1-2 vs 5-6.
    "passion": dict(groups=[("34", "FST 3-4"), ("56", "FST 5-6")], root="passion"),
}
# measure -> (file stem, column, pretty title).  None stem = the quality metric itself.
MEASURES = {
    "mse":        (None, "MSE",             "MSE"),
    "ssim":       (None, "SSIM",            "SSIM"),
    "lpips":      (None, "LPIPS",           "LPIPS"),
    "encoderstd": ("uq_metrics_biasbase",   "encoderstd_unc", "Encoder std"),
    "dropout":    ("uq_metrics_biasbase",   "dropout_unc",    "MC-dropout"),
    "slug":       ("uq_metrics_slug_it100_sk10000", "Uncertainty", "SLUG (ours)"),
}


# PASSION results live in one CSV per (run, config) with the FST level as a column, rather
# than one file per subgroup. SLUG and the quality metrics are in passion_slug_*; the cheap
# baselines were scored separately into passion_baselines_*.
PASSION_COLS = {"mse": ("slug", "mse"), "ssim": ("slug", "ssim"),
                "lpips": ("slug", "lpips"), "slug": ("slug", "uncertainty"),
                "encoderstd": ("baselines", "encoderstd_unc"),
                "dropout": ("baselines", "dropout_unc")}


def passion_cell_mean(run, cfg, group, measure):
    if measure not in PASSION_COLS:
        return np.nan
    stem, col = PASSION_COLS[measure]
    f = Path("tables") / f"passion_{stem}_run{run}_cfg{cfg}.csv"
    if not f.exists():
        return np.nan
    d = pd.read_csv(f)
    levels = [3, 4] if group == "34" else [5, 6]
    sub = d[d["fitzpatrick"].isin(levels)]
    return sub[col].mean() if len(sub) else np.nan


def cell_mean(results_dir, ds, run, cfg, group, measure, metric_col):
    if ds == "passion":
        return passion_cell_mean(run, cfg, group,
                                 measure if measure != "mse" else metric_col.lower())
    stem, col, _ = MEASURES[measure]
    base = Path(results_dir) / ds / "Results" / str(run) / cfg
    if stem is None:
        f = base / f"metrics_{group}.csv"
        return pd.read_csv(f)[metric_col].mean() if f.exists() else np.nan
    f = base / f"{stem}_{group}.csv"
    if not f.exists():
        return np.nan
    return pd.read_csv(f)[col].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="outputs/uq_baselines_performance")
    ap.add_argument("--dataset", default="fitz", choices=list(DATASETS))
    ap.add_argument("--metric", default="MSE", choices=["MSE", "SSIM", "LPIPS"])
    ap.add_argument("--methods", nargs="+", default=["mse", "encoderstd", "dropout", "slug"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--trend", action="store_true",
                    help="join the per-configuration means with a line and mark them, as the "
                         "introductory figure does. Makes the direction of the effect legible "
                         "at a glance, which matters when the figure is the first thing a "
                         "reader sees.")
    ap.add_argument("--height", type=float, default=3.5)
    a = ap.parse_args()
    cfg_ds = DATASETS[a.dataset]
    groups = cfg_ds["groups"]
    pal = PALETTES[a.dataset]
    LIGHT, DARK = pal["pair"]
    BANDS = pal["bands"]
    CONFIGS = CONFIG_LABELS[a.dataset]

    # the quality panel follows --metric; uncertainty panels do not
    methods = [("mse" if m == "mse" else m) for m in a.methods]
    titles = {m: (a.metric if m in ("mse", "ssim", "lpips") else MEASURES[m][2]) for m in methods}

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n, a.height))
    axes = np.atleast_1d(axes)
    missing = []
    for ax, meas in zip(axes, methods):
        key = a.metric.lower() if meas in ("mse", "ssim", "lpips") else meas
        trend = {gi: ([], []) for gi in range(len(groups))}
        for ci, (cfg, _) in enumerate(CONFIGS):
            ax.axvspan(ci - 0.5, ci + 0.5, color=BANDS[ci], zorder=0, lw=0)
            for gi, (g, _) in enumerate(groups):
                vals = [cell_mean(a.results_dir, cfg_ds["root"], r, cfg, g, key, a.metric)
                        for r in RUNS]
                vals = [v for v in vals if not np.isnan(v)]
                if not vals:
                    missing.append((titles[meas], cfg, g)); continue
                pos = ci + (gi - 0.5) * 0.34
                bp = ax.boxplot([vals], positions=[pos], widths=0.27, patch_artist=True,
                                medianprops=dict(color=INK, lw=1.2),
                                flierprops=dict(marker="o", ms=2.6, mfc="none",
                                                mec=INK2, mew=0.7),
                                whiskerprops=dict(color=INK2, lw=0.9),
                                capprops=dict(color=INK2, lw=0.9),
                                boxprops=dict(facecolor=[LIGHT, DARK][gi],
                                              edgecolor=INK, lw=0.9))
                trend[gi][0].append(pos)
                trend[gi][1].append(float(np.mean(vals)))
        if a.trend:
            for gi in range(len(groups)):
                xs, ys = trend[gi]
                if len(xs) > 1:
                    ax.plot(xs, ys, color=[LIGHT, DARK][gi], lw=1.6, zorder=2.5,
                            solid_capstyle="round")
                    ax.plot(xs, ys, marker="o", ms=3.4, mfc=[LIGHT, DARK][gi],
                            mec=INK, mew=0.7, ls="none", zorder=2.6)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([t for _, t in CONFIGS], fontsize=10.5, color=INK2)
        ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
        ax.set_title(titles[meas], fontsize=13, color=INK, pad=8, fontname="cmb10")
        ax.tick_params(axis="y", colors=INK2, labelsize=10, length=3.5, width=0.7)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(False)          # bands are at zorder 0, grid must sit above them
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)

    handles = [Patch(facecolor=c, edgecolor=INK, lw=0.9, label=lab)
               for c, (_, lab) in zip([LIGHT, DARK], groups)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.02), columnspacing=2.2)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = a.out or f"tables/final/fig_bias_{a.dataset}_{a.metric.lower()}.pdf"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {out}")
    if missing:
        print(f"  NOTE: {len(missing)} panel/config/group cells had no data yet, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
