"""Per-image scatter of reconstruction error against uncertainty, by subgroup and training
configuration.

Replaces the submitted correlation_*.pdf figures. Three things change.

1. The submitted figures label the rows "dark skin tones (FST:1-2)" and "light skin tones
   (FST:5-6)". Those are the wrong way round in both: FST 1-2 IS the lighter group and FST 5-6
   the darker one. The row that carries the larger errors is the darker group, which the marker
   colour already encoded correctly, so only the text was wrong. Fixed here.

2. Both Pearson and Spearman are reported. The relation is monotone but not linear, so a single
   linear coefficient understates how well the score ranks images, which is what every
   downstream use of it actually needs.

3. Typography and palette follow figure_bias_sweep.py, so the two figures read as one family:
   Computer Modern, a sequential tan-to-brown ramp for the ORDERED skin-tone groups and the
   Colorbrewer Dark2 pair for the NOMINAL imaging centres.

Usage:
  python analysis/figure_correlation.py --dataset fitz --method slug \
      --out tables/final/fig_corr_fitz_slug.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

PALETTES = {
    "fitz":    dict(pair=("#e0a970", "#855028"), bands=["#fff7ef", "#fae9da", "#e6dcd4"]),
    "ai4skin": dict(pair=("#d95f02", "#1b9e77"), bands=["#f7dfcc", "#e4e5d8", "#cedcd8"]),
}
CONFIG_LABELS = {
    "fitz":    [("0", "A: Light"), ("0.5", "B: Mixed"), ("1", "C: Dark")],
    "ai4skin": [("0", "A: HUSC"), ("0.5", "B: Mixed"), ("1", "C: HCUV")],
}
DATASETS = {
    "fitz": dict(root="fitz", groups=[("light_skin", "FST 1-2 (lighter skin)"),
                                      ("dark_skin",  "FST 5-6 (darker skin)")]),
    "ai4skin": dict(root="ai4skin", groups=[("HUSC", "HUSC"), ("HCUV", "HCUV")]),
}
# method -> (file stem, error column, uncertainty column, pretty name)
METHODS = {
    "slug":       ("uq_metrics_slug_it100_sk10000", "Error", "Uncertainty", "SLUG (ours)"),
    "dropout":    ("uq_metrics_biasbase", "dropout_err", "dropout_unc", "MC-dropout"),
    "encoderstd": ("uq_metrics_biasbase", "encoderstd_err", "encoderstd_unc", "Encoder std."),
}


def load(results_dir, ds, run, cfg, group, method):
    stem, ecol, ucol, _ = METHODS[method]
    f = Path(results_dir) / ds / "Results" / str(run) / cfg / f"{stem}_{group}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d[ecol].to_numpy(), d[ucol].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="outputs/uq_baselines_performance")
    ap.add_argument("--dataset", default="fitz", choices=list(DATASETS))
    ap.add_argument("--method", default="slug", choices=list(METHODS))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    pal = PALETTES[a.dataset]
    configs = CONFIG_LABELS[a.dataset]
    pretty = METHODS[a.method][3]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.1))
    for gi, (g, glab) in enumerate(ds["groups"]):
        for ci, (cfg, clab) in enumerate(configs):
            ax = axes[gi, ci]
            ax.set_facecolor(pal["bands"][ci])
            got = load(a.results_dir, ds["root"], a.run, cfg, g, a.method)
            if got is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color=INK2)
                continue
            e, u = got
            ax.scatter(e, u, s=7, c=pal["pair"][gi], alpha=0.65, linewidths=0, zorder=3)
            # faint least-squares line, to read the trend without implying a model
            b, c0 = np.polyfit(e, u, 1)
            xs = np.linspace(e.min(), e.max(), 2)
            ax.plot(xs, b * xs + c0, color=INK, lw=0.9, alpha=0.55, zorder=4)
            r = pearsonr(e, u)[0]
            rho = spearmanr(e, u).correlation
            ax.set_title(rf"$r={r:.2f}$,  $\rho={rho:.2f}$", fontsize=11.5, color=INK, pad=5)
            ax.grid(color=GRID, lw=0.5, alpha=0.9, zorder=1)
            ax.set_axisbelow(False)
            ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            if gi == 1:
                ax.set_xlabel(clab, fontsize=11, color=INK, labelpad=6)
            if ci == 0:
                ax.set_ylabel(glab, fontsize=11, color=INK, labelpad=6)

    fig.suptitle(f"{pretty}: per-image reconstruction error (MSE) against uncertainty",
                 fontsize=13.5, color=INK, fontname="cmb10", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = a.out or f"tables/final/fig_corr_{a.dataset}_{a.method}.pdf"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
