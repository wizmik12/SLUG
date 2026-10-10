"""Graphical abstract.

Deliberately NOT a redraw of the pipeline figure. That figure answers "how does the method
work" with boxes, arrows and equations; a reader of the abstract wants "does it work, on what,
and what does it cost". So it is laid out as a diagram, one row per dataset:

    [input] -> [VAE] -> [reconstruction]   |   error vs SLUG   |   cost and bias sweep

  rows      Fitzpatrick17k, AI4SkIN, and PASSION as the external set
  column 2  the per-image error/uncertainty scatter, both subgroups, with Spearman rho
  column 3  memory vs detection AUROC for all six methods, and the training-composition sweep

Every number is read from the reported runs; nothing is hard-coded except the AUROC table,
which is transcribed from tables/final/auroc_mse.tex and checked against it.

On the example images: each row shows TWO pairs, a well-reconstructed one from the advantaged
subgroup and a badly-reconstructed one from the disadvantaged subgroup, with the MSE and the
SLUG score printed under each. That is the paper's claim shown on two images -- the error rises
and the score rises with it -- and it is the contrast a reader sees first.

These are low and high examples, not typical ones, and the panel says so; the subgroup medians
are printed on the same line so the reader has the typical gap too (0.0046 vs 0.0066 on
Fitzpatrick17k). The magnitude of the gap is carried by those numbers and by the scatter, never
by the choice of photograph.

A graphical abstract is the most widely reproduced image of a paper -- journal listings, search
results, slides -- so the examples are chosen to be anatomically neutral and non-identifiable:
no faces, nothing graphic. Pass --examples to override.

PASSION reconstructions are not saved by passion_slug_experiment.py; regenerate the handful
used here with analysis/passion_save_recons.py first.

Usage:
  python analysis/figure_graphical_abstract.py --out tables/final/graphical_abstract.pdf
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from figure_bias_sweep import PALETTES, cell_mean

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, INK2, GRID, BOX = "#0b0b0b", "#52514e", "#d8d7d2", "#f1f0ed"
# figure_ablation.py's validated pair; blue is unused by every dataset palette,
# so the method accent never competes with a subgroup colour
OURS = "#2a78d6"
# sampled from MedIA/pipeline.pdf so the strip reads as the same system as Fig. 2:
# its stage fills, its periwinkle block arrows and its cyan latent.
PIPE_MODEL, PIPE_METHOD, PIPE_OUT = "#e9ede8", "#e5efed", "#cde2f8"
PIPE_ARROW, PIPE_Z = "#99acff", "#5ce1e6"

p = argparse.ArgumentParser()
p.add_argument("--results", default="outputs/uq_baselines_performance")
p.add_argument("--passion_recons", default="tables/final/passion_recons")
p.add_argument("--out", default="tables/final/graphical_abstract.pdf")
cli = p.parse_args()

SLUG_STEM = "uq_metrics_slug_it100_sk10000"
CELL = "0_seed0/0.5"

DATASETS = [
    dict(key="fitz", title="Fitzpatrick17k", sub="clinical skin photographs",
         groups=[("light_skin", "FST 1-2", "#ffdbb6"), ("dark_skin", "FST 5-6", "#855028")],
         low=("light_skin", "5c8f96482399c24e535060315171ac3f"),
         high=("dark_skin", "44e0f40bb5f8525d9c19015926c89063")),
    dict(key="ai4skin", title="AI4SkIN", sub="histopathology patches",
         groups=[("HUSC", "HUSC", "#d95f02"), ("HCUV", "HCUV", "#1b9e77")],
         low=("HUSC", "patches/HUSC/AI4SKIN_HUSC_330_A/patch_09048.jpg"),
         high=("HCUV", "patches/HCUV/AI4SKIN_HCUV_156_A/patch_00677.jpg")),
    dict(key="passion", title="PASSION", sub="external set,\nunseen cohort",
         groups=[("34", "FST 3-4", "#e0a970"), ("56", "FST 5-6", "#6b3d1e")],
         low=("34", "AA00971287_7.jpg"),
         high=("56", "AA00970113_1.jpg")),
]

# mean AUROC over the four subgroups, transcribed from tables/final/auroc_mse.tex
METHODS = [
    ("Encoder std.",  0.004, np.mean([0.075, 0.063, 0.137, 0.455]), False),
    ("MC-dropout",    0.004, np.mean([0.716, 0.833, 0.728, 0.677]), False),
    ("Low-rank LLA",  1.0,   np.mean([0.911, 0.953, 0.846, 0.737]), False),
    ("SWAG",          3.0,   np.mean([0.970, 0.971, 0.854, 0.682]), False),
    ("Deep ensemble", 5.0,   np.mean([0.990, 0.985, 0.890, 0.851]), False),
    ("SLUG (ours)",   0.018, np.mean([0.979, 0.981, 0.886, 0.838]), True),
]
PLACE = {
    "Encoder std.":  (2.1, 0.000, "left", "center"),
    "MC-dropout":    (2.1, 0.000, "left", "center"),
    "SLUG (ours)":   (1.0, 0.060, "center", "bottom"),
    "Low-rank LLA":  (0.75, 0.000, "right", "center"),
    "SWAG":          (1.0, -0.055, "center", "top"),
    "Deep ensemble": (1.3, 0.055, "right", "bottom"),
}


def group_frame(ds, group):
    """Per-image MSE and SLUG uncertainty for one subgroup."""
    if ds == "passion":
        d = pd.read_csv("tables/passion_slug_run0_cfg0.5.csv")
        lv = [3, 4] if group == "34" else [5, 6]
        d = d[d["fitzpatrick"].isin(lv)]
        return pd.DataFrame({"name": d["file"], "MSE": d["mse"], "unc": d["uncertainty"]})
    base = f"{cli.results}/{ds}/Results/{CELL}"
    m = pd.read_csv(f"{base}/metrics_{group}.csv")
    u = pd.read_csv(f"{base}/{SLUG_STEM}_{group}.csv")
    d = m.merge(u, on="Image Name")
    return pd.DataFrame({"name": d["Image Name"], "MSE": d["MSE"], "unc": d["Uncertainty"]})


def example_pair(ds, group, name):
    """The saved [input | reconstruction] pair, already in training preprocessing."""
    if ds == "passion":
        f = f"{cli.passion_recons}/passion_recon_{name}.png"
    else:
        # the result files encode the image path with '/' replaced by '_'
        f = f"{cli.results}/{ds}/Results/{CELL}/imgs/{group}_recon_{name.replace('/', '_')}.png"
    assert os.path.exists(f), f"missing reconstruction: {f}\n(run analysis/passion_save_recons.py?)"
    a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
    w = a.shape[1] // 2
    return np.clip(a[:, :w], 0, 1), np.clip(a[:, w:], 0, 1)


ROWS = [D for D in DATASETS if D["key"] != "passion"]
EXT = next(D for D in DATASETS if D["key"] == "passion")
_e = pd.concat([group_frame("passion", g) for g, _, _ in EXT["groups"]])
EXT_RHO, EXT_N = spearmanr(_e["MSE"], _e["unc"]).correlation, len(_e)
print(f"  external  PASSION n={EXT_N}  rho={EXT_RHO:.2f}")

fig = plt.figure(figsize=(15.2, 6.35))
gs = fig.add_gridspec(2, 3, width_ratios=[2.08, 0.90, 1.00],
                      left=0.012, right=0.985, top=0.648, bottom=0.058,
                      wspace=0.26, hspace=0.38)

# ------------------------------------------------------------------ rows: the diagram
cgs = gs[:, 1].subgridspec(2, 1, hspace=0.62)   # scatter column, spaced on its own
print("row examples (low / high error, with the subgroup medians):")
for r, D in enumerate(ROWS):
    frames = {g: group_frame(D["key"], g) for g, _, _ in D["groups"]}
    meds = [(lab, frames[g]["MSE"].median()) for g, lab, _ in D["groups"]]

    shown = []
    for which in ("low", "high"):
        g, name = D[which]
        lab = dict((gg, ll) for gg, ll, _ in D["groups"])[g]
        col = dict((gg, cc) for gg, _, cc in D["groups"])[g]
        row = frames[g].set_index("name").loc[name]
        shown.append(dict(pair=example_pair(D["key"], g, name), label=lab,
                          colour="#e0a970" if col == "#ffdbb6" else col,
                          mse=float(row["MSE"]), slug=float(row["unc"]) / 1e6,
                          verdict="well reconstructed" if which == "low" else
                                  "poorly reconstructed"))
    print(f"  {D['title']:<15s} "
          + "  |  ".join(f"{e['label']} MSE {e['mse']:.4f} SLUG {e['slug']:.1f}"
                         for e in shown)
          + f"   medians {meds[0][1]:.4f}/{meds[1][1]:.4f}")

    # --- column 0, drawn in inches so the image tiles come out square.
    # The dataset name sits to the LEFT of the tiles: the row is about two inches tall and
    # the labels underneath (MSE, SLUG, the verdict) need that vertical space.
    ax = fig.add_subplot(gs[r, 0]); ax.set_axis_off()
    pos = ax.get_position()
    W, H = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    # derive the tile side from the width actually available, so nothing spills into the
    # scatter column when the grid ratios change
    gap, sep, lab_w = 0.30, 0.46, 1.46
    S = min(1.16, (W - lab_w - 2 * gap - sep - 0.04) / 4)
    cy = H - 0.10 - S / 2

    ax.text(0, cy + 0.30, D["title"], fontsize=11.5, color=INK, fontname="cmb10",
            ha="left", va="center")
    ax.text(0, cy + 0.07, D["sub"], fontsize=8.5, color=INK2, ha="left", va="center")
    # the statistics under the dataset name sit in their own soft panels: a heading and its
    # values read as one unit, and the two units stay apart without needing a rule between them
    stats = [("median MSE", f"{meds[0][1]:.4f} ({meds[0][0]})\n{meds[1][1]:.4f} ({meds[1][0]})", 2)]
    if D["key"] == "fitz":
        # this model is also the one applied to PASSION, so the external result is noted here
        stats.append(("external cohort", f"PASSION,  $\\rho = {EXT_RHO:.2f}$", 1))
    bx_w, pad, lh = lab_w - 0.10, 0.055, 0.126
    top = cy - 0.13
    for head, body, nlines in stats:
        h = pad * 2 + lh * (1 + nlines)
        # clip_on=False: the lower panel extends past the axes box, and patches (unlike
        # text) are clipped to it by default, which left it invisible
        ax.add_patch(FancyBboxPatch((0, top - h), bx_w, h, zorder=1, clip_on=False,
                                    boxstyle="round,pad=0,rounding_size=0.030",
                                    facecolor=BOX, edgecolor="none"))
        ax.text(pad + 0.02, top - pad - lh * 0.55, head, fontsize=7.6, color=INK,
                ha="left", va="center", fontname="cmb10", zorder=2)
        ax.text(pad + 0.02, top - pad - lh * (1 + nlines * 0.5), body, fontsize=7.3,
                color=INK2, ha="left", va="center", linespacing=1.5, zorder=2)
        top -= h + 0.09

    x = lab_w
    for e in shown:
        for im, xi in zip(e["pair"], (x, x + S + gap)):
            ax.imshow(im, extent=(xi, xi + S, cy - S / 2, cy + S / 2), aspect="auto", zorder=3)
            ax.add_patch(plt.Rectangle((xi, cy - S / 2), S, S, fill=False, zorder=4,
                                       edgecolor=e["colour"], linewidth=1.5))
        ax.add_patch(FancyArrow(x + S + 0.05, cy, gap - 0.16, 0, width=0.010,
                                head_width=0.075, head_length=0.09, color=INK2, zorder=3,
                                length_includes_head=True))
        ax.text(x + S + gap / 2, cy + 0.12, "VAE", fontsize=7.5, color=INK2, ha="center",
                va="bottom", zorder=4)
        xm = x + S + gap / 2
        if r == 0:
            ax.text(x + S / 2, cy + S / 2 + 0.07, "input", fontsize=8.5, color=INK2,
                    ha="center", va="bottom")
            ax.text(x + S + gap + S / 2, cy + S / 2 + 0.07, "reconstruction", fontsize=8.5,
                    color=INK2, ha="center", va="bottom")
        ax.text(xm, cy - S / 2 - 0.09, f"{e['label']}, {e['verdict']}", fontsize=8.5,
                color=INK2, ha="center", va="top")
        ax.text(xm, cy - S / 2 - 0.27, f"MSE {e['mse']:.4f}", fontsize=9.5, color=INK,
                ha="center", va="top")
        ax.text(xm, cy - S / 2 - 0.45, f"SLUG {e['slug']:.1f}", fontsize=9.5, color=OURS,
                ha="center", va="top", fontname="cmb10")
        x += 2 * S + gap + sep



    # --- column 1, the scatter
    ax = fig.add_subplot(cgs[r])
    for g, lab, col in D["groups"]:
        d = frames[g]
        dot = "#e0a970" if col == "#ffdbb6" else col      # the palette tone is too pale at 4px
        ax.scatter(d["MSE"], d["unc"] / 1e6, s=7, c=dot, alpha=0.65, linewidths=0,
                   label=lab, zorder=3)
    allg = pd.concat(frames.values())
    rho = spearmanr(allg["MSE"], allg["unc"]).correlation   # on ALL points, not the view
    rr = pearsonr(allg["MSE"], allg["unc"])[0]
    hi = allg["MSE"].quantile(0.995)
    if allg["MSE"].max() > 3 * hi:
        ax.set_xlim(-0.03 * hi, 1.05 * hi)
    ax.set_title(rf"error vs SLUG   $r={rr:.2f}$,  $\rho={rho:.2f}$", fontsize=11.5, color=INK,
                 fontname="cmb10", pad=6)
    ax.set_xlabel("reconstruction error (MSE)", fontsize=11, color=INK, labelpad=5)
    ax.set_ylabel(r"SLUG  ($\times 10^{6}$)", fontsize=11, color=INK, labelpad=5)
    ax.grid(color=GRID, lw=0.5, alpha=0.9); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right",
              markerscale=2.6, handletextpad=0.25, borderpad=0.1)

# ------------------------------------------------------------------ cost
rgs = gs[:, 2].subgridspec(2, 1, hspace=0.80)
ax = fig.add_subplot(rgs[0])
for name, mem, au, ours in METHODS:
    ax.scatter([mem], [au], s=210 if ours else 60, c=OURS if ours else INK2,
               marker="*" if ours else "o", zorder=4, linewidths=0)
    fx, dy, ha, va = PLACE[name]
    ax.annotate(name, (mem * fx, au + dy), fontsize=9.5, ha=ha, va=va,
                color=OURS if ours else INK2, fontname="cmb10" if ours else None)
ax.set_xscale("log"); ax.set_xlim(0.0024, 18); ax.set_ylim(0.0, 1.12)
ax.axhline(0.5, color=GRID, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(0.0029, 0.515, "chance", fontsize=8.5, color=INK2, va="bottom")
ax.annotate("ranks images backwards", (0.004, 0.183), xytext=(0.0115, 0.055), fontsize=8.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7), va="center")
ax.set_xlabel("memory kept at inference (multiples of $p$)", fontsize=11, color=INK, labelpad=5)
ax.set_ylabel("AUROC\nworst 5% of images", fontsize=10.5, color=INK, labelpad=4,
              linespacing=1.45)
ax.set_title("same detection, a fraction of the memory", fontsize=11, color=INK,
             fontname="cmb10", pad=15)
# the panel averages the four subgroups of the two internal datasets; say so, since the rows
# above show per-dataset numbers and a reader will otherwise ask which these are
ax.text(0.5, 1.025, "mean over the four subgroups of Fitzpatrick17k and AI4SkIN",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.grid(color=GRID, lw=0.5, alpha=0.9); ax.set_axisbelow(True)
ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)

# ------------------------------------------------------------------ bias sweep
# Same palette as the bias-sweep figures in the paper: the pair colours for the subgroups and
# the per-configuration bands, so the two read as the same experiment.
ax = fig.add_subplot(rgs[1])
PAL = PALETTES["fitz"]
CFGS = [("0", "A\n100% light"), ("0.5", "B\nmixed"), ("1", "C\n100% dark")]
for i, band in enumerate(PAL["bands"]):
    ax.axvspan(i - 0.5, i + 0.5, color=band, zorder=0, lw=0)
for k, (g, lab, _) in enumerate(ROWS[0]["groups"]):
    # keep the 10 runs separate so the spread between them can be drawn, rather than
    # averaging them away as the bar version did
    runs = np.array([[cell_mean(cli.results, "fitz", run, c, g, "slug", "MSE")
                      for c, _ in CFGS] for run in range(10)]) / 1e6
    mu, sd = np.nanmean(runs, 0), np.nanstd(runs, 0)
    ax.fill_between(range(3), mu - sd, mu + sd, color=PAL["pair"][k], alpha=0.30,
                    lw=0, zorder=2)
    ax.plot(range(3), mu, "-o", color=PAL["pair"][k], lw=2.0, ms=5.5, label=lab, zorder=3,
            markeredgecolor=INK2, markeredgewidth=0.6)
    print(f"  bias sweep {lab}: " + "  ".join(f"{m:.2f}+-{v:.2f}" for m, v in zip(mu, sd)))
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in CFGS], fontsize=8.5, color=INK2)
ax.set_xlim(-0.5, 2.5)
ax.margins(y=0.22)                               # headroom for the legend
ax.set_ylabel(r"SLUG  ($\times 10^{6}$)", fontsize=11, color=INK, labelpad=5)
ax.set_xlabel("training composition  (mean $\\pm$ 1 s.d., 10 runs)",
              fontsize=11, color=INK, labelpad=5)
ax.set_title("and follows who is under-represented (Fitzpatrick17k)",
             fontsize=11, color=INK,
             fontname="cmb10", pad=6)
ax.grid(color=GRID, lw=0.5, alpha=0.9, axis="y", zorder=1); ax.set_axisbelow(False)
ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", ncol=2,
          handletextpad=0.4, columnspacing=1.2, borderpad=0.1)

# ------------------------------------------------------------------ minimal separators
# Hairlines read from the grid, not guessed: one between consecutive dataset rows (left of
# the rule only, so they do not cut the cost panel) and one between the evidence and the cost.
box = [[gs[r, c].get_position(fig) for c in range(3)] for r in range(2)]
xrule = (box[0][1].x1 + box[0][2].x0) / 2
xcol0 = (box[0][0].x1 + box[0][1].x0) / 2
y = (box[0][0].y0 + box[1][0].y1) / 2
fig.add_artist(plt.Line2D([0.012, xcol0], [y, y], color=GRID, lw=0.7, zorder=0))
fig.add_artist(plt.Line2D([xrule, xrule], [box[1][2].y0 - 0.03, 0.700],
                          color=GRID, lw=0.7, zorder=0))

fig.text(0.012, 0.963,
         "Bayesian generative models can flag performance loss and bias in medical imaging",
         fontsize=16, color=INK, fontname="cmb10", va="center")
# ------------------------------------------------------------------ how it works
# Borrowed from Fig. 2 (pipeline.pdf) on purpose: the same three stages, the same stage
# fills, the same periwinkle block arrows, the same frozen VAE with its cyan latent. Only
# the idiom is shared -- Fig. 2 carries the algorithm boxes, the complexities and its own
# scatter, none of which is repeated here. Type stays Computer Modern, matching the
# manuscript and the data figures rather than the slide font pipeline.pdf was built in.
mx = fig.add_axes([0.012, 0.722, 0.973, 0.198]); mx.set_axis_off()
MW = 0.973 * fig.get_figwidth()
MH = 0.198 * fig.get_figheight()
mx.set_xlim(0, MW); mx.set_ylim(0, MH)
gy = MH * 0.52
BW, BH = MW * 0.235, MH * 0.74
cxs = [MW * 0.155, MW * 0.50, MW * 0.845]
for cx, fc in zip(cxs, (PIPE_MODEL, PIPE_METHOD, PIPE_OUT)):
    mx.add_patch(FancyBboxPatch((cx - BW / 2, gy - BH / 2), BW, BH, zorder=1,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fc, edgecolor="none"))

# 1. the frozen VAE, as Fig. 2 draws it: two trapezoids, a cyan latent, a snowflake
cx = cxs[0]
for sgn in (-1, 1):
    mx.add_patch(plt.Polygon(np.array([[sgn * 0.17, -0.20], [sgn * 0.70, -0.40],
                                       [sgn * 0.70, 0.40], [sgn * 0.17, 0.20]])
                             + np.array([cx, gy]), closed=True, facecolor="#ffffff",
                             edgecolor=INK2, lw=0.9, zorder=3))
mx.add_patch(matplotlib.patches.Ellipse((cx, gy), 0.30, 0.44, facecolor=PIPE_Z,
                                        edgecolor="none", zorder=4))
mx.text(cx, gy, "$z$", fontsize=9, color=INK, ha="center", va="center", zorder=5)
mx.text(cx - 0.44, gy, "encoder", fontsize=7.5, color=INK2, ha="center", va="center",
        rotation=90, zorder=5)
mx.text(cx + 0.44, gy, "decoder", fontsize=7.5, color=INK2, ha="center", va="center",
        rotation=90, zorder=5)
# Computer Modern has no snowflake glyph, so draw the "frozen" mark rather than risk a
# missing-character box
sx, sy, sr = cx + 0.82, gy + 0.33, 0.085
for ang in (90, 30, -30):
    dx, dy = sr * np.cos(np.radians(ang)), sr * np.sin(np.radians(ang))
    mx.plot([sx - dx, sx + dx], [sy - dy, sy + dy], color="#4b9fd6", lw=1.1,
            solid_capstyle="round", zorder=5)

# 2. the mechanism itself, written as Fig. 2 writes it
cx = cxs[1]
mx.text(cx, gy + 0.38, "curvature of the loss (GGN)", fontsize=8, color=INK2,
        ha="center", va="center", zorder=4)
mx.text(cx, gy + 0.17,
        r"$\mathbf{G}\approx\mathbf{J}_{\theta^*}(x)\,"
        r"(\mathbf{I}-\mathbf{U}\mathbf{U}^{\top})\,\mathbf{J}_{\theta^*}(x)^{\top}$",
        fontsize=10.5, color=INK, ha="center", va="center", zorder=4)
mx.text(cx, gy - 0.12, "uncertainty score", fontsize=8, color=INK2,
        ha="center", va="center", zorder=4)
mx.text(cx, gy - 0.32, r"$\mathrm{SLUG}(x)=\mathrm{Tr}(\mathbf{G})$",
        fontsize=10.5, color=INK, ha="center", va="center", zorder=4)

# 3. the score
cx = cxs[2]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("slug", ["#eef4fc", OURS])
mx.imshow(np.linspace(0, 1, 256)[None], cmap=cmap, aspect="auto", zorder=3,
          extent=(cx - 0.78, cx + 0.78, gy - 0.07, gy + 0.25))
mx.add_patch(plt.Rectangle((cx - 0.78, gy - 0.07), 1.56, 0.32, fill=False,
                           edgecolor=INK2, lw=0.8, zorder=4))
mx.add_patch(plt.Polygon([[cx + 0.33, gy + 0.28], [cx + 0.23, gy + 0.43],
                          [cx + 0.43, gy + 0.43]], closed=True, facecolor=INK, lw=0, zorder=5))
for xx, tt in ((cx - 0.78, "reliable"), (cx + 0.78, "flagged")):
    mx.text(xx, gy - 0.16, tt, fontsize=8, color=INK2, ha="center", va="top", zorder=5)

for i in range(2):
    xa = (cxs[i] + cxs[i + 1]) / 2
    mx.annotate("", xy=(xa + 0.30, gy), xytext=(xa - 0.30, gy), zorder=4,
                arrowprops=dict(arrowstyle="simple,head_width=0.80,head_length=0.55,"
                                          "tail_width=0.34",
                                facecolor=PIPE_ARROW, edgecolor="none"))

CAPS = [("frozen VAE", "applied post-hoc; the trained model is never updated"),
        ("SLUG", "the trace estimated stochastically, with a Hutchinson estimator"),
        ("uncertainty", "a continuous score, one value per image, no labels")]
for cx, (head, body), lead in zip(cxs, CAPS, (False, False, True)):
    mx.text(cx, MH - 0.02, head, fontsize=11.5, color=OURS if lead else INK, ha="center",
            va="top", fontname="cmb10")
    mx.text(cx, 0.04, body, fontsize=8.8, color=INK2, ha="center", va="bottom")
fig.add_artist(plt.Line2D([0.012, 0.985], [0.700, 0.700], color=GRID, lw=0.7, zorder=0))

os.makedirs(os.path.dirname(cli.out), exist_ok=True)
fig.savefig(cli.out, bbox_inches="tight")
fig.savefig(cli.out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
print(f"-> {cli.out}")
