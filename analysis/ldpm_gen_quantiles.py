"""Generations laid out across the uncertainty range, with the most uncertain shown separately.

Usage:
  python analysis/ldpm_gen_quantiles.py --dataset ai4skin --save_dir outputs/ldpm_ai4skin
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.training import checkpoints
from diffusers import FlaxAutoencoderKL

from latent_ddpm import LatentUNet, DDPM

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False, "pdf.fonttype": 42,
})
INK, INK2 = "#0b0b0b", "#52514e"

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--t", type=int, default=500)
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--n_gen", type=int, default=1000)
p.add_argument("--ddim_steps", type=int, default=200)
p.add_argument("--n_cols", type=int, default=10, help="quantiles sampled across the range")
p.add_argument("--n_rows", type=int, default=3, help="examples per quantile")
p.add_argument("--tail_row", type=int, default=1,
               help="append a row of the most uncertain samples; the quantile rows show\n                     the progression, this shows how much detail the top end carries")
p.add_argument("--exclude", type=int, nargs="*", default=[],
               help="generation indices to skip. Fitzpatrick17k contains clinical photographs "
                    "of faces, so the model generates faces; a synthetic one can still be read "
                    "as a patient, and the nearest unused rank is substituted instead")
p.add_argument("--out", default="outputs/ldpm_figs")
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
SCORES = f"tables/ldpm_gen_uq_{cli.dataset}_p{PERC}_seed{cli.seed}_t{cli.t}.csv"
assert os.path.exists(SCORES), f"run ldpm_generation_uq.py first: {SCORES} missing"

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C, HW = Z["ztest1"].shape[-1], Z["ztest1"].shape[1]
CLIP = float(np.abs(Z["ztrain"]).max())

unet = LatentUNet(out_ch=C, base_ch=cli.base_ch)
prm = jax.tree_util.tree_map(
    jnp.asarray, checkpoints.restore_checkpoint(ckpt_dir=CELL, target=None)["params"])
ABAR = np.asarray(DDPM(T=1000).abar)
vae, vae_p = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)


@jax.jit
def eps_theta(z, t):
    return unet.apply({"params": prm}, z, t, train=False)


@jax.jit
def decode(z):
    return jnp.transpose(vae.apply({"params": vae_p}, z / SCALE, method=vae.decode).sample,
                         (0, 2, 3, 1))


print(f"generating {cli.n_gen} samples, {cli.ddim_steps} DDIM steps "
      f"(z0 clipped to +-{CLIP:.2f})", flush=True)
t0 = time.time()
zg = np.asarray(jax.random.normal(jax.random.PRNGKey(20260929), (cli.n_gen, HW, HW, C)))
ts = np.linspace(999, 0, cli.ddim_steps + 1).astype(int)
for i in range(cli.ddim_steps):
    tc, tn = int(ts[i]), int(ts[i + 1])
    ac, an = ABAR[tc], (ABAR[tn] if tn > 0 else 1.0)
    e = np.concatenate([np.asarray(eps_theta(jnp.asarray(zg[j:j + 100]),
                                             jnp.full((len(zg[j:j + 100]),), tc, jnp.int32)))
                        for j in range(0, len(zg), 100)])
    z0p = np.clip((zg - np.sqrt(1 - ac) * e) / np.sqrt(ac), -CLIP, CLIP)
    zg = np.sqrt(an) * z0p + np.sqrt(1 - an) * e
print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

d = pd.read_csv(SCORES).sort_values("slug").reset_index(drop=True)
n = cli.n_cols * cli.n_rows
# evenly spaced in RANK, so the columns are quantiles of the distribution rather than of the
# value range, which a few extreme scores would otherwise dominate
pick = np.linspace(0, len(d) - 1, n).round().astype(int)
if cli.exclude:
    banned = set(cli.exclude)
    taken = set()
    for i, r in enumerate(pick):
        while d.idx.iloc[r] in banned or r in taken:        # walk to the nearest free rank
            r = r + 1 if r + 1 < len(d) else r - 1
        pick[i] = r
        taken.add(r)
    print(f"excluded {sorted(banned)}; substituted the nearest free ranks")
sel = d.iloc[pick]
# the quantile rows show the progression but give the top end only a couple of tiles, which
# undersells how much detail is there; this row is the most uncertain samples outright
tail = d.iloc[-cli.n_cols:] if cli.tail_row else d.iloc[:0]
order = np.concatenate([sel.idx.values, tail.idx.values])

pix = np.concatenate([np.asarray(decode(jnp.asarray(order[i:i + 16])))
                      for i in range(0, len(order), 16)]) if False else np.concatenate(
    [np.asarray(decode(jnp.asarray(zg[order[i:i + 16]]))) for i in range(0, len(order), 16)])
pix = np.clip(pix * 0.5 + 0.5, 0, 1)
quant_pix, tail_pix = pix[:n], pix[n:]

BAND = {4: "quartile", 5: "quintile", 10: "decile", 20: "ventile",
        100: "percentile"}.get(cli.n_cols, f"1/{cli.n_cols} band of the ranking")
R = cli.n_rows + (2 if cli.tail_row else 0)
fig = plt.figure(figsize=(cli.n_cols * 0.63,
                          cli.n_rows * 0.67 + (1.05 if cli.tail_row else 0)))
gs = fig.add_gridspec(R, cli.n_cols,
                      height_ratios=[1] * cli.n_rows + ([0.55, 1] if cli.tail_row else []),
                      hspace=0.06, wspace=0.08)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(cli.n_cols)]
                 for i in range(cli.n_rows)])
for a, im, (_, r) in zip(axes.T.ravel(), quant_pix, sel.iterrows()):
    a.imshow(im, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#d8d7d2")
tail_axes = []
if cli.tail_row:
    for j, (im, (_, r)) in enumerate(zip(tail_pix, tail.iterrows())):
        a = fig.add_subplot(gs[cli.n_rows + 1, j])
        a.imshow(im, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#855028")
        tail_axes.append(a)
fig.subplots_adjust(top=0.78, bottom=0.05)
fig.suptitle(f"{'AI4SkIN' if cli.dataset=='ai4skin' else 'Fitzpatrick17k'}: "
             f"unconditional generations ordered by SLUG",
             fontsize=10, color=INK, y=0.995)

# an arrow across the quantile block: the columns are ordered, and that is the whole point
top = axes[0][0].get_position().y1
fig.add_artist(plt.Line2D([axes[0][0].get_position().x0, axes[0][-1].get_position().x1],
                          [top + 0.038] * 2, color=INK2, lw=0.8))
fig.add_artist(matplotlib.patches.FancyArrow(
    axes[0][-1].get_position().x1 - 0.012, top + 0.038, 0.012, 0, width=0.0005,
    head_width=0.012, head_length=0.012, color=INK2, transform=fig.transFigure,
    length_includes_head=True))
cxm = (axes[0][0].get_position().x0 + axes[0][-1].get_position().x1) / 2
fig.text(cxm, top + 0.048, "increasing uncertainty", fontsize=8, color=INK2,
         ha="center", va="bottom")
fig.text(cxm, top + 0.115,
         f"each column: {cli.n_rows} samples from the same {BAND}",
         fontsize=7.5, color=INK2, ha="center", va="bottom")
def frame(axlist, color, pad=0.010):
    p = [a.get_position() for a in axlist]
    x0, x1 = min(q.x0 for q in p) - pad, max(q.x1 for q in p) + pad
    y0, y1 = min(q.y0 for q in p) - pad, max(q.y1 for q in p) + pad
    fig.add_artist(matplotlib.patches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, zorder=0,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor="none", edgecolor=color, lw=0.9))
    return x0, y0, x1, y1

frame(list(axes.ravel()), "#b9b8b3")
if cli.tail_row:
    tx0, _, tx1, ty1 = frame(tail_axes, "#855028")
    fig.text((tx0 + tx1) / 2, ty1 + 0.012,
             f"the {cli.n_cols} most uncertain (top {100 * cli.n_cols / len(d):.3g}%)",
             fontsize=8, color="#855028", ha="center", va="bottom")

os.makedirs(cli.out, exist_ok=True)
dest = f"{cli.out}/gen_uq_quantiles_{cli.dataset}_seed{cli.seed}.png"
fig.savefig(dest, dpi=400, bbox_inches="tight")
print(f"-> {dest}")
print(sel[["idx", "slug"]].to_string(index=False))
