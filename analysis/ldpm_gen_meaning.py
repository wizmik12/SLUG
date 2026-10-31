"""What the decoder-only score tracks on generated samples.

Usage:
  python analysis/ldpm_gen_meaning.py --dataset fitz --save_dir outputs/ldpm_fitz
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr, rankdata
from flax.training import checkpoints
from diffusers import FlaxAutoencoderKL

from latent_ddpm import LatentUNet, DDPM

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--n_gen", type=int, default=1000)
p.add_argument("--ddim_steps", type=int, default=200)
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
SCORES = f"tables/ldpm_gen_uq_{cli.dataset}_p{PERC}_seed{cli.seed}_t500.csv"

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C, HW = Z["ztest1"].shape[-1], Z["ztest1"].shape[1]
CLIPV = float(np.abs(Z["ztrain"]).max())

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


@jax.jit
def encode(x_nhwc):
    xc = jnp.transpose(x_nhwc, (0, 3, 1, 2))
    return vae.apply({"params": vae_p}, xc, method=vae.encode).latent_dist.mean * SCALE


def batched(fn, x, bs=16):
    return np.concatenate([np.asarray(fn(jnp.asarray(x[i:i + bs])))
                           for i in range(0, len(x), bs)])


print(f"regenerating {cli.n_gen} samples", flush=True)
zg = np.asarray(jax.random.normal(jax.random.PRNGKey(20260929), (cli.n_gen, HW, HW, C)))
ts = np.linspace(999, 0, cli.ddim_steps + 1).astype(int)
for i in range(cli.ddim_steps):
    tc, tn = int(ts[i]), int(ts[i + 1])
    ac = ABAR[tc]
    an = ABAR[tn] if tn > 0 else 1.0
    e = np.concatenate([np.asarray(eps_theta(jnp.asarray(zg[j:j + 100]),
                                             jnp.full((len(zg[j:j + 100]),), tc, jnp.int32)))
                        for j in range(0, len(zg), 100)])
    z0p = np.clip((zg - np.sqrt(1 - ac) * e) / np.sqrt(ac), -CLIPV, CLIPV)
    zg = np.sqrt(an) * z0p + np.sqrt(1 - an) * e

gen = batched(decode, zg)                      # generated images
z_re = batched(encode, gen)                    # re-encoded
gen_rt = batched(decode, z_re)                 # round-tripped images

pix_rt = ((gen_rt - gen) ** 2).mean(axis=(1, 2, 3))
lat_rt = ((z_re - zg) ** 2).mean(axis=(1, 2, 3))
gx = np.abs(np.diff(gen, axis=2)).mean(axis=(1, 2, 3))
gy = np.abs(np.diff(gen, axis=1)).mean(axis=(1, 2, 3))
detail = 0.5 * (gx + gy)
contrast = gen.std(axis=(1, 2, 3))

d = pd.read_csv(SCORES).sort_values("idx").reset_index(drop=True)
assert len(d) == len(gen)
d["pixel_roundtrip"] = pix_rt
d["latent_roundtrip"] = lat_rt
d["detail"] = detail
d["contrast"] = contrast
dest = f"tables/ldpm_gen_meaning_{cli.dataset}.csv"
d.to_csv(dest, index=False)


def partial(a, b, c):
    ra, rb, rc = map(rankdata, (a, b, c))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return spearmanr(res(ra), res(rb)).correlation


print(f"\n=== {cli.dataset}: what the decoder-only uncertainty tracks on {len(d)} generations ===")
print(f"  {'target':>20s} {'rho(SLUG,.)':>12s} {'rho(detail,.)':>14s} {'partial | detail':>17s}")
for col in ["pixel_roundtrip", "latent_roundtrip"]:
    if col not in d:
        continue
    print(f"  {col:>20s} {spearmanr(d.slug, d[col]).correlation:>12.3f} "
          f"{spearmanr(d.detail, d[col]).correlation:>14.3f} "
          f"{partial(d.slug, d[col], d.detail):>17.3f}")
print(f"\n  rho(SLUG, detail)   = {spearmanr(d.slug, d.detail).correlation:+.3f}")
print(f"  rho(SLUG, contrast) = {spearmanr(d.slug, d.contrast).correlation:+.3f}")
print(f"\n  mean pixel round-trip {pix_rt.mean():.5f} | latent round-trip {lat_rt.mean():.5f}")
print(f"  wrote {dest}")
