"""Look at what the latent DDPM actually learned.

Two grids, and the second is the one that matters:

  samples_*.png       unconditional DDIM samples from pure noise. A sanity check that the
                      model learned the data distribution rather than a grey blur.

  recon_*.png         the quantity the UQ experiment is actually built on: noise a real test
                      image to t, denoise it in ONE step, decode. Rows are timesteps. This is
                      what "reconstruction error" means for this model, so if these look
                      uniformly good or uniformly destroyed at the chosen t, the error target
                      has no spread and the experiment cannot work -- exactly the failure the
                      pretrained SD UNet hit.

A third panel, vae_roundtrip, decodes the ground-truth latent. It is the floor: no diffusion
model can beat it, because the SD VAE's own encode/decode loss is already in every row.

Usage
-----
  python analysis/ldpm_sample.py --dataset fitz --save_dir outputs/ldpm_fitz --seed 0
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
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
p.add_argument("--n_samples", type=int, default=16)
p.add_argument("--per_row", type=int, default=8)
p.add_argument("--ddim_steps", type=int, default=100)
p.add_argument("--recon_t", type=int, nargs="+", default=[100, 200, 400, 600])
p.add_argument("--n_recon", type=int, default=8)
p.add_argument("--clip", type=float, default=None,
               help="clip the z0 prediction during sampling; default = the training "
                    "latents' max |value|")
p.add_argument("--out", default="outputs/ldpm_figs")
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
os.makedirs(cli.out, exist_ok=True)

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C = Z["ztest1"].shape[-1]
HW = Z["ztest1"].shape[1]

unet = LatentUNet(out_ch=C, base_ch=cli.base_ch)
prm = jax.tree_util.tree_map(
    jnp.asarray, checkpoints.restore_checkpoint(ckpt_dir=CELL, target=None)["params"])
ddpm = DDPM(T=1000)
ABAR = np.asarray(ddpm.abar)
vae, vae_p = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)


@jax.jit
def eps_theta(z, t):
    return unet.apply({"params": prm}, z, t, train=False)


@jax.jit
def decode(z):
    out = vae.apply({"params": vae_p}, z / SCALE, method=vae.decode).sample
    return jnp.transpose(out, (0, 2, 3, 1))


def to_png(x):
    """[-1,1] NHWC -> uint8."""
    return np.clip((np.asarray(x) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def grid(rows, path, pad=2):
    """rows: list of lists of HxWx3 uint8."""
    h, w = rows[0][0].shape[:2]
    R, Cn = len(rows), max(len(r) for r in rows)
    out = np.full((R * h + (R + 1) * pad, Cn * w + (Cn + 1) * pad, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            y, x = pad + i * (h + pad), pad + j * (w + pad)
            out[y:y + h, x:x + w] = im
    Image.fromarray(out).save(path)
    print(f"  -> {path}", flush=True)


# ------------------------------------------------------------------ unconditional samples
# Clipping the z0 prediction is not cosmetic. At t=999 the cosine schedule has
# abar = 2.4e-9, so z0_pred = (z_t - sqrt(1-abar) eps)/sqrt(abar) divides by 5e-5 and
# amplifies any denoiser error ~20,000x. Without the clip the trajectory diverges to latent
# std 132 against real latents at 0.79, and every sample decodes to saturated noise -- which
# says nothing about the model, only about the sampler. diffusers does this as clip_sample.
CLIP = cli.clip if cli.clip is not None else float(np.abs(Z["ztrain"]).max())
print(f"DDIM sampling {cli.n_samples} images in {cli.ddim_steps} steps "
      f"(z0 clipped to +-{CLIP:.2f})", flush=True)
key = jax.random.PRNGKey(1234)
z = jax.random.normal(key, (cli.n_samples, HW, HW, C))
ts = np.linspace(999, 0, cli.ddim_steps + 1).astype(int)
for i in range(cli.ddim_steps):
    t_cur, t_nxt = int(ts[i]), int(ts[i + 1])
    a_cur = ABAR[t_cur]
    a_nxt = ABAR[t_nxt] if t_nxt > 0 else 1.0
    e = eps_theta(z, jnp.full((cli.n_samples,), t_cur, jnp.int32))
    z0p = np.clip((z - np.sqrt(1 - a_cur) * e) / np.sqrt(a_cur), -CLIP, CLIP)
    z = np.sqrt(a_nxt) * z0p + np.sqrt(1 - a_nxt) * e
imgs = to_png(decode(jnp.asarray(z)))
print(f"  generated latent std {np.std(z):.3f} vs real {Z['ztrain'].std():.3f} "
      f"(a large gap means the trajectory diverged)", flush=True)
R = cli.per_row
grid([list(imgs[i:i + R]) for i in range(0, cli.n_samples, R)],
     f"{cli.out}/samples_{cli.dataset}_p{PERC}_seed{cli.seed}.png")

# Real vs generated, the standard way to look at a generative model: same panel, same
# decoder, same size. The top block is REAL training images put through the frozen SD VAE,
# so both halves carry the autoencoder's own losses and the only difference on show is the
# diffusion model.
real = to_png(decode(jnp.asarray(Z["ztrain"][:cli.n_samples])))
sep = [np.full_like(real[0], 255)] * R
grid([list(real[i:i + R]) for i in range(0, cli.n_samples, R)]
     + [sep]
     + [list(imgs[i:i + R]) for i in range(0, cli.n_samples, R)],
     f"{cli.out}/real_vs_gen_{cli.dataset}_p{PERC}_seed{cli.seed}.png")
print("  real_vs_gen: top block = real (VAE round-trip), bottom block = generated")


# ------------------------------------------------------------------ one-step reconstruction
z0 = Z["ztest1"][:cli.n_recon]
rows, labels = [], []
rows.append(list(to_png(decode(jnp.asarray(z0)))))
labels.append("VAE round-trip (floor)")
for t in cli.recon_t:
    tt = np.full(len(z0), t)
    a = ABAR[t]
    eps = np.asarray(jax.random.normal(jax.random.PRNGKey(0), z0.shape))
    zt = np.sqrt(a) * z0 + np.sqrt(1 - a) * eps
    e = np.asarray(eps_theta(jnp.asarray(zt), jnp.asarray(tt, jnp.int32)))
    rec = (zt - np.sqrt(1 - a) * e) / np.sqrt(a)
    err = ((rec - z0) ** 2).mean(axis=(1, 2, 3))
    rows.append(list(to_png(decode(jnp.asarray(rec)))))
    labels.append(f"t={t} (abar={a:.3f}) latent MSE {err.mean():.4f} cv {err.std()/err.mean():.3f}")
grid(rows, f"{cli.out}/recon_{cli.dataset}_p{PERC}_seed{cli.seed}.png")
print("\nrows top to bottom:")
for i, l in enumerate(labels):
    print(f"  {i}: {l}")
