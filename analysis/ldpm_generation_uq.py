"""Score unconditional generations of the latent diffusion model with SLUG.

Writes tables/ldpm_gen_uq_<dataset>_p<subset>_seed<seed>_t<t>.csv, which
analysis/ldpm_gen_quantiles.py and analysis/ldpm_gen_meaning.py read.

Usage:
  python analysis/ldpm_generation_uq.py --dataset fitz --save_dir outputs/ldpm_fitz \
      --scope decoder_only --n_gen 1000 --n_real 512
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
from flax.training import checkpoints
from torch.utils.data import Dataset, DataLoader
from diffusers import FlaxAutoencoderKL

import Helpers as hf
from latent_ddpm import LatentUNet, DDPM
from uncertainty.sketched_lanczos_fast import low_memory_lanczos_score_fun_fast

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--n_gen", type=int, default=1000)
p.add_argument("--n_real", type=int, default=512, help="real latents, for the comparison")
p.add_argument("--ddim_steps", type=int, default=200)
p.add_argument("--t", type=int, default=500, help="scoring timestep")
p.add_argument("--fit_images", type=int, default=1000)
p.add_argument("--fit_batch_size", type=int, default=8)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--scope", choices=["latent", "decoder_only"], default="decoder_only",
               help="'latent' scores the denoiser, 'decoder_only' the SD VAE decoder at the "
                    "generated latent. With no input image there is no encoder, so the "
                    "decoder is the whole generative path being scored.")
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
os.makedirs("tables", exist_ok=True)

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C, HW = Z["ztest1"].shape[-1], Z["ztest1"].shape[1]
ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])
CLIP = float(np.abs(Z["ztrain"]).max())

unet = LatentUNet(out_ch=C, base_ch=cli.base_ch)
prm = jax.tree_util.tree_map(
    jnp.asarray, checkpoints.restore_checkpoint(ckpt_dir=CELL, target=None)["params"])
ABAR = np.asarray(DDPM(T=1000).abar)
vae, vae_p = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)


@jax.jit
def eps_theta(z, t):
    return unet.apply({"params": prm}, z, t, train=False)


# ----------------------------------------------------------------- generate
print(f"generating {cli.n_gen} samples, {cli.ddim_steps} DDIM steps "
      f"(z0 clipped to +-{CLIP:.2f})", flush=True)
t0 = time.time()
zg = np.asarray(jax.random.normal(jax.random.PRNGKey(20260929), (cli.n_gen, HW, HW, C)))
ts = np.linspace(999, 0, cli.ddim_steps + 1).astype(int)
for i in range(cli.ddim_steps):
    tc, tn = int(ts[i]), int(ts[i + 1])
    ac = ABAR[tc]
    an = ABAR[tn] if tn > 0 else 1.0
    e = np.concatenate([np.asarray(eps_theta(jnp.asarray(zg[j:j + 100]),
                                             jnp.full((len(zg[j:j + 100]),), tc, jnp.int32)))
                        for j in range(0, len(zg), 100)])
    z0p = np.clip((zg - np.sqrt(1 - ac) * e) / np.sqrt(ac), -CLIP, CLIP)
    zg = np.sqrt(an) * z0p + np.sqrt(1 - an) * e
print(f"  done in {(time.time()-t0)/60:.1f} min | generated latent std {zg.std():.3f} "
      f"vs real {Z['ztrain'].std():.3f}", flush=True)

zreal = Z["ztest1"][:cli.n_real]


# ----------------------------------------------------------------- SLUG
class LatentDS(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], i


class DDPMAdapter:
    """x is [...,C+1]: channels :C are z_t, the last is a constant plane holding t/T."""
    def apply(self, variables, x, rng=None, train=False, **kw):
        return (unet.apply({"params": variables["params"]}, x[..., :C],
                           jnp.round(x[..., 0, 0, C] * 1000).astype(jnp.int32), train=False),)


class DecoderOnlyAdapter:
    """z0 -> pixels through the SD VAE decoder, with the decoder's parameters in the posterior.

    The diffusion model only supplies the latents: it replaces the N(0, I) prior with a
    learned one. The posterior is fitted on the latents of real training images, so it is
    independent of what is scored afterwards.
    """

    def apply(self, variables, x, rng=None, train=False, **kw):
        p = variables["params"]
        vp = {**FROZEN_VAE, "decoder": p["decoder"],
              "post_quant_conv": p["post_quant_conv"]}
        rec = vae.apply({"params": vp}, x / SCALE, method=vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


def pack(zt, t):
    t = np.broadcast_to(np.asarray(t, np.float32).reshape(-1, 1, 1, 1) / 1000.0,
                        zt.shape[:-1] + (1,))
    return np.concatenate([zt, t], axis=-1).astype(np.float32)


def noise_at(z0, t, seed=0):
    eps = np.asarray(jax.random.normal(jax.random.PRNGKey(seed), z0.shape), np.float32)
    a = ABAR[np.asarray(t, np.int32).reshape(-1)].reshape(-1, 1, 1, 1)
    return (np.sqrt(a) * z0 + np.sqrt(1 - a) * eps).astype(np.float32), eps


FROZEN_VAE = {"encoder": None, "quant_conv": None}   # filled below, once vae_p is on device

rng = np.random.default_rng(0)
idx = rng.choice(len(ztrain), min(cli.fit_images, len(ztrain)), replace=False)

if cli.scope == "decoder_only":
    # diffusers returns the VAE weights COMMITTED to the host; jnp.asarray and a bare
    # device_put are no-ops on a committed array, so the device has to be named.
    vdev = jax.device_put(vae_p, jax.devices()[0])
    FROZEN_VAE = {"encoder": vdev["encoder"], "quant_conv": vdev["quant_conv"]}
    theta = {"decoder": vdev["decoder"], "post_quant_conv": vdev["post_quant_conv"]}
    adapter = DecoderOnlyAdapter()
    fit_x = ztrain[idx].astype(np.float32)   # the datapoint is the clean latent itself
    out_shape = (128, 128, 3)
else:
    theta = prm
    adapter = DDPMAdapter()
    t_fit = rng.integers(0, 1000, len(idx))
    zt_fit, _ = noise_at(ztrain[idx], t_fit, seed=7)
    fit_x = pack(zt_fit, t_fit)
    out_shape = (HW, HW, C)

npar = sum(x.size for x in jax.tree_util.tree_leaves(theta))
loader = DataLoader(LatentDS(fit_x), batch_size=cli.fit_batch_size, shuffle=False,
                    num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)
print(f"fitting the {cli.scope} posterior on {len(idx)} real training latents "
      f"({npar:,} parameters, output dim {int(np.prod(out_shape)):,})", flush=True)
t0 = time.time()
fns = low_memory_lanczos_score_fun_fast(
    adapter, {"params": theta, "batch_stats": {}}, loader,
    output_shape=out_shape, sketch_size=cli.sketch_size,
    lanczos_iterations=cli.lanczos_iterations, seed=1,
    train_batch_size=cli.fit_batch_size, subsample_trainset=len(idx),
    hutchinson_samples=cli.hutchinson_samples)
print(f"  fit: {(time.time()-t0)/60:.1f} min", flush=True)


def slug(z0, label):
    tarr = np.full(len(z0), cli.t)
    zt, _ = noise_at(z0, tarr)
    # decoder-only scores the clean latent; the diffusion model only supplied it
    x = z0.astype(np.float32) if cli.scope == "decoder_only" else pack(zt, tarr)
    t0 = time.time()
    s = np.asarray(fns["approx"](x, jax.random.PRNGKey(0), cli.hutchinson_samples))
    print(f"  {label}: {len(z0)} in {(time.time()-t0)/60:.1f} min", flush=True)
    return s


slug_gen = slug(zg, "generated")
slug_real = slug(zreal, "real")

dest = f"tables/ldpm_gen_uq_{cli.dataset}_p{PERC}_seed{cli.seed}_t{cli.t}.csv"
pd.DataFrame(dict(idx=np.arange(len(zg)), slug=slug_gen)).to_csv(dest, index=False)

print(f"\n=== {cli.dataset} p={PERC} seed{cli.seed}, t={cli.t}, {cli.n_gen} generations ===")
print(f"  SLUG generated  mean {slug_gen.mean():.4g}  std {slug_gen.std():.4g}")
print(f"  SLUG real       mean {slug_real.mean():.4g}  std {slug_real.std():.4g}")
print(f"  wrote {dest}")
