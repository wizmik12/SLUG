"""Score generated samples with the decoder-only posterior, sharded across GPUs.

Usage:
  python analysis/ldpm_score_generations.py --dataset fitz --save_dir outputs/ldpm_fitz --mode fit
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
from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun
from uncertainty.sketched_lanczos_fast import build_fast_scorer
from uncertainty.sketch import SRFT_sketch, get_optimal_padding

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--mode", required=True, choices=["fit", "score"])
p.add_argument("--shard", type=int, default=0)
p.add_argument("--n_shards", type=int, default=1)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--n_gen", type=int, default=12000)
p.add_argument("--ddim_steps", type=int, default=200)
p.add_argument("--gen_batch", type=int, default=250)
p.add_argument("--fit_images", type=int, default=1000)
p.add_argument("--fit_batch_size", type=int, default=4)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--lanczos_eigenvectors", type=int, default=90)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--posterior_seed", type=int, default=1)
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
CACHE = os.path.join(cli.save_dir, f"gen_posterior_decoder_seed{cli.seed}.npz")

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C, HW = Z["ztest1"].shape[-1], Z["ztest1"].shape[1]
CLIPV = float(np.abs(Z["ztrain"]).max())
ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])

vae, vae_p_host = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)
# diffusers returns arrays COMMITTED to the host; the device has to be named explicitly or
# ravel_pytree later tries to concatenate across platforms.
vae_p = jax.device_put(vae_p_host, jax.devices()[0])
FROZEN = {"encoder": vae_p["encoder"], "quant_conv": vae_p["quant_conv"]}
THETA = {"decoder": vae_p["decoder"], "post_quant_conv": vae_p["post_quant_conv"]}
N_PAR = sum(x.size for x in jax.tree_util.tree_leaves(THETA))


class DecoderOnlyAdapter:
    """z0 -> pixels through the SD VAE decoder, decoder parameters in the posterior."""

    def apply(self, variables, x, rng=None, train=False, **kw):
        p_ = variables["params"]
        vp = {**FROZEN, "decoder": p_["decoder"], "post_quant_conv": p_["post_quant_conv"]}
        rec = vae.apply({"params": vp}, x / SCALE, method=vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


class LatentDS(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], i


# ------------------------------------------------------------------ fit
if cli.mode == "fit":
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ztrain), min(cli.fit_images, len(ztrain)), replace=False)
    loader = DataLoader(LatentDS(ztrain[idx].astype(np.float32)),
                        batch_size=cli.fit_batch_size, shuffle=False, num_workers=0,
                        collate_fn=hf.numpy_collate, drop_last=True)
    print(f"fitting decoder-only posterior on {len(idx)} real training latents "
          f"({N_PAR:,} parameters)", flush=True)
    t0 = time.time()
    fns = low_memory_lanczos_score_fun(
        DecoderOnlyAdapter(), {"params": THETA, "batch_stats": {}}, loader,
        output_shape=(128, 128, 3), sketch_size=cli.sketch_size,
        lanczos_iterations=cli.lanczos_iterations,
        lanczos_eigenvectors=cli.lanczos_eigenvectors,
        seed=cli.posterior_seed, train_batch_size=cli.fit_batch_size,
        subsample_trainset=len(idx))
    ev = np.asarray(fns["_posterior"]["eigenvec"])
    np.savez(CACHE, eigenvec=ev, n_params=N_PAR, sketch_size=cli.sketch_size,
             seed=cli.posterior_seed)
    print(f"FIT done in {(time.time()-t0)/60:.1f} min | eigenvec {ev.shape} "
          f"= {ev.size/N_PAR:.4f}p = {ev.size*4/2**20:.1f} MiB -> {CACHE}")
    sys.exit(0)

# ------------------------------------------------------------------ score
assert os.path.exists(CACHE), f"fit first: {CACHE} missing"
z = np.load(CACHE)
assert int(z["n_params"]) == N_PAR
key_sketch, _ = jax.random.split(jax.random.PRNGKey(int(z["seed"])))
sketch_op = SRFT_sketch(key_sketch, N_PAR, int(z["sketch_size"]),
                        padding=get_optimal_padding(N_PAR))
post = {"eigenvec": jnp.asarray(z["eigenvec"]), "sketch_op": sketch_op,
        "params": THETA, "batch_stats": {}}
score = build_fast_scorer(DecoderOnlyAdapter(), post, (128, 128, 3),
                          hutchinson_samples=cli.hutchinson_samples)

unet = LatentUNet(out_ch=C, base_ch=cli.base_ch)
prm = jax.tree_util.tree_map(
    jnp.asarray, checkpoints.restore_checkpoint(ckpt_dir=CELL, target=None)["params"])
ABAR = np.asarray(DDPM(T=1000).abar)


@jax.jit
def eps_theta(zz, t):
    return unet.apply({"params": prm}, zz, t, train=False)


# Generate deterministically, in blocks, keeping only this shard's indices. The key and
# schedule match ldpm_generation_uq.py so the sets are the same images.
mine = np.arange(cli.shard, cli.n_gen, cli.n_shards)
print(f"shard {cli.shard}/{cli.n_shards}: {len(mine)} of {cli.n_gen} generations", flush=True)
zall = np.asarray(jax.random.normal(jax.random.PRNGKey(20260929),
                                    (cli.n_gen, HW, HW, C)))[mine]
ts = np.linspace(999, 0, cli.ddim_steps + 1).astype(int)
t0 = time.time()
for i in range(cli.ddim_steps):
    tc, tn = int(ts[i]), int(ts[i + 1])
    ac = ABAR[tc]
    an = ABAR[tn] if tn > 0 else 1.0
    e = np.concatenate([np.asarray(eps_theta(jnp.asarray(zall[j:j + cli.gen_batch]),
                                             jnp.full((len(zall[j:j + cli.gen_batch]),), tc,
                                                      jnp.int32)))
                        for j in range(0, len(zall), cli.gen_batch)])
    z0p = np.clip((zall - np.sqrt(1 - ac) * e) / np.sqrt(ac), -CLIPV, CLIPV)
    zall = np.sqrt(an) * z0p + np.sqrt(1 - an) * e
print(f"  generated in {(time.time()-t0)/60:.1f} min", flush=True)

t0, out = time.time(), []
for i in range(0, len(zall), 64):
    chunk = zall[i:i + 64].astype(np.float32)
    out.append(np.asarray(score["approx"](chunk, jax.random.PRNGKey(0),
                                          cli.hutchinson_samples)))
    done = sum(len(o) for o in out)
    if done % 256 < 64:
        el = time.time() - t0
        print(f"  {done}/{len(zall)}  {el/done:.1f} s/img  "
              f"eta {(len(zall)-done)*el/done/3600:.2f} h", flush=True)
    if (i // 64) % 20 == 0:
        jax.clear_caches()
s = np.concatenate(out)
os.makedirs("tables/gen12k", exist_ok=True)
dest = f"tables/gen12k/{cli.dataset}_shard{cli.shard}of{cli.n_shards}.csv"
pd.DataFrame(dict(idx=mine, slug=s)).to_csv(dest, index=False)
print(f"wrote {dest} ({len(s)} rows, {(time.time()-t0)/60:.1f} min)")
