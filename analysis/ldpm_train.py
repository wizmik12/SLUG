"""Train a latent diffusion model under a controlled training composition.

This is the answer to "more modern architectures should have been evaluated" that the
pretrained Stable Diffusion UNet could not give. A pretrained model has a training set we do
not control and cannot inspect, so the paper's central claim -- that the uncertainty tracks
performance loss on the subgroup a model was NOT trained on -- is untestable on it. Training
our own latent diffusion model restores the 0 / 50 / 100 composition sweep, at the cost of
training a model rather than downloading one.

The architecture is modern in the ways the reviewer asked for: a latent diffusion model with
self-attention, trained on the DDPM objective, rather than the convolutional VAE of the
original submission. The SD VAE (83.7M params, pretrained, frozen) is the encoder; only the
UNet is trained and only the UNet enters the posterior.

Latents are cached per (dataset, subset, run) and shared across seeds, so the five ensemble
members encode the data once between them. Both an image and its horizontal flip are encoded,
which preserves the flip augmentation the VAE training used -- encoding once would otherwise
freeze one arbitrary flip per image for the whole run.

No EMA of the weights. It would improve sample quality, but it would also mean the Laplace
posterior is fitted at one parameter vector and evaluated at another, and every baseline here
(SWAG, ensemble, MC-dropout) is defined relative to the trained weights. The relative
comparisons these experiments make are unaffected.

Usage
-----
  python analysis/ldpm_train.py --dataset fitz --subset 0.5 --run 0 --seed 0 \\
      --save_dir outputs/ldpm_fitz
"""
import os
# Same XLA autotuner fix as everywhere else in this codebase: above batch size 8 the
# autotuner picks a numerically broken convolution on this GPU/cuDNN pair.
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader
from diffusers import FlaxAutoencoderKL

import Helpers as hf
import data as _data
from latent_ddpm import LatentUNet, DDPM

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--subset", type=float, required=True, help="perc_attr: 0, 0.5 or 1")
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0, help="init/shuffling seed; the SPLIT is --run")
p.add_argument("--save_dir", required=True)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--image_size", type=int, default=128)
p.add_argument("--train_size", type=int, default=None, help="default: 1668 fitz / 2000 ai4skin")
p.add_argument("--test_size", type=int, default=512)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--dropout", type=float, default=0.0,
               help="> 0 builds the MC-dropout variant, as the VAE study does")
p.add_argument("--encode_batch", type=int, default=8)
p.add_argument("--scale", type=float, default=0.18215,
               help="SD's latent scale factor. Fixed across cells so MSEs are comparable.")
cli = p.parse_args()

if cli.train_size is None:
    cli.train_size = 1668 if cli.dataset == "fitz" else 2000
PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
TAG = f"{cli.dataset}_p{PERC}_r{cli.run}"
CELL = os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                    f"seed{cli.seed}" + ("_dropout" if cli.dropout > 0 else ""))
LAT = os.path.join(cli.save_dir, "latents", f"{TAG}.npz")
os.makedirs(os.path.dirname(LAT), exist_ok=True)
os.makedirs(CELL, exist_ok=True)


# ----------------------------------------------------------------- data -> latents
def build_args():
    class A:
        pass
    a = A()
    a.dataset = cli.dataset
    a.image_size = cli.image_size
    a.train_size = cli.train_size
    a.dataset_root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
    a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
    a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    return a


args = build_args()
root = (args.dataset_root if cli.dataset == "fitz" else args.ai4skin_patches_root)

if not os.path.exists(LAT):
    print(f"encoding {TAG} -> {LAT}", flush=True)
    vae, vae_p = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)

    @jax.jit
    def encode(x_nhwc):
        xc = jnp.transpose(x_nhwc, (0, 3, 1, 2))
        return vae.apply({"params": vae_p}, xc, method=vae.encode).latent_dist.mean * cli.scale

    def encode_all(ds, flip=False):
        ld = DataLoader(ds, batch_size=cli.encode_batch, shuffle=False, num_workers=4,
                        collate_fn=hf.numpy_collate)
        out, ids, t0 = [], [], time.time()
        for imgs, nm in ld:
            x = np.asarray(imgs, np.float32)
            if flip:
                x = x[:, :, ::-1, :]
            out.append(np.asarray(encode(jnp.asarray(x))))
            ids.extend([str(s) for s in nm])
        print(f"  {len(ids)} images in {time.time()-t0:.0f}s", flush=True)
        return np.concatenate(out), np.asarray(ids)

    tr, te1, te2, g1, g2 = _data.train_test_setting(root, cli.subset, cli.run, cli.test_size, args)
    ztr, _ = encode_all(tr)
    ztr_f, _ = encode_all(tr, flip=True)
    zt1, n1 = encode_all(te1)
    zt2, n2 = encode_all(te2)
    np.savez(LAT, ztrain=ztr, ztrain_flip=ztr_f, ztest1=zt1, ztest2=zt2,
             names1=n1, names2=n2, group1=g1, group2=g2, scale=cli.scale)
    print(f"  latent std (train) = {ztr.std():.3f}  (1.0 is ideal for the DDPM schedule)")
    del vae, vae_p
    jax.clear_caches()

Z = np.load(LAT, allow_pickle=True)
ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])   # flip augmentation, pre-encoded
print(f"train latents {ztrain.shape} | test {Z['ztest1'].shape} ({Z['group1']}) + "
      f"{Z['ztest2'].shape} ({Z['group2']})", flush=True)


# ----------------------------------------------------------------- model
class TrainState(train_state.TrainState):
    key: jax.Array


model = LatentUNet(out_ch=ztrain.shape[-1], base_ch=cli.base_ch, dropout=cli.dropout)
ddpm = DDPM(T=1000)
rng = jax.random.PRNGKey(cli.seed)
rng, k_init, k_drop = jax.random.split(rng, 3)
variables = model.init({"params": k_init, "dropout": k_drop},
                       jnp.asarray(ztrain[:2]), jnp.zeros((2,), jnp.int32), train=False)
n_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
print(f"LatentUNet: {n_params:,} params, dropout={cli.dropout}", flush=True)

sched = optax.join_schedules(
    [optax.linear_schedule(0.0, cli.lr, cli.warmup),
     optax.cosine_decay_schedule(cli.lr, max(1, cli.steps - cli.warmup), 0.05)],
    [cli.warmup])
state = TrainState.create(apply_fn=model.apply, params=variables["params"],
                          key=k_drop,
                          tx=optax.chain(optax.clip_by_global_norm(1.0),
                                         optax.adamw(sched, weight_decay=1e-4)))

ABAR = ddpm.abar


@jax.jit
def train_step(state, z0, key):
    k_t, k_e, k_d = jax.random.split(key, 3)
    t = jax.random.randint(k_t, (z0.shape[0],), 0, ddpm.T)
    eps = jax.random.normal(k_e, z0.shape)
    a = ABAR[t][:, None, None, None]
    zt = jnp.sqrt(a) * z0 + jnp.sqrt(1 - a) * eps

    def loss_fn(prm):
        pred = state.apply_fn({"params": prm}, zt, t, train=True, rngs={"dropout": k_d})
        return jnp.mean((pred - eps) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


ztrain_j = jnp.asarray(ztrain)
N = ztrain_j.shape[0]
print(f"training {cli.steps} steps, batch {cli.batch_size} "
      f"({cli.steps * cli.batch_size / N:.0f} epochs)", flush=True)
t0, losses = time.time(), []
for step in range(cli.steps):
    rng, k_b, k_s = jax.random.split(rng, 3)
    idx = jax.random.randint(k_b, (cli.batch_size,), 0, N)
    state, loss = train_step(state, ztrain_j[idx], k_s)
    losses.append(float(loss))
    if (step + 1) % 2000 == 0:
        el = time.time() - t0
        print(f"  step {step+1}/{cli.steps}  loss={np.mean(losses[-2000:]):.4f}  "
              f"{el/(step+1)*1000:.0f} ms/step  eta {(cli.steps-step-1)*el/(step+1)/60:.1f} min",
              flush=True)

checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(CELL),
                            target={"params": state.params}, step=cli.steps, overwrite=True,
                            keep=1)
np.save(os.path.join(CELL, "loss_curve.npy"), np.asarray(losses, np.float32))
print(f"\ndone in {(time.time()-t0)/60:.1f} min | final loss {np.mean(losses[-2000:]):.4f} "
      f"-> {CELL}", flush=True)
