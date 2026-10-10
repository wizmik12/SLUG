"""Decoder-only SLUG on the real test images, both subgroups.

ldpm_generation_uq.py scores the decoder-only posterior on the first test group only, because
there it is a side quantity next to the generations. The manuscript reports it per subgroup
like every other table, so this scores both groups.

The posterior is over the autoencoder's decoder alone, fitted on the latents of real training
images, and the target is the decoder's own reconstruction error ||decode(z0) - x||^2. No
diffusion model is involved: this is the autoencoder being scored against its own failure.

Usage:
  python analysis/ldpm_decoder_real.py --dataset fitz --save_dir outputs/ldpm_fitz
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from diffusers import FlaxAutoencoderKL

import Helpers as hf
import data as _d
from uncertainty.sketched_lanczos_fast import low_memory_lanczos_score_fun_fast

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--fit_images", type=int, default=1000)
p.add_argument("--fit_batch_size", type=int, default=4)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--n_score", type=int, default=512)
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])

vae, vae_p_host = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)
# diffusers returns arrays committed to the host; the device must be named explicitly
vae_p = jax.device_put(vae_p_host, jax.devices()[0])
FROZEN = {"encoder": vae_p["encoder"], "quant_conv": vae_p["quant_conv"]}
THETA = {"decoder": vae_p["decoder"], "post_quant_conv": vae_p["post_quant_conv"]}


class DecoderOnlyAdapter:
    def apply(self, variables, x, rng=None, train=False, **kw):
        q = variables["params"]
        vp = {**FROZEN, "decoder": q["decoder"], "post_quant_conv": q["post_quant_conv"]}
        rec = vae.apply({"params": vp}, x / SCALE, method=vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


class LatentDS(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], i


@jax.jit
def decode(z):
    return jnp.transpose(vae.apply({"params": vae_p}, z / SCALE, method=vae.decode).sample,
                         (0, 2, 3, 1))


ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])
rng = np.random.default_rng(0)
idx = rng.choice(len(ztrain), min(cli.fit_images, len(ztrain)), replace=False)
loader = DataLoader(LatentDS(ztrain[idx].astype(np.float32)), batch_size=cli.fit_batch_size,
                    shuffle=False, num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)
npar = sum(x.size for x in jax.tree_util.tree_leaves(THETA))
print(f"fitting decoder-only posterior on {len(idx)} training latents ({npar:,} params)",
      flush=True)
t0 = time.time()
fns = low_memory_lanczos_score_fun_fast(
    DecoderOnlyAdapter(), {"params": THETA, "batch_stats": {}}, loader,
    output_shape=(128, 128, 3), sketch_size=cli.sketch_size,
    lanczos_iterations=cli.lanczos_iterations, seed=1,
    train_batch_size=cli.fit_batch_size, subsample_trainset=len(idx),
    hutchinson_samples=cli.hutchinson_samples)
print(f"  fit: {(time.time()-t0)/60:.1f} min", flush=True)


def group_pixels(ds, cached_names, n):
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=hf.numpy_collate)
    xs, ids = [], []
    for x, nm in ld:
        xs.append(np.asarray(x, np.float32)); ids.extend([str(v) for v in nm])
    assert ids[:n] == [str(v) for v in cached_names[:n]], "test-set order drifted from the cache"
    return np.concatenate(xs)[:n]


class A:
    pass


a = A()
a.dataset, a.image_size = cli.dataset, 128
a.train_size = 1668 if cli.dataset == "fitz" else 2000
a.dataset_root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
root = a.dataset_root if cli.dataset == "fitz" else a.ai4skin_patches_root
_, t1, t2, g1, g2 = _d.train_test_setting(root, cli.subset, cli.run, cli.n_score, a)

rows = []
for ds, zg, names, gname in [(t1, Z["ztest1"], Z["names1"], str(Z["group1"])),
                             (t2, Z["ztest2"], Z["names2"], str(Z["group2"]))]:
    n = min(cli.n_score, len(zg))
    x = group_pixels(ds, names, n)
    z0 = zg[:n].astype(np.float32)
    t0 = time.time()
    s = np.asarray(fns["approx"](z0, jax.random.PRNGKey(0), cli.hutchinson_samples))
    rec = np.concatenate([np.asarray(decode(jnp.asarray(z0[i:i + 16])))
                          for i in range(0, len(z0), 16)])
    err = ((rec - x) ** 2).mean(axis=(1, 2, 3))
    thr = np.quantile(err, 0.95)
    print(f"  {gname}: {n} images in {(time.time()-t0)/60:.1f} min | "
          f"pearson {pearsonr(s, err)[0]:+.3f} spearman {spearmanr(s, err).correlation:+.3f} "
          f"AUROC5% {roc_auc_score(err >= thr, s):.3f}", flush=True)
    rows.append(pd.DataFrame(dict(file=[str(v) for v in names[:n]], group=gname,
                                  slug=s, decode_err=err)))
    jax.clear_caches()

out = pd.concat(rows, ignore_index=True)
dest = f"tables/ldpm_decoder_real_{cli.dataset}_p{PERC}.csv"
out.to_csv(dest, index=False)
print(f"wrote {dest} ({len(out)} rows)")
