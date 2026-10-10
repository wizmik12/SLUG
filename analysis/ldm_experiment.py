"""SLUG on the Stable Diffusion autoencoder: a modern, attention-based generative model.

Answers the reviewer's point that "more modern architectures should have been evaluated".

Design. The model is pretrained on LAION, so its training set is unavailable and the Laplace
posterior needs one. We fit on COCO val2017 as a proxy for the training distribution and score
medical images as held-out data; fitting on the medical images would be fitting on test data.

Two phases so the posterior is fitted ONCE and shared by every scoring shard:
  --mode fit     writes the sketched eigenvectors to <cache>/ldm_posterior.npz
  --mode score   loads them, rebuilds the SRFT from the same seed, scores one shard

Scoring uses sketched_lanczos_fast (validated against the original on the conv VAE: 0.03%
agreement at rank 9, Spearman 1.0000). The original scorer is not usable here -- it retraces
the model VJP per image, which on this architecture costs 462 s/image against 6.7 s/image.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd, jax, jax.numpy as jnp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import FlaxAutoencoderKL

import Helpers as hf
from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun
from uncertainty.sketched_lanczos_fast import build_fast_scorer
from uncertainty.sketch import SRFT_sketch, get_optimal_padding

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True, choices=["fit", "score"])
p.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
p.add_argument("--coco", default="data_natural/val2017")
p.add_argument("--cache", default="outputs/ldm")
p.add_argument("--image_size", type=int, default=128)
p.add_argument("--fit_images", type=int, default=500)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--lanczos_eigenvectors", type=int, default=90)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--fit_batch_size", type=int, default=4)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--domain", default=None,
               help="score mode: coco | fitz_light | fitz_dark | husc | hcuv")
p.add_argument("--n_score", type=int, default=512)
cli = p.parse_args()
os.makedirs(cli.cache, exist_ok=True)
CACHE = os.path.join(cli.cache, "ldm_posterior.npz")


class SDVAEAdapter:
    def __init__(self, vae):
        self.vae = vae
    def apply(self, variables, x, rng=None, train=False, **kw):
        p_ = variables["params"]
        xc = jnp.transpose(x, (0, 3, 1, 2))
        lat = self.vae.apply({"params": p_}, xc, method=self.vae.encode).latent_dist.mean
        rec = self.vae.apply({"params": p_}, lat, method=self.vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


def to_arr(im):
    im = im.convert("RGB").resize((cli.image_size, cli.image_size))
    return (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.5


class PassthroughDS(Dataset):
    """Wraps the baseline torch Dataset so both sources expose (image, name)."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        x, nm = self.ds[i]
        return np.asarray(x, np.float32), str(nm)


class PathDS(Dataset):
    """Decodes ONCE into memory. Lanczos iterates the loader n_iter times, so decoding in
    __getitem__ means 100x the JPEG work; 500 images at 128px is only ~98 MB held."""
    def __init__(self, paths):
        self.names = [os.path.basename(p_) for p_ in paths]
        t = time.time()
        self.x = np.stack([to_arr(Image.open(p_)) for p_ in paths])
        print(f"  cached {len(self.x)} images in {time.time()-t:.0f}s "
              f"({self.x.nbytes/2**20:.0f} MiB)", flush=True)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.names[i]


def domain_paths(name, n, skip=0):
    """COCO by path; the MEDICAL domains come from data.train_test_setting so that the SD VAE
    is scored on EXACTLY the 512 images per subgroup used by the baseline tables. Drawing an
    independent random sample instead would confound any SD-VAE-vs-conv-VAE difference with
    which images happened to be picked, and would make subgroup disparities incomparable."""
    if name == "coco":
        f = sorted(glob.glob(os.path.join(cli.coco, "*.jpg")))
        rng = np.random.default_rng(0)
        if len(f) > n + skip:
            f = list(rng.permutation(f))
        return f[skip:skip + n]

    import data as _data

    class _A:
        pass
    a = _A()
    a.image_size = cli.image_size
    a.train_size = 1668
    a.block_widths = (1, 2, 4, 8)
    a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
    a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    if name.startswith("fitz"):
        a.dataset = "fitz"
        root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
    else:
        a.dataset = "ai4skin"
        root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    a.dataset_root = root
    # (perc_attr=0.5, run=0) is the split the 0_seed* main-table checkpoints were built on
    _, t1, t2, g1, g2 = _data.train_test_setting(root, 0.5, 0, n, a)
    ds = t1 if name in ("fitz_light", "husc") else t2
    print(f"  using the baseline test set for {g1 if ds is t1 else g2} ({len(ds)} images)",
          flush=True)
    return ds


vae, params = FlaxAutoencoderKL.from_pretrained(cli.model, from_pt=True, dtype=jnp.float32)
model = SDVAEAdapter(vae)
params_dict = {"params": params, "batch_stats": {}}
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

if cli.mode == "fit":
    paths = domain_paths("coco", cli.fit_images)
    print(f"fitting on {len(paths)} COCO images | {n_params:,} params", flush=True)
    loader = DataLoader(PathDS(paths), batch_size=cli.fit_batch_size, shuffle=False,
                        num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)
    t0 = time.time()
    fns = low_memory_lanczos_score_fun(
        model, params_dict, loader,
        output_shape=(cli.image_size, cli.image_size, 3),
        sketch_size=cli.sketch_size,
        lanczos_iterations=cli.lanczos_iterations,
        lanczos_eigenvectors=cli.lanczos_eigenvectors,
        seed=cli.seed, train_batch_size=cli.fit_batch_size,
        subsample_trainset=cli.fit_images)
    ev = np.asarray(fns["_posterior"]["eigenvec"])
    np.savez(CACHE, eigenvec=ev, n_params=n_params, sketch_size=cli.sketch_size,
             seed=cli.seed, image_size=cli.image_size)
    stored = ev.shape[0] * ev.shape[1]
    print(f"FIT done in {(time.time()-t0)/60:.1f} min | eigenvec {ev.shape} "
          f"= {stored/n_params:.4f}p = {stored*4/2**20:.1f} MiB -> {CACHE}")
    sys.exit(0)

# ------------------------------------------------------------------ score
z = np.load(CACHE)
assert int(z["n_params"]) == n_params and int(z["image_size"]) == cli.image_size
key_sketch, _ = jax.random.split(jax.random.PRNGKey(int(z["seed"])))
sketch_op = SRFT_sketch(key_sketch, n_params, int(z["sketch_size"]),
                        padding=get_optimal_padding(n_params))
post = {"eigenvec": jnp.asarray(z["eigenvec"]), "sketch_op": sketch_op,
        "params": params, "batch_stats": {}}
score = build_fast_scorer(model, post, (cli.image_size, cli.image_size, 3),
                          hutchinson_samples=cli.hutchinson_samples)

paths = domain_paths(cli.domain, cli.n_score, skip=cli.fit_images if cli.domain == "coco" else 0)
print(f"scoring {len(paths)} images from '{cli.domain}'", flush=True)
DS = PathDS(paths) if isinstance(paths, list) else PassthroughDS(paths)
loader = DataLoader(DS, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=hf.numpy_collate)

import lpips as lpips_lib, torch
from skimage.metrics import structural_similarity as ssim_fn
lp = lpips_lib.LPIPS(net="vgg")
key = jax.random.PRNGKey(0)
rows, t0, nb = [], time.time(), 0
for imgs, names in loader:
    imgs = np.asarray(imgs)
    u = np.asarray(score(imgs, key, cli.hutchinson_samples))
    rec = np.asarray(model.apply(params_dict, jnp.asarray(imgs))[0])
    a = torch.from_numpy(np.transpose(imgs, (0, 3, 1, 2))).float()
    b = torch.from_numpy(np.transpose(rec, (0, 3, 1, 2))).float()
    with torch.no_grad():
        lv = lp(a, b).squeeze().cpu().numpy().reshape(-1)
    for j, nm in enumerate(names):
        rows.append(dict(file=nm, domain=cli.domain,
                         mse=float(((rec[j]-imgs[j])**2).mean()),
                         ssim=float(ssim_fn(imgs[j], rec[j], channel_axis=2, data_range=2.0)),
                         lpips=float(lv[j]), uncertainty=float(u[j])))
    nb += 1
    if nb % 25 == 0:
        jax.clear_caches()
    if nb % 50 == 0:
        el = time.time() - t0
        print(f"  {nb}/{len(paths)}  {el/nb:.1f} s/img  eta {(len(paths)-nb)*el/nb/3600:.2f} h",
              flush=True)
out = pd.DataFrame(rows)
# Resolution goes in the path. Without it a 256px run silently overwrites the 128px
# results for the same domain, which happened on 2026-09-30.
dest = (f"tables/ldm_{cli.domain}.csv" if cli.image_size == 128
        else f"tables/res{cli.image_size}/ldm_{cli.domain}.csv")
os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
os.makedirs("tables", exist_ok=True)
out.to_csv(dest, index=False)
from scipy.stats import pearsonr, spearmanr
print(f"\nwrote {dest} ({len(out)} rows)")
for m in ["mse", "ssim", "lpips"]:
    print(f"  SLUG vs {m:5s}: pearson={pearsonr(out.uncertainty, out[m])[0]:+.3f} "
          f"spearman={spearmanr(out.uncertainty, out[m]).correlation:+.3f}")
