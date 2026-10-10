"""Does SLUG flag BAD unconditional generations from the latent diffusion model?

The counterpart of analysis/realism_vs_uncertainty.py, which asks the same question of the
VAE. Generated samples have no reference, so there is no reconstruction error to predict; the
question is whether the epistemic uncertainty of the denoiser at a generated latent tells you
anything about whether that sample is any good.

Two independent quality targets, because each has a weakness the other does not:

  realism      Kynkaanniemi et al. 2019 (arXiv:1904.06991). For each REAL sample, the radius
               to its k-th nearest real neighbour defines a hypersphere; a generated sample
               scores max_i radius_i / ||phi_g - phi_r,i||, so R > 1 means it lands inside the
               estimated real manifold. Adapts to local density, which matters for medical
               data where the rare presentations are legitimate but sparse. Weakness: it is
               computed in ImageNet-pretrained InceptionV3 features, and dermatology is a
               substantial domain shift -- the standard criticism of FID-family metrics on
               medical images.

  self_recon   Re-noise the generated latent to the SAME t used for scoring and denoise one
               step: how well does the model reconstruct its OWN sample. Needs no external
               feature space and no real data at all, so it is immune to the domain-shift
               objection, but it is not independent of the model being probed.

A negative correlation with realism, or a positive one with self_recon, is the result that
supports using the score to filter generations.

Real test images are scored by the identical procedure, so "are synthetic images more
uncertain than real ones" is asked without changing anything but the input.

Usage
-----
  python analysis/ldpm_generation_uq.py --dataset fitz --save_dir outputs/ldpm_fitz \\
      --n_gen 1000 --n_real 512
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
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from flax.training import checkpoints
from torch.utils.data import Dataset, DataLoader
from diffusers import FlaxAutoencoderKL
from torchvision.models import inception_v3, Inception_V3_Weights

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
p.add_argument("--n_real", type=int, default=512, help="real images, for the comparison")
p.add_argument("--n_manifold", type=int, default=1000, help="real images defining the manifold")
p.add_argument("--ddim_steps", type=int, default=200)
p.add_argument("--t", type=int, default=500, help="must match the ldpm_uq.py scoring timestep")
p.add_argument("--k", type=int, default=3, help="realism: k-th nearest neighbour")
p.add_argument("--fit_images", type=int, default=1000)
p.add_argument("--fit_batch_size", type=int, default=8)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--n_show", type=int, default=12, help="images per extreme in the figure")
p.add_argument("--scope", choices=["latent", "decoder_only"], default="decoder_only",
               help="'latent' scores the denoiser, i.e. uncertainty about how the sample was "
                    "produced. 'decoder_only' scores the SD VAE decoder at the generated "
                    "latent, i.e. uncertainty about turning that latent into an image. The "
                    "second is the direct analogue of the VAE's decoder-only generation "
                    "experiment (uncertainty/decoder_only_generation.py): there is no input "
                    "image and therefore no encoder, so the decoder is the whole generative "
                    "path being scored.")
p.add_argument("--out", default="outputs/ldpm_figs")
cli = p.parse_args()

PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
CELL = os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                    f"seed{cli.seed}"))
os.makedirs(cli.out, exist_ok=True)
os.makedirs("tables", exist_ok=True)

Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
C, HW = Z["ztest1"].shape[-1], Z["ztest1"].shape[1]
ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])
CLIP = float(np.abs(Z["ztrain"]).max())

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
    return jnp.transpose(vae.apply({"params": vae_p}, z / SCALE, method=vae.decode).sample,
                         (0, 2, 3, 1))


def decode_all(z, bs=16):
    return np.concatenate([np.asarray(decode(jnp.asarray(z[i:i + bs])))
                           for i in range(0, len(z), bs)])


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


# ----------------------------------------------------------------- real images
def real_pixels(n):
    class A:
        pass
    a = A()
    a.dataset, a.image_size, a.train_size = cli.dataset, 128, (1668 if cli.dataset == "fitz" else 2000)
    a.dataset_root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
    a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
    a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    import data as _d
    root = a.dataset_root if cli.dataset == "fitz" else a.ai4skin_patches_root
    tr, _, _, _, _ = _d.train_test_setting(root, cli.subset, cli.run, 512, a)
    ld = DataLoader(tr, batch_size=32, shuffle=False, num_workers=4, collate_fn=hf.numpy_collate)
    out = []
    for x, _ in ld:
        out.append(np.asarray(x, np.float32))
        if sum(len(o) for o in out) >= n:
            break
    return np.concatenate(out)[:n]


gen_pix = decode_all(zg)
real_pix = real_pixels(cli.n_manifold)
zreal = Z["ztest1"][:cli.n_real]           # held-out real latents, for the score comparison


def test_group1_pixels(n):
    """The held-out images that zreal is the encoding of, in the same order.

    For a decoder-only posterior the matched error target is the DECODER's own error,
    ||decode(z0) - x||^2, not how well the denoiser denoises: the score is about the decoder,
    so the error it should predict is the decoder's. The order is asserted against the cached
    names, since a silent mismatch would join every uncertainty to the wrong image.
    """
    class A:
        pass
    a = A()
    a.dataset, a.image_size = cli.dataset, 128   # the latents were encoded at 128
    a.train_size = 1668 if cli.dataset == "fitz" else 2000
    a.dataset_root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
    a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
    a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    import data as _d
    root = a.dataset_root if cli.dataset == "fitz" else a.ai4skin_patches_root
    _, t1, _, _, _ = _d.train_test_setting(root, cli.subset, cli.run, 512, a)
    ld = DataLoader(t1, batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=hf.numpy_collate)
    xs, ids = [], []
    for x, nm in ld:
        xs.append(np.asarray(x, np.float32)); ids.extend([str(v) for v in nm])
    assert ids[:n] == [str(v) for v in Z["names1"][:n]], "test-set order drifted from the cache"
    return np.concatenate(xs)[:n]


real_x = test_group1_pixels(cli.n_real)


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

    The counterpart of the VAE's decoder-only scope. When there is no input image there is no
    encoder and no reconstruction, so the object to be uncertain about is the map from a latent
    to an image. The diffusion model's only role here is to supply the latents: it replaces the
    N(0, I) prior the VAE version samples from with a learned one, which is the point of using
    a latent diffusion model at all.

    The posterior is fitted on the latents of REAL training images, which is standard Laplace
    fitting around the training distribution and is independent of what is scored afterwards.
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
    # the datapoint is the clean latent itself: no noising, no timestep
    fit_x = ztrain[idx].astype(np.float32)
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


def slug_and_recon(z0, label):
    tarr = np.full(len(z0), cli.t)
    zt, _ = noise_at(z0, tarr)
    # decoder-only scores the clean latent; the diffusion model only supplied it
    x = z0.astype(np.float32) if cli.scope == "decoder_only" else pack(zt, tarr)
    t0 = time.time()
    s = np.asarray(fns["approx"](x, jax.random.PRNGKey(0), cli.hutchinson_samples))
    e = np.concatenate([np.asarray(eps_theta(jnp.asarray(zt[i:i + 64]),
                                             jnp.full((len(zt[i:i + 64]),), cli.t, jnp.int32)))
                        for i in range(0, len(zt), 64)])
    a = ABAR[cli.t]
    rec = (zt - np.sqrt(1 - a) * e) / np.sqrt(a)
    print(f"  {label}: {len(z0)} in {(time.time()-t0)/60:.1f} min", flush=True)
    return s, ((rec - z0) ** 2).mean(axis=(1, 2, 3))


slug_gen, recon_gen = slug_and_recon(zg, "generated")
slug_real, recon_real = slug_and_recon(zreal, "real")


# ----------------------------------------------------------------- realism
def inception_features(imgs, device, batch=32):
    w = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=w, aux_logits=True, init_weights=False).to(device).eval()
    net.fc = torch.nn.Identity()
    x = np.asarray(imgs, np.float32)
    if x.min() < -0.01:
        x = (x + 1.0) / 2.0
    x = np.clip(x, 0, 1).transpose(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch])
            t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
            outs.append(net(((t - mean) / std).to(device)).cpu().numpy())
    return np.concatenate(outs, 0)


def realism(feat_real, feat_gen, k=3):
    """Kynkaanniemi realism, via cdist: the [R,R,2048] broadcast the VAE script uses would be
    8 GB at R=1000."""
    dr = cdist(feat_real, feat_real)
    np.fill_diagonal(dr, np.inf)
    radii = np.sort(dr, axis=1)[:, k - 1]
    dg = cdist(feat_gen, feat_real)
    return (radii[None, :] / np.maximum(dg, 1e-12)).max(axis=1)


print("computing realism scores (InceptionV3, CPU)", flush=True)
dev = "cpu"
fr = inception_features(real_pix, dev)
fg = inception_features(gen_pix, dev)
R_gen = realism(fr, fg, cli.k)


# ----------------------------------------------------------------- report
df = pd.DataFrame(dict(idx=np.arange(len(zg)), slug=slug_gen, realism=R_gen,
                       self_recon=recon_gen))
dest = f"tables/ldpm_gen_uq_{cli.dataset}_p{PERC}_seed{cli.seed}_t{cli.t}.csv"
df.to_csv(dest, index=False)

print(f"\n=== {cli.dataset} p={PERC} seed{cli.seed}, t={cli.t}, {cli.n_gen} generations ===")
print(f"  SLUG   generated  mean {slug_gen.mean():.4g}  std {slug_gen.std():.4g}")
print(f"  SLUG   real       mean {slug_real.mean():.4g}  std {slug_real.std():.4g}")
print(f"  realism  mean {R_gen.mean():.3f}  frac on-manifold (R>1) {(R_gen > 1).mean():.3f}")
print()
from sklearn.metrics import roc_auc_score
rec_real = decode_all(zreal)
decode_err = ((rec_real - real_x) ** 2).mean(axis=(1, 2, 3))
thr = np.quantile(decode_err, 0.95)
print("\n  REAL images, the decoder's own reconstruction error ||decode(z0) - x||^2:")
print(f"    mean {decode_err.mean():.5f}")
print(f"    SLUG vs that error : pearson {pearsonr(slug_real, decode_err)[0]:+.3f}  "
      f"spearman {spearmanr(slug_real, decode_err).correlation:+.3f}")
print(f"    AUROC, worst 5%    : {roc_auc_score(decode_err >= thr, slug_real):.3f}")
pd.DataFrame(dict(file=[str(v) for v in Z["names1"][:cli.n_real]], slug=slug_real,
                  decode_err=decode_err)).to_csv(
    f"tables/ldpm_gen_real_{cli.dataset}_p{PERC}_seed{cli.seed}.csv", index=False)
print()
print("  GENERATED images, correlation of SLUG with each quality target (Pearson / Spearman):")
for nm, v, want in [("realism  (want NEGATIVE)", R_gen, "neg"),
                    ("self_recon (want POSITIVE)", recon_gen, "pos")]:
    print(f"    {nm:28s} {pearsonr(slug_gen, v)[0]:+.3f} / "
          f"{spearmanr(slug_gen, v).correlation:+.3f}")
print(f"    {'realism vs self_recon':28s} {pearsonr(R_gen, recon_gen)[0]:+.3f} / "
      f"{spearmanr(R_gen, recon_gen).correlation:+.3f}   (the two targets, against each other)")

lo, hi = df.nsmallest(cli.n_show, "slug"), df.nlargest(cli.n_show, "slug")
print(f"\n  least uncertain {cli.n_show}: realism {lo.realism.mean():.3f}  "
      f"self_recon {lo.self_recon.mean():.4f}")
print(f"  most  uncertain {cli.n_show}: realism {hi.realism.mean():.3f}  "
      f"self_recon {hi.self_recon.mean():.4f}")


def to_png(x):
    return np.clip((np.asarray(x) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def grid(rows, path, pad=2):
    h, w = rows[0][0].shape[:2]
    Rn, Cn = len(rows), max(len(r) for r in rows)
    out = np.full((Rn * h + (Rn + 1) * pad, Cn * w + (Cn + 1) * pad, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            out[pad + i * (h + pad):pad + i * (h + pad) + h,
                pad + j * (w + pad):pad + j * (w + pad) + w] = im
    Image.fromarray(out).save(path)
    print(f"  -> {path}")


n = cli.n_show
half = n // 2
imgs_lo, imgs_hi = to_png(gen_pix[lo.idx.values]), to_png(gen_pix[hi.idx.values])
sep = [np.full_like(imgs_lo[0], 255)] * half
grid([list(imgs_lo[:half]), list(imgs_lo[half:n])] + [sep]
     + [list(imgs_hi[:half]), list(imgs_hi[half:n])],
     f"{cli.out}/gen_uq_{cli.dataset}_seed{cli.seed}.png")
print(f"\n  figure: top two rows = LEAST uncertain, bottom two = MOST uncertain")
print(f"  wrote {dest}")
