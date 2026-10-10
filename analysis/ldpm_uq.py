"""Uncertainty methods on the trained latent diffusion model, 50/50 setting.

Produces the same comparison as tables/final/corr_mse.tex -- SLUG against MC-dropout, low-rank
LLA, SWAG and a deep ensemble -- but on a latent diffusion model instead of the convolutional
VAE, which is what the reviewer asked for.

WHY THIS MODEL IS A BETTER TEST OF THE METHOD THAN THE VAE
The GGN in this codebase uses H = I (uncertainty/autodiff.py). For the VAE that is the
Gaussian likelihood term only: the KL and the perceptual term are silently dropped, so the
"Laplace posterior" is of a different objective than the one trained. For the DDPM objective
||eps - eps_theta(z_t,t)||^2, H = I is EXACT. The approximation the original submission had to
defend simply is not present here.

OPERATING POINT
The network's input is z_t, so the datapoint IS z_t -- there is no separate noising step to
make deterministic. The timestep travels as a constant 5th channel (x[...,4] = t/T), because
the SLUG machinery passes a single array per datapoint. During the posterior fit t is drawn
uniformly, matching the training objective the Hessian is supposed to describe; at scoring
time t is fixed so every image is compared at the same operating point.

ERROR TARGET
One-step DDIM, z0_hat = (z_t - sqrt(1-abar) eps_theta) / sqrt(abar), decoded to pixels by the
frozen SD VAE. z0_hat is AFFINE in eps_theta, so the reconstruction's GGN is the denoiser's
times (1-abar)/abar: the uncertainty computed on eps_theta is the uncertainty of the
reconstruction, up to a constant that is identical for every image.

Usage
-----
  python analysis/ldpm_uq.py --dataset fitz --save_dir outputs/ldpm_fitz --methods all
  python analysis/ldpm_uq.py --dataset fitz --save_dir outputs/ldpm_fitz --methods sweep_t
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
import optax
from jax import flatten_util
from flax.training import train_state, checkpoints
from torch.utils.data import Dataset, DataLoader
from diffusers import FlaxAutoencoderKL

import Helpers as hf
import data as _data
from latent_ddpm import LatentUNet, DDPM
from uncertainty.sketched_lanczos_fast import low_memory_lanczos_score_fun_fast

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
p.add_argument("--save_dir", required=True)
p.add_argument("--subset", type=float, default=0.5)
p.add_argument("--run", type=int, default=0)
p.add_argument("--seed", type=int, default=0, help="the model scored by the single-model methods")
p.add_argument("--ens_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--methods", default="all",
               help="all | sweep_t | comma-separated: slug,lla,dropout,ensemble,swag")
p.add_argument("--t", type=int, default=500,
               help="scoring timestep. 500 is the SNR=1 point of the cosine schedule "
                    "(abar=0.492, so abar/(1-abar)=0.97): the latent is half signal, half "
                    "noise. Chosen on that principle rather than by maximising the error "
                    "spread, which would be tuning on the quantity being predicted.")
p.add_argument("--scope", choices=["latent", "pixel", "decoder"], default="latent",
               help="Where the trace is taken. 'latent' differentiates eps_theta only, so "
                    "SLUG sums per-LATENT-dimension variances (1,024 dims). 'pixel' chains "
                    "the frozen SD VAE decoder into the Jacobian, so SLUG sums per-PIXEL "
                    "variances (49,152) exactly as Eq. 8-9 of the paper define it, and the "
                    "score then lives in the same space as the pixel MSE it is correlated "
                    "against, but the decoder acts as a fixed per-image multiplier that the "
                    "posterior does not account for. 'decoder' additionally puts the SD "
                    "decoder's parameters IN the posterior, which is the formulation used for "
                    "the VAE experiments, where the decoder is part of what the posterior "
                    "covers rather than a constant the Jacobian passes through.")
p.add_argument("--out_tag", default="",
               help="suffix for the output CSV, so several --methods runs can go in "
                    "parallel without overwriting one another; merge with merge_ldpm.py")
p.add_argument("--base_ch", type=int, default=128)
p.add_argument("--image_size", type=int, default=128)
p.add_argument("--train_size", type=int, default=None)
p.add_argument("--test_size", type=int, default=512)
p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
# posterior
p.add_argument("--fit_images", type=int, default=1000)
p.add_argument("--fit_batch_size", type=int, default=8)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--hutchinson_samples", type=int, default=500)
# sampling baselines
p.add_argument("--mc_samples", type=int, default=20)
p.add_argument("--swag_samples", type=int, default=20)
p.add_argument("--swa_steps", type=int, default=2000)
p.add_argument("--swa_lr", type=float, default=1e-3)
p.add_argument("--swag_collect_interval", type=int, default=20)
p.add_argument("--swag_n_vec", type=int, default=1,
               help="K deviation columns. Memory is (2+K)p, so K=1 is the 3p SWAG the "
                    "VAE tables report.")
cli = p.parse_args()

if cli.train_size is None:
    cli.train_size = 1668 if cli.dataset == "fitz" else 2000
PERC = str(int(cli.subset)) if cli.subset in (0.0, 1.0) else str(cli.subset)
LAT = os.path.join(cli.save_dir, "latents", f"{cli.dataset}_p{PERC}_r{cli.run}.npz")
T_TOTAL = 1000
ddpm = DDPM(T=T_TOTAL)
ABAR = np.asarray(ddpm.abar)
ABAR_J = jnp.asarray(ABAR)
METH = ([m for m in "slug lla dropout ensemble swag".split()]
        if cli.methods == "all" else cli.methods.split(","))


def cell(seed, dropout=False):
    return os.path.abspath(os.path.join(cli.save_dir, "Results", str(cli.run), PERC,
                                        f"seed{seed}" + ("_dropout" if dropout else "")))


def load_params(seed, dropout=False):
    d = checkpoints.restore_checkpoint(ckpt_dir=cell(seed, dropout), target=None)
    if d is None:
        raise SystemExit(f"no checkpoint in {cell(seed, dropout)}")
    return jax.tree_util.tree_map(jnp.asarray, d["params"])


Z = np.load(LAT, allow_pickle=True)
SCALE = float(Z["scale"])
GROUPS = [(str(Z["group1"]), Z["ztest1"], Z["names1"]),
          (str(Z["group2"]), Z["ztest2"], Z["names2"])]
ztrain = np.concatenate([Z["ztrain"], Z["ztrain_flip"]])
C = Z["ztest1"].shape[-1]
ZSH = Z["ztest1"].shape
print(f"{cli.dataset} p={PERC} run={cli.run} | groups: "
      f"{GROUPS[0][0]} ({len(GROUPS[0][1])}), {GROUPS[1][0]} ({len(GROUPS[1][1])})", flush=True)


# ----------------------------------------------------------------- model + adapter
unet = LatentUNet(out_ch=C, base_ch=cli.base_ch, dropout=0.0)
unet_do = LatentUNet(out_ch=C, base_ch=cli.base_ch, dropout=0.1)


class DDPMAdapter:
    """Exposes eps_theta in the (variables, x)->(out,) shape the SLUG machinery expects.

    x is [..., C+1]: channels :C carry z_t, the last channel is a constant plane holding
    t/T. Packing t into the array is what lets a per-datapoint timestep travel through a
    DataLoader that yields one array per example.
    """

    def __init__(self, net):
        self.net = net

    def apply(self, variables, x, rng=None, train=False, **kw):
        z, t = x[..., :C], x[..., 0, 0, C]
        t = jnp.round(t * T_TOTAL).astype(jnp.int32)
        return (self.net.apply({"params": variables["params"]}, z, t, train=False),)


class DDPMPixelAdapter:
    """z_t -> eps_theta -> one-step z0_hat -> FROZEN SD VAE decode -> pixels.

    The posterior still covers ONLY the UNet parameters: the VAE weights enter as closure
    constants, so jax.vjp(f, params) back-propagates THROUGH the decoder but never with
    respect to it. Chaining a frozen deterministic map adds no parameter uncertainty; it
    only re-expresses the same uncertainty in pixel units.

    Why this matters: the latent adapter takes the trace over 1,024 latent dimensions, while
    the error it is correlated against (pixel MSE/SSIM/LPIPS) is measured after decoding. The
    paper defines SLUG as the sum of per-PIXEL predictive variances, so this variant is the
    faithful one -- at the cost of a 48x larger output dimension and a VJP through an 83.7M
    parameter decoder.
    """

    def apply(self, variables, x, rng=None, train=False, **kw):
        z, tt = x[..., :C], x[..., 0, 0, C]
        ti = jnp.round(tt * T_TOTAL).astype(jnp.int32)
        eps = unet.apply({"params": variables["params"]}, z, ti, train=False)
        a = ABAR_J[ti][:, None, None, None]
        z0h = (z - jnp.sqrt(1 - a) * eps) / jnp.sqrt(a)
        rec = vae.apply({"params": vae_p}, z0h / SCALE, method=vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


class DDPMDecoderAdapter:
    """z_t -> eps_theta -> z0_hat -> SD VAE decode -> pixels, with the DECODER PARAMETERS IN
    THE POSTERIOR.

    This is the formulation the VAE experiments use, where the trace is over pixels and the
    decoder's weights are part of what the Laplace approximation is uncertain about. The
    'pixel' scope differentiates through the decoder but keeps its weights fixed, so the
    decoder contributes a per-image multiplicative factor that the posterior never sees; that
    factor turns out to be a measure of image texture, which predicts reconstruction error in
    some subgroups and not others, and it therefore helps or hurts depending on the subgroup.
    Including the weights removes that asymmetry, because the decoder's contribution is
    projected by (I - U U^T) like everything else.

    Differentiated: the UNet (27.2M) plus the decoder and post_quant_conv (49.5M). The encoder
    is not on the decode path at all and stays frozen in the closure.
    """

    def __init__(self, frozen_vae):
        self.frozen = frozen_vae

    def apply(self, variables, x, rng=None, train=False, **kw):
        p = variables["params"]
        z, tt = x[..., :C], x[..., 0, 0, C]
        ti = jnp.round(tt * T_TOTAL).astype(jnp.int32)
        eps = unet.apply({"params": p["unet"]}, z, ti, train=False)
        a = ABAR_J[ti][:, None, None, None]
        z0h = (z - jnp.sqrt(1 - a) * eps) / jnp.sqrt(a)
        vp = {**self.frozen, "decoder": p["decoder"],
              "post_quant_conv": p["post_quant_conv"]}
        rec = vae.apply({"params": vp}, z0h / SCALE, method=vae.decode).sample
        return (jnp.transpose(rec, (0, 2, 3, 1)),)


def pack(zt, t):
    """[N,H,W,C] + scalar-or-array t -> [N,H,W,C+1] with the t/T plane appended."""
    t = np.broadcast_to(np.asarray(t, np.float32).reshape(-1, 1, 1, 1) / T_TOTAL,
                        zt.shape[:-1] + (1,))
    return np.concatenate([zt, t], axis=-1).astype(np.float32)


def noise_at(z0, t, seed=0):
    """z_t with per-image independent noise, fixed by `seed` so every method sees the same
    operating point -- otherwise a method could look better purely by drawing easier noise."""
    eps = np.asarray(jax.random.normal(jax.random.PRNGKey(seed), z0.shape), np.float32)
    t = np.asarray(t, np.float32).reshape(-1, 1, 1, 1)
    a = ABAR[np.asarray(t, np.int32).reshape(-1)].reshape(-1, 1, 1, 1)
    return (np.sqrt(a) * z0 + np.sqrt(1 - a) * eps).astype(np.float32), eps


def z0_hat(eps_pred, zt, t):
    a = ABAR[np.asarray(t, np.int32).reshape(-1)].reshape(-1, 1, 1, 1)
    return (zt - np.sqrt(1 - a) * eps_pred) / np.sqrt(a)


@jax.jit
def _fwd(prm, x):
    return unet.apply({"params": prm}, x[..., :C],
                      jnp.round(x[..., 0, 0, C] * T_TOTAL).astype(jnp.int32), train=False)


def predict(prm, x, bs=32):
    return np.concatenate([np.asarray(_fwd(prm, jnp.asarray(x[i:i + bs])))
                           for i in range(0, len(x), bs)])


# ----------------------------------------------------------------- sweep_t
if cli.methods == "sweep_t":
    prm = load_params(cli.seed)
    print(f"\n{'t':>6s} {'abar':>7s} {'latent MSE':>11s} {'cv':>7s}  "
          f"(cv is the spread ACROSS images: the signal any score must rank)")
    for t in [50, 100, 200, 300, 400, 500, 600, 700]:
        line = f"{t:>6d} {ABAR[t]:>7.3f}"
        for gname, z0, _ in GROUPS:
            zt, _ = noise_at(z0, np.full(len(z0), t))
            e = ((z0_hat(predict(prm, pack(zt, np.full(len(z0), t))), zt,
                         np.full(len(z0), t)) - z0) ** 2).mean(axis=(1, 2, 3))
            line += f" | {gname[:9]:>9s} {e.mean():>9.4f} {e.std()/e.mean():>6.3f}"
        print(line, flush=True)
    sys.exit(0)


# ----------------------------------------------------------------- error target
print(f"\nscoring at t={cli.t} (abar={ABAR[cli.t]:.3f})", flush=True)
vae, vae_p = FlaxAutoencoderKL.from_pretrained(cli.vae, from_pt=True, dtype=jnp.float32)


@jax.jit
def decode(z):
    out = vae.apply({"params": vae_p}, z / SCALE, method=vae.decode).sample
    return jnp.transpose(out, (0, 2, 3, 1))


def raw_images():
    """The ORIGINAL pixels for the two test groups, re-read through the same split call the
    latents were built from, so image i of the CSV is image i of the latent cache."""
    class A:
        pass
    a = A()
    a.dataset, a.image_size, a.train_size = cli.dataset, cli.image_size, cli.train_size
    a.dataset_root = "/work/work_mik/2026Diff_UQ/FitzFairness/data/images/"
    a.ai4skin_patches_csv = "/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv"
    a.ai4skin_patches_root = "/work/work_mik/2026Diff_UQ/FusoFairness/data/"
    root = a.dataset_root if cli.dataset == "fitz" else a.ai4skin_patches_root
    _, t1, t2, _, _ = _data.train_test_setting(root, cli.subset, cli.run, cli.test_size, a)
    out = []
    for ds, cached in ((t1, Z["names1"]), (t2, Z["names2"])):
        ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=hf.numpy_collate)
        xs, ids = [], []
        for x, nm in ld:
            xs.append(np.asarray(x, np.float32))
            ids.extend([str(s) for s in nm])
        # The pixels and the latents come from two separate passes over the split. If the
        # order ever diverged, every per-image metric would be joined to the wrong image and
        # the correlations would silently go to zero.
        assert list(ids) == [str(s) for s in cached], "test-set order drifted from the cache"
        out.append(np.concatenate(xs))
    return out


IMGS = raw_images()
prm0 = load_params(cli.seed)

import lpips as lpips_lib, torch
from skimage.metrics import structural_similarity as ssim_fn
lp = lpips_lib.LPIPS(net="vgg")

RESULTS = {}
for gi, (gname, z0, names) in enumerate(GROUPS):
    tarr = np.full(len(z0), cli.t)
    zt, _ = noise_at(z0, tarr)
    rec_lat = z0_hat(predict(prm0, pack(zt, tarr)), zt, tarr)
    rec_pix = np.concatenate([np.asarray(decode(jnp.asarray(rec_lat[i:i + 16])))
                              for i in range(0, len(rec_lat), 16)])
    x = IMGS[gi]
    a = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()
    b = torch.from_numpy(np.transpose(rec_pix, (0, 3, 1, 2))).float()
    with torch.no_grad():
        lv = np.concatenate([lp(a[i:i + 32], b[i:i + 32]).squeeze().cpu().numpy().reshape(-1)
                             for i in range(0, len(a), 32)])
    RESULTS[gname] = pd.DataFrame(dict(
        file=[str(s) for s in names], group=gname,
        latent_mse=((rec_lat - z0) ** 2).mean(axis=(1, 2, 3)),
        mse=((rec_pix - x) ** 2).mean(axis=(1, 2, 3)),
        ssim=[ssim_fn(x[i], rec_pix[i], channel_axis=2, data_range=2.0) for i in range(len(x))],
        lpips=lv))
    print(f"  {gname}: latent MSE {RESULTS[gname].latent_mse.mean():.4f} | "
          f"pixel MSE {RESULTS[gname].mse.mean():.4f} | LPIPS {lv.mean():.4f}", flush=True)


# ----------------------------------------------------------------- SLUG / LLA
class LatentDS(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], i


def fit_and_score(sketch_size, label, n_iter=None, n_vec=None):
    """sketch_size=None is the dense low-rank LLA; an int is SLUG's sketched posterior.

    The two are compared on MEMORY, not rank, which is the axis the SLU paper uses. SLUG
    stores n_vec x sketch_size floats; dense LLA stores n_vec x n_params. At 27.2M params,
    SLUG's 90 x 10,000 = 0.033p leaves the dense competitor rank 0, so LLA is given rank 1
    (= 1p) -- a 30x memory ADVANTAGE, the same handicap the VAE table reports."""
    n_iter = cli.lanczos_iterations if n_iter is None else n_iter
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ztrain), min(cli.fit_images, len(ztrain)), replace=False)
    # t ~ U[0,T) over the FIT set: the Hessian must describe the objective actually trained,
    # which averages over timesteps. A single-t fit would describe a different loss.
    t_fit = rng.integers(0, T_TOTAL, len(idx))
    zt_fit, _ = noise_at(ztrain[idx], t_fit, seed=7)
    loader = DataLoader(LatentDS(pack(zt_fit, t_fit)), batch_size=cli.fit_batch_size,
                        shuffle=False, num_workers=0, collate_fn=hf.numpy_collate,
                        drop_last=True)
    t0 = time.time()
    if cli.scope == "decoder":
        # diffusers' from_pretrained returns arrays already COMMITTED to the host
        # (TFRT_CPU_0), while the UNet checkpoint is on the GPU, and ravel_pytree then tries
        # to concatenate across platforms. jnp.asarray and a bare jax.device_put are both
        # no-ops on a committed array: the device has to be named explicitly.
        vdev = jax.device_put(vae_p, jax.devices()[0])
        adapter = DDPMDecoderAdapter({"encoder": vdev["encoder"],
                                      "quant_conv": vdev["quant_conv"]})
        theta = {"unet": jax.device_put(prm0), "decoder": vdev["decoder"],
                 "post_quant_conv": vdev["post_quant_conv"]}
    elif cli.scope == "pixel":
        adapter, theta = DDPMPixelAdapter(), prm0
    else:
        adapter, theta = DDPMAdapter(unet), prm0
    out_shape = ((ZSH[1], ZSH[2], C) if cli.scope == "latent"
                 else (cli.image_size, cli.image_size, 3))
    npar = sum(x.size for x in jax.tree_util.tree_leaves(theta))
    print(f"  scope={cli.scope}: {npar:,} parameters in the posterior", flush=True)
    fns = low_memory_lanczos_score_fun_fast(
        adapter, {"params": theta, "batch_stats": {}}, loader,
        output_shape=out_shape,
        sketch_size=sketch_size, lanczos_iterations=n_iter, lanczos_eigenvectors=n_vec,
        seed=1, train_batch_size=cli.fit_batch_size, subsample_trainset=len(idx),
        hutchinson_samples=cli.hutchinson_samples)
    print(f"  {label} fit [{cli.scope}, output dim "
          f"{int(np.prod(out_shape)):,}]: {(time.time()-t0)/60:.1f} min", flush=True)
    for gname, zg, _ in GROUPS:
        tarr = np.full(len(zg), cli.t)
        ztg, _ = noise_at(zg, tarr)
        t1 = time.time()
        s = np.asarray(fns["approx"](pack(ztg, tarr), jax.random.PRNGKey(0),
                                     cli.hutchinson_samples))
        RESULTS[gname][label] = s
        print(f"  {label} {gname}: {(time.time()-t1)/60:.1f} min, "
              f"{(time.time()-t1)/len(zg):.2f} s/img", flush=True)
    jax.clear_caches()


if "slug" in METH:
    print("\n--- SLUG ---", flush=True)
    fit_and_score(cli.sketch_size, "slug")
if "lla" in METH:
    print("\n--- low-rank LLA, memory-matched (rank 1 = 1p) ---", flush=True)
    fit_and_score(None, "lla", n_iter=2, n_vec=1)


# ----------------------------------------------------------------- sampling baselines
def sample_variance(param_list, label, dropout_rngs=None):
    """Variance of the one-step reconstruction across a set of weight samples, averaged over
    latent dimensions -- the same scalarisation ensemble_var.py uses for the VAE."""
    for gname, zg, _ in GROUPS:
        tarr = np.full(len(zg), cli.t)
        ztg, _ = noise_at(zg, tarr)
        xg = pack(ztg, tarr)
        preds = []
        for m, prm in enumerate(param_list):
            if dropout_rngs is None:
                e = predict(prm, xg)
            else:
                e = np.concatenate([
                    np.asarray(unet_do.apply({"params": prm}, jnp.asarray(xg[i:i + 32, ..., :C]),
                                             jnp.full((len(xg[i:i + 32]),), cli.t, jnp.int32),
                                             train=True,
                                             rngs={"dropout": jax.random.PRNGKey(dropout_rngs[m])}))
                    for i in range(0, len(xg), 32)])
            preds.append(z0_hat(e, ztg, tarr))
        RESULTS[gname][label] = np.stack(preds).var(axis=0).mean(axis=(1, 2, 3))
    print(f"  {label}: {len(param_list)} samples", flush=True)


if "ensemble" in METH:
    print("\n--- deep ensemble ---", flush=True)
    sample_variance([load_params(s) for s in cli.ens_seeds], "ensemble")

if "dropout" in METH:
    print("\n--- MC-dropout ---", flush=True)
    pdo = load_params(cli.seed, dropout=True)
    sample_variance([pdo] * cli.mc_samples, "dropout",
                    dropout_rngs=list(range(cli.mc_samples)))

if "swag" in METH:
    print("\n--- SWAG ---", flush=True)
    # Constant-LR SGD from the trained weights: the stationary spread of the iterates is what
    # SWAG fits a Gaussian to. Snapshots every `swag_collect_interval` STEPS, as in Maddox.
    class TS(train_state.TrainState):
        pass

    st = TS.create(apply_fn=unet.apply, params=prm0,
                   tx=optax.chain(optax.clip(1.0), optax.sgd(cli.swa_lr, momentum=0.99)))

    @jax.jit
    def swa_step(st, z0b, key):
        k_t, k_e = jax.random.split(key)
        t = jax.random.randint(k_t, (z0b.shape[0],), 0, T_TOTAL)
        eps = jax.random.normal(k_e, z0b.shape)
        a = ABAR_J[t][:, None, None, None]
        ztb = jnp.sqrt(a) * z0b + jnp.sqrt(1 - a) * eps
        loss, g = jax.value_and_grad(
            lambda prm: jnp.mean((unet.apply({"params": prm}, ztb, t, train=False) - eps) ** 2)
        )(st.params)
        return st.apply_gradients(grads=g), loss

    flat0, unravel = flatten_util.ravel_pytree(prm0)
    run_sum = jnp.zeros_like(flat0)
    run_sq = jnp.zeros_like(flat0)
    D, n_snap = [], 0
    ztr_j = jnp.asarray(ztrain)
    key = jax.random.PRNGKey(123)
    for s in range(cli.swa_steps):
        key, kb, ks = jax.random.split(key, 3)
        idx = jax.random.randint(kb, (64,), 0, ztr_j.shape[0])
        st, loss = swa_step(st, ztr_j[idx], ks)
        if (s + 1) % cli.swag_collect_interval == 0:
            f, _ = flatten_util.ravel_pytree(st.params)
            run_sum, run_sq, n_snap = run_sum + f, run_sq + f ** 2, n_snap + 1
            D.append(np.asarray(f - run_sum / n_snap))
            if len(D) > cli.swag_n_vec:
                D.pop(0)
    mean = run_sum / n_snap
    std = jnp.sqrt(jnp.maximum(run_sq / n_snap - mean ** 2, 0.0))
    Dm = jnp.asarray(np.stack(D, axis=1))
    K = Dm.shape[1]
    print(f"  {n_snap} snapshots, K={K}, memory (2+{K})p; "
          f"iterate std mean={float(std.mean()):.2e}", flush=True)
    kk = jax.random.PRNGKey(7)
    samples = []
    for _ in range(cli.swag_samples):
        kk, k1, k2 = jax.random.split(kk, 3)
        # Maddox full SWAG: theta = theta_SWA + (1/sqrt2) Sigma_d^{1/2} z1
        #                                      + (1/sqrt(2(K-1))) D z2
        # The low-rank term is undefined at K=1, so K=1 is diagonal SWAG -- the same guard
        # analysis/swag_experiment.py uses, and the same nv=1 (3p) setting the VAE tables
        # report, so the two SWAG rows mean the same thing.
        w = mean + std * jax.random.normal(k1, mean.shape) / np.sqrt(2)
        if K > 1:
            w = w + (Dm @ jax.random.normal(k2, (K,))) / np.sqrt(2 * (K - 1))
        samples.append(unravel(w))
    sample_variance(samples, "swag")


# ----------------------------------------------------------------- write
os.makedirs("tables", exist_ok=True)
out = pd.concat(list(RESULTS.values()), ignore_index=True)
dest = (f"tables/ldpm_{cli.dataset}_p{PERC}_r{cli.run}_t{cli.t}"
        + (f"_{cli.out_tag}" if cli.out_tag else "") + ".csv")
out.to_csv(dest, index=False)
print(f"\nwrote {dest} ({len(out)} rows)")

from scipy.stats import pearsonr, spearmanr
meth_cols = [c for c in out.columns
             if c in ("slug", "lla", "dropout", "ensemble", "swag")]
for gname, df in RESULTS.items():
    print(f"\n=== {gname} ===")
    print(f"  {'method':>10s} " + " ".join(f"{m:>16s}" for m in ["mse", "ssim", "lpips"]))
    for mcol in meth_cols:
        cells = []
        for em in ["mse", "ssim", "lpips"]:
            cells.append(f"r={pearsonr(df[mcol], df[em])[0]:+.3f}/"
                         f"{spearmanr(df[mcol], df[em]).correlation:+.3f}")
        print(f"  {mcol:>10s} " + " ".join(f"{c:>16s}" for c in cells))
