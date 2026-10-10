"""Score PASSION with SLUG -- replicating the submitted paper's external-validation protocol.

Protocol (manuscript.tex sec. "External evaluation: PASSION dataset", Fig. slug_passion):
  * 3 training configurations   A = Light (100% light), B = Mixed (50/50), C = Dark (100% dark)
  * 10 independent runs each    -> the 30 checkpoints at fitz/Results/<run>/<subset>/
  * group PASSION by FST 3,4,5,6 and report per-group MSE and SLUG uncertainty
  * box plots over the 10 runs

Differences from the submitted version, both deliberate:
  * the posterior is fitted at the REFERENCE configuration (100 Lanczos iterations, sketch
    10,000, rank 90) rather than the submitted rank 9, matching the rest of the revision;
  * FST groups are balanced by subsampling, and the SAME images are scored by all 30
    checkpoints (random_state=0), so config/run differences are not confounded by which
    images happened to be drawn.

Setup. The VAE is trained on Fitzpatrick17k; the SLUG posterior is fitted on that SAME
training set (the posterior is a property of the model and its training data). PASSION is
then scored purely as held-out test data. No retraining, no refitting per subgroup.

Why PASSION is the right third dataset. It is entirely Fitzpatrick skin types 3-6 -- the end
of the spectrum Fitzpatrick17k is thin on -- so every image is out of distribution for this
model. The question is therefore NOT the ID-vs-OOD separation the reviewers already found
unconvincing, but the finer one: WITHIN a shifted domain, does the uncertainty identify which
subgroups (country, skin type) the model reconstructs worse, with no labels and no ground
truth? Per-image MSE/SSIM/LPIPS are recorded alongside so that question can be answered.

Output: tables/passion_slug_seed{S}.csv with one row per image:
  file, subject_id, country, fitzpatrick, sex, conditions_PASSION, mse, ssim, lpips, uncertainty
"""
import os
# XLA's GPU conv autotuner picks a numerically broken algorithm above batch 8 on this
# GPU/cuDNN combo (see src/ensemble_var.py:17 and main.py -- confirmed 2026-08-11). Without
# this, uncertainties at batch>=16 come out ~100x inflated with INVERTED rankings.
# Must be set before `import jax`.
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import jax
import torch
from torch.utils.data import DataLoader, Dataset
from flax.training import checkpoints
from PIL import Image

import data
import Helpers as hf
from VAE import VAE
from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun

p = argparse.ArgumentParser()
p.add_argument("--passion_dir", default="/work/work_mik/2026/SLUG/PASSION_MICCAI_2024/prepared128")
p.add_argument("--results_dir", default="outputs/uq_baselines_performance")
p.add_argument("--dataset_root", default="/work/work_mik/2026Diff_UQ/FitzFairness/data/images/")
p.add_argument("--run_dir", default="0", help="bias-sweep run 0..9")
p.add_argument("--subset", default="0.5", choices=["0", "0.5", "1"],
               help="training configuration: 0 = A/Light, 0.5 = B/Mixed, 1 = C/Dark")
p.add_argument("--image_size", type=int, default=128)
p.add_argument("--latent_channels", type=int, default=128)
p.add_argument("--ch_multi", type=int, default=64)
p.add_argument("--num_res_blocks", type=int, default=1)
p.add_argument("--nepoch", type=int, default=500)
p.add_argument("--test_size", type=int, default=512)
p.add_argument("--lanczos_iterations", type=int, default=100)
p.add_argument("--lanczos_eigenvectors", type=int, default=90)
p.add_argument("--sketch_size", type=int, default=10000)
p.add_argument("--fit_batch_size", type=int, default=8)
p.add_argument("--fit_train_size", type=int, default=1668)
p.add_argument("--hutchinson_samples", type=int, default=500)
p.add_argument("--score_batch_size", type=int, default=4,
               help="MUST match the in-domain runs (low_rank_lla_memory_matched.py uses 4) "
                    "so PASSION numbers are comparable with fitz/ai4skin. Also stays clear "
                    "of the batch>=16 regime where the XLA conv autotuner picks a wrong "
                    "kernel under memory pressure and reconstructions are silently wrong.")
p.add_argument("--per_fst", type=int, default=480,
               help="balanced images per FST group (480 = all of FST 6, the smallest)")
p.add_argument("--max_images", type=int, default=0, help="if >0, cap total images (debug)")
p.add_argument("--out", default=None)
cli = p.parse_args()


class PassionDS(Dataset):
    def __init__(self, root, files, image_size):
        self.root, self.files = root, files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        im = Image.open(os.path.join(self.root, "images_resized", f)).convert("RGB")
        return hf.image_to_numpy(im), f


man = pd.read_csv(os.path.join(cli.passion_dir, "manifest.csv"))
man = man.dropna(subset=["country", "fitzpatrick"])
man["fitzpatrick"] = man["fitzpatrick"].astype(int)
man = man[man.fitzpatrick.isin([3, 4, 5, 6])]          # paper's Fig.: FST 3-6 (FST 1 has 11 imgs)
if cli.per_fst > 0:
    man = (man.groupby("fitzpatrick", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), cli.per_fst), random_state=0)))
if cli.max_images > 0:
    man = man.head(cli.max_images)
man = man.sort_values("file").reset_index(drop=True)
print(f"scoring {len(man)} PASSION images, balanced over FST 3-6")
print(man.groupby("fitzpatrick").agg(images=("file", "size"),
                                     subjects=("subject_id", "nunique")).to_string())


class ArgsShim:
    pass

args = ArgsShim()
args.dataset = "fitz"
args.dataset_root = cli.dataset_root
args.image_size = cli.image_size
args.train_size = cli.fit_train_size
args.block_widths = (1, 2, 4, 8)

# The posterior MUST be fitted on THIS checkpoint's own training split: main.py passes
# (perc_attr, run) into train_test_setting, so a checkpoint at Results/<run>/<subset>/ was
# trained on exactly that split. Hardcoding (0.5, 0) silently fits the posterior to the
# wrong training data -- the same bug that invalidated an earlier bias scoring pass.
perc_attr = float(cli.subset)
perc_attr = int(perc_attr) if perc_attr in (0.0, 1.0) else perc_attr    # mirror main.py:52
split_run = int(cli.run_dir) if cli.run_dir.isdigit() else 0
print(f"  fit split: perc_attr={perc_attr} run={split_run} (must match the checkpoint)")
train_dataset, _, _, _, _ = data.train_test_setting(
    cli.dataset_root, perc_attr, split_run, cli.test_size, args)
train_loader = DataLoader(train_dataset, batch_size=cli.fit_batch_size, shuffle=True,
                          num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)

model = VAE(channel_in=3, ch=cli.ch_multi, blocks=args.block_widths,
            latent_channels=cli.latent_channels, num_res_blocks=cli.num_res_blocks,
            deep_model=False, use_dropout=False)
base = os.path.join(cli.results_dir, "fitz", "Results", cli.run_dir, cli.subset)
ckpt = os.path.abspath(os.path.join(base, "model",
                                    f"fitz17_{cli.latent_channels}_{cli.image_size}{cli.nepoch}"))
dummy = np.zeros((1, cli.image_size, cli.image_size, 3), dtype=np.float32)
_ = model.init(jax.random.PRNGKey(0), dummy, train=True, rng=jax.random.PRNGKey(0))
sd = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=None)
if sd is None:
    raise SystemExit(f"no checkpoint at {ckpt}")
params_dict = {"params": sd["params"], "batch_stats": sd["batch_stats"]}
print(f"checkpoint: {ckpt}")

t0 = time.time()
fns = low_memory_lanczos_score_fun(
    model, params_dict, train_loader,
    output_shape=(cli.image_size, cli.image_size, 3),
    sketch_size=cli.sketch_size,
    lanczos_iterations=cli.lanczos_iterations,
    lanczos_eigenvectors=cli.lanczos_eigenvectors,
    train_batch_size=cli.fit_batch_size,
    subsample_trainset=cli.fit_train_size,
)
score_fn = fns["approx"]
print(f"FIT done in {(time.time()-t0)/3600:.2f} h")

# ---------------------------------------------------------------- score + metrics
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

lp = lpips_lib.LPIPS(net="vgg")
ds = PassionDS(cli.passion_dir, man["file"].tolist(), cli.image_size)
loader = DataLoader(ds, batch_size=cli.score_batch_size, shuffle=False,
                    num_workers=0, collate_fn=hf.numpy_collate)

key = jax.random.PRNGKey(0)
rows, t0, n, nb = [], time.time(), 0, 0
for imgs, files in loader:
    imgs = np.asarray(imgs)
    key, sub = jax.random.split(key)
    unc = np.asarray(score_fn(imgs, sub, cli.hutchinson_samples))
    recs = np.asarray(model.apply(params_dict, imgs,
                                  rng=jax.random.PRNGKey(0), train=False)[0])
    a = torch.from_numpy(np.transpose(imgs, (0, 3, 1, 2))).float()
    b = torch.from_numpy(np.transpose(recs, (0, 3, 1, 2))).float()
    with torch.no_grad():
        lpv = lp(a * 2 - 1, b * 2 - 1).squeeze().cpu().numpy().reshape(-1)
    for j, f in enumerate(files):
        rows.append(dict(file=f,
                         mse=float(np.mean((imgs[j] - recs[j]) ** 2)),
                         ssim=float(ssim_fn(imgs[j], recs[j], channel_axis=2, data_range=1.0)),
                         lpips=float(lpv[j]),
                         uncertainty=float(unc[j])))
    n += len(files)
    nb += 1
    if nb % 25 == 0:
        # JAX accumulates host-side buffers and compiled-executable caches across repeated
        # calls; over 1920 images this reached ~90 GB per process and exhausted a 503 GB
        # machine when six ran concurrently. src/train_vae.py:349 clears on the same cadence.
        jax.clear_caches()
    if n % (cli.score_batch_size * 5) == 0:
        el = time.time() - t0
        print(f"  {n}/{len(man)}  {el/n:.2f} s/img  eta {(len(man)-n)*el/n/3600:.2f} h", flush=True)

out = pd.DataFrame(rows).merge(man, on="file", how="left")
dest = cli.out or f"tables/passion_slug_run{cli.run_dir}_cfg{cli.subset}.csv"
os.makedirs(os.path.dirname(dest), exist_ok=True)
out.to_csv(dest, index=False)
print(f"\nwrote {dest}  ({len(out)} rows)")

from scipy.stats import pearsonr, spearmanr
for m in ["mse", "ssim", "lpips"]:
    print(f"  SLUG vs {m:5s}: pearson={pearsonr(out.uncertainty, out[m])[0]:+.3f} "
          f"spearman={spearmanr(out.uncertainty, out[m]).correlation:+.3f}")
print("\n-- by FST (the paper's axis) --")
print(out.groupby("fitzpatrick")[["mse", "ssim", "lpips", "uncertainty"]].mean().round(5).to_string())
print("\n-- by country (extra axis PASSION also supports) --")
print(out.groupby("country")[["mse", "uncertainty"]].mean().round(5).to_string())
