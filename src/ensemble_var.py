"""
Ensemble variance ('ensemble_var' UQ score), batch1 task 2.1 (revision_part1.md Sec 4).

For a dataset, loads all 5 seed checkpoints (a genuine M=5 deep ensemble: same frozen
split across seeds via --run=0, only initialisation/shuffling differs -- see Sec 0.c/1),
reconstructs every test image with all 5 members, computes per-pixel variance across
members, averages to one scalar per image.

Writes a separate ensemble_var_{group}.csv (image_id, ensemble_var) per dataset/group,
rather than injecting a column into each seed's already-written metrics_{group}.csv --
those files are written by 5 independent training processes and read-modify-write races
are not worth the risk. Join on image_id at analysis time instead.

Usage: python ensemble_var.py --dataset fitz --nepoch 500 --results_dir ../outputs/uq_baselines_performance
"""
import os
# Disable XLA's GPU convolution algorithm autotuning: on this GPU/cuDNN combo it picks a
# numerically broken algorithm once batch size exceeds 8, causing forward-pass outputs to
# degenerate (confirmed 2026-08-11: batch_size=16 gave near-constant reconstructions, std
# ~0.09 vs the correct ~0.39, under jax.jit specifically -- eager execution was unaffected).
# Must be set before `import jax`. See main.py's identical fix.
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import csv
import time

import jax
import numpy as np
from flax.training import checkpoints
from torch.utils.data import DataLoader

import Helpers as hf
import data
from VAE import VAE

parser = argparse.ArgumentParser(description="Ensemble variance across batch1 seeds")
parser.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
parser.add_argument("--nepoch", type=int, required=True, help="must match uq_baselines_performance_train.sh's NEPOCH")
parser.add_argument("--results_dir", type=str, required=True, help="path to outputs/uq_baselines_performance")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--ch_multi", type=int, default=64)
parser.add_argument("--num_res_blocks", type=int, default=1)
parser.add_argument("--latent_channels", type=int, default=128)
parser.add_argument("--block_widths", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--test_size", type=int, default=512)
parser.add_argument("--dataset_root", type=str, default="/work/work_mik/2026Diff_UQ/FitzFairness/data/images/")
parser.add_argument("--ai4skin_patches_csv", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv")
parser.add_argument("--ai4skin_patches_root", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/")
cli_args = parser.parse_args()


# Minimal shim so data.train_test_setting (which expects the full argparse-style
# Namespace from Helpers.set_parser) gets the fields it actually reads.
class ArgsShim:
    pass


args = ArgsShim()
args.dataset = cli_args.dataset
args.dataset_root = cli_args.dataset_root
args.ai4skin_patches_csv = cli_args.ai4skin_patches_csv
args.ai4skin_patches_root = cli_args.ai4skin_patches_root
args.image_size = cli_args.image_size
args.train_size = 1668 if cli_args.dataset == "fitz" else 2000  # unused for test-only loading, kept for parity

save_dir = os.path.join(cli_args.results_dir, cli_args.dataset)
perc_attr = 0.5  # config B / mixed, batch1's only arm

# subset=0.5 gives the same frozen split regardless of seed (--run=0 is what's fixed);
# only need one call to get the test sets and group names.
_, test_dataset_attr1, test_dataset_attr2, attr1, attr2 = data.train_test_setting(
    cli_args.dataset_root, perc_attr, 0, cli_args.test_size, args,
)

model = VAE(channel_in=3, ch=cli_args.ch_multi, blocks=cli_args.block_widths,
            latent_channels=cli_args.latent_channels, num_res_blocks=cli_args.num_res_blocks,
            deep_model=False, use_dropout=False)


def load_checkpoint(seed):
    ckpt_dir = os.path.join(save_dir, "Results", f"0_seed{seed}", "0.5", "model",
                             f"fitz17_{cli_args.latent_channels}_{cli_args.image_size}{cli_args.nepoch}")
    if not os.path.isdir(ckpt_dir):
        return None
    # model.init() warmup before restore(target=None): a bare restore as the first JAX/GPU
    # op triggers Orbax's generic-topology fallback and blows up memory (confirmed
    # 2026-08-04, see revision_part1.md/check_diag_convergence.py). Warming up first avoids it.
    dummy = np.zeros((1, cli_args.image_size, cli_args.image_size, 3), dtype=np.float32)
    _ = model.init(jax.random.PRNGKey(0), dummy, train=True, rng=jax.random.PRNGKey(0))
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    return {"params": state_dict["params"], "batch_stats": state_dict["batch_stats"]}


def reconstruct_all(params_dict, loader):
    @jax.jit
    def forward(imgs, params, batch_stats):
        recon, _, _ = model.apply({"params": params, "batch_stats": batch_stats},
                                   x=imgs, rng=jax.random.PRNGKey(0), train=False)
        return recon

    names_all, recons_all = [], []
    for imgs, names in loader:
        recon = forward(imgs, params_dict["params"], params_dict["batch_stats"])
        recons_all.append(np.asarray(recon))
        names_all.extend(names)
    return names_all, np.concatenate(recons_all, axis=0)


def run_group(test_dataset, group_name):
    loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=hf.numpy_collate)

    per_seed_recons = {}
    ref_names = None
    for seed in cli_args.seeds:
        params_dict = load_checkpoint(seed)
        if params_dict is None:
            print(f"  seed {seed}: checkpoint not found, skipping this member")
            continue
        start = time.time()
        names, recons = reconstruct_all(params_dict, loader)
        print(f"  seed {seed}: reconstructed {len(names)} images in {time.time()-start:.1f}s")
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise RuntimeError(
                f"Image order/identity mismatch between seeds for group {group_name} -- "
                f"the test set is supposed to be identical across seeds (same --run=0). "
                f"Check that all checkpoints were trained with the same frozen split."
            )
        per_seed_recons[seed] = recons

    members = list(per_seed_recons.values())
    if len(members) < 2:
        print(f"  group {group_name}: fewer than 2 members available, cannot compute ensemble variance")
        return

    stacked = np.stack(members, axis=0)  # [M, N, H, W, C]
    per_pixel_var = stacked.var(axis=0)  # [N, H, W, C]
    ensemble_var = per_pixel_var.mean(axis=(1, 2, 3))  # [N]

    out_path = os.path.join(save_dir, f"ensemble_var_{group_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "ensemble_var", "n_members"])
        for name, val in zip(ref_names, ensemble_var):
            writer.writerow([name, float(val), len(members)])
    print(f"  wrote {out_path} ({len(ref_names)} images, {len(members)} members)")


print(f"=== {cli_args.dataset}, group {attr1} ===")
run_group(test_dataset_attr1, attr1)
print(f"=== {cli_args.dataset}, group {attr2} ===")
run_group(test_dataset_attr2, attr2)
