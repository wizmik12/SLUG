"""
Dense low-rank Linearized Laplace (`low_rank_lla` in Miani et al., "Sketched Lanczos
uncertainty score", NeurIPS 2024), run at a MEMORY BUDGET matched to SLUG.

Why this and not the matched-rank comparison: the SLU paper matches methods on *memory*
(budget fixed at 3p, with a 10p appendix), not on rank. That is the whole point of the
method -- sketching buys more rank per byte, "trading sketching noise for better spectral
coverage". Matching rank instead would discard exactly the advantage SLU is designed to
have, so it is the wrong axis for a faithful comparison.

Accounting (persistent uncertainty summary, float32):
    dense low-rank LLA : n_eigenvec * n_params        floats
    SLUG (sketched)    : n_eigenvec * sketch_size     floats
With n_params=50,636,355 and sketch_size=5,000,000, SLUG's shipped config
(lanczos_iterations=10 -> 9 kept eigenvectors) costs 9*5e6 = 45e6 floats ~ 0.89p, so its
memory-matched dense competitor is rank ~1. Miani's headline 3p budget gives dense rank 3
against SLUG rank ~30.

Note both `--lanczos_iterations` and `--lanczos_eigenvectors` are exposed: the stored summary
is (n_params x lanczos_eigenvectors), while lanczos_iterations only affects transient fit
cost. Lanczos with n_iter=1 degenerates (empty off-diagonal), so use n_iter >= 2 and truncate.

Usage: python analysis/low_rank_lla_memory_matched.py --results_dir <...> \
    --dataset fitz --seed 0 --lanczos_iterations 2 --lanczos_eigenvectors 1
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import numpy as np
from flax.training import checkpoints
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

import Helpers as hf
import data
from VAE import VAE
from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun


class ArgsShim:
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_dir", type=str, default=None,
                   help="Results/<run_dir>/<subset>/ . Default None -> '0_seed{seed}' "
                        "(the init-varying checkpoints). The bias sweep uses '0'..'9' "
                        "(split-varying, main.py:66 with seed=42).")
    p.add_argument("--subset", type=str, default="0.5",
                   help="training-composition cell: 0 | 0.5 | 1")
    p.add_argument("--nepoch", type=int, default=500)
    p.add_argument("--lanczos_iterations", type=int, default=2)
    p.add_argument("--lanczos_eigenvectors", type=int, default=1)
    p.add_argument("--sketch_size", type=int, default=0,
                   help="0 = DENSE (low_rank_lla baseline, eigenvectors at full dimension). "
                        ">0 = SKETCHED (this is SLUG). Miani's slu.sh uses 10000; our shipped "
                        "config uses 5000000.")
    p.add_argument("--hutchinson_samples", type=int, default=500)
    p.add_argument("--test_size", type=int, default=512)
    p.add_argument("--score_batch_size", type=int, default=4)
    p.add_argument("--fit_train_size", type=int, default=1668)
    p.add_argument("--fit_batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--ch_multi", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=1)
    p.add_argument("--latent_channels", type=int, default=128)
    p.add_argument("--block_widths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dataset_root", type=str, default="/work/work_mik/2026Diff_UQ/FitzFairness/data/images/")
    p.add_argument("--ai4skin_patches_csv", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv")
    p.add_argument("--ai4skin_patches_root", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/")
    cli = p.parse_args()

    args = ArgsShim()
    args.dataset = cli.dataset
    args.dataset_root = cli.dataset_root
    args.ai4skin_patches_csv = cli.ai4skin_patches_csv
    args.ai4skin_patches_root = cli.ai4skin_patches_root
    args.image_size = cli.image_size
    args.train_size = cli.fit_train_size

    # The split MUST match the checkpoint being scored. main.py:76 passes (perc_attr, run)
    # into train_test_setting, so a checkpoint at Results/<run>/<subset>/ was trained and
    # evaluated on the split defined by exactly those two values. Hardcoding (0.5, 0) here
    # silently scores the wrong images: IDs then fail to join against metrics_<group>.csv and
    # most rows become NaN (observed as n=32 / n=110 instead of 512, and as a scipy
    # "length at least 2" crash where the overlap was empty).
    perc_attr = float(cli.subset)
    perc_attr = int(perc_attr) if perc_attr in (0.0, 1.0) else perc_attr   # mirror main.py:52
    split_run = int(cli.run_dir) if (cli.run_dir is not None and cli.run_dir.isdigit()) else 0
    print(f"  split: perc_attr={perc_attr} run={split_run} (must match the checkpoint)")
    train_dataset, test_a1, test_a2, attr1, attr2 = data.train_test_setting(
        cli.dataset_root, perc_attr, split_run, cli.test_size, args)
    train_loader = DataLoader(train_dataset, batch_size=cli.fit_batch_size, shuffle=True,
                              num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)

    model = VAE(channel_in=3, ch=cli.ch_multi, blocks=cli.block_widths,
                latent_channels=cli.latent_channels, num_res_blocks=cli.num_res_blocks,
                deep_model=False, use_dropout=False)
    save_dir = os.path.join(cli.results_dir, cli.dataset)
    run_dir = cli.run_dir if cli.run_dir is not None else f"0_seed{cli.seed}"
    base = os.path.join(save_dir, "Results", run_dir, cli.subset)
    ckpt = os.path.join(base, "model",
                        f"fitz17_{cli.latent_channels}_{cli.image_size}{cli.nepoch}")
    dummy = np.zeros((1, cli.image_size, cli.image_size, 3), dtype=np.float32)
    _ = model.init(jax.random.PRNGKey(0), dummy, train=True, rng=jax.random.PRNGKey(0))
    sd = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=None)
    params_dict = {"params": sd["params"], "batch_stats": sd["batch_stats"]}

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params_dict["params"]))
    stored = cli.lanczos_eigenvectors * n_params
    print(f"=== dense low-rank LLA: {cli.lanczos_eigenvectors} eigenvector(s) x {n_params} params "
          f"= {stored/1e6:.1f}e6 floats = {stored/n_params:.2f}p = {stored*4/2**30:.2f} GiB ===")

    sketch = None if cli.sketch_size <= 0 else cli.sketch_size
    kind = "DENSE low_rank_lla" if sketch is None else f"SKETCHED SLUG (sketch={sketch:,})"
    stored_ours = cli.lanczos_eigenvectors * (n_params if sketch is None else sketch)
    print(f"  variant: {kind}; stored summary = {cli.lanczos_eigenvectors} x "
          f"{n_params if sketch is None else sketch:,} = {stored_ours/n_params:.4f}p "
          f"= {stored_ours*4/2**20:.1f} MiB")
    fns = low_memory_lanczos_score_fun(
        model, params_dict, train_loader,
        output_shape=(cli.image_size, cli.image_size, 3),
        sketch_size=sketch,
        lanczos_iterations=cli.lanczos_iterations,
        lanczos_eigenvectors=cli.lanczos_eigenvectors,
        train_batch_size=cli.fit_batch_size,
        subsample_trainset=cli.fit_train_size,
    )
    score_fn = fns["approx"]

    for group, test_ds in [(attr1, test_a1), (attr2, test_a2)]:
        loader = DataLoader(test_ds, batch_size=cli.score_batch_size, shuffle=False,
                            num_workers=0, collate_fn=hf.numpy_collate)
        key = jax.random.PRNGKey(0)
        ids, scores = [], []
        for imgs, batch_ids in loader:
            key, sub = jax.random.split(key)
            # vmap'd with in_axes=(0, None, None): all three args must be positional
            s = np.asarray(score_fn(np.asarray(imgs), sub, cli.hutchinson_samples))
            scores.extend([float(x) for x in s])
            ids.extend(list(batch_ids))
        scores = np.asarray(scores)

        mse_by_id = {}
        mfile = os.path.join(base, f"metrics_{group}.csv")
        if os.path.exists(mfile):
            with open(mfile) as f:
                for row in csv.DictReader(f):
                    mse_by_id[row["Image Name"]] = float(row["MSE"])
        mse = np.array([mse_by_id.get(i, np.nan) for i in ids])
        ok = ~np.isnan(mse)
        print(f"  {cli.dataset}/{group}: n={ok.sum()} "
              f"pearson={pearsonr(scores[ok], mse[ok])[0]:.4f} "
              f"spearman={spearmanr(scores[ok], mse[ok])[0]:.4f}")

        tag = (f"lowranklla_k{cli.lanczos_eigenvectors}" if sketch is None
               else f"slug_it{cli.lanczos_iterations}_sk{cli.sketch_size}")
        out = os.path.join(base, f"uq_metrics_{tag}_{group}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Image Name", "Error", "Uncertainty"])
            for i, img_id in enumerate(ids):
                w.writerow([img_id, mse[i], scores[i]])
        print(f"  wrote {out}")
