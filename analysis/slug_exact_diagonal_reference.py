"""Exact per-pixel diagonal, one vector-Jacobian product per output, as a reference without sampling noise.

Usage:
  python analysis/slug_exact_diagonal_reference.py --results_dir ../outputs/uq_baselines_performance \
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import numpy as np
import pandas as pd
from flax.training import checkpoints
from torch.utils.data import DataLoader

import Helpers as hf
import data
from VAE import VAE
from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun


class ArgsShim:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
    parser.add_argument("--nepoch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_images", type=int, default=1, help="how many of the drawn images to compute the exact diagonal for -- expensive, keep small")
    parser.add_argument("--test_size_draw", type=int, default=32, help="must match slug_sample_convergence.py's --n_images so the same images get drawn")
    parser.add_argument("--fit_train_size", type=int, default=1668)
    parser.add_argument("--fit_train_batch_size", type=int, default=8)
    parser.add_argument("--lanczos_iterations", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--ch_multi", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--block_widths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dataset_root", type=str, default="/work/work_mik/2026Diff_UQ/FitzFairness/data/images/")
    parser.add_argument("--ai4skin_patches_csv", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv")
    parser.add_argument("--ai4skin_patches_root", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/")
    parser.add_argument("--out_csv", type=str, default=None)
    cli = parser.parse_args()

    args = ArgsShim()
    args.dataset = cli.dataset
    args.dataset_root = cli.dataset_root
    args.ai4skin_patches_csv = cli.ai4skin_patches_csv
    args.ai4skin_patches_root = cli.ai4skin_patches_root
    args.image_size = cli.image_size
    args.train_size = cli.fit_train_size

    train_dataset, test_a1, test_a2, attr1, attr2 = data.train_test_setting(
        cli.dataset_root, 0.5, 0, cli.test_size_draw, args,
    )
    train_loader = DataLoader(train_dataset, batch_size=cli.fit_train_batch_size, shuffle=True,
                               num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)
    test_loader = DataLoader(test_a1, batch_size=cli.test_size_draw, shuffle=False,
                              num_workers=0, collate_fn=hf.numpy_collate)

    model = VAE(channel_in=3, ch=cli.ch_multi, blocks=cli.block_widths, latent_channels=cli.latent_channels,
                num_res_blocks=cli.num_res_blocks, deep_model=False, use_dropout=False)

    result_dir = cli.dataset
    ckpt_dir = os.path.join(cli.results_dir, result_dir, "Results", f"0_seed{cli.seed}", "0.5", "model",
                             f"fitz17_{cli.latent_channels}_{cli.image_size}{cli.nepoch}")
    dummy = np.zeros((1, cli.image_size, cli.image_size, 3), dtype=np.float32)
    _ = model.init(jax.random.PRNGKey(0), dummy, train=True, rng=jax.random.PRNGKey(0))
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    params_dict = {"params": state_dict["params"], "batch_stats": state_dict["batch_stats"]}

    print(f"=== Fitting SLUG (full model, standard scope, same config as slug_sample_convergence.py) ===")
    output_shape = (cli.image_size, cli.image_size, 3)
    uncertainty_functions = low_memory_lanczos_score_fun(
        model, params_dict, train_loader, output_shape=output_shape,
        lanczos_iterations=cli.lanczos_iterations, train_batch_size=cli.fit_train_batch_size,
        subsample_trainset=cli.fit_train_size,
    )
    exact_fn = uncertainty_functions["exact"]  # per-pixel diagonal, output_shape, no sampling

    test_imgs, test_ids = next(iter(test_loader))
    test_imgs = np.asarray(test_imgs)[:cli.n_images]
    test_ids = list(test_ids[:cli.n_images])
    print(f"Computing EXACT per-pixel diagonal for {len(test_imgs)} image(s) "
          f"({np.prod(output_shape)} one-hot JVPs each -- this is slow)")

    import time
    rows = []
    for i, (img, img_id) in enumerate(zip(test_imgs, test_ids)):
        start = time.time()
        diag_map = np.asarray(exact_fn(img[None]))[0]  # (H, W, C)
        exact_trace = float(diag_map.sum())
        elapsed = time.time() - start
        print(f"  image {i} ({img_id}): exact_trace={exact_trace:.6g}, took {elapsed:.1f}s")
        rows.append({"image_id": img_id, "exact_trace": exact_trace, "time_s": elapsed})

    if cli.out_csv:
        pd.DataFrame(rows).to_csv(cli.out_csv, index=False)
        print(f"wrote {cli.out_csv}")
