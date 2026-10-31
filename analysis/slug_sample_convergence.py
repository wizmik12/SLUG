"""Sweep the number of Hutchinson samples used by the trace estimator.

Usage:
  python analysis/slug_sample_convergence.py --results_dir ../outputs/uq_baselines_performance \
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
from scipy.stats import spearmanr
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
    parser.add_argument("--n_images", type=int, default=48)
    parser.add_argument("--score_batch_size", type=int, default=4,
                         help="images scored per vmap call -- memory scales with this "
                              "(vmap materializes this many parallel backward graphs), "
                              "4 is the validated-safe value from the main pipeline")
    parser.add_argument("--sample_counts", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument("--n_repeats_at_500", type=int, default=3)
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
        cli.dataset_root, 0.5, 0, cli.n_images, args,
    )
    train_loader = DataLoader(train_dataset, batch_size=cli.fit_train_batch_size, shuffle=True,
                               num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)
    test_loader = DataLoader(test_a1, batch_size=cli.n_images, shuffle=False,
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

    print(f"=== Fitting SLUG (full model, standard scope) ===")
    output_shape = (cli.image_size, cli.image_size, 3)
    uncertainty_functions = low_memory_lanczos_score_fun(
        model, params_dict, train_loader, output_shape=output_shape,
        lanczos_iterations=cli.lanczos_iterations, train_batch_size=cli.fit_train_batch_size,
        subsample_trainset=cli.fit_train_size,
    )
    score_fun = uncertainty_functions["approx"]

    def score_in_batches(images, key, n_samples, batch_size=cli.score_batch_size):
        # vmap materializes one backward graph per image in the call; memory scales with
        # batch_size, not len(images) -- score in small chunks, same as the main pipeline's
        # batch_size_uq, to avoid OOM.
        out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            key, subkey = jax.random.split(key)
            out.append(np.asarray(score_fun(chunk, subkey, n_samples)))
        return np.concatenate(out, axis=0)

    test_imgs, test_ids = next(iter(test_loader))
    test_imgs = np.asarray(test_imgs)[:cli.n_images]
    test_ids = list(test_ids[:cli.n_images])
    print(f"Scoring {len(test_imgs)} fixed test images at sample counts {cli.sample_counts}")

    import time as _time
    results = {}
    timings = {}
    key = jax.random.PRNGKey(999)
    for n in cli.sample_counts:
        key, subkey = jax.random.split(key)
        start_time = _time.time()
        scores = score_in_batches(test_imgs, subkey, n)
        elapsed = _time.time() - start_time
        print(f"  N={n}: took {elapsed:.1f}s ({elapsed/len(test_imgs):.3f}s/image), "
              f"mean={scores.mean():.4g}, std={scores.std():.4g}")
        results[f"N{n}"] = scores
        timings[n] = elapsed

    print(f"\n=== Repeating N=500 {cli.n_repeats_at_500}x to measure estimator noise ===")
    repeats_500 = []
    for r in range(cli.n_repeats_at_500):
        key, subkey = jax.random.split(key)
        scores = score_in_batches(test_imgs, subkey, 500)
        repeats_500.append(scores)
        results[f"N500_repeat{r}"] = scores
    repeats_500 = np.stack(repeats_500, axis=0)  # [n_repeats, n_images]
    per_image_repeat_std = repeats_500.std(axis=0)
    per_image_repeat_mean = repeats_500.mean(axis=0)
    across_image_std = repeats_500.mean(axis=0).std()
    print(f"  Mean per-image relative estimator std at N=500 (repeat-std / |mean|): "
          f"{np.mean(per_image_repeat_std / np.abs(per_image_repeat_mean)):.4f}")
    print(f"  Across-image std of the (repeat-averaged) score: {across_image_std:.4g}")
    print(f"  Ratio (estimator noise / signal spread): "
          f"{np.mean(per_image_repeat_std) / across_image_std:.4f}  (want << 1)")

    reference_n = max(cli.sample_counts)
    reference_scores = results[f"N{reference_n}"]

    print(f"\n=== Speed vs. performance (reference = N={reference_n}) ===")
    print(f"{'N':>6s} {'time(s)':>10s} {'s/image':>9s} {'rho_vs_ref':>11s} {'time_per_rho_point':>19s}")
    speed_perf_rows = []
    for n in cli.sample_counts:
        rho = 1.0 if n == reference_n else spearmanr(results[f"N{n}"], reference_scores)[0]
        t = timings[n]
        # "cost per unit of correlation gained above trivial (rho=0)" -- a crude
        # marginal-value proxy: lower is a better time/performance tradeoff point.
        cost_per_rho = t / rho if rho > 0 else float("inf")
        print(f"{n:6d} {t:10.1f} {t/len(test_imgs):9.3f} {rho:11.4f} {cost_per_rho:19.1f}")
        speed_perf_rows.append({"N": n, "time_s": t, "s_per_image": t / len(test_imgs), "rho_vs_reference": rho})

    if cli.out_csv:
        df = pd.DataFrame(results)
        df.insert(0, "image_id", test_ids)
        df.to_csv(cli.out_csv, index=False)

        speed_perf_csv = cli.out_csv.replace(".csv", "_speed_vs_performance.csv")
        pd.DataFrame(speed_perf_rows).to_csv(speed_perf_csv, index=False)
        print(f"wrote {speed_perf_csv}")
        print(f"\nwrote {cli.out_csv}")
