"""
Driver for the Fitzpatrick17k / AI4SkIN VAE UQ bias study.

Trains two model variants per (run, subset) cell — "plain" (SLUG + encoder-std,
via --eval_uq) and "dropout" (MC-dropout, via --use_dropout, since this codebase
bakes dropout into the architecture at construction time, not as an eval-time
toggle) — for each of 10 independent runs x 3 training configs (Dataset A/B/C:
perc_attr in {0, 0.5, 1}), matching the paper's protocol. Wraps main.py as a
subprocess per (run, subset, variant) call.

Output layout
-------------
  <study_dir>/
    <dataset>/plain/Results/<run>/<perc_attr>/...    (checkpoint, results_run.pkl,
                                                        uq_metrics_<attr>.csv,
                                                        uq_metrics_encoderstd_<attr>.csv)
    <dataset>/dropout/Results/<run>/<perc_attr>/...  (checkpoint, results_run.pkl,
                                                        uq_metrics_dropout_<attr>.csv)

Usage
-----
  # Smoke test: 1 run, 1 subset, 5 epochs, both variants
  python run_study.py --config ../configs/fitzpatrick17k.yaml --study-dir ../outputs/fitz_smoke \\
      --nepoch-override 5 --runs-override 1 --subsets-override 0

  # Full sweep
  python run_study.py --config ../configs/fitzpatrick17k.yaml --study-dir ../outputs/fitz_study

  # Retry/resume only specific cells (already-trained checkpoints are loaded, not retrained)
  python run_study.py --config ../configs/fitzpatrick17k.yaml --study-dir ../outputs/fitz_study \\
      --only "3:0.5,7:1"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def perc_attr_str(subset: float) -> str:
    """Mirror main.py's perc_attr formatting exactly, so checkpoint paths match."""
    if subset in (0.0, 1.0):
        return str(int(subset))
    return str(subset)


def run(cmd, description="", cuda_visible=None):
    print(f"\n{'-' * 65}")
    print(f"> {description}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'-' * 65}")
    env = os.environ.copy()
    if cuda_visible is not None:
        # This is the ONLY thing that actually restricts which GPU JAX sees:
        # main.py's own --graphic_card handling sets CUDA_VISIBLE_DEVICES too
        # late (after jax.devices() has already been called at import time),
        # so it must be set here, in the subprocess's environment, before the
        # interpreter starts.
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    result = subprocess.run([str(c) for c in cmd], env=env)
    return result.returncode == 0


def checkpoint_path(save_dir, run_idx, subset, latent_channels, image_size, nepoch):
    perc = perc_attr_str(subset)
    # "fitz17_" prefix is hardcoded in train_vae.py regardless of dataset.
    return (Path(save_dir) / "Results" / str(run_idx) / perc / "model"
            / f"fitz17_{latent_channels}_{image_size}{nepoch}")


def build_common_args(cfg, gpu):
    args = [
        "--dataset", cfg["dataset"],
        "--image_size", cfg["image_size"],
        "--ch_multi", cfg["ch_multi"],
        "--num_res_blocks", cfg["num_res_blocks"],
        "--latent_channels", cfg["latent_channels"],
        "--batch_size", cfg["batch_size"],
        "--nepoch", cfg["nepoch"],
        "--lr", cfg["lr"],
        "--kl_scale", cfg["kl_scale"],
        "--feature_scale", cfg["feature_scale"],
        "--train_size", cfg["train_size"],
        "--test_size", cfg["test_size"],
        "--batch_size_uq", cfg["batch_size_uq"],
        "--graphic_card", str(gpu),
    ]
    args += ["--block_widths"] + [str(b) for b in cfg["block_widths"]]

    if cfg["dataset"] == "fitz":
        args += ["--dataset_root", cfg["dataset_root"]]
    elif cfg["dataset"] == "ai4skin":
        args += ["--ai4skin_patches_csv", cfg["ai4skin_patches_csv"],
                  "--ai4skin_patches_root", cfg["ai4skin_patches_root"]]

    passion_root = cfg.get("passion_root")
    if passion_root and os.path.isdir(passion_root):
        args += ["--test_set_root", passion_root]
    elif passion_root:
        print(f"  note: passion_root={passion_root!r} not found on this machine, skipping external validation")

    return args


def run_variant(main_py, cfg, run_idx, subset, save_dir, variant, gpu, force_retrain):
    exp_name = f"{cfg['dataset']}_{variant}_run{run_idx}_p{perc_attr_str(subset)}"
    ckpt = checkpoint_path(save_dir, run_idx, subset, cfg["latent_channels"], cfg["image_size"], cfg["nepoch"])

    cmd = [sys.executable, main_py,
           "--experiment_name", exp_name,
           "--run", run_idx,
           "--subset", subset,
           "--save_dir", save_dir]
    cmd += build_common_args(cfg, gpu)

    if ckpt.exists() and not force_retrain:
        print(f"  checkpoint exists ({ckpt}), loading instead of retraining")
        cmd += ["--load_model"]

    if variant == "plain":
        cmd += ["--eval_uq"]
    elif variant == "dropout":
        cmd += ["--use_dropout"]
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return run(cmd, description=exp_name, cuda_visible=gpu)


def main():
    parser = argparse.ArgumentParser(description="Run the Fitzpatrick17k/AI4SkIN VAE UQ bias study")
    parser.add_argument("--config", required=True, help="Base YAML config (data paths, model arch, training hyper-params)")
    parser.add_argument("--study-dir", required=True, help="Root directory for all experiment outputs")
    parser.add_argument("--gpu", default="0", help="Physical GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--runs-override", type=int, default=None, help="Override number of runs (default: 10)")
    parser.add_argument("--subsets-override", default=None,
                        help="Comma-separated subset values to run, e.g. '0' or '0,0.5,1' (default: all three)")
    parser.add_argument("--nepoch-override", type=int, default=None, help="Override training epochs (for smoke tests)")
    parser.add_argument("--train-size-override", type=int, default=None,
                        help="Override training set size (SLUG's Lanczos setup scans this fully -- keep small for smoke tests)")
    parser.add_argument("--test-size-override", type=int, default=None,
                        help="Override test set size (SLUG scoring cost is proportional to this -- keep small for smoke tests)")
    parser.add_argument("--variants", default="plain,dropout", help="Comma-separated variants to run")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a checkpoint already exists")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run:subset tokens to restrict to, e.g. '3:0.5,7:1'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.nepoch_override is not None:
        cfg["nepoch"] = args.nepoch_override
    if args.train_size_override is not None:
        cfg["train_size"] = args.train_size_override
    if args.test_size_override is not None:
        cfg["test_size"] = args.test_size_override

    n_runs = args.runs_override if args.runs_override is not None else 10
    runs = list(range(n_runs))
    subsets = [float(s) for s in args.subsets_override.split(",")] if args.subsets_override else [0.0, 0.5, 1.0]
    variants = args.variants.split(",")

    only = None
    if args.only:
        only = set()
        for tok in args.only.split(","):
            r, s = tok.split(":")
            only.add((int(r), float(s)))

    # orbax requires absolute checkpoint paths -- resolve before any subprocess uses it.
    study_dir = Path(args.study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    main_py = str(Path(__file__).parent / "main.py")

    n_ok, n_fail, n_skip = 0, 0, 0
    failures = []

    for run_idx in runs:
        for subset in subsets:
            if only is not None and (run_idx, subset) not in only:
                n_skip += len(variants)
                continue
            for variant in variants:
                save_dir = str(study_dir / cfg["dataset"] / variant)
                print(f"\n{'#' * 65}")
                print(f"  {cfg['dataset']} | variant={variant} | run={run_idx} | subset={subset}")
                print(f"{'#' * 65}")
                ok = run_variant(main_py, cfg, run_idx, subset, save_dir, variant, args.gpu, args.force_retrain)
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append((cfg["dataset"], variant, run_idx, subset))

    print(f"\n{'=' * 65}")
    print(f"Sweep complete: {n_ok} ok, {n_fail} failed, {n_skip} skipped (via --only)")
    if failures:
        print("Failures:")
        for f_ in failures:
            print(f"  {f_}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
