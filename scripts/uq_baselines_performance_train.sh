#!/bin/bash
# Batch 1, first real experiment: 2 datasets x 5 seeds, config B (subset=0.5), perceptual
# loss ON (feature_scale=1, per revision_part1.md Sec 0.c), same data split per dataset
# (--run fixed at 0, only --seed varies -- see main.py run/seed decoupling).
#
# batch_size=16 (revised from an earlier batch_size=1 attempt): with feature_scale=1 (VGG16
# perceptual loss) on this 24GB GPU, 32 and 16 both originally OOM'd under JAX's default
# memory preallocation; batch_size=1 was used as a stopgap. Retesting with
# XLA_PYTHON_CLIENT_PREALLOCATE=false showed batch_size=16 actually fits -- the earlier
# failure was a preallocation artifact, not a hard capacity limit. 16 trains far faster per
# epoch than 1 (~104 steps/epoch vs 1668) and gives much cleaner convergence curves.
#
# NEPOCH=500: chosen from full 400-epoch batch_size=16 calibration runs (outputs/
# calib_fitz_bs16, outputs/calib_ai4skin_bs16). AI4SkIN was still improving meaningfully at
# epoch 400 (MSE 0.0064->0.0057 from epoch 300->400), so 400 was extended to 500 -- which is
# not an arbitrary round number: the LR schedule's own decay_steps=500*len(train_loader) in
# train_vae.py reaches its designed floor at exactly epoch 500 for batch_size=16
# (len(train_loader)=1668/16~=104, 500*104=52000 steps=500 epochs). Fitz's converged
# training MSE (~0.0087) at epoch 400 was already close to the published test-set blended
# average (light 0.0074 / dark 0.0111 ~= 0.0093), a good sanity check.
#
# Parallelized one seed per GPU: each GPU runs its seed fitz job then its seed ai4skin job,
# sequentially; the 5 seeds run concurrently across the GPUs listed in GPUS. This turns
# about 10 run-durations of wall-clock into about 2. Re-check nvidia-smi for currently-idle
# GPUs before running -- this is a shared machine and availability changes; never include
# GPU 0 without separately confirming it is actually free (see revision_part1.md Sec 0.c).
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID is set here (not left to main.py, which sets it too late,
# after jax has already been imported and devices enumerated) so CUDA_VISIBLE_DEVICES=N
# actually lands on nvidia-smi's GPU N, not whatever GPU CUDA's default fastest-first
# ordering calls N. Confirmed today: without this, all 5 processes landed on GPUs
# different from the ones requested (though still one distinct physical GPU each).
#
# Usage: NEPOCH=500 GPUS="1 2 3 4 5" bash scripts/uq_baselines_performance_train.sh
set -eu

if [ -z "${NEPOCH:-}" ]; then
  echo "Set NEPOCH (from the calibration runs convergence point) before running" >&2
  exit 1
fi
GPUS="${GPUS:-1 2}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/uq_baselines_performance"
mkdir -p "$OUT_DIR"

gpu_arr=($GPUS)
n_gpus=${#gpu_arr[@]}

run_seed() {
  local seed="$1" gpu="$2"
  echo "=== seed=$seed on GPU $gpu: fitz ==="
  CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${PY:-python}" "$SRC_DIR/main.py" \
    --experiment_name "uq_baselines_performance_fitz_seed${seed}" --run 0 --seed "$seed" --subset 0.5 \
    --save_dir "$OUT_DIR/fitz" \
    --dataset fitz --image_size 128 --ch_multi 64 --num_res_blocks 1 --latent_channels 128 \
    --batch_size 16 --nepoch "$NEPOCH" --lr 0.0001 --kl_scale 0.5 --feature_scale 1 \
    --train_size 1668 --test_size 512 --batch_size_uq 8 --graphic_card "$gpu" \
    --block_widths 1 2 4 8 \
    --dataset_root /work/work_mik/2026Diff_UQ/FitzFairness/data/images/ \
    --eval_metrics --n_recon_examples 8 \
    > "$OUT_DIR/fitz_seed${seed}.log" 2>&1

  echo "=== seed=$seed on GPU $gpu: ai4skin ==="
  CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${PY:-python}" "$SRC_DIR/main.py" \
    --experiment_name "uq_baselines_performance_ai4skin_seed${seed}" --run 0 --seed "$seed" --subset 0.5 \
    --save_dir "$OUT_DIR/ai4skin" \
    --dataset ai4skin --image_size 128 --ch_multi 64 --num_res_blocks 1 --latent_channels 128 \
    --batch_size 16 --nepoch "$NEPOCH" --lr 0.0001 --kl_scale 0.5 --feature_scale 1 \
    --train_size 2000 --test_size 512 --batch_size_uq 8 --graphic_card "$gpu" \
    --block_widths 1 2 4 8 \
    --ai4skin_patches_csv /work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv \
    --ai4skin_patches_root /work/work_mik/2026Diff_UQ/FusoFairness/data/ \
    --eval_metrics --n_recon_examples 8 \
    > "$OUT_DIR/ai4skin_seed${seed}.log" 2>&1

  echo "=== seed=$seed done ==="
}

pids=()
for i in 0 1 2 3 4; do
  gpu="${gpu_arr[$((i % n_gpus))]}"
  run_seed "$i" "$gpu" &
  pids+=($!)
  # Stagger launches so the first-compile memory spike of each process does not overlap
  # across processes that might land on the same GPU (if n_gpus < 5).
  sleep 5
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "Batch 1: one or more seeds FAILED, check $OUT_DIR for per-seed logs"
  exit 1
fi
echo "Batch 1 training complete: 10 runs (5 seeds x 2 datasets) in $OUT_DIR"
