#!/bin/bash
# Batch 1, post-hoc evaluation: for each of the 10 plain checkpoints produced by
# uq_baselines_performance_train.sh (5 seeds x 2 datasets), run the three per-checkpoint UQ evaluations:
#   1. SLUG (--eval_uq)          -- also produces encoder_std for free, same branch
#   2. MC-dropout (--use_dropout, --load_model) -- no separate dropout training needed:
#      nn.Dropout has no learnable params, so the plain checkpoint loads directly into a
#      dropout-enabled model instance and MC-dropout runs purely at eval time (confirmed
#      2026-08-05). Do NOT retrain a dropout variant.
#   3. LAE (--eval_lae)          -- native-JAX diagonal Hutchinson estimator, decoder-only
#      by default (see uncertainty/diagonal_laplace.py); pass --lae-decoder-layers to
#      override once that CLI wiring lands (currently only the Python default applies).
#
# NEPOCH/architecture args must match whatever uq_baselines_performance_train.sh was actually run with, or
# --load_model will look in the wrong checkpoint directory and fail to find anything.
#
# Requires the corresponding uq_baselines_performance_train.sh checkpoint to already exist; run this only
# after (or, per-seed, as soon as) that seed's training has finished -- see the
# --only-if-done guard below, which skips any seed whose checkpoint directory is missing.
#
# Usage: NEPOCH=500 GPUS="1 2 3 4 5" bash scripts/uq_baselines_performance_posthoc_eval.sh
set -eu

if [ -z "${NEPOCH:-}" ]; then
  echo "Set NEPOCH (must match the uq_baselines_performance_train.sh run) before running" >&2
  exit 1
fi
GPUS="${GPUS:-1 2}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/uq_baselines_performance"

gpu_arr=($GPUS)
n_gpus=${#gpu_arr[@]}

common_args() {
  # $1=dataset $2=seed $3=gpu
  local dataset="$1" seed="$2" gpu="$3"
  if [ "$dataset" = "fitz" ]; then
    echo "--experiment_name uq_baselines_performance_fitz_seed${seed} --run 0 --seed ${seed} --subset 0.5 \
      --save_dir ${OUT_DIR}/fitz --dataset fitz --image_size 128 --ch_multi 64 \
      --num_res_blocks 1 --latent_channels 128 --batch_size 16 --nepoch ${NEPOCH} \
      --lr 0.0001 --kl_scale 0.5 --feature_scale 1 --train_size 1668 --test_size 512 \
      --batch_size_uq 4 --graphic_card ${gpu} --block_widths 1 2 4 8 \
      --dataset_root /work/work_mik/2026Diff_UQ/FitzFairness/data/images/ --load_model"
  else
    echo "--experiment_name uq_baselines_performance_ai4skin_seed${seed} --run 0 --seed ${seed} --subset 0.5 \
      --save_dir ${OUT_DIR}/ai4skin --dataset ai4skin --image_size 128 --ch_multi 64 \
      --num_res_blocks 1 --latent_channels 128 --batch_size 16 --nepoch ${NEPOCH} \
      --lr 0.0001 --kl_scale 0.5 --feature_scale 1 --train_size 2000 --test_size 512 \
      --batch_size_uq 4 --graphic_card ${gpu} --block_widths 1 2 4 8 \
      --ai4skin_patches_csv /work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv \
      --ai4skin_patches_root /work/work_mik/2026Diff_UQ/FusoFairness/data/ --load_model"
  fi
}

checkpoint_exists() {
  # $1=dataset $2=seed
  local ckpt="${OUT_DIR}/$1/Results/0_seed$2/0.5/model/fitz17_128_128${NEPOCH}"
  [ -d "$ckpt" ]
}

eval_seed() {
  local seed="$1" gpu="$2"
  for dataset in fitz ai4skin; do
    if ! checkpoint_exists "$dataset" "$seed"; then
      echo "=== seed=$seed $dataset: checkpoint not found yet, skipping (rerun later) ==="
      continue
    fi
    local args; args="$(common_args "$dataset" "$seed" "$gpu")"

    echo "=== seed=$seed on GPU $gpu: $dataset SLUG+encoder_std ==="
    CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
      "${PY:-python}" "$SRC_DIR/main.py" $args --eval_uq \
      > "$OUT_DIR/${dataset}_seed${seed}_slug.log" 2>&1

    echo "=== seed=$seed on GPU $gpu: $dataset MC-dropout ==="
    CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
      "${PY:-python}" "$SRC_DIR/main.py" $args --use_dropout \
      > "$OUT_DIR/${dataset}_seed${seed}_mcdropout.log" 2>&1

    echo "=== seed=$seed on GPU $gpu: $dataset LAE ==="
    CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
      "${PY:-python}" "$SRC_DIR/main.py" $args --eval_lae \
      --lae_n_samples_diag 30 --lae_s_samples 500 \
      > "$OUT_DIR/${dataset}_seed${seed}_lae.log" 2>&1
  done
  echo "=== seed=$seed post-hoc eval done ==="
}

pids=()
for i in 0 1 2 3 4; do
  gpu="${gpu_arr[$((i % n_gpus))]}"
  eval_seed "$i" "$gpu" &
  pids+=($!)
  sleep 5
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "Batch 1 post-hoc eval: one or more seeds FAILED, check $OUT_DIR for per-seed logs"
  exit 1
fi
echo "Batch 1 post-hoc eval complete"
