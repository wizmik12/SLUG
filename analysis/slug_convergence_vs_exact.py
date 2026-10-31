"""Join the sample-count sweep against the exact reference.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

conv = pd.read_csv("tables/slug_sample_convergence_fitz_seed0.csv")
exact = pd.read_csv("tables/slug_exact_diagonal_fitz_seed0.csv")
df = conv.merge(exact[["image_id", "exact_trace"]], on="image_id", how="inner")
print(f"{len(df)} images have both a sampled estimate and an exact trace\n")

Ns = [10, 50, 100, 200, 500, 1000, 2000]
print(f"{'N':>6s} {'mean |rel err|':>15s} {'max |rel err|':>14s} {'spearman vs exact':>18s}")
for N in Ns:
    est = df[f"N{N}"].values
    rel = np.abs(est - df.exact_trace.values) / df.exact_trace.values
    rho = spearmanr(est, df.exact_trace.values)[0]
    print(f"{N:6d} {rel.mean():15.4%} {rel.max():14.4%} {rho:18.4f}")

reps = [f"N500_repeat{i}" for i in range(3) if f"N500_repeat{i}" in df.columns]
if reps:
    R = df[["N500"] + reps].values
    cv = R.std(axis=1, ddof=1) / R.mean(axis=1)
    print(f"\nN=500 run-to-run reproducibility over {R.shape[1]} independent draws:")
    print(f"  per-image coefficient of variation: mean {cv.mean():.4%}, max {cv.max():.4%}")
    rhos = [spearmanr(R[:, i], df.exact_trace.values)[0] for i in range(R.shape[1])]
    print(f"  spearman vs exact across repeats: {np.mean(rhos):.4f} +/- {np.std(rhos, ddof=1):.4f}")
