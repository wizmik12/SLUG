# Uncertainty methods in this repo — what each one actually computes

Every method here scores a test image by the **same underlying quantity**,
`trace(J Sigma J^T)` — how much the reconstruction moves under parameter perturbations the
training data does not rule out. They differ in **three independent axes**, and conflating
them has caused real confusion in this project, so they are spelled out here.

| axis | options |
|---|---|
| **curvature source** | exact GGN `G = sum_n J_n^T J_n` (H = I) · empirical Fisher `diag(F) = (1/N) sum_n (grad l_n)^2` |
| **posterior structure** | diagonal · low-rank `I - U_k U_k^T` · full covariance `(G + tau I)^-1` |
| **parameter scope** | last layer (`Conv_1`, 4,803) · decoder (25.3M) · full network (50.6M) |

`p = 50,636,355` parameters. Memory is the persistent summary in float32, as multiples of `p`
(this is the axis Miani et al. match on: budget 3p, 10p in their appendix). "GGN-vps" is the
number of full-dataset GGN-vector products needed to FIT — the dominant compute cost, and
**not** proportional to memory.

| # | name / file | curvature | structure | scope | memory | GGN-vps to fit |
|---|---|---|---|---|---|---|
| 1 | **SLUG** `sketched_lanczos_uncertainty.py` | GGN | low-rank, **sketched** (9 of 10 Lanczos vecs, sketch 5e6) | full | **0.89p** | 10 |
| 2 | **low-rank LLA** same file, `sketch_size=None` | GGN | low-rank, **dense** eigenvectors | full | **k·p** (1p at k=1, 3p at k=3) | k..30 |
| 3 | **EF-diag** (`lae`) `diagonal_laplace.py` | **empirical Fisher** | diagonal | decoder or full | **1.00p** | **0** (one exact pass, no sampling) |
| 4 | **diagonal_lla (Miani)** `analysis/diagonal_lla_miani.py` | **GGN** | diagonal | full | **1.00p** | **10,000** |
| 5 | **LLLA-diag** `last_layer_laplace.py` | GGN (**exact**, no sampling) | diagonal | **last layer** | tiny | 0 (exact) |
| 6 | **LLLA-full** same file | GGN (exact) | **full covariance** | last layer | tiny | 0 (exact) |
| 7 | **SLUG-form** same file | GGN (exact) | low-rank `I - U_k U_k^T`, **exact** eigenvectors | last layer | tiny | 0 (exact) |
| 8 | **SWAG** `analysis/swag_experiment.py` | — (SGD trajectory) | diagonal + low-rank | full | (2+K)p | 0 (extra training) |
| 9 | **jacnorm** `analysis/jacobian_norm_ablation.py` | — | none (null baseline) | last layer | 0 | 0 |

## Things that are easy to get wrong

**(3) and (4) are both "diagonal Laplace at 1p" and they are NOT the same method.**
Different curvature source, and §9 shows the difference is decisive: the exact-GGN diagonal is
*anti-correlated* with error (−0.40) while the empirical-Fisher diagonal scores 0.83–0.94.
Never label (3) as `diagonal_lla`.

**Do not call (3) `LAE`.** That is the established acronym for *Laplacian Autoencoders*
(Miani, Warburg, Moreno-Muñoz, Skafte, Hauberg, NeurIPS 2022) — a **training-time** Bayesian
autoencoder trained by MC-EM, which we cannot apply. Use `EF-diag`.

**"LLA" is ambiguous.** In Miani's SLU paper it means **Linearized** Laplace over the whole
network (methods 1–2 here). In our §9 notes `LLLA` means **last-layer** Laplace (methods 5–7).
Different objects.

**Memory-matched is not compute-matched.** (2) and (4) both cost 3p / 1p of storage, but (2)
fits in 3 GGN-vps and (4) needs 10,000 — a ~3,000× gap for the same memory. Lanczos converges
on the **spectral gap** (here λ₁/λ₂ ≈ 5.6e7, so 2–3 iterations suffice); Hutchinson estimates
50.6M independent scalars whose per-entry variance is that row's off-diagonal mass, converging
only as 1/√N. The more low-rank the GGN, the easier for Lanczos and the harder for Hutchinson.

**Hutchinson appears in two places; only one of them failed.** *Fitting* the GGN diagonal
(method 4) is the hard one. *Scoring* `trace(J Sigma J^T)` uses Hutchinson over the
49,152-dim output space in methods 1–4 and 8, and works fine there — verified at 1.28% error
against an exact reference. Methods 5–7 use no sampling at either stage, which is why they were
usable to isolate the diagonal-vs-correlated question.

**Lanczos vector storage depends on sketching.** In the sketched path `skvs` is
`[n_iter, sketch_size]`, so iterations are nearly free. In the **dense** path it is
`[n_iter, p]` — 30 iterations = 6.08 GB and OOMs. Iterations are only "free" for method 1.
