"""Implements METHODS.md row 3: the EMPIRICAL FISHER diagonal, `diag(F) = (1/N) sum_n
(grad l_n)^2`, computed exactly in one pass with no sampling.

This is NOT Miani's `diagonal_lla` (row 4), which is the GGN diagonal estimated by
Hutchinson with 10,000 Rademacher probes. Different matrices, and the difference is
decisive: Sec 9 shows the exact-GGN diagonal is ANTI-correlated with error (-0.40) while
this empirical-Fisher diagonal scores 0.83-0.94.

Do NOT call this `LAE` -- that is the acronym for Laplacian Autoencoders (Miani et al.,
NeurIPS 2022), a training-time method we cannot apply. Use `EF-diag`.

`estimate_ggn_diagonal` below IS the Hutchinson GGN-diagonal estimator (row 4's fitter),
kept for comparison; it needs ~10,000 probes = ~97 GPU-h per checkpoint at this scale.

See src/uncertainty/METHODS.md for the full taxonomy.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from uncertainty.autodiff import get_ggn_vector_product_dataloader
from uncertainty.sketched_lanczos_uncertainty import compute_num_params, get_subset_loader

"""
Native-JAX diagonal Laplace baseline ("lae" score), reusing SLUG's own GGN-vector-product
machinery (uncertainty/autodiff.py) instead of porting Miani et al.'s LaplaceAE to JAX/torch.

SLUG approximates the predictive covariance with a low-rank correlated posterior
(I - U U^T). This module estimates the *diagonal* Laplace posterior instead -- the same
target quantity Miani et al.'s Laplacian Autoencoder computes exactly via layer-wise
backprop, but obtained here via a Hutchinson/Bekas-Kokiopoulou-Saad stochastic diagonal
estimator (diag(G) ~= E_v[v * Gv]) applied to the identical black-box GGN-vector product
`get_ggn_vector_product_dataloader` that SLUG's Lanczos sketch is built on. This guarantees
both scores are computed from the literal same J/G operator (when using the full parameter
set), isolating diagonal-vs-low-rank-correlated posterior as the only difference between them.

Also supports restricting the diagonal to a decoder-only or last-decoder-layers-only subset
(`decoder_layers`), holding the rest of the network fixed:
  - decoder-only matches Miani et al.'s own primary post-hoc fitter (`fit_laplace_to_decoder`
    in the cloned LaplaceAE repo), so it's not a memory hack, it's the more faithful
    reproduction of their method.
  - last-layers-only is the standard "last-layer Laplace" simplification (Kristiadi et al.
    2020), and empirically dominates decoder-only for *this* architecture: the VAE's final
    decoder layers (Conv_1 + ResUp_3, i.e. the layers nearest the RGB output) are ~190K
    params, 0.37% of the full 50.6M -- vs. ~50% for decoder-only. Fewer estimated entries
    also means better per-entry Hutchinson statistics for the same sample budget, which is
    the actual open problem this batch hit (see revision_part1.md Sec 0.b/2.2 and the
    2026-08-05 negative-diagonal-entries finding).

See revision_part1.md Sec 0.b / 2.2 for the design rationale.
"""


def estimate_ggn_diagonal(
        ggn_vector_product,
        n_params,
        key,
        n_samples,
        sample_dist="rademacher",
        record_every=1,
    ):
    """Hutchinson/Bekas et al. (2007) stochastic diagonal estimator: diag(G) ~= E_v[v * Gv].

    Returns (diag_estimate, running_means) where running_means is a list of the running-mean
    diag estimate recorded every `record_every` samples, for a convergence check -- each
    sample costs one full-dataset GGN-vector product (same order as one Lanczos iteration).
    """
    if sample_dist not in ("rademacher", "gaussian"):
        raise ValueError(f"Unknown sample_dist {sample_dist!r}")

    keys = jax.random.split(key, n_samples)
    diag_estimate = jnp.zeros(n_params)
    running_means = []

    for s in range(n_samples):
        if sample_dist == "rademacher":
            v = jax.random.rademacher(keys[s], shape=(n_params,)).astype(jnp.float32)
        else:
            v = jax.random.normal(keys[s], shape=(n_params,))

        Gv = ggn_vector_product(v)
        # Numerically stable running mean: avoids re-summing n_samples large vectors.
        diag_estimate = diag_estimate + (v * Gv - diag_estimate) / (s + 1)

        if (s + 1) % record_every == 0 or s == n_samples - 1:
            running_means.append(np.asarray(diag_estimate))
            frac_neg = float(jnp.mean(diag_estimate < 0))
            print(f"  diag GGN Hutchinson sample {s + 1}/{n_samples}: "
                  f"mean={float(jnp.mean(diag_estimate)):.6g} "
                  f"min={float(jnp.min(diag_estimate)):.6g} "
                  f"max={float(jnp.max(diag_estimate)):.6g} "
                  f"frac_negative={frac_neg:.4f}")

    return diag_estimate, running_means


def estimate_empirical_fisher_diagonal(model_apply_full, free_params, dataloader):
    """Empirical-Fisher diagonal: diag(F) = (1/N) sum_n (grad_theta l_n(theta))^2, via one
    deterministic pass over the training set (vmap(grad) per batch) -- no Hutchinson
    sampling, guaranteed non-negative by construction (each term is a square).

    This is a related but distinct quantity from the GGN diagonal (Kunstner et al. 2019,
    "Limitations of the Empirical Fisher Approximation") -- used here because the Hutchinson
    GGN-diagonal estimator (estimate_ggn_diagonal) failed to converge for this model's
    off-diagonal-heavy GGN (conv weight sharing inflates off-diagonal mass relative to the
    diagonal -- see revision_part1.md Sec 0.g), not because it is claimed to be a closer
    approximation to the true GGN diagonal. Per-example loss is the Gaussian-likelihood
    (squared-error) term, matching H=I (manuscript Sec 3.4 fix, revision_part1.md Sec 0.c).
    """
    def single_example_sq_loss(free_p, x):
        x_batched = x[None]
        recon = model_apply_full(x_batched, free_p)
        return jnp.sum((recon[0] - x) ** 2)

    batched_grad_fn = jax.jit(jax.vmap(jax.grad(single_example_sq_loss), in_axes=(None, 0)))

    diag_sum = None
    n_total = 0
    for X, _ in dataloader:
        X = jnp.asarray(X)
        per_example_grads = batched_grad_fn(free_params, X)  # pytree, leaves have a leading batch axis
        leaves = jax.tree_util.tree_leaves(per_example_grads)
        batch_size = leaves[0].shape[0]
        flat = jnp.concatenate([leaf.reshape(batch_size, -1) for leaf in leaves], axis=1)
        batch_sq_sum = jnp.sum(flat ** 2, axis=0)
        diag_sum = batch_sq_sum if diag_sum is None else diag_sum + batch_sq_sum
        n_total += batch_size

    diag_estimate = diag_sum / n_total
    print(f"  empirical Fisher diagonal: n_examples={n_total} "
          f"mean={float(jnp.mean(diag_estimate)):.6g} "
          f"min={float(jnp.min(diag_estimate)):.6g} "
          f"max={float(jnp.max(diag_estimate)):.6g}")
    return diag_estimate


def _split_decoder_params(params, decoder_layers):
    """Split VAE params into (free_subset, merge_fn) for a given decoder-layer restriction.

    decoder_layers=None -> the entire 'decoder' subtree is free, 'encoder' is fixed
        (matches Miani et al.'s decoder-only post-hoc fitter).
    decoder_layers=list[str] -> only those params['decoder'][key] entries are free;
        everything else (encoder + the rest of the decoder) is fixed
        (last-layer-Laplace style restriction, e.g. ['Conv_1', 'ResUp_3']).
    decoder_layers='all' -> every parameter (encoder + decoder) is free, matching SLUG's
        own scope exactly, for a same-parameter-subset comparison if that's ever wanted
        instead of the decoder-only default.
    """
    if decoder_layers == "all":
        return params, lambda free: free

    decoder = params["decoder"]
    if decoder_layers is None:
        free = decoder
        fixed_decoder = {}
    else:
        missing = [k for k in decoder_layers if k not in decoder]
        if missing:
            raise ValueError(f"decoder_layers {missing} not found in decoder params; "
                              f"available: {list(decoder.keys())}")
        free = {k: decoder[k] for k in decoder_layers}
        fixed_decoder = {k: v for k, v in decoder.items() if k not in decoder_layers}

    def merge(free_subset):
        return {**params, "decoder": {**fixed_decoder, **free_subset}}

    return free, merge


def get_ggn_vector_product_subset(params_dict, model, dataloader, decoder_layers):
    """Same computation as autodiff.get_ggn_vector_product_dataloader, restricted to a
    parameter subset (see _split_decoder_params). Kept local to this file rather than
    modifying the shared autodiff.py, so SLUG's own (full-parameter) usage is untouched.
    """
    full_params = params_dict["params"]
    batch_stats = params_dict["batch_stats"]
    free_params, merge = _split_decoder_params(full_params, decoder_layers)

    model_apply = lambda data, p: model.apply(
        {"params": merge(p), "batch_stats": batch_stats},
        x=data, rng=jax.random.PRNGKey(0), train=False,
    )[0]

    devectorize_fun = flatten_util.ravel_pytree(free_params)[1]
    flatten_param = jnp.array(flatten_util.ravel_pytree(free_params)[0])

    @jax.jit
    def ggn_tree_product_batch(X, tree):
        model_on_data = functools.partial(model_apply, X)
        _, J_tree = jax.jvp(model_on_data, (free_params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, free_params)
        JtHJ_tree = model_on_data_vjp(J_tree)[0]
        return JtHJ_tree

    @jax.jit
    def ggn_vector_product_batch(X, v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product_batch(X, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    def ggn_vector_product_dataloader(v):
        # No jax.pure_callback here -- see the matching comment in
        # autodiff.get_ggn_vector_product_dataloader. Same reasoning: this is only
        # ever called from plain Python (the Hutchinson sampling loop below, or
        # lanczos.py's Lanczos loop), never from a traced context, so the callback
        # layer is unnecessary and was the dominant per-call cost (~40x).
        result = jnp.zeros_like(v)
        for X, _ in dataloader:
            X = jnp.asarray(X)
            result += ggn_vector_product_batch(X, v)
        return result

    return ggn_vector_product_dataloader, free_params, merge, model_apply


def diagonal_laplace_score_fun(
        model,
        params_dict,
        train_loader,
        output_shape=(128, 128, 3),
        n_samples_diag=30,
        tau=1.0,
        seed=1,
        train_batch_size=8,
        subsample_trainset=None,
        sample_dist="rademacher",
        decoder_layers=None,
        diag_method="empirical_fisher",
    ):
    """Build the diagonal-Laplace ('lae') per-image score function.

    Mirrors uncertainty.sketched_lanczos_uncertainty.low_memory_lanczos_score_fun's structure
    (same jacobianT_vector_product pattern, same vmap-over-batch scoring convention) so the
    two scores are comparable in kind.

    `decoder_layers`: which parameters the diagonal is fitted over -- see
    _split_decoder_params. Default None = decoder-only (Miani et al.'s own default scope).
    Pass a list (e.g. ['Conv_1', 'ResUp_3']) to restrict further to specific decoder layers
    (last-layer-Laplace style), or 'all' to match SLUG's full encoder+decoder scope exactly.
    Note: empirical Fisher's cost doesn't depend on Hutchinson sample count, so it no longer
    needs the last-layers restriction the way the Hutchinson path did to have any hope of
    converging -- decoder-only (or even 'all') is affordable again with diag_method='empirical_fisher'.

    `diag_method`: 'empirical_fisher' (default) -- one deterministic pass, no sampling,
    guaranteed non-negative (see estimate_empirical_fisher_diagonal for why this replaced
    the Hutchinson estimator as the default). 'hutchinson' -- the original stochastic
    GGN-diagonal estimator (estimate_ggn_diagonal); kept available since it's a closer
    approximation to the true target quantity in principle, but empirically failed to
    converge within a practical sample budget for this model (revision_part1.md Sec 0.g) --
    use only for comparison/documentation purposes, not as the batch1 default.

    `tau` is the prior precision / damping term (posterior precision ~= diag_ggn + tau).
    Choose it via a held-out grid search (see revision_part1.md Sec 2.2 step 2) before
    treating results as final -- the default of 1.0 here is a placeholder, not a tuned value.
    """
    trainset_size = len(train_loader.dataset) if subsample_trainset is None else subsample_trainset
    train_loader = get_subset_loader(
        train_loader,
        trainset_size,
        batch_size=train_batch_size,
        drop_last=True,
    )

    ggn_vector_product, free_params, merge, model_apply_full = get_ggn_vector_product_subset(
        params_dict, model, train_loader, decoder_layers,
    )
    n_params = compute_num_params(free_params)

    running_means = None
    if diag_method == "empirical_fisher":
        print(f"Estimating diagonal via empirical Fisher (one deterministic pass), "
              f"decoder_layers={decoder_layers!r}, n_params={n_params}...")
        start = time.time()
        diag_ggn = estimate_empirical_fisher_diagonal(model_apply_full, free_params, train_loader)
        print(f"Empirical Fisher diagonal estimate took {time.time() - start:.3f} seconds")
    elif diag_method == "hutchinson":
        key_diag, key_score = jax.random.split(jax.random.PRNGKey(seed))
        print(f"Estimating diagonal GGN with {n_samples_diag} Hutchinson samples "
              f"({sample_dist}), decoder_layers={decoder_layers!r}, n_params={n_params}...")
        start = time.time()
        diag_ggn, running_means = estimate_ggn_diagonal(
            ggn_vector_product, n_params, key_diag, n_samples_diag, sample_dist=sample_dist,
        )
        print(f"Diagonal GGN estimate took {time.time() - start:.3f} seconds")
    else:
        raise ValueError(f"Unknown diag_method {diag_method!r}, expected 'empirical_fisher' or 'hutchinson'")

    if bool(jnp.any(diag_ggn < 0)):
        n_neg = int(jnp.sum(diag_ggn < 0))
        print(f"WARNING: {n_neg}/{n_params} diagonal entries are negative "
              f"(expected for an unconverged Hutchinson estimate on a PSD matrix; "
              f"should be impossible for diag_method='empirical_fisher' -- investigate if seen).")

    inv_precision = 1.0 / (diag_ggn + tau)  # posterior variance per free parameter

    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    @jax.jit
    def model_apply(datapoint, free_p):
        datapoint = jnp.expand_dims(datapoint, 0)
        return model_apply_full(datapoint, free_p)

    def get_jacobianT_vector_product(datapoint, output_shape):
        model_on_data = functools.partial(model_apply, datapoint)
        _, model_on_data_vjp = jax.vjp(model_on_data, free_params)

        @jax.jit
        def jacobianT_vector_product(vector):
            batch_shape = (1, *output_shape)
            vector = vector.reshape(batch_shape)
            Jt_vector = model_on_data_vjp(vector)[0]
            return vectorize_fun(Jt_vector)

        return jacobianT_vector_product

    output_dim = 1
    for d in output_shape:
        output_dim *= d

    def get_uncertainty_trace_diag(jacobianT_vector_product, inv_precision, output_dim, key, hutchinson_samples):
        keys = jax.random.split(key, hutchinson_samples)

        @jax.jit
        def add_sample(k, trace_estimate):
            key = keys[k]
            vec = jax.random.normal(key, shape=(output_dim,))
            a = jacobianT_vector_product(vec)
            single_sample = jnp.sum((a ** 2) * inv_precision)
            trace_estimate += single_sample / hutchinson_samples
            return trace_estimate

        trace_estimate = jax.lax.fori_loop(0, hutchinson_samples, add_sample, 0.0)
        return trace_estimate

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def score_fun_diagonal_laplace(datapoint, key, hutchinson_samples=None):
        datapoint = jnp.asarray(datapoint)
        jacobianT_vector_product = get_jacobianT_vector_product(datapoint, output_shape)
        return get_uncertainty_trace_diag(
            jacobianT_vector_product, inv_precision, output_dim, key, hutchinson_samples,
        )

    return {
        "approx": score_fun_diagonal_laplace,
        "diag_ggn": diag_ggn,
        "running_means": running_means,
        "tau": tau,
        "decoder_layers": decoder_layers,
        "diag_method": diag_method,
        "n_params": n_params,
    }
