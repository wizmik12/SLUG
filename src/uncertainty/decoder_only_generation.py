"""
Random-generation experiment: sample z ~ N(0, I) directly from the VAE prior (bypassing
the encoder entirely -- there's no real input to encode), decode through the trained
decoder, and score each generated image's epistemic uncertainty using a DECODER-ONLY
posterior for both SLUG and LAE.

This is new for SLUG, which normally always scores using the full encoder+decoder scope
and always starts from a real input image (re-running the encoder to get the reconstruction
whose uncertainty is being measured). LAE already defaults to decoder-only parameters, but
its existing per-image scoring path still runs the encoder forward on a real image; that
doesn't apply when there's no real input at all. Both methods here get a z-direct
jacobianT_vector_product that starts from a sampled z and never touches the encoder.

Training-set FITTING is unchanged for both: the GGN/Fisher diagonal characterizing "how
sensitive is the decoder around the training distribution" is still estimated from real
training images (get_ggn_vector_product_subset, estimate_empirical_fisher_diagonal) -- only
the per-image SCORING step changes, and only for the generated-image case. A matching
real-image, decoder-only scoring path is also provided (reusing model_apply_full, which
already exists for LAE) so generated-vs-real can be compared under the IDENTICAL restricted
posterior, not just against the old full-model SLUG numbers.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from uncertainty.diagonal_laplace import (
    get_ggn_vector_product_subset,
    estimate_empirical_fisher_diagonal,
)
from uncertainty.lanczos import low_memory_lanczos
from uncertainty.sketch import SRFT_sketch, get_optimal_padding
from uncertainty.sketched_lanczos_uncertainty import compute_num_params, get_subset_loader

_vectorize = lambda tree: flatten_util.ravel_pytree(tree)[0]


def sample_latent(key, n_samples, fm_size, latent_channels):
    """z ~ N(0, I), the VAE's prior -- same shape the encoder's mu would have."""
    return jax.random.normal(key, shape=(n_samples, fm_size, fm_size, latent_channels))


def decode_batch(decoder_module, params_dict, z_batch):
    """Full decoder forward on a batch of sampled latents -- the 'random generation' images.

    decoder_module must be a standalone Decoder(...) instance (not model.decoder -- Flax
    submodules defined in setup() are only bound during a trace/apply call on the parent,
    not accessible as a plain attribute on an unbound VAE() instance).
    """
    @jax.jit
    def _decode(z):
        return decoder_module.apply(
            {"params": params_dict["params"]["decoder"], "batch_stats": params_dict["batch_stats"]["decoder"]},
            z, train=False,
        )
    return _decode(z_batch)


def fit_decoder_only_posteriors(
        model, params_dict, train_loader, decoder_layers=None,
        sketch_size=1000, lanczos_iterations=10, lanczos_eigenvectors=None,
        seed=1, train_batch_size=8, subsample_trainset=None,
    ):
    """Fits both SLUG (low-rank, sketched Lanczos) and LAE (diagonal, empirical Fisher)
    decoder-only posteriors from the same GGN-vector-product-subset setup, since both are
    ultimately approximating the same restricted precision matrix, just with different
    structure. Real training images throughout -- this is standard MAP/Laplace fitting
    around the training distribution, independent of what gets scored afterward.
    """
    if lanczos_eigenvectors is None:
        lanczos_eigenvectors = int(0.9 * lanczos_iterations)

    trainset_size = len(train_loader.dataset) if subsample_trainset is None else subsample_trainset
    sub_loader = get_subset_loader(train_loader, trainset_size, batch_size=train_batch_size, drop_last=True)

    ggn_vector_product, free_params, merge, model_apply_full = get_ggn_vector_product_subset(
        params_dict, model, sub_loader, decoder_layers,
    )
    n_params = compute_num_params(free_params)

    # --- SLUG: sketched Lanczos low-rank fit, restricted to the decoder-only operator ---
    sketch_padding = get_optimal_padding(n_params)
    key_sketch, key_lanczos = jax.random.split(jax.random.PRNGKey(seed))
    print(f"[decoder-only SLUG] n_params={n_params}, sketch_size={sketch_size}, "
          f"lanczos_iterations={lanczos_iterations}")
    sketch_op = SRFT_sketch(key_sketch, n_params, sketch_size, padding=sketch_padding)
    eigenvec, eigenval = low_memory_lanczos(key_lanczos, ggn_vector_product, n_params, lanczos_iterations, sketch_op)

    U, S, _ = np.linalg.svd(eigenvec @ jnp.diag(eigenval), full_matrices=False)
    if lanczos_eigenvectors < len(S):
        threshold = sorted(S, reverse=True)[lanczos_eigenvectors]
        eigenvec = jnp.asarray(U[:, S > threshold])
        eigenval = jnp.asarray(S[S > threshold])
    else:
        eigenvec = jnp.asarray(U)
        eigenval = jnp.asarray(S)

    @jax.jit
    def inv_sqrt_approx_ggn_vector_product(vector):
        return (sketch_op @ vector).T @ eigenvec

    # --- LAE: empirical Fisher diagonal fit, same free_params/subset ---
    print(f"[decoder-only LAE] n_params={n_params}, empirical Fisher (one deterministic pass)")
    diag_fisher = estimate_empirical_fisher_diagonal(model_apply_full, free_params, sub_loader)

    return {
        "free_params": free_params,
        "merge": merge,
        "model_apply_full": model_apply_full,   # real-image, full-forward, decoder-only-grad
        "n_params": n_params,
        "slug_inv_sqrt_ggn_vp": inv_sqrt_approx_ggn_vector_product,
        "lae_diag_fisher": diag_fisher,
    }


def _jacobianT_vector_product_from_z(decoder_module, fit, batch_stats_decoder, z, output_shape):
    """Decoder-only, z-direct VJP: skips the encoder entirely.

    decoder_module must be a standalone Decoder(...) instance -- see decode_batch's
    docstring for why (Flax setup()-submodules aren't attribute-accessible unbound)."""
    merge = fit["merge"]

    def decoder_on_z(p):
        decoder_params = merge(p)["decoder"]
        z_batched = z[None]
        # Decoder.__call__ returns the image tensor directly (unlike VAE.__call__, which
        # returns a (recon, mu, log_var) tuple) -- no [0] unpacking here, stays (1,H,W,C).
        return decoder_module.apply({"params": decoder_params, "batch_stats": batch_stats_decoder}, z_batched, train=False)

    _, vjp_fn = jax.vjp(decoder_on_z, fit["free_params"])

    @jax.jit
    def jacobianT_vector_product(vector):
        batch_shape = (1, *output_shape)
        vector = vector.reshape(batch_shape)
        return _vectorize(vjp_fn(vector)[0])

    return jacobianT_vector_product


def _jacobianT_vector_product_from_image(fit, datapoint, output_shape):
    """Real-image, full-forward, decoder-only-grad VJP (same construction LAE's own
    per-image scoring already uses) -- for a real-vs-generated comparison under the
    identical restricted posterior."""
    model_apply_full = fit["model_apply_full"]
    model_on_data = functools.partial(model_apply_full, datapoint[None])
    _, vjp_fn = jax.vjp(model_on_data, fit["free_params"])

    @jax.jit
    def jacobianT_vector_product(vector):
        batch_shape = (1, *output_shape)
        vector = vector.reshape(batch_shape)
        return _vectorize(vjp_fn(vector)[0])

    return jacobianT_vector_product


def _hutchinson_trace_low_rank(jacobianT_vector_product, inv_sqrt_ggn_vp, output_dim, key, hutchinson_samples):
    keys = jax.random.split(key, hutchinson_samples)

    @jax.jit
    def add_sample(k, trace_estimate):
        vec = jax.random.normal(keys[k], shape=(output_dim,))
        col = jacobianT_vector_product(vec)
        variance_I = jnp.sum(col ** 2)
        col = inv_sqrt_ggn_vp(col)
        variance_P = jnp.sum(col ** 2)
        return trace_estimate + (variance_I - variance_P) / hutchinson_samples

    return jax.lax.fori_loop(0, hutchinson_samples, add_sample, 0.0)


def _hutchinson_trace_diag(jacobianT_vector_product, inv_precision, output_dim, key, hutchinson_samples):
    keys = jax.random.split(key, hutchinson_samples)

    @jax.jit
    def add_sample(k, trace_estimate):
        vec = jax.random.normal(keys[k], shape=(output_dim,))
        a = jacobianT_vector_product(vec)
        return trace_estimate + jnp.sum((a ** 2) * inv_precision) / hutchinson_samples

    return jax.lax.fori_loop(0, hutchinson_samples, add_sample, 0.0)


def score_slug_from_z(decoder_module, fit, batch_stats_decoder, z_batch, output_shape, key, hutchinson_samples=500):
    output_dim = int(np.prod(output_shape))

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def score_fun(z, key, hutchinson_samples):
        jtvp = _jacobianT_vector_product_from_z(decoder_module, fit, batch_stats_decoder, z, output_shape)
        return _hutchinson_trace_low_rank(jtvp, fit["slug_inv_sqrt_ggn_vp"], output_dim, key, hutchinson_samples)

    return score_fun(z_batch, key, hutchinson_samples)


def score_lae_from_z(decoder_module, fit, batch_stats_decoder, z_batch, output_shape, key, tau=1.0, hutchinson_samples=500):
    output_dim = int(np.prod(output_shape))
    inv_precision = 1.0 / (fit["lae_diag_fisher"] + tau)

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def score_fun(z, key, hutchinson_samples):
        jtvp = _jacobianT_vector_product_from_z(decoder_module, fit, batch_stats_decoder, z, output_shape)
        return _hutchinson_trace_diag(jtvp, inv_precision, output_dim, key, hutchinson_samples)

    return score_fun(z_batch, key, hutchinson_samples)


def score_slug_from_images(fit, image_batch, output_shape, key, hutchinson_samples=500):
    output_dim = int(np.prod(output_shape))

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def score_fun(datapoint, key, hutchinson_samples):
        jtvp = _jacobianT_vector_product_from_image(fit, datapoint, output_shape)
        return _hutchinson_trace_low_rank(jtvp, fit["slug_inv_sqrt_ggn_vp"], output_dim, key, hutchinson_samples)

    return score_fun(image_batch, key, hutchinson_samples)


def score_lae_from_images(fit, image_batch, output_shape, key, tau=1.0, hutchinson_samples=500):
    output_dim = int(np.prod(output_shape))
    inv_precision = 1.0 / (fit["lae_diag_fisher"] + tau)

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def score_fun(datapoint, key, hutchinson_samples):
        jtvp = _jacobianT_vector_product_from_image(fit, datapoint, output_shape)
        return _hutchinson_trace_diag(jtvp, inv_precision, output_dim, key, hutchinson_samples)

    return score_fun(image_batch, key, hutchinson_samples)
