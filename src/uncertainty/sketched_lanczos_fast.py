"""Fast-scoring variant of SLUG (see sketched_lanczos_uncertainty.py).

The fit is IDENTICAL -- same Lanczos, same sketch, same PCA. Only the scoring
path is restructured.

Why: in the original, `score_fun_trace_approx` is vmapped but NOT jitted (the
`@jax.jit` on it is commented out), so the inner `@jax.jit` fragments -- the
model VJP and the Hutchinson body -- are dispatched and recompiled through
vmap's batching rules on every call. On the 50M conv VAE that overhead hides
behind cheap compiles. On an attention model (e.g. diffusers FlaxAutoencoderKL)
each compile is expensive and the path either blows up in XLA or degrades to
~460 s/image.

Here the whole per-image score is ONE top-level jitted function taking the
datapoint as an argument (not a closure), so the model is linearized and
compiled once and reused for every image. Images are then mapped over in a
plain Python loop of compiled calls.

Bit-exactness: the original vmaps with in_axes=(0,None,None), i.e. every image
shares the same Hutchinson key -- common random numbers across images. That is
preserved here, so results match the original to floating-point reassociation.
"""

import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from uncertainty.autodiff import get_ggn_vector_product_dataloader
from uncertainty.lanczos import low_memory_lanczos
from uncertainty.sketch import SRFT_sketch, get_optimal_padding
from uncertainty.sketched_lanczos_uncertainty import compute_num_params, get_subset_loader


def low_memory_lanczos_score_fun_fast(
        model,
        params_dict,
        train_loader,
        output_shape=(128, 128, 3),
        sketch_size=1000,
        sketch_padding=None,
        lanczos_iterations=5,
        lanczos_eigenvectors=None,
        seed=1,
        train_batch_size=32,
        subsample_trainset=None,
        hutchinson_samples=500,
    ):
    if lanczos_eigenvectors is None:
        lanczos_eigenvectors = int(0.9 * lanczos_iterations)

    n_params = compute_num_params(params_dict["params"])
    if sketch_padding is None and sketch_size is not None:
        sketch_padding = get_optimal_padding(n_params)

    trainset_size = len(train_loader.dataset) if subsample_trainset is None else subsample_trainset
    train_loader = get_subset_loader(
        train_loader, trainset_size, batch_size=train_batch_size, drop_last=True,
    )

    # ---------------------------------------------------------------- fit
    ggn_vector_product = get_ggn_vector_product_dataloader(params_dict, model, train_loader)

    key_sketch, key_lanczos = jax.random.split(jax.random.PRNGKey(seed))
    if sketch_size is None:
        print(f"NO sketch (dense low-rank LLA): eigenvectors kept at full dimension {n_params}")
        sketch_op = None
    else:
        print(f"Use srft sketch with num params {n_params} and padding {sketch_padding}")
        sketch_op = SRFT_sketch(key_sketch, n_params, sketch_size, padding=sketch_padding)

    print(f"Starting Lanczos, which will do {lanczos_iterations} GGN vps")
    start = time.time()
    eigenvec, eigenval = low_memory_lanczos(
        key_lanczos, ggn_vector_product, n_params, lanczos_iterations, sketch_op)
    print(f"Lanczos {lanczos_iterations} iterations, dataset size {trainset_size}, "
          f"{n_params} params -> {time.time()-start:.3f} s")

    U, S, _ = np.linalg.svd(eigenvec @ jnp.diag(eigenval), full_matrices=False)
    if lanczos_eigenvectors < len(S):
        threshold = sorted(S, reverse=True)[lanczos_eigenvectors]
        eigenvec = U[:, S > threshold]
        eigenval = S[S > threshold]
    else:
        eigenvec, eigenval = U, S
    eigenvec = jnp.asarray(eigenvec)

    if sketch_op is None:
        def inv_sqrt(vector):
            return vector.T @ eigenvec
    else:
        def inv_sqrt(vector):
            return (sketch_op @ vector).T @ eigenvec

    # ------------------------------------------------------------- scoring
    params = params_dict["params"]
    batch_stats = params_dict["batch_stats"]
    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    output_dim = int(np.prod(output_shape))
    batch_shape = (1, *output_shape)

    # NOTE: no inner jit anywhere below. One compile for the whole per-image score.
    @functools.partial(jax.jit, static_argnums=(2,))
    def score_one_both(datapoint, key, n_samples):
        """Returns BOTH Hutchinson terms, not only their difference.

        variance_I = ||J^T v||^2        estimates trace(J J^T) = ||J||_F^2: the model's total
                                        output sensitivity on this image. No posterior in it.
        variance_P = ||U_k^T J^T v||^2  the part of that sensitivity lying inside the top-k
                                        subspace the training data pinned down.

        The SLUG score is the difference, which is an absolute variance and therefore scales
        with image complexity. The RATIO (I - P)/I is the *fraction* of the image's
        sensitivity that the posterior leaves unconstrained, which divides complexity out and
        is the natural candidate for an out-of-distribution signal.
        """
        def model_on_data(p_):
            return model.apply(
                {"params": p_, "batch_stats": batch_stats},
                x=jnp.expand_dims(datapoint, 0),
                rng=jax.random.PRNGKey(0),
                train=False,
            )[0]

        _, vjp_fn = jax.vjp(model_on_data, params)
        keys = jax.random.split(key, n_samples)

        def body(k, acc):
            aI, aP = acc
            vec = jax.random.normal(keys[k], shape=(output_dim,))
            col = vectorize_fun(vjp_fn(vec.reshape(batch_shape))[0])
            vI = jnp.sum(col ** 2)
            vP = jnp.sum(inv_sqrt(col) ** 2)
            return (aI + vI / n_samples, aP + vP / n_samples)

        return jax.lax.fori_loop(0, n_samples, body, (0.0, 0.0))

    def score_fun_both(datapoints, key, n_samples=None):
        """(slug_score, variance_I, variance_P) per image."""
        n = hutchinson_samples if n_samples is None else n_samples
        out = [score_one_both(jnp.asarray(d), key, n) for d in datapoints]
        vI = jnp.stack([o[0] for o in out])
        vP = jnp.stack([o[1] for o in out])
        return vI - vP, vI, vP

    @functools.partial(jax.jit, static_argnums=(2,))
    def score_one(datapoint, key, n_samples):
        def model_on_data(p):
            return model.apply(
                {"params": p, "batch_stats": batch_stats},
                x=jnp.expand_dims(datapoint, 0),
                rng=jax.random.PRNGKey(0),
                train=False,
            )[0]

        _, vjp_fn = jax.vjp(model_on_data, params)
        keys = jax.random.split(key, n_samples)

        def body(k, acc):
            vec = jax.random.normal(keys[k], shape=(output_dim,))
            col = vectorize_fun(vjp_fn(vec.reshape(batch_shape))[0])
            variance_I = jnp.sum(col ** 2)
            col = inv_sqrt(col)
            variance_P = jnp.sum(col ** 2)
            return acc + (variance_I - variance_P) / n_samples

        return jax.lax.fori_loop(0, n_samples, body, 0.0)

    def score_fun_trace_approx(datapoints, key, n_samples=None):
        """Same signature/semantics as the original vmapped scorer."""
        n = hutchinson_samples if n_samples is None else n_samples
        datapoints = jnp.asarray(datapoints)
        # common random numbers across images, exactly as in_axes=(0,None,None)
        return jnp.stack([score_one(d, key, n) for d in datapoints])

    return {"approx": score_fun_trace_approx, "score_one": score_one}


def build_fast_scorer(model, posterior, output_shape, hutchinson_samples=500):
    """Build the fast scorer on an ALREADY-FITTED posterior.

    `posterior` is the '_posterior' entry returned by
    sketched_lanczos_uncertainty.low_memory_lanczos_score_fun. This lets the original and
    fast scorers be compared on identical eigenvectors, which is the only way to attribute a
    difference to the scoring path rather than to Lanczos re-fit noise.
    """
    eigenvec = posterior["eigenvec"]
    sketch_op = posterior["sketch_op"]
    batch_stats = posterior["batch_stats"]
    params = posterior["params"]

    if sketch_op is None:
        def inv_sqrt(vector):
            return vector.T @ eigenvec
    else:
        def inv_sqrt(vector):
            return (sketch_op @ vector).T @ eigenvec

    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]
    output_dim = int(np.prod(output_shape))
    batch_shape = (1, *output_shape)

    @functools.partial(jax.jit, static_argnums=(2,))
    def score_one_both(datapoint, key, n_samples):
        """Returns BOTH Hutchinson terms, not only their difference.

        variance_I = ||J^T v||^2        estimates trace(J J^T) = ||J||_F^2: the model's total
                                        output sensitivity on this image. No posterior in it.
        variance_P = ||U_k^T J^T v||^2  the part of that sensitivity lying inside the top-k
                                        subspace the training data pinned down.

        The SLUG score is the difference, which is an absolute variance and therefore scales
        with image complexity. The RATIO (I - P)/I is the *fraction* of the image's
        sensitivity that the posterior leaves unconstrained, which divides complexity out and
        is the natural candidate for an out-of-distribution signal.
        """
        def model_on_data(p_):
            return model.apply(
                {"params": p_, "batch_stats": batch_stats},
                x=jnp.expand_dims(datapoint, 0),
                rng=jax.random.PRNGKey(0),
                train=False,
            )[0]

        _, vjp_fn = jax.vjp(model_on_data, params)
        keys = jax.random.split(key, n_samples)

        def body(k, acc):
            aI, aP = acc
            vec = jax.random.normal(keys[k], shape=(output_dim,))
            col = vectorize_fun(vjp_fn(vec.reshape(batch_shape))[0])
            vI = jnp.sum(col ** 2)
            vP = jnp.sum(inv_sqrt(col) ** 2)
            return (aI + vI / n_samples, aP + vP / n_samples)

        return jax.lax.fori_loop(0, n_samples, body, (0.0, 0.0))

    def score_fun_both(datapoints, key, n_samples=None):
        """(slug_score, variance_I, variance_P) per image."""
        n = hutchinson_samples if n_samples is None else n_samples
        out = [score_one_both(jnp.asarray(d), key, n) for d in datapoints]
        vI = jnp.stack([o[0] for o in out])
        vP = jnp.stack([o[1] for o in out])
        return vI - vP, vI, vP

    @functools.partial(jax.jit, static_argnums=(2,))
    def score_one(datapoint, key, n_samples):
        def model_on_data(p):
            return model.apply(
                {"params": p, "batch_stats": batch_stats},
                x=jnp.expand_dims(datapoint, 0),
                rng=jax.random.PRNGKey(0),
                train=False,
            )[0]

        _, vjp_fn = jax.vjp(model_on_data, params)
        keys = jax.random.split(key, n_samples)

        def body(k, acc):
            vec = jax.random.normal(keys[k], shape=(output_dim,))
            col = vectorize_fun(vjp_fn(vec.reshape(batch_shape))[0])
            variance_I = jnp.sum(col ** 2)
            col = inv_sqrt(col)
            variance_P = jnp.sum(col ** 2)
            return acc + (variance_I - variance_P) / n_samples

        return jax.lax.fori_loop(0, n_samples, body, 0.0)

    def score_fun_trace_approx(datapoints, key, n_samples=None):
        n = hutchinson_samples if n_samples is None else n_samples
        return jnp.stack([score_one(jnp.asarray(d), key, n) for d in datapoints])

    # Returned as a SINGLE callable so the seven existing call sites keep working unchanged;
    # the both-terms variant hangs off it as an attribute rather than altering the signature.
    score_fun_trace_approx.both = score_fun_both
    return score_fun_trace_approx
