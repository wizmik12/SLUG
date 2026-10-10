"""NOTE: GGN-vector product primitive shared by every method here. `HJ_tree = J_tree` sets
H = I, i.e. the GGN of the PIXEL-MSE likelihood only -- the VGG perceptual term the model
was trained with (Helpers.py:28-31) is deliberately NOT in the curvature, because
exp(-perceptual) is not a normalised density and has no likelihood interpretation. This is
the documented root cause of these scores being "MSE-flavoured": strong MSE correlations,
weak LPIPS ones. See src/uncertainty/METHODS.md for the full taxonomy.
"""
import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util

import time


#####################################
# Generalize Gauss Newtown products #

def get_ggn_vector_product_dataloader(
        params_dict,
        model: flax.linen.Module,
        dataloader
    ):
    params = params_dict['params']
    batch_stats = params_dict['batch_stats']
    model_apply = lambda data, p: model.apply(
        {'params': p, 'batch_stats': batch_stats},
        x=data,
        rng=jax.random.PRNGKey(0),
        train=False,
        )[0]

    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.array(flatten_util.ravel_pytree(params)[0])

    @jax.jit
    def ggn_tree_product_batch(X, tree):
        model_on_data = functools.partial(model_apply, X)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        
        HJ_tree = J_tree

        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree
    
    @jax.jit
    def ggn_vector_product_batch(X, v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product_batch(X, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    x_init, _ = next(iter(dataloader))
    x_init = jnp.asarray(x_init)

    start = time.time()
    ggn_vector_product_batch(x_init, flatten_param)
    print(f"One BATCH GGN vp took {time.time()-start:.3f} seconds")
    start = time.time()
    ggn_vector_product_batch(x_init, jnp.ones_like(flatten_param))
    print(f"Again (now compiled)...... it took {time.time()-start:.3f} seconds")
    
    
    def ggn_vector_product_dataloader(v):
        """
        Plain Python loop over the (host-side) dataloader, calling the jitted
        per-batch product each iteration.

        NOT wrapped in jax.pure_callback: that wrapping was needed only when this
        function could be invoked from inside a jax.lax.fori_loop / other traced
        context, where a host-side dataloader loop can't be traced directly. Since
        the caller (uncertainty/lanczos.py's Lanczos loop) is itself now a plain
        Python loop (see lanczos.py comment), mv_prod is always called from
        untraced Python, so pure_callback is unnecessary here.

        This matters a lot for speed, not just style: pure_callback demotes `v`
        to a host numpy array for the callback body, so every one of the ~200
        per-batch calls inside had to re-upload the full parameter vector
        (~50M floats here) to the GPU instead of reusing the resident device
        array. Measured effect on an isolated GPU (confirmed 2026-08-06, see
        revision_part1.md): the callback-wrapped version took 342s for a single
        mv_prod call over 25 batches, vs ~8s (25 x 0.186s compiled batch calls +
        ~3s dataloader) once the callback layer is removed and v stays a JAX
        device array throughout -- a ~40x difference, and the dominant cost in
        SLUG's real-scale Lanczos fit.
        """
        result = jnp.zeros_like(v)
        for X, _ in dataloader:
            X = jnp.asarray(X)
            result_batch = ggn_vector_product_batch(X, v)
            result += result_batch
        return result

    return ggn_vector_product_dataloader




def get_jacobianT_vector_product(
        params_dict,
        model: flax.linen.Module,
        datapoint: jax.Array,
        output_shape,
    ):
    datapoint = jnp.expand_dims(datapoint, 0)
    B = datapoint.shape[0] # this will be 1
    
    params = params_dict['params']
    batch_stats = params_dict['batch_stats']
    model_on_data = lambda p: model.apply(
        {'params': p, 'batch_stats': batch_stats},
        x=datapoint,
        rng=jax.random.PRNGKey(0),
        train=False,
        )[0]

    _, model_on_data_vjp = jax.vjp(model_on_data, params)
    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    @jax.jit
    def jacobianT_vector_product(vector):
        # data times output space -> parameter space 
        batch_shape = (B, *output_shape)
        vector = vector.reshape(batch_shape)
        Jt_vector = model_on_data_vjp(vector)[0]
        return vectorize_fun(Jt_vector)
    
    return jacobianT_vector_product
