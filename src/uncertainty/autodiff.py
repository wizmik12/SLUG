"""
GGN-vector products, shared by every method here.

H = I, so the curvature assumes a Gaussian likelihood (MSE)
"""
import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util

import time


#####################################
# Generalized Gauss Newton products #
#####################################

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
        Iterates over the dataloader using a plain Python loop, calling the
        JIT-compiled matrix-vector product for each batch.
        
        There is no need for jax.pure_callback here because the Lanczos loop
        runs in regular Python rather than a traced JAX context. This also
        avoids unnecessary CPU-GPU transfers: without the callback, the input
        vector stays on the GPU and can be reused across batches instead of
        being repeatedly transferred back and forth.
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
