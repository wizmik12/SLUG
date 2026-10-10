"""Implements METHODS.md rows 1-2: SLUG (sketched low-rank) and, with sketch_size=None,
the dense low-rank Linearized Laplace baseline (`low_rank_lla` in Miani et al.).

Naming: "LLA" in the SLU paper means *Linearized* Laplace over the whole network -- NOT
last-layer Laplace (that is last_layer_laplace.py, which we call LLLA).

Gotcha: Lanczos vector storage is [n_iter, sketch_size] in the sketched path (iterations
nearly free) but [n_iter, n_params] in the DENSE path -- 30 iterations = 6.08 GB and OOMs.

See src/uncertainty/METHODS.md for the full taxonomy.
"""

import torch
import jax
import jax.numpy as jnp
from jax import flatten_util
import numpy as np
import time
import functools

from uncertainty.autodiff import get_ggn_vector_product_dataloader
from uncertainty.lanczos import low_memory_lanczos
from uncertainty.sketch import SRFT_sketch, get_optimal_padding


def compute_num_params(params):
    vector_params = flatten_util.ravel_pytree(params)[0]
    return vector_params.shape[0]

def get_subset_loader(
        dataloader,
        n_samples: int,
        batch_size = 128,
        shuffle: bool = False,
        drop_last: bool = True,
        seed = 0
    ):
    if len(dataloader.dataset) < n_samples:
        raise ValueError(f"Can't return a subset of size {n_samples} from a dataset of size {len(dataloader.dataset)}")
    sub_dataset, _ = torch.utils.data.random_split(
        dataloader.dataset, (n_samples, len(dataloader.dataset)-n_samples), generator=torch.Generator().manual_seed(seed)
    )
    return torch.utils.data.DataLoader(
                sub_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=0,
                pin_memory=False, 
                drop_last=drop_last
            )


def low_memory_lanczos_score_fun(
        model, 
        params_dict, 
        train_loader, 
        output_shape = (128,128,3),
        sketch_size = 1000,
        sketch_padding = None,
        lanczos_iterations = 5,
        lanczos_eigenvectors = None, #this has to be less (or equal) than "lanczos_iterations"
        seed = 1,
        train_batch_size = 32, # reduce this it you get memory error
        subsample_trainset = None
    ):
    if lanczos_eigenvectors is None:
        lanczos_eigenvectors = int(0.9 * lanczos_iterations)
    

    n_params = compute_num_params(params_dict["params"])
    if sketch_padding is None and sketch_size is not None:
        start = time.time()
        sketch_padding = get_optimal_padding(n_params)
        print(f"Computing optimal padding took {time.time()-start} seconds")

    trainset_size = len(train_loader.dataset) if subsample_trainset is None else subsample_trainset
    # subsample train dataloader
    train_loader = get_subset_loader(
        train_loader,
        trainset_size,
        batch_size = train_batch_size,
        drop_last = True, #this is quite important for compilation reasons
    )

    # define efficient GGN vector product
    ggn_vector_product = get_ggn_vector_product_dataloader(
        params_dict,
        model,
        train_loader
    )

    #start = time.time()
    #ggn_vector_product(jax.random.normal(jax.random.PRNGKey(0), shape=(n_params,)))
    #print(f"One GGN vp took {time.time()-start} seconds")
    #start = time.time()
    #ggn_vector_product(jax.random.normal(jax.random.PRNGKey(1), shape=(n_params,)))
    #print(f"Again (now compiled)...... it took {time.time()-start} seconds")

    # define the sketch operator (if needed).
    # sketch_size=None runs Lanczos WITHOUT sketching and keeps full-dimension eigenvectors:
    # that is the dense low-rank Linearized Laplace baseline (`low_rank_lla` in Miani et al.,
    # "Sketched Lanczos uncertainty score", NeurIPS 2024). It is the memory-matched
    # competitor to SLUG: the stored summary costs n_eigenvec * n_params floats here, versus
    # n_eigenvec * sketch_size for the sketched version -- which is exactly the trade the SLU
    # paper is about. Default (an int) preserves the existing sketched behaviour.
    key_sketch, key_lanczos = jax.random.split(jax.random.PRNGKey(seed))
    if sketch_size is None:
        print(f"NO sketch (dense low-rank LLA): eigenvectors kept at full dimension {n_params}")
        sketch_op = None
    else:
        print(f"Use srft sketch with num params {n_params} and padding {sketch_padding} --> fake num params = {sketch_padding+n_params} must NOT have prime factors >127 (thanks JAX fft)")
        sketch_op = SRFT_sketch(key_sketch, n_params, sketch_size, padding=sketch_padding)
        
    # perform Lanzos and find eigenval/eigenvec pairs
    print(f"Starting Lanzcos, which will do {lanczos_iterations} GGN vps")
    start = time.time()
    eigenvec, eigenval = low_memory_lanczos(key_lanczos, ggn_vector_product, n_params, lanczos_iterations, sketch_op)
    print(f"Lanczos {lanczos_iterations} iterations, dataset size {trainset_size}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
    print(f"returned {len(eigenval)} eigenvals = {eigenval[:5]} ... {eigenval[-5:]}")

    # orthogonnalize and select the first (good) 'n_eigenvec' vectors
    start = time.time()
    print("Doing PCA...")
    U, S, _ = np.linalg.svd(eigenvec @ jnp.diag(eigenval), full_matrices=False)
    if lanczos_eigenvectors<len(S):
        threshold = sorted(S, reverse=True)[lanczos_eigenvectors]
        eigenvec = U[:, S > threshold] # achtung, these are not really eigenvectors
        eigenval = S[S > threshold]
    else:
        eigenvec = U
        eigenval = S
    eigenvec = jnp.asarray(eigenvec)
    eigenval = jnp.asarray(eigenval)
    print(f"PCA took {time.time()-start:.3f} seconds")

    if sketch_op is None:
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return vector.T @ eigenvec
    else:
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return (sketch_op @ vector).T @ eigenvec



    #build efficient Jacobian vector product
    batch_stats = params_dict['batch_stats']
    @jax.jit
    def model_apply(datapoint, params):
        datapoint = jnp.expand_dims(datapoint, 0)
        return model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x=datapoint,
            rng=jax.random.PRNGKey(0),
            train=False,
            )[0]
    params = params_dict['params']
    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    def get_jacobianT_vector_product(model_apply, datapoint, output_shape):

        model_on_data = functools.partial(model_apply, datapoint)
        _, model_on_data_vjp = jax.vjp(model_on_data, params)

        @jax.jit
        def jacobianT_vector_product(vector):
            # data times output space -> parameter space 
            batch_shape = (1, *output_shape)
            vector = vector.reshape(batch_shape)
            Jt_vector = model_on_data_vjp(vector)[0]
            return vectorize_fun(Jt_vector)
        
        return jacobianT_vector_product
    

    # build covariance "observers"
    def get_uncertainty_diagonal(
        J_vector_product,
        M_vector_product,
        output_dim
    ):
        base_vectors = jnp.concatenate((jnp.zeros((output_dim-1,)), jnp.ones((1,)), jnp.zeros((output_dim-1,))))
        get_vec = lambda k: jax.lax.dynamic_slice(base_vectors, (output_dim-k-1,), (output_dim,))
        def get_norm(i):
            e_i = get_vec(i)
            col_i = J_vector_product(e_i)
            variance_I = jnp.sum(col_i ** 2)
            col_i = M_vector_product(col_i)
            variance_P = jnp.sum(col_i ** 2)
            diagonal_i = variance_I - variance_P
            return diagonal_i
        
        def scan_body(carry, i):
            return carry, get_norm(i)
        
        _, diagonal = jax.lax.scan(scan_body, None, jnp.arange(0, output_dim))
        
        return diagonal
    
    def get_uncertainty_trace_approx(
        J_vector_product,
        M_vector_product,
        output_dim,
        key,
        hutchinson_samples = None
    ):
        keys = jax.random.split(key, hutchinson_samples)
        @jax.jit
        def add_diagonal_sample(k, trace_estimate):
            key = keys[k]
            vec = jax.random.normal(key, shape=(output_dim,))
            col = J_vector_product(vec)
            variance_I = jnp.sum(col ** 2)
            col = M_vector_product(col)
            variance_P = jnp.sum(col ** 2)
            single_sample_trace_estimate = variance_I - variance_P
            trace_estimate += single_sample_trace_estimate/hutchinson_samples 
            return trace_estimate
        
        trace_estimate = jax.lax.fori_loop(0, hutchinson_samples, add_diagonal_sample, 0.)          
        
        return trace_estimate

    
    output_dim = 1
    for d in output_shape:
        output_dim *= d
        
      


    @jax.vmap
    @jax.jit
    def score_fun_diagonal(datapoint):
        datapoint = jnp.asarray(datapoint)

        jacobianT_vector_product = get_jacobianT_vector_product(model_apply, datapoint, output_shape)
        
        variance_diagonal = get_uncertainty_diagonal(
            jacobianT_vector_product,
            inv_sqrt_approx_ggn_vector_product,
            output_dim,
        )
        variance_diagonal = variance_diagonal.reshape(output_shape)
        return variance_diagonal
    

    @functools.partial(jax.vmap, in_axes=(0,None,None)) # Consider 0 for the key if batchsize is >1
    #@jax.jit
    def score_fun_trace_approx(datapoint, key, hutchinson_samples = None):
        datapoint = jnp.asarray(datapoint)
        print("output shape", output_shape)
        jacobianT_vector_product = get_jacobianT_vector_product(model_apply, datapoint, output_shape)
        
        variance_trace = get_uncertainty_trace_approx(
            jacobianT_vector_product,
            inv_sqrt_approx_ggn_vector_product,
            output_dim,
            key,
            hutchinson_samples
        )
        return variance_trace
    
    # '_posterior' is additive and used only by validation//diagnostics: it exposes the
    # fitted summary so an alternative scorer can be built on the SAME fit. Lanczos at this
    # scale is not bit-reproducible across runs (it returns eigenvalues of both signs, i.e.
    # it has lost orthogonality), so comparing two scorers each with its own fit conflates
    # scorer differences with fit noise.
    return {'exact': score_fun_diagonal, 'approx': score_fun_trace_approx,
            '_posterior': {'eigenvec': eigenvec, 'sketch_op': sketch_op,
                           'batch_stats': batch_stats, 'params': params}}

