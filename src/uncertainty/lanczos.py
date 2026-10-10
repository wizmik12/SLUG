import os, time
import jax
import jax.numpy as jnp
from scipy import linalg
from matfree import decomp

######################
# low memory lanczos #
# performed without reothogonalization.

def low_memory_lanczos(
        key: jax.random.PRNGKey, 
        mv_prod, 
        dim: int, 
        n_iter: int, 
        sketch_op = None
    ):
    key, key_lancz = jax.random.split(key)
    if sketch_op is None:
        skvs, alphas, betas = low_memory_lanczos_to_tridiag(key_lancz, mv_prod, dim, n_iter)
    else:
        skvs, alphas, betas = low_memory_lanczos_to_tridiag_sketch(key_lancz, mv_prod, dim, n_iter, sketch_op)
        
    eig_val, eig_vec = linalg.eigh_tridiagonal(jnp.array(alphas), jnp.array(betas), lapack_driver='stebz')
    eig_vec, eig_val = jnp.array(eig_vec), jnp.array(eig_val)
    sketched_eig_vec = skvs.T @ eig_vec
    # flip eigenvalues and eigenvectors so that they are in decreasing order
    eig_val = jnp.flip(eig_val)
    sketched_eig_vec = jnp.flip(sketched_eig_vec, axis=1)
    return sketched_eig_vec, eig_val


#@partial(jax.jit, static_argnames=['mv_prod', 'dim', 'n_iter'])
def low_memory_lanczos_to_tridiag(key, mv_prod, dim, n_iter):
    v_old = jax.random.normal(key, shape=(dim, ))
    v_old /= jax.numpy.sqrt(v_old.T @ v_old)

    skvs = jnp.zeros((n_iter, v_old.shape[0]))
    skvs = skvs.at[0].set(v_old)
    alphas = jnp.zeros(n_iter)
    betas = jnp.zeros(n_iter - 1)

    wp = mv_prod(v_old)
    a = jnp.dot(wp, v_old)
    w = wp - a * v_old
    alphas = alphas.at[0].set(a)

    def lanczos_step(i, state):
        (w, betas, skvs, alphas, v_old) = state
        b = jnp.linalg.norm(w)
        betas = betas.at[i - 1].set(b)
        v_new = w / b
        skvs = skvs.at[i].set(v_new)
        wp = mv_prod(v_new)
        a = jnp.dot(wp, v_new)
        alphas = alphas.at[i].set(a)
        w = wp - a * v_new - b * v_old
        v_old = v_new
        return (w, betas, skvs, alphas, v_old)

    # Plain Python loop, not jax.lax.fori_loop: mv_prod goes through
    # jax.pure_callback (it scans the training set on the host per call), and tracing
    # that inside a fori_loop -- fusing n_iter sequential host round-trips into one
    # compiled XLA program -- causes severe compile-time blowup in practice (confirmed
    # 2026-08-05, see revision_part1.md Sec 0.f). A Python loop dispatches each
    # iteration separately instead, avoiding the fusion pathology. Same operations,
    # same result, just not traced as a single fused loop.
    state = (w, betas, skvs, alphas, v_old)
    _t0 = time.time()
    for i in range(1, n_iter):
        state = lanczos_step(i, state)
        if os.environ.get("LANCZOS_PROGRESS"):
            el = time.time() - _t0
            print(f"    lanczos {i+1}/{n_iter}  {el/i:.1f}s/iter  "
                  f"eta {(n_iter-1-i)*el/i/60:.1f} min", flush=True)
    (w, betas, skvs, alphas, v_old) = state

    return skvs, alphas, betas


#@partial(jax.jit, static_argnames=['mv_prod', 'dim', 'n_iter', 'sketch_op'])
def low_memory_lanczos_to_tridiag_sketch(key, mv_prod, dim, n_iter, sketch_op):
    v_old = jax.random.normal(key, shape=(dim, ))
    v_old /= jax.numpy.sqrt(v_old.T @ v_old)

    skv_old = sketch_op @ v_old
    skvs = jnp.zeros((n_iter, skv_old.shape[0]))
    skvs = skvs.at[0].set(skv_old)
    alphas = jnp.zeros(n_iter)
    betas = jnp.zeros(n_iter - 1)

    wp = mv_prod(v_old)
    a = jnp.dot(wp, v_old)
    w = wp - a * v_old
    alphas = alphas.at[0].set(a)

    def lanczos_step(i, state):
        (w, betas, skvs, alphas, v_old) = state
        b = jnp.linalg.norm(w)
        betas = betas.at[i - 1].set(b)
        v_new = w / b
        skvs = skvs.at[i].set(sketch_op @ v_new)
        wp = mv_prod(v_new)
        a = jnp.dot(wp, v_new)
        alphas = alphas.at[i].set(a)
        w = wp - a * v_new - b * v_old
        v_old = v_new
        return (w, betas, skvs, alphas, v_old)

    # See the comment in low_memory_lanczos_to_tridiag above -- same fix, same reason.
    state = (w, betas, skvs, alphas, v_old)
    _t0 = time.time()
    for i in range(1, n_iter):
        state = lanczos_step(i, state)
        if os.environ.get("LANCZOS_PROGRESS"):
            el = time.time() - _t0
            print(f"    lanczos {i+1}/{n_iter}  {el/i:.1f}s/iter  "
                  f"eta {(n_iter-1-i)*el/i/60:.1f} min", flush=True)
    (w, betas, skvs, alphas, v_old) = state

    return skvs, alphas, betas
