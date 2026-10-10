"""
SWAG (Maddox et al., NeurIPS 2019, "A Simple Baseline for Bayesian Uncertainty in Deep
Learning") for the VAE -- a sampling-based Bayesian baseline that, unlike SGHMC/BVAE, is
compatible with this model's training loss.

Why SWAG and not SGHMC: SGHMC needs grad of a *normalised* log-likelihood. This VAE trains on
  loss = kl_scale*KL + MSE + feature_scale*VGG_perceptual        (Helpers.py:28-31)
and exp(-perceptual) is not a normalised density in x -- its partition function depends on
theta and is intractable, so an SGHMC chain would not target the claimed posterior. SWAG never
evaluates a density; it only *observes the SGD trajectory*. So collection runs under the exact
training loss, perceptual term included, and characterises the model you actually trained.

Procedure: continue training from the checkpoint at a CONSTANT learning rate, record the
iterates, fit a Gaussian to them, then sample weights and measure reconstruction disagreement.

  mean  theta_SWA = (1/T) sum_t theta_t
  diag  Sigma_d   = (1/T) sum_t theta_t^2  -  theta_SWA^2
  score per image = mean over pixels of the across-sample variance of the reconstruction

That score is deliberately the SAME definition ensemble_var uses (ensemble_var.py:140-142:
per_pixel_var.mean(axis=(1,2,3))), so SWAG and the deep ensemble are directly comparable --
both measure decoder disagreement, one across posterior samples, one across training runs.

Memory: theta_SWA + second moment = 2p. (Full SWAG adds K deviation columns for (2+K)p; at the
standard K=20 that is 22p, far outside the 3p budget used for the low-rank LLA comparison --
worth reporting rather than glossing.)

Deviations from the paper, stated rather than hidden:
  * Maddox collects under SGD; we use Adam at constant LR, matching the optimiser the model was
    trained with, to stay in the same basin.
  * BatchNorm statistics are not recomputed per sampled weight vector (the paper recommends it);
    we reuse the collected running statistics. Cheap and adequate here since batch_stats are
    already running averages, but it is an approximation.
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import flatten_util
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

import Helpers as hf
import data
from VAE import VAE
from train_vae import TrainState


class ArgsShim:
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--dataset", choices=["fitz", "ai4skin"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nepoch", type=int, default=500)
    p.add_argument("--swa_epochs", type=int, default=40, help="collection epochs at constant LR")
    p.add_argument("--swa_lr", type=float, default=1e-3,
                   help="CONSTANT collection LR. Miani's repo uses 1e-3 (bash/scoring/*/swag.sh). "
                        "Too small -> degenerate (all samples identical); too large -> leaves the "
                        "basin. Our training used peak 1e-4, end 1e-5, so 1e-3 is aggressive here.")
    p.add_argument("--swag_optimizer", choices=["sgd", "adam"], default="sgd",
                   help="sgd+momentum matches Miani/Maddox; adam matches how this VAE was trained")
    p.add_argument("--swag_momentum", type=float, default=0.99, help="Miani: 0.99")
    p.add_argument("--swag_collect_interval", type=int, default=100,
                   help="collect a snapshot every N optimiser STEPS (Miani: 100). 0 = per epoch.")
    p.add_argument("--swag_n_vec", type=int, default=3,
                   help="rank of the low-rank deviation matrix D. Miani uses 3 (3p budget) and "
                        "10 (10p budget) -- n_vec doubles as the memory multiplier there. 0 = "
                        "diagonal-only SWAG.")
    p.add_argument("--n_samples", type=int, default=30, help="weight samples drawn at test time")
    p.add_argument("--var_scale", type=float, default=1.0, help="scale on the sampled std")
    p.add_argument("--test_size", type=int, default=512)
    p.add_argument("--train_size", type=int, default=1668)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--kl_scale", type=float, default=0.5)
    p.add_argument("--feature_scale", type=float, default=1.0)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--ch_multi", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=1)
    p.add_argument("--latent_channels", type=int, default=128)
    p.add_argument("--block_widths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dataset_root", type=str, default="/work/work_mik/2026Diff_UQ/FitzFairness/data/images/")
    p.add_argument("--ai4skin_patches_csv", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/metadata/patches.csv")
    p.add_argument("--ai4skin_patches_root", type=str, default="/work/work_mik/2026Diff_UQ/FusoFairness/data/")
    cli = p.parse_args()

    args = ArgsShim()
    args.dataset = cli.dataset
    args.dataset_root = cli.dataset_root
    args.ai4skin_patches_csv = cli.ai4skin_patches_csv
    args.ai4skin_patches_root = cli.ai4skin_patches_root
    args.image_size = cli.image_size
    args.train_size = cli.train_size
    args.kl_scale = cli.kl_scale
    args.feature_scale = cli.feature_scale
    args.use_dropout = False

    train_dataset, test_a1, test_a2, attr1, attr2 = data.train_test_setting(
        cli.dataset_root, 0.5, 0, cli.test_size, args)
    train_loader = DataLoader(train_dataset, batch_size=cli.batch_size, shuffle=True,
                              num_workers=0, collate_fn=hf.numpy_collate, drop_last=True)

    model = VAE(channel_in=3, ch=cli.ch_multi, blocks=cli.block_widths,
                latent_channels=cli.latent_channels, num_res_blocks=cli.num_res_blocks,
                deep_model=False, use_dropout=False)
    save_dir = os.path.join(cli.results_dir, cli.dataset)
    base = os.path.join(save_dir, "Results", f"0_seed{cli.seed}", "0.5")
    ckpt = os.path.join(base, "model", f"fitz17_{cli.latent_channels}_{cli.image_size}{cli.nepoch}")
    dummy = np.zeros((1, cli.image_size, cli.image_size, 3), dtype=np.float32)
    _ = model.init(jax.random.PRNGKey(0), dummy, train=True, rng=jax.random.PRNGKey(0))
    sd = checkpoints.restore_checkpoint(ckpt_dir=ckpt, target=None)

    # constant LR -- SGD/Adam stops converging and instead explores the basin; that stationary
    # spread IS the thing SWAG fits a Gaussian to.
    # Cache the ORIGINAL batch_stats BEFORE collection mutates state.batch_stats. Evaluating
    # theta_hat afterwards with post-collection statistics measures the wrong model -- observed
    # as the same checkpoint reporting MSE 0.007995 in one run and 0.006247 in another.
    batch_stats_0 = sd["batch_stats"]

    if cli.swag_optimizer == "sgd":
        optimizer = optax.chain(optax.clip(1.0), optax.sgd(cli.swa_lr, momentum=cli.swag_momentum))
    else:
        optimizer = optax.chain(optax.clip(1.0), optax.adam(cli.swa_lr))
    state = TrainState.create(apply_fn=model.apply, params=sd["params"],
                              batch_stats=sd["batch_stats"], tx=optimizer)

    @jax.jit
    def train_step(state, imgs, rng, rng_dropout):
        loss_fn = lambda prm: hf.elbo(model, prm, state.batch_stats, imgs, rng, rng_dropout, args, 1)
        ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads, batch_stats=ret[1][2]["batch_stats"]), ret[0]

    flat0, unravel = flatten_util.ravel_pytree(state.params)
    n_params = flat0.size
    print(f"=== SWAG: {n_params:,} params, {cli.swa_epochs} epochs @ constant lr={cli.swa_lr:g}, "
          f"{cli.n_samples} samples; memory = 2p = {2*n_params*4/2**20:.0f} MiB ===")

    # Snapshots every `swag_collect_interval` optimiser STEPS (Miani: 100), not per epoch --
    # far more snapshots for the same compute, which is what makes a rank-K deviation matrix
    # meaningful. D keeps only the most recent K deviations (a deque), as in Maddox.
    run_sum = jnp.zeros_like(flat0)
    run_sqsum = jnp.zeros_like(flat0)
    D = []                      # low-rank deviation columns, at most swag_n_vec of them
    T = 0
    rng = jax.random.PRNGKey(cli.seed + 1000)
    step = 0

    def snapshot(run_sum, run_sqsum, D, T):
        flat, _ = flatten_util.ravel_pytree(state.params)
        run_sum = run_sum + flat
        run_sqsum = run_sqsum + flat ** 2
        T += 1
        if cli.swag_n_vec > 0:
            D.append(np.asarray(flat - run_sum / T))   # deviation from the running mean
            if len(D) > cli.swag_n_vec:
                D.pop(0)
        return run_sum, run_sqsum, D, T

    for ep in range(cli.swa_epochs):
        losses = []
        for imgs, _ in train_loader:
            rng, k1, k2 = jax.random.split(rng, 3)
            state, loss = train_step(state, imgs, k1, k2)
            losses.append(float(loss))
            step += 1
            if cli.swag_collect_interval > 0 and step % cli.swag_collect_interval == 0:
                run_sum, run_sqsum, D, T = snapshot(run_sum, run_sqsum, D, T)
        if cli.swag_collect_interval == 0:
            run_sum, run_sqsum, D, T = snapshot(run_sum, run_sqsum, D, T)
        if ep % 5 == 0 or ep == cli.swa_epochs - 1:
            print(f"  epoch {ep+1}/{cli.swa_epochs}  loss={np.mean(losses):.5f}  snapshots={T}")

    if T < 2:
        raise SystemExit(f"only {T} snapshots collected -- lower --swag_collect_interval")
    mean = run_sum / T
    var = jnp.maximum(run_sqsum / T - mean ** 2, 0.0)
    std = jnp.sqrt(var)
    Dmat = jnp.asarray(np.stack(D, axis=1)) if D else None   # [p, K]
    K = Dmat.shape[1] if Dmat is not None else 0
    mem = (2 + K) * n_params
    print(f"  {T} snapshots, low-rank K={K}; memory = mean+diag+D = (2+{K})p "
          f"= {mem*4/2**30:.2f} GiB = {mem/n_params:.0f}p")
    print(f"  SWA std over iterates: mean={float(std.mean()):.3e} max={float(std.max()):.3e} "
          f"(zero => degenerate collection, raise --swa_lr)")

    # Drift check. The `Error` column below is the ORIGINAL checkpoint's MSE, but theta_SWA
    # moves away from theta_hat during collection (more so at higher swa_lr / more epochs). If
    # the SWA mean is a materially different model, we would be scoring one model's uncertainty
    # against another model's errors. Report both MSEs so that is visible rather than assumed.
    theta_hat = unravel(flat0)
    theta_swa = unravel(mean)
    print(f"  drift ||theta_SWA - theta_hat||/||theta_hat|| = "
          f"{float(jnp.linalg.norm(mean - flat0) / jnp.linalg.norm(flat0)):.4e}")

    @jax.jit
    def reconstruct(params, imgs):
        recon, _, _ = model.apply({"params": params, "batch_stats": state.batch_stats},
                                  x=imgs, rng=jax.random.PRNGKey(0), train=False)
        return recon

    @jax.jit
    def reconstruct_orig(params, imgs):
        # theta_hat with its ORIGINAL batch_stats -- the reference model, unchanged.
        recon, _, _ = model.apply({"params": params, "batch_stats": batch_stats_0},
                                  x=imgs, rng=jax.random.PRNGKey(0), train=False)
        return recon

    sample_keys = jax.random.split(jax.random.PRNGKey(cli.seed + 2000), cli.n_samples)

    for group, test_ds in [(attr1, test_a1), (attr2, test_a2)]:
        loader = DataLoader(test_ds, batch_size=cli.eval_batch_size, shuffle=False,
                            num_workers=0, collate_fn=hf.numpy_collate)
        # Stream over samples and accumulate the first two moments of the reconstructions
        # instead of stacking [S, N, H, W, C]. Holding all S weight vectors (S x 202 MiB) or
        # all S reconstruction sets at once OOMs; this keeps memory at 2 accumulators of
        # [N, H, W, C] regardless of S.
        ids, run_sum, run_sqsum = None, None, None
        for si, k in enumerate(sample_keys):
            # Maddox full SWAG: theta = theta_SWA + (1/sqrt2) Sigma_d^{1/2} z1
            #                                  + (1/sqrt(2(K-1))) D z2
            k_a, k_b = jax.random.split(k)
            pert = std * jax.random.normal(k_a, shape=mean.shape)
            if Dmat is not None and K > 1:
                z2 = jax.random.normal(k_b, shape=(K,))
                pert = pert / jnp.sqrt(2.0) + (Dmat @ z2) / jnp.sqrt(2.0 * (K - 1))
            prm = unravel(mean + cli.var_scale * pert)
            recons, batch_ids_all = [], []
            for imgs, batch_ids in loader:
                recons.append(np.asarray(reconstruct(prm, np.asarray(imgs))))
                batch_ids_all.extend(list(batch_ids))
            r = np.concatenate(recons, axis=0).astype(np.float64)
            del prm, recons
            if run_sum is None:
                ids, run_sum, run_sqsum = batch_ids_all, np.zeros_like(r), np.zeros_like(r)
            run_sum += r
            run_sqsum += r ** 2
            del r
        S = cli.n_samples
        var_map = np.maximum(run_sqsum / S - (run_sum / S) ** 2, 0.0)   # [N,H,W,C]
        scores = var_map.mean(axis=(1, 2, 3))                            # [N]
        del run_sum, run_sqsum, var_map

        mse_by_id = {}
        mfile = os.path.join(base, f"metrics_{group}.csv")
        if os.path.exists(mfile):
            with open(mfile) as f:
                for row in csv.DictReader(f):
                    mse_by_id[row["Image Name"]] = float(row["MSE"])
        mse = np.array([mse_by_id.get(i, np.nan) for i in ids])
        ok = ~np.isnan(mse)

        # MSE of the checkpoint vs the SWA mean, on these same images -- if these diverge, the
        # comparison against the stored `Error` column is no longer apples-to-apples.
        def mean_mse(prm, fn):
            tot, n = 0.0, 0
            for imgs, _ in loader:
                imgs = np.asarray(imgs)
                r = np.asarray(fn(prm, imgs))
                tot += float(((r - imgs) ** 2).mean(axis=(1, 2, 3)).sum()); n += len(imgs)
            return tot / n
        m_hat = mean_mse(theta_hat, reconstruct_orig)   # original batch_stats
        m_swa = mean_mse(theta_swa, reconstruct)        # collected batch_stats
        print(f"  {cli.dataset}/{group}: MSE checkpoint={m_hat:.6f} SWA-mean={m_swa:.6f} "
              f"({100*(m_swa-m_hat)/m_hat:+.1f}%)")

        print(f"  {cli.dataset}/{group}: n={ok.sum()} score_mean={scores.mean():.4g} "
              f"pearson={pearsonr(scores[ok], mse[ok])[0]:.4f} "
              f"spearman={spearmanr(scores[ok], mse[ok])[0]:.4f}")

        # include n_vec: memory is (2+K)p, so runs at different K are different methods
        # and must not overwrite one another (K=0 -> 2p floor, K=1 -> 3p).
        out = os.path.join(base, f"uq_metrics_swag_nv{cli.swag_n_vec}_{group}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Image Name", "Error", "Uncertainty"])
            for i, img_id in enumerate(ids):
                w.writerow([img_id, mse[i], scores[i]])
        print(f"  wrote {out}")
