import torch.nn.functional as F
import jax
import torch
import numpy as np
from jax import jit
from vgg19 import feature_loss

# Function to evaluate the model (MSE metric)
def eval_model(model, params, batch_stats, imgs, rng, rng_dropout, use_dropout):
    if use_dropout:
        recon_imgs, mu, logvar = model.apply({'params': params, 'batch_stats': batch_stats}, x=imgs,rng=rng, rngs={'dropout': rng_dropout}, train=False)
    else:
        recon_imgs, mu, logvar = model.apply({'params': params, 'batch_stats': batch_stats}, x=imgs, rng=rng, train=False)
    loss = ((recon_imgs - imgs) ** 2).mean(axis=0).mean()  # Mean over batch, mean over pixels
    return loss, mu, logvar

# Compute the elbo for traning
def elbo(model, params, batch_stats, imgs, rng, rng_dropout, args, scale_factor=1):
    if args.use_dropout:
        output = model.apply({'params': params, 'batch_stats': batch_stats}, x=imgs,rng=rng, rngs={'dropout': rng_dropout}, train=True, mutable=['batch_stats'])
    else:
        output = model.apply({'params': params, 'batch_stats': batch_stats}, x=imgs,rng=rng, train=True, mutable=['batch_stats'])

    recon_imgs, mu, logvar = output[0]
    new_model_state = output[1]
    mse_loss = ((recon_imgs - imgs) ** 2).mean(axis=0).mean()
    kl_loss =  -0.5 * (1 + logvar - jax.numpy.pow(mu,2) - jax.numpy.exp(logvar)).mean()
    loss = args.kl_scale * kl_loss + mse_loss
    if args.feature_scale > 0:
                        feat_loss = feature_loss(recon_imgs, imgs)
                        loss += args.feature_scale * feat_loss
    return loss * scale_factor, (mse_loss, kl_loss, new_model_state)

def set_parser(parser):
    # Experiment configuration
    parser.add_argument("--experiment_name", "-expn", help="Experiment save name", type=str, required=True)
    parser.add_argument("--dataset", help="Which dataset to train/eval on", type=str,
                        choices=['fitz', 'ai4skin'], default='fitz')
    parser.add_argument("--dataset_root", "-dr", help="Dataset root dir (fitz only)", type=str, default='../imgs/')
    parser.add_argument("--ai4skin_patches_csv", help="Path to AI4SkIN patches.csv metadata", type=str, default=None)
    parser.add_argument("--ai4skin_patches_root", help="Root dir that AI4SkIN patch_path values are relative to", type=str, default=None)
    parser.add_argument("--train_size", help="Fixed training set size (resampled with replacement)", type=int, default=1668)
    parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
    parser.add_argument("--graphic_card", "-gc", type=str, default='1')
    parser.add_argument("--use_wandb", "-wb", action='store_true',
                        help="Use weight&biases to store and track logging data while training and test.")

    # External dataset
    parser.add_argument("--test_set_root", help="Root to external dataset", type=str, default=None)

    # run configuration
    parser.add_argument("--run", help="Seed for data resampling (train subset + test set draw). "
                         "Keep this FIXED across a set of runs that should share the same split "
                         "(e.g. multiple seeds/initializations) -- vary --seed instead.", type=int, default=None)
    parser.add_argument("--seed", help="Seed for model initialisation and batch shuffling, "
                         "independent of --run/data. Vary this across independent-initialization "
                         "runs while keeping --run fixed to hold the data split constant.", type=int, default=42)
    parser.add_argument("--subset", help="computed subset", type=float, default=None)

    # hyperparameters
    ## architecture
    parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
    parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)
    parser.add_argument("--num_res_blocks", '-nrb',
                        help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)
    parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
    parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                        type=int, nargs='+', default=(1, 2, 4, 8))
    parser.add_argument("--deep_model", '-dm', action='store_true',
                        help="Deep Model adds an additional res-identity block to each down/up sampling stage")

    ## other
    parser.add_argument("--test_size", help="test size", type=int, default=512)
    parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
    parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
    parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

    # Uncertainty experiment
    parser.add_argument("--load_model", '-cp', action='store_true',
                        help="Load pre-trained model")
    parser.add_argument("--eval_uq", '-uq', action='store_true',
                        help="Evaluate UQ")
    parser.add_argument("--use_dropout", '-dropout', action='store_true',
                        help="Evaluate UQ")
    parser.add_argument("--plot_uq", action='store_true',
                        help="Evaluate UQ")
    parser.add_argument("--eval_lae", action='store_true',
                        help="Evaluate the diagonal-Laplace ('lae') baseline uncertainty")
    parser.add_argument("--eval_metrics", action='store_true',
                        help="Compute per-image MSE/SSIM/LPIPS and save reconstruction examples")
    parser.add_argument("--n_recon_examples", help="Reconstruction examples to save per test group", type=int, default=8)
    parser.add_argument("--lae_n_samples_diag", help="Hutchinson samples for the diagonal GGN estimate", type=int, default=30)
    parser.add_argument("--lae_tau", help="Diagonal-Laplace prior precision / damping", type=float, default=1.0)
    parser.add_argument("--lae_s_samples", help="Hutchinson samples for the per-image trace score (mirrors SLUG's hardcoded 500)", type=int, default=500)
    parser.add_argument("--lae_decoder_layers", type=str, default=None,
                        help="Which params the LAE diagonal is fitted over: omit for decoder-only "
                             "(default, matches Miani et al.'s primary fitter, ~25M params), 'all' for "
                             "encoder+decoder (matches SLUG's own scope), or a comma-separated list of "
                             "decoder layer names for last-layer-Laplace (e.g. 'Conv_1,ResUp_3', ~190K params)")
    parser.add_argument("--lae_diag_method", type=str, default="empirical_fisher",
                        choices=["empirical_fisher", "hutchinson"],
                        help="How the LAE diagonal is estimated: 'empirical_fisher' (default, one "
                             "deterministic pass, no sampling, guaranteed non-negative) or 'hutchinson' "
                             "(the original stochastic GGN-diagonal estimator, kept for comparison -- "
                             "failed to converge in practice, see revision_part1.md Sec 0.g)")
    parser.add_argument("--batch_size_uq", "-bs_uq", help="UQ batch size", type=int, default=128)

    return parser.parse_args()

# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255
    img = (img-0.5)/0.5
    return img

# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
