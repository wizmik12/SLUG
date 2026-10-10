import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map, tree_leaves
import numpy as np
from flax.training import checkpoints
import imageio

import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from VAE import VAE
import vgg19
import Helpers as hf

import wandb

import torchvision.utils as vutils
import torch

import tqdm
import os
import csv
import time

from uncertainty.sketched_lanczos_uncertainty import low_memory_lanczos_score_fun
from uncertainty.diagonal_laplace import diagonal_laplace_score_fun

from PIL import Image
from skimage.metrics import structural_similarity
import lpips

"""
Train/Load/Evaluate VAE for the paper experiments.
"""

# Trainer state (add batchstats for batchnorm)
class TrainState(train_state.TrainState):
  batch_stats: Any

class TrainState_dropout(train_state.TrainState):
  batch_stats: Any
  key: Any


class TrainerModule:

    def __init__(self, channel_in, train_loader, test_images, save_path, args, use_wandb, seed=42):
        super().__init__()
        # Arguments
        self.lr = args.lr # Learning rate
        self.seed = seed # Random seed
        self.args = args # Command-line parameters
        self.use_wandb = use_wandb
        # Paths
        self.save_path = save_path
        self.save_path_model = save_path + '/model/'
        # Image loaders
        self.train_loader = train_loader
        self.exmp_imgs, _ = next(iter(train_loader)) # Batch example to init the model
        self.test_images_attr1, self.test_images_attr2 = test_images
        print('images:' ,self.test_images_attr2.shape, self.test_images_attr1.shape)
        # Create empty  (Note: no parameters yet)
        self.model = VAE(channel_in=channel_in,
                ch=args.ch_multi,
                blocks=args.block_widths,
                latent_channels=args.latent_channels,
                num_res_blocks=args.num_res_blocks,
                deep_model=args.deep_model,
                use_dropout=args.use_dropout)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()
        # Function to compute uncertainty
        self.uncertainty_functions=None
        # Params/batch_stats snapshot used by evaluate_uncertainty/plot_uncertainty.
        # load_model() sets this from a checkpoint; if we trained in this same
        # process instead, it's built from self.state on first use (see below).
        self.state_dict = None
        # Function to compute the diagonal-Laplace ('lae') baseline uncertainty
        self.lae_functions=None
        # LPIPS network, lazily constructed (downloads pretrained weights on first use)
        self._lpips_fn=None

    def create_functions(self): # Train and evaluation jitted functions
        # Training function
        def train_step(state, imgs, rng, rng_dropout):
            loss_fn = lambda params: hf.elbo(self.model, params, state.batch_stats, imgs, rng, rng_dropout, self.args, 1)
            # Compute loss and gradients
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  # Get loss and gradients for loss
            loss = ret[0]
            loss_mse = ret[1][0]
            loss_kl = ret[1][1] 
            new_model_state = ret[1][2]
            
            # Clip gradients
            state = state.apply_gradients(grads=grads,batch_stats=new_model_state['batch_stats'])  # Optimizer update step
            return state, loss, loss_mse, loss_kl
        
        self.train_step = jax.jit(train_step)
        # Eval function
        def eval_step(state, imgs, rng, rng_dropout):
            return hf.eval_model(self.model, state.params, state.batch_stats, imgs, rng, rng_dropout, self.args.use_dropout)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, dropout_rng, init_rng = jax.random.split(rng,3)
        self.rng = rng # random number generation
        variables = self.model.init(init_rng, self.exmp_imgs, train=True, rng=init_rng)
        params = variables['params']
        batch_stats = variables['batch_stats']
        # Scheduler for learning rate
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=self.lr,
            warmup_steps=100,
            decay_steps=500*len(self.train_loader),
            end_value=1e-5
        )
        # Optimizer
        optimizer = optax.chain(
            optax.clip(1.0),  # Clip gradients at 1
            optax.adam(lr_schedule))
        # Initialize training state
        if self.args.use_dropout:
            self.state = TrainState_dropout.create(apply_fn=self.model.apply, params=params, key=dropout_rng, batch_stats=batch_stats, tx=optimizer)
        else:
            self.state = TrainState.create(apply_fn=self.model.apply, params=params, batch_stats=batch_stats, tx=optimizer)
        print("Number of parameters: ", sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def train_model(self, num_epochs, test_loader_attr1, test_loader_attr2):
        # Train model for defined number of epochs
        train_log_path = os.path.join(self.save_path, 'train_log.csv')
        with open(train_log_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "mse_loss", "kl_loss"])

        for epoch_idx in tqdm.tqdm(range(1, num_epochs+1)):
            avg_loss, avg_mse, avg_kl = self.train_epoch(epoch=epoch_idx)
            log_dict = {"epoch": epoch_idx, "loss": avg_loss, "mse_loss": avg_mse, "kl_loss": avg_kl, "sample": jax.random.normal(self.rng) }

            with open(train_log_path, mode="a", newline="") as f:
                csv.writer(f).writerow([epoch_idx, float(avg_loss), float(avg_mse), float(avg_kl)])

            if epoch_idx % 100 == 0:
                print(avg_loss)
                loss_attr1, _, _ = self.eval_model(test_loader_attr1)
                loss_attr2, _, _ = self.eval_model(test_loader_attr2)
                self.save_images(net=self.model, state=self.state, test_images= self.test_images_attr1, subset=f'{epoch_idx}_attr1')
                self.save_images(net=self.model, state=self.state, test_images= self.test_images_attr2, subset=f'{epoch_idx}_attr2')
                log_dict.update({'loss_attr1': loss_attr1, 'loss_attr2': loss_attr2})

            # log epoch results
            if self.use_wandb:
                wandb.log(log_dict)

        return avg_loss

    def avg_loss(self, loss):
        losses_np = np.stack(jax.device_get(loss))
        return losses_np.mean()

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        losses_mse, losses_kl = [], []
        for imgs, _ in self.train_loader:
            # split random key
            self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
            # train step
            self.state, loss, loss_mse, loss_kl = self.train_step(self.state, imgs, params_key, dropout_key) 
            # Store loss of the current epoch
            losses.append(loss)
            losses_mse.append(loss_mse)
            losses_kl.append(loss_kl)
        avg_loss = self.avg_loss(losses)
        avg_mse = self.avg_loss(losses_mse)
        avg_kl = self.avg_loss(losses_kl)
        return avg_loss, avg_mse, avg_kl
    
    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        mus = []
        vars = []
        batch_sizes = []
        for imgs, _ in data_loader:
            # split random key
            self.rng, rng, rng_dropout = jax.random.split(self.rng, 3)
            # eval step
            loss, mu, logvar = self.eval_step(self.state, imgs, rng, rng_dropout)
            losses.append(loss)
            mus.append(mu)
            vars.append(np.exp(logvar))
            batch_sizes.append(imgs[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        mus_np = np.vstack(jax.device_get(mus))
        vars_np = np.vstack(jax.device_get(vars))
        batch_sizes_np = np.stack(batch_sizes)

        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        avg_mus = mus_np.mean()
        avg_vars = vars_np.mean(0)
        return avg_loss, avg_mus, avg_vars
    
    def save_images(self, net, state, test_images, subset):
        # Set network to evaluation mode
        def forward_fn(images, rng, rng_dropout=None):
            if self.args.use_dropout:
                recon_img, mu, log_var = net.apply({'params': state.params, 'batch_stats': state.batch_stats}, x=images,rng=rng, rngs={'dropout': rng_dropout}, train=False)
            else:
                recon_img, mu, log_var = net.apply({'params': state.params, 'batch_stats': state.batch_stats}, x=images, rng=rng, train=False)
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn) # jitted!
         #split rng
        self.rng, rng, rng_dropout = jax.random.split(self.rng, 3)
        # Forward pass
        recon_img, _, _ = forward(test_images, rng, rng_dropout)
        # Concat recon and original for posterior viz
        img_cat = np.array(jnp.concatenate([recon_img, test_images], axis=1))
        # Convert JAX array to torch for viz
        img_cat =  np.transpose(img_cat, axes=[0, 3, 1, 2])
        img_cat_torch = torch.from_numpy(img_cat)
        # Save the image using vutils of torch
        save_path = self.save_path + f'fitz17_{self.args.latent_channels}_{self.args.image_size}{subset}.png'
        vutils.save_image(img_cat_torch, save_path, normalize=True)

        return img_cat
    
    def save_model(self):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.save_path_model, target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats}, prefix=f'fitz17_{self.args.latent_channels}_{self.args.image_size}', step=self.args.nepoch)

    def load_model(self):
        # Load model. We use different checkpoint for pretrained models
        self.state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.save_path_model,f'fitz17_{self.args.latent_channels}_{self.args.image_size}{self.args.nepoch}'), target=None)
        self.rng, dropout_key = jax.random.split(self.rng)
        if self.args.use_dropout:
            self.state = TrainState_dropout.create(apply_fn=self.model.apply,
                                       params=self.state_dict['params'],
                                       key = dropout_key,
                                       batch_stats=self.state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.adam(self.lr))   # Default optimizer
        else:
            self.state = TrainState.create(apply_fn=self.model.apply,
                                        params=self.state_dict['params'],
                                        batch_stats=self.state_dict['batch_stats'],
                                        tx=self.state.tx if self.state else optax.adam(self.lr)   # Default optimizer
                                        )
    
    def evaluate_uncertainty_mcdropout(self, loader, subset='_w', mc_samples=20):
        """
        Evaluate uncertainty using Monte Carlo dropout.
        Saves MSE and epistemic uncertainty per image.
        """
        metrics_file = os.path.join(self.save_path, f"uq_metrics_dropout_{subset}.csv")

        with open(metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Error", "Uncertainty"])

        def mc_dropout_forward(images, rng, rng_dropout, num_samples=20):
            """Forward pass with MC Dropout enabled (train=True)."""
            preds = []
            for i in range(num_samples):
                rng_dropout = jax.random.fold_in(rng, i)
                recon_img, _, _ = self.model.apply(
                    {'params': self.state.params, 'batch_stats': self.state.batch_stats},
                    x=images,
                    rng=rng,
                    rngs={'dropout': rng_dropout},
                    train=False  # Enable dropout during inference
                )
                preds.append(recon_img)
            preds = jnp.stack(preds)  # [mc_samples, B, H, W, C]
            mean_pred = preds.mean(axis=0)
            uncertainty = preds.std(axis=0).mean(axis=(1, 2, 3))  # scalar uncertainty per image
            return mean_pred, uncertainty

        i = 0
        for imgs, names in loader:
            self.rng, rng, dropout_rng = jax.random.split(self.rng, 3)

            start = time.time()
            recon_imgs, uq_score = mc_dropout_forward(imgs, rng, dropout_rng, num_samples=mc_samples)
            print(f"MC Dropout on one batch took {time.time() - start:.3f} seconds")

            mse_score = ((recon_imgs - imgs) ** 2).mean(axis=(1, 2, 3))

            # Save reconstructed vs input image grid for visualization
            img_cat = np.array(jnp.concatenate([recon_imgs, imgs], axis=1))
            img_cat = np.transpose(img_cat, axes=[0, 3, 1, 2])
            img_cat_torch = torch.from_numpy(img_cat)
            save_path_im = self.save_path + f'/imgs/im_recon_{self.args.latent_channels}_{self.args.image_size}.png'
            vutils.save_image(img_cat_torch, save_path_im, normalize=True)

            with open(metrics_file, mode="a", newline="") as f:
                i += 1
                if i % 25 == 0:
                    jax.clear_caches()
                print(f'Batch #{i}')
                for name_batch, mse_batch, uq_batch in zip(names, mse_score, uq_score):
                    writer = csv.writer(f)
                    writer.writerow([name_batch, mse_batch.item(), uq_batch.item()])

        print("UQ evaluation with MC Dropout finished!")


    def evaluate_uncertainty_encoderstd(self, loader, subset='_w'):
        """
        Evaluate uncertainty using the encoder's per-image latent variance.
        Saves MSE and per-image encoder std (mean exp(logvar) over the latent) per image.
        Plain-model only (no dropout rngs needed).
        """
        metrics_file = os.path.join(self.save_path, f"uq_metrics_encoderstd_{subset}.csv")

        with open(metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Error", "Uncertainty"])

        def forward_fn(images, rng):
            recon_img, mu, log_var = self.model.apply(
                {'params': self.state.params, 'batch_stats': self.state.batch_stats},
                x=images, rng=rng, train=False
            )
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn)

        i = 0
        for imgs, names in loader:
            self.rng, rng = jax.random.split(self.rng)
            recon_imgs, mu, log_var = forward(imgs, rng)

            mse_score = ((recon_imgs - imgs) ** 2).mean(axis=(1, 2, 3))
            uq_score = jnp.exp(log_var).mean(axis=(1, 2, 3))  # per-image mean latent variance

            with open(metrics_file, mode="a", newline="") as f:
                i += 1
                if i % 25 == 0:
                    jax.clear_caches()
                print(f'Batch #{i}')
                for name_batch, mse_batch, uq_batch in zip(names, mse_score, uq_score):
                    writer = csv.writer(f)
                    writer.writerow([name_batch, mse_batch.item(), uq_batch.item()])

        print("UQ evaluation with encoder std finished!")


    def plot_uncertainty(self, loader, subset='_w'):
        # Compute qualitative maps of the UQ score
        if self.state_dict is None:
            # Trained in this process rather than loaded from a checkpoint.
            self.state_dict = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        if self.uncertainty_functions is None:
            # Initialize low memory lanczos 
            self.uncertainty_functions = low_memory_lanczos_score_fun(
            self.model, 
            self.state_dict, 
            self.train_loader, 
            output_shape = self.test_images_attr2[0].shape,  # the shape of a single image
            sketch_padding = None,
            sketch_size = 5000000, 
            lanczos_iterations = 10,
            lanczos_eigenvectors = None, 
            seed = 1,
            train_batch_size = 8, # the bigger the value the faster GGV-vector product will be. Too big value will cause memory error. Search for the biggest value that fit in your memory
            subsample_trainset = None # None, which correspond to full datase. Subsample the dataset will make the GGV-vector product faster but will also decrease the quality of the uncertainty score
        )
            
        # Forward function
        def forward_fn(images, rng):
            recon_img, mu, log_var = self.model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats}, x=images, rng=rng, train=False)
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn) # jitted!

        for imgs, names in loader:
            # Forward pass!
            recon_imgs, _, _ = forward(imgs, rng=jax.random.PRNGKey(0))
    
            start = time.time()
            ## Compute UQ pixelwise
            uq_scores = self.uncertainty_functions['exact'](imgs)
            print(f"scoring one batch took {time.time()-start:.3f} seconds")

            ## Normalize UQ
            uq_scores = jax.numpy.asarray(uq_scores)
            uq_scores = np.asarray(uq_scores)
            uq_scores_norm = ((uq_scores/uq_scores.max()) - 0.5)/0.5

            ## Compute MSE
            im_mse = ((recon_imgs - imgs) ** 2)
            im_mse_norm = ((im_mse/im_mse.max()) -0.5)/0.5

            # Save UQ
            save_path_im = self.save_path + f'/imgs/{subset}_im_{names}_uqmap_{uq_scores.mean()}.png'
            image_ =  (uq_scores_norm[0]*0.5)+0.5
            plt.imsave(save_path_im, image_)

            # Save MSE
            save_path_im = self.save_path + f'/imgs/{subset}_im_{names}_msemap_{im_mse.mean()}.png'
            image_ =  (im_mse_norm[0]*0.5)+0.5
            plt.imsave(save_path_im, image_)

            # Save the reconstructed img
            save_path_im = self.save_path + f'/imgs/{subset}_im_{names}_recon_{uq_scores.mean()}.png'
            recon_imgs = jax.numpy.asarray(recon_imgs)
            recon_imgs = np.asarray(recon_imgs)
            image_ =  (recon_imgs[0]*0.5)+0.5
            plt.imsave(save_path_im, image_)

            # Concat image
            img_cat = np.array(jnp.concatenate([imgs, recon_imgs, im_mse_norm, uq_scores_norm], axis=1))
            img_cat =  np.transpose(img_cat, axes=[0, 3, 1, 2])
            img_cat_torch = torch.from_numpy(img_cat)

            # Save UQ, img, and reconstructed image together
            save_path_im = self.save_path + f'/imgs/{subset}_im_{names}_uq_{uq_scores.mean()}.png'
            vutils.save_image(img_cat_torch, save_path_im, normalize=True)

        print("UQ evaluation finished!")
        

    def evaluate_uncertainty(self, loader, subset='_w'):
        # Compute quantitative metrics for UQ (SLUG score)
        if self.state_dict is None:
            # Trained in this process rather than loaded from a checkpoint.
            self.state_dict = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        if self.uncertainty_functions is None:
            # Initialize low memory lanczos 
            self.uncertainty_functions = low_memory_lanczos_score_fun(
            self.model, 
            self.state_dict, 
            self.train_loader, 
            output_shape = self.test_images_attr2[0].shape,  # the shape of a single image
            sketch_padding = None,
            sketch_size =  5000000, 
            lanczos_iterations = 10,
            lanczos_eigenvectors = None, 
            seed = 1,
            train_batch_size = 8, # the bigger the value the faster GGV-vector product will be. Too big value will cause memory error. Search for the biggest value that fit in your memory
            subsample_trainset = None # None, which correspond to full datase. Subsample the dataset will make the GGV-vector product faster but will also decrease the quality of the uncertainty score
        )
        
        # Load the dataframe
        metrics_file = os.path.join(self.save_path, f"uq_metrics_{subset}.csv")

        # Initialize CSV file to store MSE and SLUG metric
        with open(metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Error", "Uncertainty"])

        # Forward function
        def forward_fn(images, rng):
            recon_img, mu, log_var = self.model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats}, x=images, rng=rng, train=False)
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn) # jitted!

        # Test loop     
        for imgs, names in loader:

            recon_imgs, _, _ = forward(imgs, rng=jax.random.PRNGKey(0))
            img_cat = np.array(jnp.concatenate([recon_imgs, imgs], axis=1))

            # Convert JAX array to torch for viz
            img_cat =  np.transpose(img_cat, axes=[0, 3, 1, 2])
            img_cat_torch = torch.from_numpy(img_cat)

            # Save the image using vutils of torch
            save_path_im = self.save_path + f'/imgs/im_recon_{self.args.latent_channels}_{self.args.image_size}.png'
            vutils.save_image(img_cat_torch, save_path_im, normalize=True)
        
            self.rng, _ = jax.random.split(self.rng)
            start = time.time()
            # Compute SLUG score!
            scores = self.uncertainty_functions['approx'](imgs, self.rng, 500) # 500 samples
            uq_score = scores
            print(f"scoring one batch took {time.time()-start:.3f} seconds")

            mse_score = ((recon_imgs - imgs) ** 2).mean(axis=(1,2,3))

            with open(metrics_file, mode="a", newline="") as f:
                    i+=1
                    if i % 25 == 0:
                        jax.clear_caches()
                    
                    print(f'Batch #{i}')
                    
                    for name_batch, mse_batch, uq_batch in zip(names, mse_score, uq_score):
                        writer = csv.writer(f)
                        writer.writerow([name_batch, mse_batch, uq_batch])

        print("UQ evaluation finished!")


    def evaluate_uncertainty_lae(self, loader, subset='_w', n_samples_diag=30, tau=1.0,
                                  s_samples=500, decoder_layers=None, diag_method='empirical_fisher'):
        # Compute quantitative metrics for the diagonal-Laplace ('lae') baseline.
        # decoder_layers=None -> decoder-only (Miani et al.'s primary fitter's scope);
        # 'all' -> full encoder+decoder, matching SLUG's own scope exactly;
        # a list -> last-layer-Laplace style restriction. diag_method='empirical_fisher'
        # (default) -> one deterministic pass, no sampling, guaranteed non-negative;
        # 'hutchinson' -> original stochastic estimator, kept for comparison (failed to
        # converge in practice). See uncertainty/diagonal_laplace.py and
        # revision_part1.md Sec 0.b/0.d/0.g/2.2.
        if self.state_dict is None:
            self.state_dict = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        if self.lae_functions is None:
            self.lae_functions = diagonal_laplace_score_fun(
                self.model,
                self.state_dict,
                self.train_loader,
                output_shape=self.test_images_attr2[0].shape,
                n_samples_diag=n_samples_diag,
                tau=tau,
                seed=1,
                train_batch_size=8,
                subsample_trainset=None,
                decoder_layers=decoder_layers,
                diag_method=diag_method,
            )

        metrics_file = os.path.join(self.save_path, f"uq_metrics_lae_{subset}.csv")
        with open(metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Error", "Uncertainty"])

        def forward_fn(images, rng):
            recon_img, mu, log_var = self.model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats}, x=images, rng=rng, train=False)
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn)

        i = 0
        for imgs, names in loader:
            recon_imgs, _, _ = forward(imgs, rng=jax.random.PRNGKey(0))

            self.rng, _ = jax.random.split(self.rng)
            start = time.time()
            scores = self.lae_functions['approx'](imgs, self.rng, s_samples)
            uq_score = scores
            print(f"scoring one batch (lae) took {time.time()-start:.3f} seconds")

            mse_score = ((recon_imgs - imgs) ** 2).mean(axis=(1,2,3))

            with open(metrics_file, mode="a", newline="") as f:
                i += 1
                if i % 25 == 0:
                    jax.clear_caches()

                print(f'Batch #{i} (lae)')

                for name_batch, mse_batch, uq_batch in zip(names, mse_score, uq_score):
                    writer = csv.writer(f)
                    writer.writerow([name_batch, mse_batch, uq_batch])

        print("UQ evaluation (lae) finished!")


    def evaluate_metrics(self, loader, subset='_w', n_recon_examples=8):
        # Per-image MSE, SSIM, LPIPS on a test loader, plus a handful of saved
        # input/reconstruction pairs for visual QA. Images are stored NHWC in [-1, 1]
        # (Helpers.image_to_numpy), hence data_range=2.0 for SSIM and no rescaling needed
        # for LPIPS (which expects [-1, 1] by convention).
        if self._lpips_fn is None:
            print("Loading LPIPS network (net='vgg', matching the perceptual-loss backbone)...")
            # Deliberately kept on CPU: JAX's GPU memory preallocation and torch's own CUDA
            # allocator don't coordinate, and JAX (already running the VAE) claims most of the
            # GPU first -- moving LPIPS to CUDA here reliably OOMs. LPIPS on a single small
            # (128x128) image pair is cheap enough on CPU that this isn't worth fighting.
            self._lpips_fn = lpips.LPIPS(net='vgg').eval()

        metrics_file = os.path.join(self.save_path, f"metrics_{subset}.csv")
        with open(metrics_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "MSE", "SSIM", "LPIPS"])

        def forward_fn(images, rng):
            recon_img, mu, log_var = self.model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats}, x=images, rng=rng, train=False)
            return recon_img, mu, log_var
        forward = jax.jit(forward_fn)

        recon_dir = os.path.join(self.save_path, 'imgs')
        os.makedirs(recon_dir, exist_ok=True)
        n_saved = 0
        i = 0
        for imgs, names in loader:
            recon_imgs, _, _ = forward(imgs, rng=jax.random.PRNGKey(0))
            imgs_np = np.array(imgs)
            recon_np = np.array(recon_imgs)

            mse_batch = ((recon_np - imgs_np) ** 2).mean(axis=(1, 2, 3))
            ssim_batch = [
                structural_similarity(imgs_np[k], recon_np[k], data_range=2.0, channel_axis=-1)
                for k in range(imgs_np.shape[0])
            ]

            # NHWC[-1,1] -> NCHW torch tensor for LPIPS (CPU -- see note above)
            imgs_t = torch.from_numpy(np.transpose(imgs_np, (0, 3, 1, 2))).float()
            recon_t = torch.from_numpy(np.transpose(recon_np, (0, 3, 1, 2))).float()
            with torch.no_grad():
                lpips_batch = self._lpips_fn(imgs_t, recon_t).reshape(-1).detach().cpu().numpy()

            with open(metrics_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for name, m, s, l in zip(names, mse_batch, ssim_batch, lpips_batch):
                    writer.writerow([name, float(m), float(s), float(l)])

            if n_saved < n_recon_examples:
                n_here = min(n_recon_examples - n_saved, imgs_np.shape[0])
                for k in range(n_here):
                    img_cat = np.concatenate([imgs_np[k], recon_np[k]], axis=1)  # side-by-side, HWC
                    img_cat = np.clip((img_cat * 0.5) + 0.5, 0, 1)  # [-1,1] -> [0,1]
                    # Sanitize: some datasets' image ids are nested paths (e.g. AI4SkIN's
                    # patch_path, 'patches/HUSC/.../patch_x.jpg'), which would otherwise
                    # build a path into a directory that doesn't exist and crash the whole
                    # eval (confirmed 2026-08-07: killed metrics_HCUV.csv for every ai4skin
                    # seed, and truncated metrics_HUSC.csv to the first batch only).
                    safe_name = str(names[k]).replace('/', '_')
                    plt.imsave(os.path.join(recon_dir, f'{subset}_recon_{safe_name}.png'), img_cat)
                n_saved += n_here

            i += 1
            if i % 25 == 0:
                jax.clear_caches()

        print(f"Metrics evaluation ({subset}) finished! MSE/SSIM/LPIPS -> {metrics_file}, "
              f"{n_saved} reconstruction examples -> {recon_dir}")