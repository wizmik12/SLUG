import os
# Disable XLA's GPU convolution algorithm autotuning: on this GPU/cuDNN combo it
# picks a numerically broken algorithm once batch size exceeds 8, causing forward
# passes on the same image to diverge depending on what else is in the batch
# (confirmed on this VAE architecture at 128x128: mu diverges by up to 2.7).
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader

from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import pickle

import Helpers as hf
import data
from train_vae import TrainerModule


print("=== JAX GPU TEST START ===")
print("JAX platform:", jax.default_backend())
print("Available devices:", jax.devices())
print("=== JAX GPU TEST DONE ===")

# Command-line parameters
parser = argparse.ArgumentParser(description="Training Params")
args = hf.set_parser(parser)
print("-Target Image Size %d" % args.image_size)

fm_size = args.image_size//(2 ** len(args.block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

use_wandb = args.use_wandb

# Graphic card
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.graphic_card

# Create the feature loss module if required
if args.feature_scale > 0:
    print("-VGG19 Feature Loss ON")
else:
    print("-VGG19 Feature Loss OFF")

# Experiment Configuration - perc_attr indicates the percentage of samples with the sensitive attribute in the train_set
perc_attr = int(args.subset) if args.subset == 0.0 or args.subset == 1.0 else args.subset
run = args.run

# Repeat experiment {runs} times    
print("############## Let's go with run", run)
if use_wandb:
    wandb.init(project=args.experiment_name,
            name=f"run-{run}",
            group=f"percent_attr-{perc_attr}", config=args)


# Create the save directory if it does not exist.
# Seed is folded into the run-level directory component so that multiple
# initialisation seeds sharing the same --run (data split) don't collide.
run_dir = f"{run}_seed{args.seed}" if args.seed != 42 else str(run)
save_path = f"{args.save_dir}/Results/{run_dir}/{perc_attr}/"
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path + 'imgs/'):
    os.makedirs(save_path + 'imgs/')

print("Percentage of black samples: ", perc_attr)

# Load the train/test data for this run (split in test b/w)
train_dataset, test_dataset_attr1, test_dataset_attr2, attr1, attr2 = data.train_test_setting(args.dataset_root, perc_attr, run, args.test_size, args)

# Dataloaders for fitzpatrick17k dataset.
# Shuffle order is seeded by --seed (not --run), so it varies with initialisation
# across otherwise-identical-split runs, consistent with TrainerModule's seed below.
shuffle_generator = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=hf.numpy_collate, drop_last=True, generator=shuffle_generator)
test_loader_attr1 = DataLoader(test_dataset_attr1, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)
test_loader_attr2  = DataLoader(test_dataset_attr2, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
test_images_attr1, _ = next(iter(test_loader_attr1))
test_images_attr2, _ = next(iter(test_loader_attr2))

# TrainerModule Class (where the architecture, evaluation, ... are defined)
trainer = TrainerModule(channel_in=test_images_attr1.shape[-1], train_loader=train_loader,
                        test_images = (test_images_attr1, test_images_attr2),
                        save_path=save_path,
                        use_wandb=use_wandb,
                        args=args,
                        seed=args.seed)

# Load (trained) model
if args.load_model:
    print("############## Loading model")
    trainer.load_model()
else: # train model (from scratch)
    print("############## Training model")
    trainer.train_model(num_epochs=args.nepoch, test_loader_attr2=test_loader_attr2, test_loader_attr1=test_loader_attr1)

# Evaluate model in test
mse_results_attr1, _, latent_std_results_attr1 = trainer.eval_model(test_loader_attr1)
mse_results_attr2, _, latent_std_results_attr2 = trainer.eval_model(test_loader_attr2)

# Store results of the test set
results = {"mse": {attr1: np.mean(mse_results_attr1), attr2: np.mean(mse_results_attr2)},
           "latent_std": {attr1: np.mean(latent_std_results_attr1), attr2: np.mean(latent_std_results_attr2)}}

# Save results of the fitzpatrick dataset
path_save_results = f'{save_path}/results_run.pkl'
with open(path_save_results, 'wb') as f:
    pickle.dump(results, f)

if use_wandb: # logs in wandb
    wandb.log({f"mse_{attr1}_final": results["mse"][attr1], f"latent_std_{attr1}_final": results["latent_std"][attr1],
            f"mse_{attr2}_final": results["mse"][attr2], f"latent_std_{attr2}_final": results["latent_std"][attr2]})
    wandb.finish()

# Save trained model 
if not args.load_model:
    trainer.save_model()

# Uncertainty experiment (SLUG metric)
if args.eval_uq: # compute extensive metrics for UQ
    print("Evaluate Uncertainty!")
    if args.batch_size_uq != args.batch_size: # Change the batch size for this experiment
        test_loader_attr1 = DataLoader(test_dataset_attr1, batch_size=args.batch_size_uq, shuffle=False, collate_fn=hf.numpy_collate)
        test_loader_attr2 = DataLoader(test_dataset_attr2, batch_size=args.batch_size_uq, shuffle=False, collate_fn=hf.numpy_collate)
    # Evaluate in both test sets
    trainer.evaluate_uncertainty(test_loader_attr1, subset=attr1)
    trainer.evaluate_uncertainty(test_loader_attr2, subset=attr2)
    # Encoder-std measure rides along for free on the same (plain, non-dropout) model
    trainer.evaluate_uncertainty_encoderstd(test_loader_attr1, subset=attr1)
    trainer.evaluate_uncertainty_encoderstd(test_loader_attr2, subset=attr2)

elif args.plot_uq: # plot some examples for UQ
    test_loader_attr1 = DataLoader(test_dataset_attr1, batch_size=1, shuffle=False, collate_fn=hf.numpy_collate)
    test_loader_attr2 = DataLoader(test_dataset_attr2, batch_size=1, shuffle=False, collate_fn=hf.numpy_collate)
    # Evaluate in both test sets 
    trainer.plot_uncertainty(test_loader_attr1, subset=attr1)
    trainer.plot_uncertainty(test_loader_attr2, subset=attr2)

elif args.use_dropout:
    trainer.evaluate_uncertainty_mcdropout(test_loader_attr1, subset=attr1)
    trainer.evaluate_uncertainty_mcdropout(test_loader_attr2, subset=attr2)

elif args.eval_lae: # diagonal-Laplace baseline, native-JAX Hutchinson estimator (see revision_part1.md 0.b/2.2)
    if args.batch_size_uq != args.batch_size:
        test_loader_attr1 = DataLoader(test_dataset_attr1, batch_size=args.batch_size_uq, shuffle=False, collate_fn=hf.numpy_collate)
        test_loader_attr2 = DataLoader(test_dataset_attr2, batch_size=args.batch_size_uq, shuffle=False, collate_fn=hf.numpy_collate)
    # None -> decoder-only (default); 'all' -> full encoder+decoder; comma-separated -> explicit layer list
    if args.lae_decoder_layers is None:
        decoder_layers = None
    elif args.lae_decoder_layers == 'all':
        decoder_layers = 'all'
    else:
        decoder_layers = [s.strip() for s in args.lae_decoder_layers.split(',')]
    print(f"LAE parameter subset: decoder_layers={decoder_layers!r}, diag_method={args.lae_diag_method!r}")
    trainer.evaluate_uncertainty_lae(test_loader_attr1, subset=attr1, n_samples_diag=args.lae_n_samples_diag, tau=args.lae_tau, s_samples=args.lae_s_samples, decoder_layers=decoder_layers, diag_method=args.lae_diag_method)
    trainer.evaluate_uncertainty_lae(test_loader_attr2, subset=attr2, n_samples_diag=args.lae_n_samples_diag, tau=args.lae_tau, s_samples=args.lae_s_samples, decoder_layers=decoder_layers, diag_method=args.lae_diag_method)

# MSE/SSIM/LPIPS + reconstruction examples -- independent of the (mutually exclusive)
# UQ branches above, so it can run alongside any of them in the same invocation.
if args.eval_metrics:
    print("Evaluate MSE/SSIM/LPIPS!")
    trainer.evaluate_metrics(test_loader_attr1, subset=attr1, n_recon_examples=args.n_recon_examples)
    trainer.evaluate_metrics(test_loader_attr2, subset=attr2, n_recon_examples=args.n_recon_examples)

# External test dataset (fairness in the PASSION dataset and OOD on the ISIC dataset).
if args.test_set_root is not None:
    test_dataset = data.load_external_test(args)
    root_data = args.test_set_root
    if 'passion' in root_data: # Fairness metrics on the PASSION dataset
        print("PASSION dataset")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_uq, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)
        if args.use_dropout:
            trainer.evaluate_uncertainty_mcdropout(test_loader, subset='PASSION')
        else:
            trainer.evaluate_uncertainty(test_loader, subset='PASSION')

    elif 'ISIC' in root_data: # OOD plots on the ISIC dataset
        print("ISIC dataset")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)
        trainer.plot_uncertainty(test_loader, subset='ISIC') 
    else:
        raise("Select valid external set")