import jax
import jax.numpy as jnp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from flax.core import FrozenDict
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import data
import Helpers as hf


from typing import Any, Sequence, Union
import jax
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
import torch


class Backbone(nn.Module):
  cfg: Union[Sequence[int], Sequence[str]]
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    features = []
    for v in self.cfg:
      if v == 'M':
        x = nn.max_pool(x, (2, 2), (2, 2))
      else:
        x = nn.Conv(v, (3, 3), padding=((1,1), (1,1)), strides=(1,1), dtype=self.dtype)(x)

        features.append(x)
        x = nn.relu(x)
    return features


class VGG(nn.Module):
  cfg: Union[Sequence[int], Sequence[str]]
  dtype: Any = jnp.float32

  @staticmethod
  def make_backbone(self):
    return Backbone(self.cfg, self.dtype)

  def setup(self):
    self.backbone = VGG.make_backbone(self)

  def __call__(self, inputs):
    x = self.backbone(inputs)
    return x


def _torch_to_vgg(torch_params, cfg):
  """Convert PyTorch parameters to nested dictionaries."""
  flax_params = {'params': {'backbone': {}}}
  conv_idx = 0

  tensor_iter = iter(torch_params.items())

  def next_tensor():
    _, tensor = next(tensor_iter)
    return tensor.detach().numpy()

  for layer_cfg in cfg:
    if isinstance(layer_cfg, int):
      flax_params['params']['backbone'][f'Conv_{conv_idx}'] = {
          'kernel': np.transpose(next_tensor(), (2, 3, 1, 0)),
          'bias': next_tensor(),
      }
      conv_idx += 1

  return FrozenDict(flax_params)



def vgg19():
  cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

  model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

  vgg = VGG(cfg=cfg)

  torch_params = torch.hub.load_state_dict_from_url(model_url)
  flax_params = FrozenDict(_torch_to_vgg(torch_params, cfg))
  return vgg, flax_params

# Load pretrained vgg19 for perceptual loss
vgg_features_model, params_vgg = vgg19()


# Function to extract features
def extract_features(model, params, loader):
    features = []
    labels = []
    for img_batch, label_batch in loader:
        img_batch = jnp.array(img_batch)  # Convert to JAX array
        feature_maps = model.apply(jax.lax.stop_gradient(params), img_batch)
        # Flatten the features from all layers into one vector
        flattened_features = feature_maps[-1].reshape(-1,8192)
        print(flattened_features.shape)
        features.append(flattened_features)  # Concatenate the features
        labels.append(label_batch)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features

# Function to compute the variance for each component (feature)
def compute_variance(features):
    return np.var(features, axis=0)

# Function to compute t-SNE plot
def plot_tsne(features_w, features_b):
    features = np.concatenate((features_w, features_b), axis=0)
    labels = np.concatenate((np.zeros(len(features_w)), np.ones(len(features_b))))
    pca = PCA(n_components=1024)  # Reduce the features dimensionality first with PCA
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    features_tsne = tsne.fit_transform(features_pca)

    plt.figure(figsize=(10, 8))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE of VGG Features")
    plt.savefig('tnse.pdf')
    plt.show()

# Assume you have a data loader to load your dataset
# Example (replace with your own dataset):
# Load the train/test data for this run (split in test b/w)

print("Script")
import argparse
parser = argparse.ArgumentParser(description="Training Params")
args = hf.set_parser(parser)
train_dataset, test_dataset_w, test_dataset_b = data.train_test_setting(args.dataset_root, 0, 1, 2000, args)

# Dataloaders
test_loader_w = DataLoader(test_dataset_w, batch_size=64, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)
test_loader_b = DataLoader(test_dataset_b, batch_size=64, shuffle=False, num_workers=0, collate_fn=hf.numpy_collate)

# Extract features from the model
features_w = extract_features(vgg_features_model, params_vgg, test_loader_w)
features_b = extract_features(vgg_features_model, params_vgg, test_loader_b)


print(features_w.shape)

# Compute the variance for each feature
variances_w = compute_variance(features_w).mean()
variances_b = compute_variance(features_b).mean()

print(variances_w, variances_b)


# Visualize the features using t-SNE
plot_tsne(features_w, features_b)