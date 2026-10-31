from typing import Any, Sequence, Union
import jax
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
import torch

"""
Define and load pretrained VGG19 for the perceptual loss of the VAE.
"""
# Architecture of VGG19
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

# Convert PyTorch parameters to nested dictionaries.
def _torch_to_vgg(torch_params, cfg):
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

# Define vv19 and load weight from pytorch
def vgg19():
  cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

  model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

  vgg = VGG(cfg=cfg)

  torch_params = torch.hub.load_state_dict_from_url(model_url)
  flax_params = FrozenDict(_torch_to_vgg(torch_params, cfg))
  return vgg, flax_params

# Load pretrained vgg19 for perceptual loss
vgg_features_model, params_vgg = vgg19()

# Perceptual loss
@jax.jit
def feature_loss(img1, img2):
    # Feature maps
    features_img1 = vgg_features_model.apply(jax.lax.stop_gradient(params_vgg), img1)
    features_img2 = vgg_features_model.apply(jax.lax.stop_gradient(params_vgg), img2)
    
    # MSE between feature maps
    loss = 0
    for f1, f2 in zip(features_img1, features_img2):
        loss += jnp.mean((f1 - f2) ** 2)
    
    return loss/16 # average the loss through the 16 layers of vgg19