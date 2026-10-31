import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, List, Callable
import math

# Residual (down-sampling)
class ResDown(nn.Module):
    channel_in: int
    channel_out: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, train):
        # 1 ConvBlock
        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x, use_running_average= not train) # Batch Normalization
        x = nn.elu(x)
        x_cat = nn.Conv(features=(self.channel_out // 2) + self.channel_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(2, 2),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)
        # Skip connection
        skip = x_cat[:, :, :, :self.channel_out]
        x = x_cat[:, :, :, self.channel_out:]

        # 2 ConvBlock
        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x, use_running_average= not train) # Batch Normalization
        x = nn.elu(x)
        x = nn.Conv(features=self.channel_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(1, 1),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)

        return x + skip # Skip connection

class ResUp(nn.Module):
    channel_in: int
    channel_out: int
    kernel_size: int = 3
    scale_factor: int = 2

    @nn.compact
    def __call__(self, x_in, train):
        # 1 ConvBlock
        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x_in, use_running_average= not train) # Batch Normalization
        x = nn.elu(x)
        # Up-sampling
        new_shape = (x.shape[0], x.shape[1] * self.scale_factor, x.shape[2] * self.scale_factor, x.shape[3])
        x = jax.image.resize(x, shape=new_shape, method="nearest")
        x_cat = nn.Conv(features=(self.channel_in // 2) + self.channel_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(1, 1),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)
        
        # Skip Connection        
        skip = x_cat[:, :, :, :self.channel_out]
        x = x_cat[:, :, :, self.channel_out:]
        
        # 2 ConvBlock
        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x, use_running_average= not train) # Batch Normalization
        x = nn.elu(x)
        x = nn.Conv(features=self.channel_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(1, 1),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)
        
        return x + skip # Skip connection

# Residual Block
class ResBlock(nn.Module):
    channel_in: int
    channel_out: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x_in, train):
        bttl_nk = self.channel_in // 2
        skip = self.channel_in == self.channel_out

        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x_in, use_running_average= not train)
        x = nn.elu(x_in)
        first_out = self.channel_in // 2 if self.channel_in == self.channel_out else (self.channel_in // 2) + self.channel_out
        x_cat = nn.Conv(features=first_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(1, 1),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)
        x = x_cat[:, :, :, :bttl_nk]

        skip = x_in if skip else x_cat[:, :, :, bttl_nk:]
        x = nn.BatchNorm(epsilon=1e-04, momentum=0.9)(x, use_running_average= not train)
        x = nn.elu(x)
        x = nn.Conv(features=self.channel_out,
                             kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(1, 1),
                             padding=((self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)))(x)
        return x + skip

# Encoder sub-network
class Encoder(nn.Module):
    channels: int # Input channels
    ch: int = 64 # Latent channels
    blocks: Tuple[int, int, int, int] = (1, 2, 4, 8)
    latent_channels: int = 256
    num_res_blocks: int = 1
    deep_model: bool = False

    # Reparameterization trick! 
    def sample(self, mu, log_var, rng):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, std.shape) 
        return mu + eps * std

    @nn.compact
    def __call__(self, x, rng, train=True):
        # Initial Conv
        x = nn.Conv(features=self.blocks[0] * self.ch,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding=((1,1),(1,1)))(x)
        # Define all the layers in this list
        layers=[] 
        widths_in = list(self.blocks)
        widths_out = list(self.blocks[1:]) + [2 * self.blocks[-1]]
        for w_in, w_out in zip(widths_in, widths_out):
            if self.deep_model:
                layers.append(ResBlock(w_in * self.ch, w_in * self.ch))
            layers.append(ResDown(w_in * self.ch, w_out * self.ch))
           
        for _ in range(self.num_res_blocks):
            layers.append(ResBlock(widths_out[-1] * self.ch, widths_out[-1] * self.ch))

        # Apply all the layers!
        for layer in layers:
            x =layer(x, train)

        # Compute mean and std deviation of the latent variable z
        x = nn.elu(x)
        mu = nn.Conv(features=self.latent_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID')(x)
        log_var = nn.Conv(features=self.latent_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID')(x)

        if train: # sample
            x = self.sample(mu, log_var, rng)
        else: # degenerate on the mean
            x = mu

        return x, mu, log_var

# Decoder sub-network
class Decoder(nn.Module):
    channels: int
    ch: int = 64
    blocks: Tuple[int, int, int, int] = (1, 2, 4, 8)
    latent_channels: int = 256
    num_res_blocks: int = 1
    deep_model: bool = False
    use_dropout: bool = False

    @nn.compact
    def __call__(self, x, train=True):

        widths_out = list(self.blocks)[::-1]
        widths_in = (list(self.blocks[1:]) + [2 * self.blocks[-1]])[::-1]

        x = nn.Conv(features=widths_in[0] * self.ch,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="VALID")(x)
        
        # Define all the layers in this list
        layers = []  
        for _ in range(self.num_res_blocks):
            layers.append(ResBlock(widths_in[0] * self.ch, widths_in[0] * self.ch))

        for w_in, w_out in zip(widths_in, widths_out):
            layers.append(ResUp(w_in * self.ch, w_out * self.ch))
            if self.deep_model:
                layers.append(ResBlock(w_out * self.ch, w_out * self.ch))

        # Apply all the layers
        for layer in layers:
            x =layer(x, train)
            if self.use_dropout:
                print("Using MC Dropout!")
                x = nn.Dropout(rate=0.2)(x, deterministic=False)

        x = nn.elu(x)
        return jnp.tanh(nn.Conv(features=self.channels, kernel_size=(5, 5), strides=(1, 1), padding=((2,2)))(x))

class VAE(nn.Module):
    channel_in: int = 3
    ch: int = 64
    blocks: Tuple[int, int, int, int] = (1, 2, 4, 8)
    latent_channels: int = 256
    num_res_blocks: int = 1
    deep_model: bool = False
    train: bool = False
    use_dropout: bool = False

    def setup(self):
        self.encoder = Encoder(
            channels=self.channel_in,
            ch=self.ch,
            blocks=self.blocks,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            deep_model=self.deep_model
        )
        self.decoder = Decoder(
            channels=self.channel_in,
            ch=self.ch,
            blocks=self.blocks,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            deep_model=self.deep_model,
            use_dropout=self.use_dropout
        )

    def __call__(self, x, rng, train=True):
        encoding, mu, log_var = self.encoder(x, rng, train) # encoder
        recon_img = self.decoder(encoding, train) # decoder
        return recon_img, mu, log_var