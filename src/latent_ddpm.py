"""A small unconditional latent DDPM, for the 50/50 bias experiment.

Why train one rather than use Stable Diffusion's UNet: a pretrained model gives no control
over the training composition, so the 50/50 / 100%-light / 100%-dark configurations that the
paper's bias claim rests on are impossible. It also turned out to be a poor denoiser on these
latents -- the one-step reconstruction error had cv 5-9% at every timestep and every DDIM step
count, i.e. it measured the injected noise rather than the image, leaving nothing for an
uncertainty score to rank.

A model trained on the data does not have that problem: it denoises easy images well and hard
ones badly, which is exactly the variation the experiment needs.

Design. The SD VAE is frozen and used only as an encoder, so this operates on 16x16x4 latents
(128px images) rather than pixels. That keeps the UNet small -- a few tens of millions of
parameters against SD's 860M -- and the DDPM objective ||eps - eps_theta||^2 means H = I is
EXACT for the Gauss-Newton, unlike the VAE where the perceptual term has to be dropped.

Dropout is included so MC-dropout is available as a baseline on the same checkpoint, matching
how the VAE experiments are set up.
"""
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def timestep_embedding(t, dim: int, max_period: int = 10000):
    """Sinusoidal embedding, as in the original DDPM."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t.astype(jnp.float32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class ResBlock(nn.Module):
    out_ch: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, temb, train: bool):
        h = nn.GroupNorm(num_groups=min(32, x.shape[-1]))(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)
        # time conditioning, added as a per-channel shift
        h = h + nn.Dense(self.out_ch)(nn.swish(temb))[:, None, None, :]
        h = nn.GroupNorm(num_groups=min(32, self.out_ch))(h)
        h = nn.swish(h)
        if self.dropout > 0:
            h = nn.Dropout(self.dropout, deterministic=not train)(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)
        if x.shape[-1] != self.out_ch:
            x = nn.Conv(self.out_ch, (1, 1))(x)
        return x + h


class AttnBlock(nn.Module):
    """Self-attention over spatial positions; cheap at 16x16 and below."""

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(32, C))(x)
        h = h.reshape(B, H * W, C)
        h = nn.SelfAttention(num_heads=max(1, C // 64), qkv_features=C)(h)
        return x + h.reshape(B, H, W, C)


class LatentUNet(nn.Module):
    """eps_theta(z_t, t) on latents. Unconditional: no text encoder, no cross-attention."""
    out_ch: int = 4
    base_ch: int = 128
    ch_mult: Sequence[int] = (1, 2, 2)
    num_res_blocks: int = 2
    attn_at: Sequence[int] = (1, 2)      # resolution levels that get self-attention
    dropout: float = 0.0

    @nn.compact
    def __call__(self, z, t, train: bool = False):
        temb = timestep_embedding(t, self.base_ch)
        temb = nn.Dense(self.base_ch * 4)(temb)
        temb = nn.Dense(self.base_ch * 4)(nn.swish(temb))

        h = nn.Conv(self.base_ch, (3, 3), padding="SAME")(z)
        skips = [h]
        # ---- down
        for lvl, m in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                h = ResBlock(self.base_ch * m, self.dropout)(h, temb, train)
                if lvl in self.attn_at:
                    h = AttnBlock()(h)
                skips.append(h)
            if lvl != len(self.ch_mult) - 1:
                h = nn.Conv(h.shape[-1], (3, 3), strides=(2, 2), padding="SAME")(h)
                skips.append(h)
        # ---- middle
        h = ResBlock(h.shape[-1], self.dropout)(h, temb, train)
        h = AttnBlock()(h)
        h = ResBlock(h.shape[-1], self.dropout)(h, temb, train)
        # ---- up
        for lvl, m in reversed(list(enumerate(self.ch_mult))):
            for _ in range(self.num_res_blocks + 1):
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = ResBlock(self.base_ch * m, self.dropout)(h, temb, train)
                if lvl in self.attn_at:
                    h = AttnBlock()(h)
            if lvl != 0:
                B, H, W, C = h.shape
                h = jax.image.resize(h, (B, H * 2, W * 2, C), method="nearest")
                h = nn.Conv(C, (3, 3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=min(32, h.shape[-1]))(h)
        return nn.Conv(self.out_ch, (3, 3), padding="SAME")(nn.swish(h))


def cosine_alphas_cumprod(T: int = 1000, s: float = 0.008):
    """Nichol & Dhariwal cosine schedule; better than linear on small images."""
    x = jnp.linspace(0, T, T + 1)
    f = jnp.cos(((x / T) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    ab = f / f[0]
    betas = jnp.clip(1 - (ab[1:] / ab[:-1]), 0, 0.999)
    return jnp.cumprod(1 - betas)


class DDPM:
    """Forward noising and the one-step DDIM estimate of z0.

    The one-step estimate is AFFINE in eps_theta, so the Gauss-Newton of the reconstruction
    equals that of the noise prediction up to the scalar (1-abar)/abar. That is what lets SLUG
    be computed on the denoiser at a single timestep while the error target is a
    reconstruction.
    """

    def __init__(self, T: int = 1000):
        self.T = T
        self.abar = cosine_alphas_cumprod(T)

    def q_sample(self, z0, t, eps):
        a = self.abar[t][:, None, None, None]
        return jnp.sqrt(a) * z0 + jnp.sqrt(1 - a) * eps

    def z0_from_eps(self, zt, t, eps_pred):
        a = self.abar[t][:, None, None, None]
        return (zt - jnp.sqrt(1 - a) * eps_pred) / jnp.sqrt(a)
