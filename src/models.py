import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config import Config
from utils import patchify, depatchify, sinusoidal_timestep_embedding


class TinyAE(nn.Module):
    """
    A tiny Variational Autoencoder (VAE) for learning the latents briefly for LDMs.
    """

    def __init__(self, config: Config):
        super(TinyAE, self).__init__()

        self.z_dim = config.z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, self.z_dim, kernel_size=3, padding=1),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class TimeMLP(nn.Module):
    def __init__(self, t_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization that conditions on an external vector.
    """

    def __init__(self, dim: int, cond_dim: int):
        super(AdaptiveLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale_shift = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x  # placeholder implementation


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super(DiTBlock, self).__init__()
        self.norm1 = AdaptiveLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(dim, cond_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        res = self.norm1(x, cond)
        attn_out = self.attn(res, res, res, need_weights=False)
        x = x + attn_out
        res = self.norm2(x, cond)
        mlp_out = self.mlp_net(res)
        x = x + mlp_out
        return x


class VelocityDiT(nn.Module):
    """
    A tiny Diffusion Transformer (DiT) for learning the noise prediction briefly for LDMs.
    """

    def __init__(self, cfg: Config):
        super(VelocityDiT, self).__init__()
        self.cfg = cfg
        self.token_dim = cfg.patch_size * cfg.patch_size * cfg.z_dim

        # input projection to the embed dim
        self.in_projection = nn.Linear(self.token_dim, cfg.embed_dim)

        # need positional encodings for the tokens, learned postional embeddings
        self.n_tokens = (cfg.z_image_size // cfg.patch_size) ** 2
        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, self.n_tokens, cfg.embed_dim))

        # need MLP for the timestep
        self.time_mlp = TimeMLP(cfg.t_dim, cfg.embed_dim)

        # create the model with DIT blocks
        self.blocks = nn.ModuleList([DiTBlock(dim=cfg.embed_dim, num_heads=cfg.num_heads,
                                    cond_dim=cfg.embed_dim, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.num_layers)])

        self.out_norm = AdaptiveLayerNorm(cfg.embed_dim, cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, self.token_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        """
        Docstring for forward

        z_t: the latent of the input at timestep t
        t: the timestep 
        """

        # patchify the zt
        # (B, num_patches, token_dim)
        patches = patchify(z_t, self.cfg.patch_size)
        # in project and combine with pos embeddings
        # (B, num_patches, embed_dim)
        token_emb = self.in_projection(patches) + self.pos_embeddings

        # get the time embeddings and execute the time MLP
        time_in = sinusoidal_timestep_embedding(
            t, self.cfg.t_dim)  # (B, t_dim)
        time_emb = self.time_mlp(time_in)  # (B, embed_dim)

        # send the time conditioning and the input through the DiT blocks
        for block in self.blocks:
            token_emb = block(token_emb, time_emb)

        # outproject and depatchify
        token_emb = self.out_norm(token_emb, time_emb)
        out_patches = self.out_proj(token_emb)  # (B, num_patches, token_dim)
        z_pred = depatchify(out_patches, self.cfg.patch_size,
                            self.cfg.z_image_size, self.cfg.z_image_size)  # (B, z_dim, H, W)

        return z_pred
