import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config import Config
from utils import patchify, depatchify



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
        self.pos_embeddings = nn.Parameter(torch.zeros(1, self.n_tokens, cfg.embed_dim))

        # need MLP for the timestep
        self.time_mlp = TimeMLP(cfg.t_dim, cfg.embed_dim)
        
        # create the model with DIT blocks
        self.blocks = nn.ModuleList([]) # TODO: fill in with DiT blocks
        # self.out_adaln = AdaLN(dim, cond_dim=dim)
        self.out_proj = nn.Linear(cfg.embed_dim, self.token_dim)


    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        """
        Docstring for forward
        
        z_t: the latent of the input at timestep t
        t: the timestep 
        """

        # patchify the zt
        patches = patchify(z_t, self.cfg.patch_size)  # (B, num_patches, token_dim)
        # in project and combine with pos embeddings
        token_emb = self.in_projection(patches) + self.pos_embeddings  # (B, num_patches, embed_dim)

        # get the time embeddings and execute the time MLP
        time_in = ... # (B, t_dim), need to fill in with sinusoidal embeddings
        time_emb = self.time_mlp(time_in)  # (B, embed_dim)

        # send the time conditioning and the input through the DiT blocks
        for block in self.blocks:
            pass  # TODO: fill in with DiT block forward

        # outproject and depatchify
        out_patches = self.out_proj(token_emb)  # (B, num_patches, token_dim)
        z_pred = depatchify(out_patches, self.cfg.patch_size, self.cfg.z_image_size, self.cfg.z_image_size)  # (B, z_dim, H, W
        return z_pred