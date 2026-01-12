import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TinyAE(nn.Module):
    """
    A tiny Variational Autoencoder (VAE) for learning the latents briefly for LDMs.
    """
    def __init__(self):
        super(TinyVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 8, kernel_size=3, padding=1),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 128, kernel_size=3, padding=1),
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

class VelocityDiT(nn.Module):
    """
    A tiny Diffusion Transformer (DiT) for learning the noise prediction briefly for LDMs.
    """
    def __init__(self, input_dim=784, num_layers=4, num_heads=4, dim_feedforward=256):
        super(VelocityDiT, self).__init__()
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        pass