from dataclasses import dataclass

@dataclass
class Config:
    # for the AE
    image_size: int = 64
    z_dim: int = 4
    z_image_size: int = image_size // 4
    # for the DiT
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    t_dim: int = 128
    patch_size: int = 2