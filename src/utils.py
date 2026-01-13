import torch
import math


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Converts images into patches.
    Args:
        imgs: (B, C, H, W) tensor of images
        patch_size: size of each patch
    Returns:
        patches: (B, num_patches, C * patch_size * patch_size) tensor of patches
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    patches = imgs.view(B, C, num_patches_h, patch_size,
                        num_patches_w, patch_size).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(
        B, num_patches_h * num_patches_w,  C * patch_size * patch_size)
    return patches


def depatchify(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    Converts patches back into images.
    Args:
        patches: (B, num_patches, C * patch_size * patch_size) tensor of patches
        patch_size: size of each patch
        img_size: size of the original image (assumed square)
    Returns:
        imgs: (B, C, H, W) tensor of images
    """
    B, _num_patches, D = patches.shape
    C = D // (patch_size * patch_size)
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    imgs = patches.view(B, num_patches_h, num_patches_w, C,
                        patch_size, patch_size).permute(0, 3, 1, 4, 2, 5)
    imgs = imgs.reshape(B, C, H, W)
    return imgs


# ----------------------------
# Timestep embedding (DiT-style)
# ----------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] in [0,1] (float)
    returns: [B, dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half,
                                        device=device).float() / (half - 1)
    )
    args = t[:, None] * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
