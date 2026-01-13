import pytest
import torch
from models import VelocityDiT
from config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def model_inputs(config):
    # mock batch
    B = 2
    C = config.z_dim
    H = W = config.z_image_size
    
    # random latent at timestep t
    z_t = torch.randn(B, C, H, W)
    
    # random timesteps
    t = torch.randint(low=0, high=1000, size=(B,), dtype=torch.long)
    
    return z_t, t, B, C, H, W


def test_velocity_dit_forward_pass(config, model_inputs):
    z_t, t, B, C, H, W = model_inputs
    
    # model
    model = VelocityDiT(config)
    model.eval()
    
    with torch.no_grad():
        z_pred = model(z_t, t)
    
    # assertions
    assert z_t.shape == (B, C, H, W)
    assert t.shape == (B,)
    assert z_pred.shape == (B, C, H, W)
