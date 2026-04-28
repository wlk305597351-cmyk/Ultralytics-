import torch

from ultralytics.nn.extra_modules.mamba.TransMixer import TransMixerModule
from ultralytics.utils.torch_utils import initialize_weights


def test_transmixer_backward_survives_global_inplace_activation_init():
    """TransMixer should remain trainable after Ultralytics flips activations to inplace mode."""
    model = TransMixerModule(32)
    initialize_weights(model)

    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    y = model(x)
    y.mean().backward()
