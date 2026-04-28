import pytest
import torch

from ultralytics.nn.extra_modules.attention.DHPF import DHPF
from ultralytics.nn.extra_modules.attention.FSA import FSA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression coverage")
@pytest.mark.parametrize(
    ("module_name", "size"),
    [
        ("DHPF", (20, 20)),
        ("DHPF", (40, 40)),
        ("DHPF", (80, 80)),
        ("FSA", (20, 20)),
        ("FSA", (32, 32)),
        ("FSA", (40, 40)),
        ("FSA", (80, 80)),
    ],
)
def test_frequency_attention_amp_forward_backward(module_name, size):
    """Frequency-domain attention modules should support half-precision feature maps from AMP layers."""
    h, w = size
    if module_name == "DHPF":
        model = DHPF(64).to("cuda")
    else:
        model = FSA(64, size=(h, w)).to("cuda")

    x = torch.randn(1, 64, h, w, device="cuda", dtype=torch.float16, requires_grad=True)
    loss = model(x).square().mean()

    loss.backward()
