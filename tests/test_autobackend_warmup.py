import pytest
import torch

import ultralytics.nn.autobackend as autobackend_module
from ultralytics.nn.autobackend import AutoBackend


class _FiniteInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.names = {0: "class0"}
        self.stride = torch.tensor([32.0])
        self.yaml = {"channels": 3}

    def fuse(self, verbose=True):
        return self

    def forward(self, x, *args, **kwargs):
        if not torch.isfinite(x).all():
            raise AssertionError("warmup input must be finite")
        return self.conv(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for warmup regression coverage")
def test_autobackend_warmup_uses_finite_dummy_input(monkeypatch):
    """Warmup should use finite dummy inputs even if uninitialized allocations contain NaNs."""
    model = _FiniteInputModel().to("cuda")

    def fake_empty(*size, **kwargs):
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return torch.full(shape, float("nan"), **kwargs)

    monkeypatch.setattr(autobackend_module.torch, "empty", fake_empty)

    backend = AutoBackend(model, device=torch.device("cuda:0"), fp16=False, fuse=True, verbose=False)
    backend.warmup((1, 3, 32, 32))
