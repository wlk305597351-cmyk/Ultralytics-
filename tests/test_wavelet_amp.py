import pytest
import torch

from ultralytics.nn.extra_modules.downsample.FSConv import FSConv
from ultralytics.nn.extra_modules.downsample.HWD import HWD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression coverage")
@pytest.mark.parametrize(("module_cls", "in_ch", "out_ch"), [(FSConv, 64, 64), (HWD, 64, 64)])
def test_wavelet_downsample_amp_backward(module_cls, in_ch, out_ch):
    """Wavelet downsample modules should support AMP forward+backward on CUDA."""
    model = module_cls(in_ch, out_ch).to("cuda")
    x = torch.randn(2, in_ch, 64, 64, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(x).mean()

    loss.backward()
