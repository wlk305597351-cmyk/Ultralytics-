import pytest
import torch

from ultralytics.nn.extra_modules.transformer.CBSA import CBSA
from ultralytics.nn.extra_modules.transformer.MSLA import MSLA
from ultralytics.nn.extra_modules.transformer.MUA import MaskUnitAttention
from ultralytics.nn.extra_modules.transformer.PolaLinearAttention import PolaLinearAttention


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    ("module", "x4d"),
    [
        (MSLA(dim=64, num_heads=8).eval(), torch.randn(2, 64, 8, 8)),
        (CBSA(dim=64, num_heads=8).eval(), torch.randn(2, 64, 8, 8)),
        (
            MaskUnitAttention(dim=64, heads=8, q_stride=1, window_size=16, use_mask_unit_attn=True).eval(),
            torch.randn(2, 64, 8, 8),
        ),
        (PolaLinearAttention(dim=64, hw=(8, 8), num_heads=8).eval(), torch.randn(2, 64, 8, 8)),
    ],
)
def test_transformer_modules_accept_bchw_and_match_legacy_tokens(module: torch.nn.Module, x4d: torch.Tensor):
    x3d = _to_tokens(x4d)

    with torch.no_grad():
        y3d = module(x3d)
        y4d = module(x4d)

    assert y3d.ndim == 3
    assert y4d.shape == x4d.shape
    assert torch.allclose(_to_tokens(y4d), y3d, atol=1e-5, rtol=1e-5)
