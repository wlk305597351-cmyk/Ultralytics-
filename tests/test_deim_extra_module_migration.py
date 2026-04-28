# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.extra_modules import EdgeLAWDS, FreqLAWDS, RouterLAWDS, Faster_CGA_Block
from ultralytics.nn.tasks import parse_model


@pytest.mark.parametrize(
    ("module_cls", "init_args", "input_shape", "expected_shape"),
    [
        (EdgeLAWDS, (16, 32), (1, 16, 31, 33), (1, 32, 16, 17)),
        (FreqLAWDS, (16, 32), (1, 16, 31, 33), (1, 32, 16, 17)),
        (RouterLAWDS, (16, 32), (1, 16, 31, 33), (1, 32, 16, 17)),
        (Faster_CGA_Block, (16, 32), (1, 16, 31, 33), (1, 32, 31, 33)),
    ],
)
def test_migrated_modules_forward_shapes(module_cls, init_args, input_shape, expected_shape):
    module = module_cls(*init_args).eval()
    output = module(torch.randn(*input_shape))
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "module_name",
    ["EdgeLAWDS", "FreqLAWDS", "RouterLAWDS", "Faster_CGA_Block"],
)
def test_parse_model_supports_migrated_modules(module_name):
    model, save = parse_model(
        {
            "nc": 80,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "backbone": [
                [-1, 1, "Conv", [16, 3, 2]],
                [-1, 1, module_name, [32]],
            ],
            "head": [],
        },
        ch=3,
        verbose=False,
    )

    assert len(model) == 2
    assert isinstance(save, list)
