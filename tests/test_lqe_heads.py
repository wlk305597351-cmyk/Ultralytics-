from importlib import import_module
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model


LQE_HEADS = {
    "Segment_LQE": {
        "args": (5, 8, 32, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "mask_coefficient", "proto", "feats"},
    },
    "Segment26_LQE": {
        "args": (5, 8, 32, 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
    "OBB_LQE": {
        "args": (5, 1, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "angle", "feats"},
    },
    "OBB26_LQE": {
        "args": (5, 1, 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
    "Pose_LQE": {
        "args": (5, (17, 3), 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "kpts", "feats"},
    },
    "Pose26_LQE": {
        "args": (5, (17, 3), 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
}

LQE_PARSE_CASES = [
    ("Segment_LQE", [5, 32, 256]),
    ("Segment26_LQE", [5, 32, 256]),
    ("OBB_LQE", [5, 1]),
    ("OBB26_LQE", [5, 1]),
    ("Pose_LQE", [5, (17, 3)]),
    ("Pose26_LQE", [5, (17, 3)]),
]

LQE_YAMLS = [
    "ultralytics/cfg/models/improve/head/LQE/yolo11/yolo11-seg-LQE.yaml",
    "ultralytics/cfg/models/improve/head/LQE/yolo11/yolo11-pose-LQE.yaml",
    "ultralytics/cfg/models/improve/head/LQE/yolo11/yolo11-obb-LQE.yaml",
    "ultralytics/cfg/models/improve/head/LQE/yolo26/yolo26-seg-LQE.yaml",
    "ultralytics/cfg/models/improve/head/LQE/yolo26/yolo26-pose-LQE.yaml",
    "ultralytics/cfg/models/improve/head/LQE/yolo26/yolo26-obb-LQE.yaml",
]


def _feature_maps():
    return [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8),
    ]


def _set_stride(head):
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    return head


@pytest.mark.parametrize("class_name", sorted(LQE_HEADS))
def test_lqe_module_exports_task_heads(class_name):
    module = import_module("ultralytics.nn.extra_modules.head.LQE")
    assert hasattr(module, class_name)


@pytest.mark.parametrize(("module_name", "args"), LQE_PARSE_CASES)
def test_parse_model_supports_lqe_task_heads(module_name, args):
    model, save = parse_model(
        {
            "nc": 5,
            "kpt_shape": [17, 3],
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "reg_max": 1 if "26" in module_name else 16,
            "end2end": "26" in module_name,
            "backbone": [
                [-1, 1, "Conv", [32, 3, 2]],
                [-1, 1, "Conv", [64, 3, 2]],
                [-1, 1, "Conv", [128, 3, 2]],
            ],
            "head": [
                [[0, 1, 2], 1, module_name, args],
            ],
        },
        ch=3,
        verbose=False,
    )

    assert len(model) == 4
    assert isinstance(save, list)


@pytest.mark.parametrize(("class_name", "config"), sorted(LQE_HEADS.items()))
def test_lqe_task_heads_train_outputs_include_task_specific_keys(class_name, config):
    module = import_module("ultralytics.nn.extra_modules.head.LQE")
    head_cls = getattr(module, class_name)
    head = _set_stride(head_cls(*config["args"])).train()

    outputs = head(_feature_maps())

    assert isinstance(outputs, dict)
    assert config["train_keys"].issubset(outputs)
    if "one2many" in outputs:
        assert "boxes" in outputs["one2many"]
        assert "scores" in outputs["one2many"]
    if "one2one" in outputs:
        assert "boxes" in outputs["one2one"]
        assert "scores" in outputs["one2one"]


@pytest.mark.parametrize(("class_name", "config"), sorted(LQE_HEADS.items()))
def test_lqe_task_heads_eval_forward_runs(class_name, config):
    module = import_module("ultralytics.nn.extra_modules.head.LQE")
    head_cls = getattr(module, class_name)
    head = _set_stride(head_cls(*config["args"])).eval()

    outputs = head(_feature_maps())

    assert outputs is not None


@pytest.mark.parametrize("yaml_path", LQE_YAMLS)
def test_lqe_yaml_examples_build(yaml_path):
    path = Path(yaml_path)
    assert path.is_file()
    model = YOLO(str(path))
    assert model.model is not None
