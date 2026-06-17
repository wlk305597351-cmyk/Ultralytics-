# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.nn.tasks import parse_model
from ultralytics.utils import YAML


def _load_yaml(path: Path) -> dict:
    d = YAML.load(path)
    d["scale"] = "n"
    return d


def test_parse_model_supports_nested_module_param_dict():
    """Regression: nested {'module': ..., 'param': ...} values should resolve to callables."""
    yaml_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/improve/conv/yolo26-c3k2-conv.yaml"
    model, save = parse_model(_load_yaml(yaml_path), ch=3, verbose=False)
    assert len(model) > 0
    assert isinstance(save, list)


def test_parse_model_keeps_existing_block_param_behavior():
    """Compatibility: existing block module YAMLs should still parse unchanged."""
    yaml_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/improve/yolo26-metaformer.yaml"
    model, save = parse_model(_load_yaml(yaml_path), ch=3, verbose=False)
    assert len(model) > 0
    assert isinstance(save, list)
