import ast
from pathlib import Path


SCRIPT_PATH = Path("train_afss.py")


def test_train_afss_script_uses_explicit_afss_trainer_entry():
    module = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    imported_afss_trainer = False
    train_call = None

    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module == "ultralytics.models.yolo.detect.afss_train":
            imported_afss_trainer = any(alias.name == "AFSSDetectionTrainer" for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train":
            train_call = node

    assert imported_afss_trainer is True
    assert train_call is not None

    keywords = {keyword.arg: keyword.value for keyword in train_call.keywords if keyword.arg}
    assert isinstance(keywords["trainer"], ast.Name)
    assert keywords["trainer"].id == "AFSSDetectionTrainer"
    assert isinstance(keywords["afss"], ast.Constant)
    assert keywords["afss"].value is True
