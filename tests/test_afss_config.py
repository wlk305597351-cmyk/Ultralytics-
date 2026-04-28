import inspect

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.model import Model
from ultralytics.models.yolo import detect
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer


def test_afss_explicit_trainer_required_for_phase1():
    cfg = get_cfg()
    assert cfg.afss is False
    assert cfg.afss_warmup_epochs == 20
    assert cfg.afss_update_interval == 5
    assert cfg.afss_easy_ratio == 0.02
    assert cfg.afss_moderate_ratio == 0.40
    assert cfg.afss_easy_forced_gap == 10
    assert cfg.afss_moderate_forced_gap == 3
    assert cfg.afss_conf == 0.25
    assert cfg.afss_save_refresh_json is False
    assert cfg.afss_thresholds == {
        "detect": [0.55, 0.85],
        "obb": [0.55, 0.85],
        "segment": [0.55, 0.85],
        "pose": [0.55, 0.85],
    }

    assert inspect.signature(Model.train).parameters["trainer"].default is None

    model = YOLO("yolo26n.yaml")
    assert model._smart_load("trainer") is detect.DetectionTrainer
    assert "afss" not in inspect.getsource(detect.DetectionTrainer).lower()


def test_default_detect_trainer_does_not_create_afss_artifacts(tmp_path):
    trainer = detect.DetectionTrainer(
        overrides={
            "model": "yolo26n.yaml",
            "data": "coco8.yaml",
            "imgsz": 32,
            "epochs": 1,
            "project": str(tmp_path),
            "name": "default-detect",
            "save": False,
        }
    )

    assert type(trainer) is detect.DetectionTrainer
    assert not isinstance(trainer, AFSSDetectionTrainer)
    assert not hasattr(trainer, "afss_enabled")
    assert not (trainer.save_dir / "afss").exists()
