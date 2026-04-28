# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from types import SimpleNamespace
from unittest import mock

import torch

from tests import MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR


def test_func(*args, **kwargs):
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


def test_export():
    """Test model exporting functionality by adding a callback and verifying its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_segment():
    """Test image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo26n-seg.yaml",
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo26n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo26n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_fitness_progress_logging():
    """Test fitness delta tracking and logging format in validation flow."""
    trainer = mock.Mock()
    trainer.ema = None
    trainer.world_size = 1
    trainer.loss = torch.tensor(0.0)
    trainer.best_fitness = None
    trainer.fitness_prev_best = None
    trainer.fitness_delta = None
    trainer.fitness_improved = False

    # First validation should initialize best and mark improvement.
    trainer.validator = lambda _: {"fitness": 0.5, "metrics/mAP50-95(B)": 0.4}
    _, trainer.fitness = BaseTrainer.validate(trainer)
    assert trainer.fitness_prev_best is None
    assert trainer.fitness_delta == 0.0
    assert trainer.fitness_improved is True
    assert trainer.best_fitness == 0.5

    # Second validation with lower score should not update best and should log negative delta.
    trainer.validator = lambda _: {"fitness": 0.4, "metrics/mAP50-95(B)": 0.3}
    _, trainer.fitness = BaseTrainer.validate(trainer)
    assert trainer.fitness_prev_best == 0.5
    assert abs(trainer.fitness_delta + 0.1) < 1e-12
    assert trainer.fitness_improved is False
    assert trainer.best_fitness == 0.5

    # Third validation with higher score should update best and emit best-updated marker.
    trainer.validator = lambda _: {"fitness": 0.6, "metrics/mAP50-95(B)": 0.5}
    _, trainer.fitness = BaseTrainer.validate(trainer)
    with mock.patch("ultralytics.engine.trainer.LOGGER.info") as info_mock:
        BaseTrainer._log_fitness_progress(trainer)
        info_mock.assert_called_once()
        msg = info_mock.call_args.args[0]
        assert "Fitness: current=0.60000" in msg
        assert "prev_best=0.50000" in msg
        assert "delta=+0.10000" in msg
        assert "best=0.60000" in msg
        assert "[best updated]" in msg


def test_do_train_uses_current_train_loader_length_for_progress_bar(monkeypatch):
    """Progress bar total should follow the active train loader after callbacks rebuild it."""

    class DummyLoader:
        def __init__(self, batches: int):
            self.batches = batches
            self.num_workers = 0
            self.dataset = [None] * batches

        def __len__(self):
            return self.batches

        def __iter__(self):
            for _ in range(self.batches):
                yield {"cls": torch.zeros((1, 1)), "img": torch.zeros((1, 3, 32, 32))}

    class DummyModel:
        criterion = object()

        def __call__(self, batch):
            return torch.tensor(1.0, requires_grad=True), torch.tensor([1.0, 1.0, 1.0])

    class DummyStopper:
        possible_stop = False

        def __call__(self, epoch, fitness):
            return False

    class FakeTQDM:
        totals = []

        def __init__(self, iterable, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.totals.append(total)

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, *args, **kwargs):
            return None

    trainer = SimpleNamespace()
    trainer.args = SimpleNamespace(
        warmup_epochs=0,
        nbs=1,
        close_mosaic=0,
        compile=False,
        time=None,
        val=False,
        plots=False,
        save=False,
        imgsz=32,
    )
    trainer.world_size = 1
    trainer.train_loader = DummyLoader(10)
    trainer.save_dir = "runs/test"
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}], zero_grad=lambda: None)
    trainer.scheduler = SimpleNamespace(step=lambda: None, last_epoch=-1)
    trainer.stopper = DummyStopper()
    trainer.stop = False
    trainer.start_epoch = 0
    trainer.epochs = 1
    trainer.epoch = 0
    trainer.epoch_time_start = 0.0
    trainer.train_time_start = 0.0
    trainer.plot_idx = []
    trainer.tloss = None
    trainer.scaler = SimpleNamespace(scale=lambda x: x)
    trainer.model = DummyModel()
    trainer.ema = SimpleNamespace(update_attr=lambda *args, **kwargs: None)
    trainer.metrics = {}
    trainer.fitness = 0.0
    trainer.device = torch.device("cpu")
    trainer.amp = False
    trainer.accumulate = 1
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss")
    trainer.nan_recovery_attempts = 0

    trainer._setup_train = lambda: None
    trainer._model_train = lambda: None
    trainer.preprocess_batch = lambda batch: batch
    trainer.optimizer_step = lambda: None
    trainer.progress_string = lambda: "progress"
    trainer._get_memory = lambda fraction=False: 0.0
    trainer.validate = lambda: ({}, 0.0)
    trainer.label_loss_items = lambda tloss=None, prefix="train": {}
    trainer._log_fitness_progress = lambda: None
    trainer._clear_memory = lambda threshold=None: None
    trainer._handle_nan_recovery = lambda epoch: False
    trainer.save_metrics = lambda metrics: None
    trainer.save_model = lambda: None
    trainer.final_eval = lambda: None
    trainer.plot_metrics = lambda: None

    state = {"swapped": False}

    def run_callbacks(event):
        if event == "on_train_epoch_start" and not state["swapped"]:
            trainer.train_loader = DummyLoader(3)
            state["swapped"] = True

    trainer.run_callbacks = run_callbacks

    monkeypatch.setattr("ultralytics.engine.trainer.TQDM", FakeTQDM)
    monkeypatch.setattr("ultralytics.engine.trainer.LOGGER.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("ultralytics.engine.trainer.unset_deterministic", lambda: None)

    BaseTrainer._do_train(trainer)

    assert FakeTQDM.totals == [3]
