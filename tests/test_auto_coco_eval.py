# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
import types

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.coco_auto import (
    build_annotations,
    build_categories,
    build_image_index,
    export_auto_gt_json,
)
from ultralytics.utils.metrics import DetMetrics


def _dummy_labels(task: str) -> list[dict]:
    im_file = "/tmp/auto_coco_eval_0001.jpg"
    base = {
        "im_file": im_file,
        "shape": (100, 200),  # h, w
        "cls": np.array([[0]], dtype=np.float32),
        "bboxes": np.array([[0.2, 0.3, 0.2, 0.2]], dtype=np.float32),  # xywh normalized
        "segments": [],
        "keypoints": None,
    }
    if task == "segment":
        base["segments"] = [np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]], dtype=np.float32)]
    if task == "pose":
        base["keypoints"] = np.array([[[0.2, 0.3, 1.0], [0.4, 0.5, 0.0], [0.6, 0.7, 2.0]]], dtype=np.float32)
    return [base]


def test_build_detect_annotations():
    labels = _dummy_labels("detect")
    image_map, _ = build_image_index([labels[0]["im_file"]])
    annotations = build_annotations(labels, image_map, task="detect")

    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["image_id"] == 1
    assert ann["category_id"] == 1
    assert ann["bbox"] == pytest.approx([20.0, 20.0, 40.0, 20.0])
    assert ann["area"] == pytest.approx(800.0)


def test_det_metrics_exposes_empty_coco_results_dict():
    metrics = DetMetrics()
    assert metrics.coco_results_dict == {}


def test_build_segment_annotations():
    labels = _dummy_labels("segment")
    image_map, _ = build_image_index([labels[0]["im_file"]])
    annotations = build_annotations(labels, image_map, task="segment")

    assert len(annotations) == 1
    ann = annotations[0]
    assert len(ann["segmentation"]) == 1
    assert ann["segmentation"][0] == pytest.approx([20.0, 20.0, 60.0, 20.0, 60.0, 40.0])
    assert ann["area"] == pytest.approx(400.0)


def test_build_pose_annotations():
    labels = _dummy_labels("pose")
    image_map, _ = build_image_index([labels[0]["im_file"]])
    annotations = build_annotations(labels, image_map, task="pose", kpt_shape=[3, 3])

    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["keypoints"] == pytest.approx([40.0, 30.0, 1.0, 80.0, 50.0, 0.0, 120.0, 70.0, 2.0])
    assert ann["num_keypoints"] == 2


def test_build_categories_for_pose():
    categories = build_categories(names={0: "person"}, task="pose", kpt_shape=[3, 3])
    assert categories == [
        {
            "id": 1,
            "name": "person",
            "supercategory": "none",
            "keypoints": ["kpt_1", "kpt_2", "kpt_3"],
            "skeleton": [[1, 2], [2, 3]],
        }
    ]


@pytest.mark.parametrize(
    ("task", "filename"),
    [
        ("detect", "annotations_auto_detect_val.json"),
        ("segment", "annotations_auto_segment_val.json"),
        ("pose", "annotations_auto_pose_val.json"),
    ],
)
def test_export_auto_gt_json(task: str, filename: str, tmp_path: Path):
    labels = _dummy_labels(task)
    path = export_auto_gt_json(
        labels=labels,
        im_files=[labels[0]["im_file"]],
        names={0: "cls0"},
        task=task,
        save_dir=tmp_path,
        split="val",
        kpt_shape=[3, 3] if task == "pose" else None,
    )
    assert path == tmp_path / filename
    assert path.is_file()


def test_detection_validator_auto_coco_eval_uses_image_map():
    labels = _dummy_labels("detect")
    dataset = SimpleNamespace(im_files=[labels[0]["im_file"]], labels=labels)
    dataloader = SimpleNamespace(dataset=dataset)
    model = SimpleNamespace(names={0: "cls0"})
    validator = DetectionValidator(args={"auto_coco_eval": True, "split": "val"})
    validator.data = {"val": "custom/val.txt", "path": Path("/tmp")}
    validator.dataloader = dataloader
    validator.training = False
    validator.init_metrics(model)

    assert validator.auto_coco_eval is True
    assert validator.args.save_json is True
    assert validator.class_map == [1]
    assert validator.eval_img_ids == [1]

    predn = {
        "bboxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
    }
    validator.pred_to_json(predn, {"im_file": labels[0]["im_file"]})
    assert validator.jdict[0]["image_id"] == 1


def test_export_auto_gt_json_without_shape_falls_back_to_image_size(tmp_path: Path):
    im_file = tmp_path / "shape_fallback.jpg"
    Image.new("RGB", (320, 160)).save(im_file)
    labels = [
        {
            "im_file": str(im_file),
            "cls": np.array([[0]], dtype=np.float32),
            "bboxes": np.array([[0.5, 0.5, 0.25, 0.5]], dtype=np.float32),
            "segments": [],
            "keypoints": None,
        }
    ]
    json_path = export_auto_gt_json(
        labels=labels,
        im_files=[str(im_file)],
        names={0: "cls0"},
        task="detect",
        save_dir=tmp_path,
        split="val",
    )
    data = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert data["images"][0]["width"] == 320
    assert data["images"][0]["height"] == 160
    assert data["annotations"][0]["bbox"] == pytest.approx([120.0, 40.0, 80.0, 80.0])


def test_coco_evaluate_keypoints_passes_custom_kpt_sigmas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_json = tmp_path / "predictions.json"
    anno_json = tmp_path / "annotations.json"
    pred_json.write_text("[]", encoding="utf-8")
    anno_json.write_text("{}", encoding="utf-8")

    class FakeCOCO:
        def __init__(self, _path):
            self.path = _path

        def loadRes(self, _path):
            return {"pred": str(_path)}

    calls = []

    class FakeCOCOevalFaster:
        def __init__(self, anno, pred, iouType, lvis_style, print_function, **kwargs):
            calls.append({"iouType": iouType, "kwargs": kwargs})
            self.params = SimpleNamespace(imgIds=[])
            self.stats_as_dict = {"AP_50": 0.1, "AP_all": 0.2, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

        def evaluate(self):
            return None

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_coco_eval",
        types.SimpleNamespace(COCO=FakeCOCO, COCOeval_faster=FakeCOCOevalFaster),
    )
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.check_requirements", lambda *_args, **_kwargs: None)

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict = [{"image_id": 1}]
    validator.auto_coco_eval = True
    validator.is_lvis = False
    validator.eval_img_ids = [1]
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))
    validator.sigma = np.ones(24, dtype=np.float32) / 24

    stats = validator.coco_evaluate(
        {},
        pred_json,
        anno_json,
        iou_types=["bbox", "keypoints"],
        suffix=["Box", "Pose"],
    )
    assert "metrics/mAP50(B)" in stats
    assert "metrics/mAP50(P)" in stats
    assert validator.metrics.coco_results_dict["metrics/mAP50(B)"] == 0.1
    assert validator.metrics.coco_results_dict["metrics/mAP50(P)"] == 0.1

    keypoint_calls = [c for c in calls if c["iouType"] == "keypoints"]
    assert len(keypoint_calls) == 1
    assert "kpt_oks_sigmas" in keypoint_calls[0]["kwargs"]
    assert len(keypoint_calls[0]["kwargs"]["kpt_oks_sigmas"]) == 24


def test_coco_evaluate_handles_missing_ap_small_for_keypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_json = tmp_path / "predictions.json"
    anno_json = tmp_path / "annotations.json"
    pred_json.write_text("[]", encoding="utf-8")
    anno_json.write_text("{}", encoding="utf-8")

    class FakeCOCO:
        def __init__(self, _path):
            self.path = _path

        def loadRes(self, _path):
            return {"pred": str(_path)}

    class FakeCOCOevalFaster:
        def __init__(self, anno, pred, iouType, lvis_style, print_function, **kwargs):
            self.params = SimpleNamespace(imgIds=[])
            if iouType == "bbox":
                self.stats_as_dict = {
                    "AP_50": 0.3,
                    "AP_all": 0.2,
                    "AP_small": 0.1,
                    "AP_medium": 0.15,
                    "AP_large": 0.18,
                }
            else:
                # faster-coco-eval keypoints output has no AP_small
                self.stats_as_dict = {"AP_50": 0.4, "AP_all": 0.25, "AP_medium": 0.2, "AP_large": 0.22}

        def evaluate(self):
            return None

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_coco_eval",
        types.SimpleNamespace(COCO=FakeCOCO, COCOeval_faster=FakeCOCOevalFaster),
    )
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.check_requirements", lambda *_args, **_kwargs: None)
    warnings = []
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.LOGGER.warning", lambda msg: warnings.append(str(msg)))

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict = [{"image_id": 1}]
    validator.auto_coco_eval = True
    validator.is_lvis = False
    validator.eval_img_ids = [1]
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))
    validator.sigma = np.ones(24, dtype=np.float32) / 24

    stats = validator.coco_evaluate(
        {},
        pred_json,
        anno_json,
        iou_types=["bbox", "keypoints"],
        suffix=["Box", "Pose"],
    )
    assert stats["metrics/mAP50(B)"] == 0.3
    assert stats["metrics/mAP50(P)"] == 0.4
    assert stats["metrics/mAP_small(B)"] == 0.1
    assert not warnings


def test_coco_evaluate_populates_segment_keys_in_coco_results_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_json = tmp_path / "predictions.json"
    anno_json = tmp_path / "annotations.json"
    pred_json.write_text("[]", encoding="utf-8")
    anno_json.write_text("{}", encoding="utf-8")

    class FakeCOCO:
        def __init__(self, _path):
            self.path = _path

        def loadRes(self, _path):
            return {"pred": str(_path)}

    class FakeCOCOevalFaster:
        def __init__(self, anno, pred, iouType, lvis_style, print_function, **kwargs):
            self.params = SimpleNamespace(imgIds=[])
            self.stats_as_dict = {
                "AP_50": 0.55 if iouType == "segm" else 0.45,
                "AP_all": 0.35 if iouType == "segm" else 0.25,
                "AP_small": 0.1,
                "AP_medium": 0.2,
                "AP_large": 0.3,
            }

        def evaluate(self):
            return None

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_coco_eval",
        types.SimpleNamespace(COCO=FakeCOCO, COCOeval_faster=FakeCOCOevalFaster),
    )
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.check_requirements", lambda *_args, **_kwargs: None)

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict = [{"image_id": 1}]
    validator.auto_coco_eval = True
    validator.is_lvis = False
    validator.eval_img_ids = [1]
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))

    _ = validator.coco_evaluate({}, pred_json, anno_json, iou_types=["bbox", "segm"], suffix=["Box", "Mask"])
    assert validator.metrics.coco_results_dict["metrics/mAP50(B)"] == 0.45
    assert validator.metrics.coco_results_dict["metrics/mAP50(M)"] == 0.55


def test_coco_results_dict_contains_per_class_for_box_and_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_json = tmp_path / "predictions.json"
    anno_json = tmp_path / "annotations.json"
    pred_json.write_text("[]", encoding="utf-8")
    anno_json.write_text("{}", encoding="utf-8")

    class FakeCOCO:
        def __init__(self, _path):
            self.path = _path

        def loadRes(self, _path):
            return {"pred": str(_path)}

    class FakeCOCOevalFaster:
        def __init__(self, anno, pred, iouType, lvis_style, print_function, **kwargs):
            self.params = SimpleNamespace(
                imgIds=[],
                iouThrs=np.array([0.5, 0.75], dtype=np.float32),
                areaRngLbl=["all", "small", "medium", "large"],
                catIds=[1],
            )
            p = np.full((2, 1, 1, 4, 1), -1.0, dtype=np.float32)
            if iouType == "segm":
                p[0, 0, 0, 0, 0], p[1, 0, 0, 0, 0] = 0.7, 0.1
                p[0, 0, 0, 1, 0], p[1, 0, 0, 1, 0] = 0.2, 0.4
                p[0, 0, 0, 2, 0], p[1, 0, 0, 2, 0] = 0.3, 0.5
                p[0, 0, 0, 3, 0], p[1, 0, 0, 3, 0] = 0.4, 0.6
                self.stats_as_dict = {"AP_50": 0.7, "AP_all": 0.4, "AP_small": 0.3, "AP_medium": 0.4, "AP_large": 0.5}
            else:
                p[0, 0, 0, 0, 0], p[1, 0, 0, 0, 0] = 0.6, 0.2
                p[0, 0, 0, 1, 0], p[1, 0, 0, 1, 0] = 0.1, 0.3
                p[0, 0, 0, 2, 0], p[1, 0, 0, 2, 0] = 0.2, 0.4
                p[0, 0, 0, 3, 0], p[1, 0, 0, 3, 0] = 0.3, 0.5
                self.stats_as_dict = {"AP_50": 0.6, "AP_all": 0.4, "AP_small": 0.2, "AP_medium": 0.3, "AP_large": 0.4}
            self.eval = {"precision": p}

        def evaluate(self):
            return None

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_coco_eval",
        types.SimpleNamespace(COCO=FakeCOCO, COCOeval_faster=FakeCOCOevalFaster),
    )
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.check_requirements", lambda *_args, **_kwargs: None)

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict = [{"image_id": 1}]
    validator.auto_coco_eval = True
    validator.is_lvis = False
    validator.eval_img_ids = [1]
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))
    validator.names = {0: "dog"}
    validator.class_map = [1]

    _ = validator.coco_evaluate({}, pred_json, anno_json, iou_types=["bbox", "segm"], suffix=["Box", "Mask"])
    per_class = validator.metrics.coco_results_dict["per_class"]
    assert per_class["B"]["dog"]["AP50"] == pytest.approx(0.6)
    assert per_class["B"]["dog"]["AP75"] == pytest.approx(0.2)
    assert per_class["B"]["dog"]["AP50-95"] == pytest.approx(0.4)
    assert per_class["B"]["dog"]["APs"] == pytest.approx(0.2)
    assert per_class["B"]["dog"]["APm"] == pytest.approx(0.3)
    assert per_class["B"]["dog"]["APl"] == pytest.approx(0.4)
    assert per_class["M"]["dog"]["AP50"] == pytest.approx(0.7)


def test_coco_results_dict_per_class_pose_without_small_sets_aps_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pred_json = tmp_path / "predictions.json"
    anno_json = tmp_path / "annotations.json"
    pred_json.write_text("[]", encoding="utf-8")
    anno_json.write_text("{}", encoding="utf-8")

    class FakeCOCO:
        def __init__(self, _path):
            self.path = _path

        def loadRes(self, _path):
            return {"pred": str(_path)}

    class FakeCOCOevalFaster:
        def __init__(self, anno, pred, iouType, lvis_style, print_function, **kwargs):
            self.params = SimpleNamespace(
                imgIds=[],
                iouThrs=np.array([0.5, 0.75], dtype=np.float32),
                areaRngLbl=["all", "medium", "large"],  # no 'small' for keypoints
                catIds=[1],
            )
            p = np.full((2, 1, 1, 3, 1), -1.0, dtype=np.float32)
            p[0, 0, 0, 0, 0], p[1, 0, 0, 0, 0] = 0.5, 0.3
            p[0, 0, 0, 1, 0], p[1, 0, 0, 1, 0] = 0.2, 0.4
            p[0, 0, 0, 2, 0], p[1, 0, 0, 2, 0] = 0.1, 0.3
            self.eval = {"precision": p}
            self.stats_as_dict = {"AP_50": 0.5, "AP_all": 0.4, "AP_medium": 0.3, "AP_large": 0.2}

        def evaluate(self):
            return None

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_coco_eval",
        types.SimpleNamespace(COCO=FakeCOCO, COCOeval_faster=FakeCOCOevalFaster),
    )
    monkeypatch.setattr("ultralytics.models.yolo.detect.val.check_requirements", lambda *_args, **_kwargs: None)

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict = [{"image_id": 1}]
    validator.auto_coco_eval = True
    validator.is_lvis = False
    validator.eval_img_ids = [1]
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))
    validator.names = {0: "dog"}
    validator.class_map = [1]
    validator.sigma = np.ones(24, dtype=np.float32) / 24

    _ = validator.coco_evaluate({}, pred_json, anno_json, iou_types=["keypoints"], suffix=["Pose"])
    pose_class = validator.metrics.coco_results_dict["per_class"]["P"]["dog"]
    assert pose_class["AP50"] == pytest.approx(0.5)
    assert pose_class["AP75"] == pytest.approx(0.3)
    assert pose_class["AP50-95"] == pytest.approx(0.4)
    assert pose_class["APs"] is None
    assert pose_class["APm"] == pytest.approx(0.3)
    assert pose_class["APl"] == pytest.approx(0.2)
