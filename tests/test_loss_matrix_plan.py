import os
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss


def _default_args(**overrides):
    base = dict(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        cls_loss="bce",
        iou_loss="ciou",
        iou_aux="none",
        iou_aux_ratio=0.5,
        inner_iou_ratio=0.7,
        focaler_d=0.0,
        focaler_u=0.95,
        shapeiou_scale=0.0,
        piou_lambda=1.3,
        wiseiou_monotonous=False,
        slide_auto_iou_min=0.2,
        slide_delta=0.1,
        ema_decay=0.999,
        ema_tau=2000.0,
        focal_gamma=1.5,
        focal_alpha=0.25,
        varifocal_alpha=0.75,
        varifocal_gamma=2.0,
        qfl_beta=2.0,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _DummyDetect(nn.Module):
    def __init__(self, nc=3, reg_max=16):
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.nc = nc
        self.reg_max = reg_max


class _DummyModel(nn.Module):
    def __init__(self, args, nc=3, reg_max=16):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.args = args
        self.model = nn.ModuleList([nn.Identity(), _DummyDetect(nc=nc, reg_max=reg_max)])


def _boxes_xyxy(n: int = 4):
    a = torch.rand(n, 2)
    b = torch.rand(n, 2)
    p1 = torch.minimum(a, b)
    p2 = torch.maximum(a, b) + 0.2
    box1 = torch.cat((p1, p2), dim=1)
    box2 = box1 + 0.05
    return box1, box2


def _make_preds_and_batch(bs=2, nc=3, reg_max=16):
    feats = [torch.randn(bs, 8, 8, 8), torch.randn(bs, 8, 4, 4), torch.randn(bs, 8, 2, 2)]
    n_anchors = sum(f.shape[2] * f.shape[3] for f in feats)
    preds = {
        "feats": feats,
        "boxes": torch.randn(bs, reg_max * 4, n_anchors),
        "scores": torch.randn(bs, nc, n_anchors),
    }
    batch = {
        "batch_idx": torch.arange(bs, dtype=torch.long),
        "cls": torch.arange(bs, dtype=torch.float32) % nc,
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.30, 0.30],
                [0.45, 0.55, 0.25, 0.20],
            ],
            dtype=torch.float32,
        )[:bs],
    }
    return preds, batch


class LossMatrixPlanTest(unittest.TestCase):
    CLS_CASES = [
        dict(cls_loss="bce"),
        dict(cls_loss="slide"),
        dict(cls_loss="ema_slide"),
        dict(cls_loss="focal", focal_gamma=1.5, focal_alpha=0.25),
        dict(cls_loss="varifocal", varifocal_alpha=0.75, varifocal_gamma=2.0),
        dict(cls_loss="qualityfocal", qfl_beta=2.0),
    ]

    IOU_CASES = [
        "iou",
        "giou",
        "diou",
        "ciou",
        "eiou",
        "siou",
        "shapeiou",
        "piou",
        "piou2",
        "inner_diou",
        "focaler_diou",
        "mpdiou",
        "inner_mpdiou",
        "focaler_mpdiou",
        "wiseiou",
        "wiseiou_diou",
        "wiseiou_inner_diou",
        "wiseiou_focaler_diou",
        "wiseiou_shapeiou",
        "wiseiou_mpdiou",
    ]

    def test_cls_loss_matrix_integration_forward(self):
        for case in self.CLS_CASES:
            args = _default_args(iou_loss="ciou", iou_aux="none", **case)
            model = _DummyModel(args=args)
            criterion = v8DetectionLoss(model)
            preds, batch = _make_preds_and_batch()
            loss, loss_det = criterion.loss(preds, batch)
            with self.subTest(case=case):
                self.assertTrue(torch.isfinite(loss).all(), msg=f"non-finite loss for {case}")
                self.assertTrue(torch.isfinite(loss_det).all(), msg=f"non-finite detached loss for {case}")

    def test_iou_loss_matrix_unit_forward(self):
        pred_dist = torch.rand(1, 4, 4)
        pred_bboxes, target_bboxes = _boxes_xyxy(4)
        pred_bboxes = pred_bboxes.unsqueeze(0)
        target_bboxes = target_bboxes.unsqueeze(0)
        target_scores = torch.ones(1, 4, 1)
        fg_mask = torch.tensor([[True, True, True, True]])
        anchor_points = torch.zeros(4, 2)
        imgsz = torch.tensor([640.0, 640.0])
        stride = torch.ones(4, 1)

        for mode in self.IOU_CASES:
            args = _default_args(iou_loss=mode, iou_aux="none")
            loss_fn = BboxLoss(reg_max=1, args=args)
            liou, ldfl = loss_fn(
                pred_dist=pred_dist,
                pred_bboxes=pred_bboxes,
                anchor_points=anchor_points,
                target_bboxes=target_bboxes,
                target_scores=target_scores,
                target_scores_sum=torch.tensor(4.0),
                fg_mask=fg_mask,
                imgsz=imgsz,
                stride=stride,
            )
            with self.subTest(iou_loss=mode):
                self.assertTrue(torch.isfinite(liou), msg=f"non-finite liou for {mode}")
                self.assertTrue(torch.isfinite(ldfl), msg=f"non-finite ldfl for {mode}")

    def test_iou_aux_matrix_unit_forward(self):
        pred_dist = torch.rand(1, 4, 4)
        pred_bboxes, target_bboxes = _boxes_xyxy(4)
        pred_bboxes = pred_bboxes.unsqueeze(0)
        target_bboxes = target_bboxes.unsqueeze(0)
        target_scores = torch.ones(1, 4, 1)
        fg_mask = torch.tensor([[True, True, True, True]])
        anchor_points = torch.zeros(4, 2)
        imgsz = torch.tensor([640.0, 640.0])
        stride = torch.ones(4, 1)

        for aux in ("gcd", "nwd"):
            args = _default_args(iou_loss="ciou", iou_aux=aux, iou_aux_ratio=0.5)
            loss_fn = BboxLoss(reg_max=1, args=args)
            liou, ldfl = loss_fn(
                pred_dist=pred_dist,
                pred_bboxes=pred_bboxes,
                anchor_points=anchor_points,
                target_bboxes=target_bboxes,
                target_scores=target_scores,
                target_scores_sum=torch.tensor(4.0),
                fg_mask=fg_mask,
                imgsz=imgsz,
                stride=stride,
            )
            with self.subTest(iou_aux=aux):
                self.assertTrue(torch.isfinite(liou), msg=f"non-finite liou for {aux}")
                self.assertTrue(torch.isfinite(ldfl), msg=f"non-finite ldfl for {aux}")

    def test_backward_pass_no_nan(self):
        args = _default_args(cls_loss="focal", iou_loss="wiseiou_inner_diou", iou_aux="gcd")
        model = _DummyModel(args=args)
        criterion = v8DetectionLoss(model)
        preds, batch = _make_preds_and_batch()
        preds["boxes"].requires_grad_(True)
        preds["scores"].requires_grad_(True)
        loss, _ = criterion.loss(preds, batch)
        total = loss.sum()
        total.backward()
        self.assertIsNotNone(preds["boxes"].grad)
        self.assertIsNotNone(preds["scores"].grad)
        self.assertTrue(torch.isfinite(preds["boxes"].grad).all())
        self.assertTrue(torch.isfinite(preds["scores"].grad).all())


@unittest.skipUnless(
    os.getenv("RUN_LOSS_TRAIN_SMOKE") == "1", "Set RUN_LOSS_TRAIN_SMOKE=1 to enable training smoke tests"
)
class LossTrainSmokeTest(unittest.TestCase):
    """Optional micro training smoke tests for end-to-end loss wiring."""

    @staticmethod
    def _build_tiny_dataset(root: Path):
        (root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            for i in range(2):
                img = np.full((64, 64, 3), 127, dtype=np.uint8)
                cv2.rectangle(img, (16, 16), (48, 48), (255, 255, 255), -1)
                img_path = root / "images" / split / f"{i}.jpg"
                lbl_path = root / "labels" / split / f"{i}.txt"
                cv2.imwrite(str(img_path), img)
                lbl_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        yaml_text = f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: obj\n"
        yaml_path = root / "tiny.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        return yaml_path

    def test_micro_train_smoke_selected_combos(self):
        from ultralytics import YOLO

        combos = [
            dict(cls_loss="ema_slide", iou_loss="ciou", iou_aux="none"),
            dict(cls_loss="qualityfocal", iou_loss="inner_diou", iou_aux="gcd"),
            dict(cls_loss="bce", iou_loss="wiseiou_inner_diou", iou_aux="none"),
            dict(cls_loss="focal", iou_loss="wiseiou_mpdiou", iou_aux="nwd"),
        ]
        with tempfile.TemporaryDirectory(prefix="loss-smoke-") as td:
            root = Path(td)
            data_yaml = self._build_tiny_dataset(root)
            for i, combo in enumerate(combos):
                model = YOLO("yolo26n.pt")
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    imgsz=64,
                    batch=2,
                    workers=0,
                    device="0",
                    project=str(root / "runs"),
                    name=f"combo_{i}",
                    exist_ok=True,
                    val=False,
                    save=False,
                    **combo,
                )
