import io
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.extra_modules.conv_module.wtconv2d import WTConv2d
from ultralytics.nn.extra_modules.module.PartialNetBlock import METHOD, iRPE
from ultralytics.utils.loss import (
    BboxLoss,
    EMASlideLoss,
    FocalLoss_YOLO,
    QualityfocalLoss_YOLO,
    SlideLoss,
    VarifocalLoss_YOLO,
    _validate_detection_loss_args,
    v8DetectionLoss,
)
from ultralytics.utils.metrics import (
    bbox_focaler_iou,
    bbox_focaler_mpdiou,
    bbox_inner_iou,
    bbox_inner_mpdiou,
    bbox_iou,
    bbox_mpdiou,
)


def _boxes_xyxy(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.rand(n, 2)
    b = torch.rand(n, 2)
    p1 = torch.minimum(a, b)
    p2 = torch.maximum(a, b) + 0.2
    box1 = torch.cat((p1, p2), dim=1)
    box2 = box1 + 0.05
    return box1, box2


class _DummyDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = torch.tensor([8.0])
        self.nc = 2
        self.reg_max = 1


class _DummyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = nn.ModuleList([nn.Identity(), _DummyDetect()])


class LossMigrationTest(unittest.TestCase):
    def test_extended_iou_functions_are_available_and_finite(self):
        box1, box2 = _boxes_xyxy(8)
        mp_hw = torch.full((8, 1), 640.0**2 + 640.0**2)

        outs = [
            bbox_iou(box1, box2, xywh=False, EIoU=True),
            bbox_iou(box1, box2, xywh=False, SIoU=True),
            bbox_iou(box1, box2, xywh=False, ShapeIoU=True),
            bbox_iou(box1, box2, xywh=False, PIoU=True),
            bbox_iou(box1, box2, xywh=False, PIoU2=True),
            bbox_inner_iou(box1, box2, xywh=False, CIoU=True),
            bbox_focaler_iou(box1, box2, xywh=False, CIoU=True),
            bbox_mpdiou(box1, box2, xywh=False, mpdiou_hw=mp_hw),
            bbox_inner_mpdiou(box1, box2, xywh=False, mpdiou_hw=mp_hw),
            bbox_focaler_mpdiou(box1, box2, xywh=False, mpdiou_hw=mp_hw),
        ]
        for out in outs:
            self.assertTrue(torch.isfinite(out).all())

    def test_custom_cls_loss_wrappers_forward(self):
        pred = torch.randn(2, 5, 3)
        target = torch.rand(2, 5, 3)
        pos_mask = target > 0.5

        base = nn.BCEWithLogitsLoss(reduction="none")
        slide = SlideLoss(base)
        ema_slide = EMASlideLoss(nn.BCEWithLogitsLoss(reduction="none"))
        focal = FocalLoss_YOLO()
        vfl = VarifocalLoss_YOLO()
        qfl = QualityfocalLoss_YOLO()

        vals = [
            slide(pred, target),
            ema_slide(pred, target),
            focal(pred, target),
            vfl(pred, target),
            qfl(pred, target, pos_mask),
        ]
        for v in vals:
            self.assertTrue(torch.isfinite(v).all())

    def test_ema_slide_loss_is_serializable(self):
        loss_fn = EMASlideLoss(nn.BCEWithLogitsLoss(reduction="none"))
        # torch.save is the exact path used by trainer checkpoint serialization.
        buff = io.BytesIO()
        torch.save(loss_fn, buff)
        self.assertGreater(len(buff.getvalue()), 0)

    def test_wtconv2d_is_serializable(self):
        module = WTConv2d(16, 16)
        buff = io.BytesIO()
        torch.save(module, buff)
        self.assertGreater(len(buff.getvalue()), 0)

    def test_irpe_default_initializer_path_is_serializable(self):
        module = iRPE(head_dim=8, num_heads=2, mode="bias", method=METHOD.PRODUCT, transposed=True, num_buckets=16)
        buff = io.BytesIO()
        torch.save(module, buff)
        self.assertGreater(len(buff.getvalue()), 0)

    def test_detection_loss_logs_selected_losses(self):
        args = types.SimpleNamespace(
            cls_loss="ema_slide",
            iou_loss="wiseiou_focaler_diou",
            iou_aux="gcd",
            iou_aux_ratio=0.5,
            focaler_d=0.0,
            focaler_u=0.95,
            wiseiou_monotonous=True,
            use_wiseiou=False,
        )
        model = _DummyModel(args)
        with patch("ultralytics.utils.loss.RANK", -1), patch("ultralytics.utils.loss.LOGGER.info") as mock_info:
            _ = v8DetectionLoss(model)
        self.assertTrue(mock_info.called)
        message = mock_info.call_args[0][0]
        self.assertIn("LossConfig", message)
        self.assertIn("cls_loss=ema_slide", message)
        self.assertIn("iou_loss=wiseiou_focaler_diou", message)
        self.assertIn("iou_aux=gcd", message)
        self.assertIn("focaler_d=", message)
        self.assertIn("focaler_u=", message)
        self.assertIn("wiseiou_monotonous=True", message)

    def test_detection_loss_logging_is_rank_gated(self):
        args = types.SimpleNamespace(cls_loss="bce", iou_loss="ciou", iou_aux="none")
        model = _DummyModel(args)
        with patch("ultralytics.utils.loss.RANK", 1), patch("ultralytics.utils.loss.LOGGER.info") as mock_info:
            _ = v8DetectionLoss(model)
        mock_info.assert_not_called()

    def test_custom_loss_config_keys_are_accepted(self):
        cfg = get_cfg(overrides={"cls_loss": "bce", "iou_loss": "ciou", "iou_aux": "none"})
        self.assertEqual(cfg.cls_loss, "bce")
        self.assertEqual(cfg.iou_loss, "ciou")
        self.assertEqual(cfg.iou_aux, "none")

    def test_validate_detection_loss_args_rejects_unknown_values(self):
        args = types.SimpleNamespace(cls_loss="unknown", iou_loss="ciou", iou_aux="none")
        with self.assertRaises(ValueError):
            _validate_detection_loss_args(args)

    def test_validate_detection_loss_args_accepts_composed_iou_modes(self):
        args = types.SimpleNamespace(cls_loss="bce", iou_loss="inner_diou", iou_aux="none")
        _validate_detection_loss_args(args)
        args = types.SimpleNamespace(cls_loss="bce", iou_loss="focaler_siou", iou_aux="none")
        _validate_detection_loss_args(args)
        args = types.SimpleNamespace(cls_loss="bce", iou_loss="wiseiou_inner_diou", iou_aux="none")
        _validate_detection_loss_args(args)

    def test_bbox_loss_with_custom_modes_forward(self):
        pred_dist = torch.rand(1, 4, 4)
        pred_bboxes, target_bboxes = _boxes_xyxy(4)
        pred_bboxes = pred_bboxes.unsqueeze(0)
        target_bboxes = target_bboxes.unsqueeze(0)
        target_scores = torch.ones(1, 4, 1)
        fg_mask = torch.tensor([[True, True, True, True]])
        anchor_points = torch.zeros(4, 2)
        imgsz = torch.tensor([640.0, 640.0])
        stride = torch.ones(4, 1)

        for mode in (
            "mpdiou",
            "inner_diou",
            "focaler_diou",
            "wiseiou_diou",
            "wiseiou_inner_diou",
            "wiseiou_focaler_diou",
        ):
            args = types.SimpleNamespace(
                iou_loss=mode,
                iou_aux="gcd",
                iou_aux_ratio=0.5,
                inner_iou_ratio=0.7,
                focaler_d=0.0,
                focaler_u=0.95,
                shapeiou_scale=0.0,
                piou_lambda=1.3,
                use_wiseiou=False,
                wiseiou_monotonous=False,
            )
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
            self.assertTrue(torch.isfinite(liou), msg=f"non-finite liou for {mode}")
            self.assertTrue(torch.isfinite(ldfl), msg=f"non-finite ldfl for {mode}")
