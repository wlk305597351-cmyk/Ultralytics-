from types import SimpleNamespace

from ultralytics.afss.io import load_state
from ultralytics.afss.state import AFSSImageState
from ultralytics.models.yolo.pose.afss_train import AFSSPoseTrainer
from ultralytics.models.yolo.pose.afss_val import AFSSPoseEvaluator, aggregate_image_metrics


def make_trainer_stub(tmp_path) -> AFSSPoseTrainer:
    trainer = AFSSPoseTrainer.__new__(AFSSPoseTrainer)
    trainer.args = SimpleNamespace(
        afss=True,
        afss_warmup_epochs=20,
        afss_update_interval=5,
        afss_easy_ratio=0.02,
        afss_moderate_ratio=0.40,
        afss_easy_forced_gap=10,
        afss_moderate_forced_gap=3,
        afss_conf=0.25,
        afss_save_refresh_json=True,
        afss_thresholds={"pose": [0.55, 0.85]},
    )
    trainer.afss_task_name = "pose"
    trainer.afss_dir = tmp_path / "pose" / "train" / "afss"
    trainer.afss_state = {}
    return trainer


def test_pose_joint_score_uses_box_and_pose():
    metrics = {
        "box": {"precision": 0.93, "recall": 0.90},
        "pose": {"precision": 0.78, "recall": 0.66},
    }
    assert AFSSPoseEvaluator(save_dir=None).build_image_result(metrics)["task_score"] == 0.66


def test_pose_aggregate_image_metrics_tracks_box_and_pose():
    result = aggregate_image_metrics(
        num_gt=4,
        num_pred=5,
        matched_gt=4,
        matched_pred=4,
        matched_pose_gt=3,
        matched_pose_pred=3,
    )
    assert result["box"]["recall"] == 1.0
    assert result["box"]["precision"] == 0.8
    assert result["pose"]["recall"] == 0.75
    assert result["pose"]["precision"] == 0.6


def test_pose_evaluator_collects_joint_payload_by_image_file(tmp_path):
    evaluator = AFSSPoseEvaluator(save_dir=tmp_path)
    evaluator.store_image_metrics(
        "sample.jpg",
        aggregate_image_metrics(
            num_gt=4,
            num_pred=4,
            matched_gt=4,
            matched_pred=4,
            matched_pose_gt=3,
            matched_pose_pred=3,
        ),
    )

    assert evaluator.image_results["sample.jpg"]["metrics"]["pose"]["recall"] == 0.75
    assert evaluator.image_results["sample.jpg"]["task_score"] == 0.75


def test_pose_trainer_uses_joint_thresholds_for_state_and_snapshot(tmp_path):
    trainer = make_trainer_stub(tmp_path)
    trainer.afss_state = {
        "easy.jpg": AFSSImageState(im_file="easy.jpg", task_score=0.0, level="hard"),
        "hard.jpg": AFSSImageState(im_file="hard.jpg", task_score=0.0, level="easy"),
    }

    trainer._update_afss_state_from_results(
        {
            "easy.jpg": {
                "metrics": {
                    "box": {"precision": 0.95, "recall": 0.94},
                    "pose": {"precision": 0.92, "recall": 0.90},
                },
                "task_score": 0.90,
            },
            "hard.jpg": {
                "metrics": {
                    "box": {"precision": 0.60, "recall": 0.58},
                    "pose": {"precision": 0.30, "recall": 0.20},
                },
                "task_score": 0.20,
            },
        },
        epoch=5,
    )
    snapshot_path = trainer._write_afss_refresh_snapshot(epoch=5, active_images=["hard.jpg"])

    assert trainer.afss_state["easy.jpg"].level == "easy"
    assert trainer.afss_state["hard.jpg"].level == "hard"
    payload = load_state(snapshot_path)
    assert payload["task"] == "pose"
    assert payload["thresholds"] == {"moderate": 0.55, "easy": 0.85}
