import argparse
import csv
import os
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

# 默认使用第 0 张显卡，运行时可通过 --device 覆盖
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # 指定使用第三、四张显卡进行多卡训练
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 全流程实战教程：从零开始完成环境配置、数据集解析、模型训练到测试验证 https://www.bilibili.com/video/BV1tUFkzSEn7/


def _print_weight_loading_report(model, weight_path):
    """打印 --cfg + .pt 组合的权重加载/跳过表格。."""
    import torch

    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt.get("model", ckpt)
    if hasattr(ckpt_state, "state_dict"):
        ckpt_state = ckpt_state.state_dict()
    if hasattr(ckpt_state, "float"):
        ckpt_state = ckpt_state.float()

    model_state = model.model.state_dict()
    loaded, skipped = 0, 0
    rows = []
    for k, v in model_state.items():
        if k in ckpt_state:
            if v.shape == ckpt_state[k].shape:
                loaded += 1
            else:
                skipped += 1
                rows.append((k, str(list(v.shape)), f"shape mismatch: ckpt {list(ckpt_state[k].shape)}"))
        else:
            skipped += 1
            rows.append((k, str(list(v.shape)), "not in checkpoint"))

    print(f"\n{'=' * 70}")
    print(f"权重加载报告: {weight_path}")
    print(f"匹配加载: {loaded} 层 | 跳过: {skipped} 层")
    if rows:
        print(f"{'Layer':<45} {'Model Shape':<20} Skip Reason")
        print("-" * 90)
        for name, shape, reason in rows:
            print(f"{name:<45} {shape:<20} {reason}")
    print(f"{'=' * 70}\n")


def _write_registry(name, model_path, cfg, data, epochs, batch, imgsz, device):
    """追加一行 started 记录到 experiments/registry.csv。."""
    registry_dir = Path(__file__).parent / "experiments"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "registry.csv"

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        git_commit = "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = [
        "exp_name",
        "timestamp",
        "git_commit",
        "model",
        "cfg",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "gpu",
        "status",
        "best_mAP50",
        "best_mAP50_95",
        "log_path",
        "wandb_url",
        "notes",
    ]

    file_exists = registry_path.exists()
    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "exp_name": name,
                "timestamp": timestamp,
                "git_commit": git_commit,
                "model": model_path,
                "cfg": cfg or "",
                "data": data,
                "epochs": str(epochs),
                "batch": str(batch),
                "imgsz": str(imgsz),
                "gpu": device,
                "status": "started",
                "best_mAP50": "",
                "best_mAP50_95": "",
                "log_path": f"logs/{name}.log",
                "wandb_url": "",
                "notes": "",
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Snow 训练")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/wanglinkai/projects/Ultralytics_305597351/yolov8m.pt",
        help="权重 (.pt) 或 YAML 路径",
    )
    parser.add_argument("--cfg", type=str, default=None, help="模型架构 YAML（improve 变体用，可选）")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="数据集配置文件")
    parser.add_argument("--device", type=str, default="0", help='GPU ID，支持 "0,1"')
    parser.add_argument("--epochs", type=int, default=300, help="训练总轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--optimizer", type=str, default=None, help="优化器（默认: yolo26 用 MuSGD, 其他用 SGD）")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--name", type=str, default="exp", help="实验名")
    parser.add_argument("--project", type=str, default="train", help="输出目录")
    parser.add_argument("--cos-lr", action="store_true", default=False, help="使用余弦退火学习率")
    parser.add_argument("--resume", action="store_true", default=False, help="断点续训")
    parser.add_argument("--amp", action="store_true", default=True, help="启用自动混合精度 (默认开)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="关闭自动混合精度")
    parser.add_argument("--cls-loss", type=str, default="bce", help="分类损失类型")
    parser.add_argument("--iou-loss", type=str, default="ciou", help="IoU 损失类型")
    parser.add_argument("--cache", type=str, default="False", choices=["False", "True", "disk"], help="缓存模式")
    parser.add_argument("--close-mosaic", type=int, default=0, help="最后 N 个 epoch 关闭 Mosaic 增强")
    parser.add_argument("--dry-run", action="store_true", default=False, help="打印解析后的参数和等效命令，不真正训练")
    parser.add_argument("--afss", action="store_true", default=False, help="开启 AFSS（将来用）")

    args = parser.parse_args()

    # --- GPU 控制 ---
    # 解析 --device 后写入环境变量，model.train(device='0')
    # PyTorch 视角下始终只有 cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # --- 确定 architecture YAML 路径（用于 optimizer 默认选择） ---
    yaml_path = args.cfg if args.cfg else args.model

    # --- dry-run：打印参数不训练 ---
    if args.dry_run:
        print("[dry-run] 解析后的参数:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print("[dry-run] 未启动训练。")
        sys.exit(0)

    # --- 模型初始化 ---
    # 初始化 YOLO 模型，加载 COCO 预训练权重进行 fine-tune
    if args.cfg and args.model.endswith(".pt"):
        # --cfg 架构 + --model 预训练权重
        model = YOLO(args.cfg)
        _print_weight_loading_report(model, args.model)
        model.load(args.model)
    elif args.model.endswith(".pt"):
        model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # --- 优化器默认值 ---
    if args.optimizer is None:
        args.optimizer = "MuSGD" if "yolo26" in yaml_path else "SGD"

    # --- registry 写入 ---
    try:
        _write_registry(args.name, args.model, args.cfg, args.data, args.epochs, args.batch, args.imgsz, args.device)
    except Exception:
        pass  # registry 写入失败不阻塞训练

    # --- 正式训练 ---
    model.train(
        data=args.data,
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        device="0",
        optimizer=args.optimizer,
        patience=args.patience,
        resume=args.resume,
        amp=args.amp,
        cos_lr=args.cos_lr,
        save_period=-1,
        project=args.project,
        name=args.name,
        # trainer=AFSSDetectionTrainer,
        # afss=True, # 开启 AFSS
        # afss_save_refresh_json=False,
        # afss_warmup_epochs=20, # 前 20 个 epoch 使用全量训练集 warmup
        # afss_update_interval=5, # 每隔 5 个 epoch 刷新一次图像难度状态
        # afss_easy_ratio=0.02, # easy 样本每轮保留 2%
        # afss_moderate_ratio=0.40, # moderate 样本每轮保留 40%
        # afss_easy_forced_gap=10, # easy 样本超过 10 个 epoch 未使用则强制回看
        # afss_moderate_forced_gap=3, # moderate 样本超过 3 个 epoch 未使用则强制覆盖
        # afss_thresholds={
        #     "detect": [0.55, 0.85],
        #     "obb": [0.55, 0.85],
        #     "segment": [0.55, 0.85],
        #     "pose": [0.55, 0.85],
        # },
        # -------------------- LOSS部分(更多解释可以看LOSS-UserGuide.md) --------------------
        cls_loss=args.cls_loss,
        iou_loss=args.iou_loss,
        iou_aux="none",
        iou_aux_ratio=0.5,
    )
