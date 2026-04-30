# Ultralytics YOLO — 雪天目标检测

基于 Ultralytics YOLO 框架的目标检测项目，研究方向为雪天天气下的目标检测。

## 环境

- **Conda 环境**: `ultralytics`
- **激活命令**: `source /home/wanglinkai/miniconda3/etc/profile.d/conda.sh && conda activate ultralytics`
- **Python**: miniconda3
- **工作目录**: `/home/wanglinkai/projects/Ultralytics_305597351`

## GPU

6 张 NVIDIA GeForce RTX 4090，运行训练/推理前用 `nvidia-smi` 查看空闲 GPU，选择显存占用最低的卡。

在 Python 脚本中通过 `device=0`（或 1-5）选择 GPU，或在命令行加 `device=0`。

## 数据集

5 类目标检测：person / bicycle / car / motorcycle / bus，train 7287 / val 1713 / test 1521，总 26440 个标注框。

数据集在 `dataset/` 目录下：

```
dataset/
  images/
    train/      # 训练集 7287 张
    val/        # 验证集 1713 张
    test/       # 测试集 1521 张
  labels/
    train/      # YOLO 格式标注 (class_id cx cy w h, 归一化)
    val/
    test/
  data.yaml     # 数据集配置 (5 类)
  classes.txt
  .archive/     # 历史备份 (VOC 20 类 / COCO 80 类原始标签)
scripts/
  filter_to_5class.py   # COCO 80 → 5 类筛选脚本
  verify_labels.py      # 标签校验脚本
```

- **train/val** 来源：VOC 合成雪天图（sSnow）
- **test** 来源：师姐自己收集的真实雪天图（rSnow，多来源混合：手机拍摄、截图、网页下载等）
- 与师姐 SnowNet 论文（Visual Computer 2026）的 srSnow 数据集一致

### 已知数据特性

- **类别极不均衡**：person 占 71.3%，bus 仅 3.5%；car 17.2%，motorcycle 4.1%，bicycle 3.9%
- **标注噪声**：test 集来源多样，存在少量标注噪声（~5%），与上游论文保持一致，不做修改
- **全图标注框**：168 个框（面积 > 0.95，cx≈0.5, cy≈0.5, w≈1.0, h≈1.0）属占图特写标注，合法
- **分布偏移**：train/val 与 test 分布有偏移（person 占比 80.6% vs 52%），test 不能完全代表训练分布
- **数据集已审计完成**（5 类版本），不要再建议改类别配置或重新标注

## 预训练权重

项目根目录下有多个 YOLO 权重文件：`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`, `yolo26n.pt`

## 常用脚本

- `train.py` — 训练脚本
- `val.py` — 验证/评估脚本
- `detect.py` — 推理/检测脚本
- `export.py` — 模型导出
- `heatmap.py` — 热力图可视化
- `track.py` — 目标跟踪

## 输出目录

训练/验证/检测的输出在 `runs/` 下，按任务类型分目录：
- `runs/detect/train/` — 训练输出
- `runs/detect/val/` — 验证输出
- `runs/detect/predict/` — 检测/推理输出

## Sub-agent 调度规则

涉及以下领域的检查/操作，必须使用对应的 sub-agent，主对话只负责研究决策、模块设计、讨论：

| 领域 | Sub-agent | 状态 |
|------|-----------|------|
| 数据集检查（任何涉及 data.yaml、labels/、images/、标注质量、类别分布的审计） | `data-inspector` | 可用 |
| 训练启动（GPU 检测 → 参数确认 → 数据审计 → 冒烟测试 → detach 启动） | `trainer` | 可用 |
| 训练进度查询（读取 results.csv、估算剩余时间、崩溃检测） | `monitor` | 可用 |
| 评估/验证（val.py、mAP 计算、结果分析、实验对比） | `evaluator` | 可用 |

## 训练自动化工作流

### 触发方式

用自然语言触发 agent，无需手动编辑脚本：

| 你想做的事 | 可以这样说 |
|-----------|-----------|
| 启动训练 | "训练 yolov8s 100 epochs" / "用 yolov8m 训练，加 EMA 模块" |
| 查询进度 | "看下 {实验名} 进度" / "现在跑得怎么样" |
| 评估模型 | "评估 {实验名}" / "val 一下 {实验名}" |
| 对比实验 | "比较一下 {实验A} 和 {实验B}" |

### 实验注册表

所有实验记录在 `experiments/registry.csv`：

```
exp_name, timestamp, git_commit, model, cfg, data, epochs, batch, imgsz, gpu,
status, best_mAP50, best_mAP50_95, log_path, wandb_url, notes
```

- `status` 取值：`started` → `completed` / `crashed` / `evaluated` / `smoke_failed`
- trainer 启动时写入 `started` 行，evaluator 完成后更新 mAP 字段
- 训练日志在 `logs/{name}.log`，PID 文件在 `logs/{name}.pid`

### train.py / val.py 参数

`python train.py --help` 和 `python val.py --help` 列出完整参数。常用：

```
python train.py --model yolov8s.pt --device 1 --epochs 100 --batch 16 --name my_exp
python train.py --model yolov8m.pt --cfg improve/attention/yolov8/yolov8-EMA-1.yaml --epochs 200
python train.py --dry-run                          # 只打印参数，不训练
python val.py --model train/my_exp/weights/best.pt --device 1 --name my_exp
```

### 多卡训练

指定 `--device 0,1` 即可启动多卡：

```
python train.py --model yolov8l.pt --device 0,1 --batch 32
```

### 已知限制

- **trainer agent 启动训练后不会主动汇报进度**。训练是 detach 模式运行的（nohup），关闭 Claude Code 会话不影响训练。需要查询进度时，主动说「看下 {实验名} 进度」触发 monitor agent
- **单卡为主，不做自动 DDP 决策**。多卡仅在用户显式指定 `--device 0,1` 时启用
- **wandb 暂未接入**。registry.csv 的 `wandb_url` 字段保留，将来接 wandb 时不需改 schema
- **agent 只做编排，不提供调参建议**。超参数选择由用户决定

## 用户偏好

- 进行训练前必须先做阶段式检查（数据配置 → smoke test → 正式训练），每阶段确认后再继续
- 遇到报错先停下来报告，不要自行修复
- 解释技术问题时用简洁直白的语言，避免过于学术化的术语堆砌
- 这是实验性质的工作，不需要过度追求工程完美
- 数据集已审计完成（5 类版本），不要建议再改类别配置
