---
description: 一键启动 YOLO 训练。用法: /train <模型> [--epochs N] [--batch N] [--module MODULE] [--device N]
model: opus
---

立即调用 trainer subagent，传入以下训练请求。

用户输入: $ARGUMENTS

## 参数解析

将 $ARGUMENTS 映射为 trainer agent 的参数确认阶段：

| 用户写法 | 映射 |
|---------|------|
| `yolov8n` / `yolov8s` / `yolov8m` / `yolov8l` / `yolov8x` | `--model {name}.pt` |
| `--epochs N` | 训练轮数 |
| `--batch N` | 批次大小 |
| `--module EMA` / `CBAM` / `SE` 等 | 注入模块（trainer 会搜索 improve/ 下列出变体） |
| `--device N` | 指定 GPU（跳过 trainer 的 GPU 检测阶段） |
| `--imgsz N` | 图像尺寸 |
| `--cos-lr` | 启用余弦退火 |
| `--no-amp` | 关闭混合精度 |
| `--name X` | 实验名 |

未指定的参数使用 trainer agent 的默认值。

## 执行方式

调用 trainer agent，用以下 prompt：

"用户通过 /train 命令请求训练。参数已解析：
{解析后的参数列表}

请从 trainer workflow 的阶段 1（参数确认）开始执行。跳过阶段 0（GPU 检测）如果用户已指定 --device。展示最终的完整参数让用户确认。"

## 示例

```
/train yolov8s                           # 训练 yolov8s 300 epochs，全部默认
/train yolov8m --epochs 100              # 100 epochs
/train yolov8l --module EMA --device 2   # yolov8l + EMA 模块，GPU 2
/train yolo26n --epochs 200 --batch 32   # yolo26n，大 batch
```
