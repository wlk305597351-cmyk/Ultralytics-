#!/usr/bin/env python3
"""校验 5 类标签文件的正确性。.

校验项：
  - class_id ∈ [0, 4]
  - cx, cy, w, h ∈ [0, 1]
  - 每行正好 5 个字段
  - 图片-标签双向配对
  - 每类 box 数量分布
"""

import os

DATASET_DIR = "/home/wanglinkai/projects/Ultralytics_305597351/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus"}
NC = 5

errors = []
total_boxes_all = 0


def check_labels(split_name):
    global total_boxes_all
    label_dir = os.path.join(LABELS_DIR, split_name)
    if not os.path.isdir(label_dir):
        print(f"  ⛔ {label_dir} 不存在")
        return

    txt_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    class_counts = {i: 0 for i in range(NC)}
    total_boxes = 0
    empty_files = 0
    file_errors = 0

    for fname in txt_files:
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()

        if len(lines) == 0 or all(not l.strip() for l in lines):
            empty_files += 1
            continue

        for lno, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"[{split_name}] {fname}:{lno}  字段数={len(parts)} (期望 5): {line}")
                file_errors += 1
                continue

            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"[{split_name}] {fname}:{lno}  解析失败: {line}")
                file_errors += 1
                continue

            if cls_id < 0 or cls_id >= NC:
                errors.append(f"[{split_name}] {fname}:{lno}  class_id={cls_id} 超出 [0,{NC - 1}]: {line}")
                file_errors += 1

            for val_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0.0 or val > 1.0:
                    errors.append(f"[{split_name}] {fname}:{lno}  {val_name}={val} 超出 [0,1]: {line}")
                    file_errors += 1

            class_counts[cls_id] += 1
            total_boxes += 1

    total_boxes_all += total_boxes

    print(f"  [{split_name}]  文件:{len(txt_files)}  空文件:{empty_files}  box:{total_boxes}  错误:{file_errors}")
    for cls_id in range(NC):
        print(f"    {cls_id} ({CLASS_NAMES[cls_id]}): {class_counts[cls_id]}")
    return len(txt_files), empty_files


def check_pairing(split_name):
    """双向配对检查：每个图片对应 .txt，每个 .txt 对应图片。."""
    img_dir = os.path.join(IMAGES_DIR, split_name)
    label_dir = os.path.join(LABELS_DIR, split_name)

    img_names = set()
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMG_EXTS:
            img_names.add(stem)

    txt_names = set()
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            txt_names.add(os.path.splitext(fname)[0])

    img_only = img_names - txt_names
    txt_only = txt_names - img_names

    if img_only:
        print(f"  ⚠ [{split_name}] 图片无对应标签 ({len(img_only)} 个):")
        for name in sorted(img_only)[:10]:
            print(f"    {name}")
        if len(img_only) > 10:
            print(f"    ... 还有 {len(img_only) - 10} 个")
        return False

    if txt_only:
        print(f"  ⚠ [{split_name}] 标签无对应图片 ({len(txt_only)} 个):")
        for name in sorted(txt_only)[:10]:
            print(f"    {name}")
        if len(txt_only) > 10:
            print(f"    ... 还有 {len(txt_only) - 10} 个")
        return False

    print(f"  [{split_name}] 图片-标签配对: {len(img_names)} ↔ {len(txt_names)}  ✅")
    return True


def main():
    print("=" * 60)
    print("  5 类标签校验")
    print(f"  期望 class_id ∈ [0, {NC - 1}]")
    print("  期望 cx/cy/w/h ∈ [0, 1]")
    print("=" * 60)

    # ---- 标签内容校验 ----
    print("\n--- 标签内容校验 ---")
    total_empty = 0
    for split_name in SPLITS:
        result = check_labels(split_name)
        if result:
            total_empty += result[1]

    # ---- 图片-标签配对 ----
    print("\n--- 图片-标签配对 ---")
    all_paired = True
    for split_name in SPLITS:
        if not check_pairing(split_name):
            all_paired = False

    # ---- 汇总 ----
    print("\n--- 错误汇总 ---")
    if errors:
        print(f"  共 {len(errors)} 个错误:")
        for e in errors[:30]:
            print(f"    {e}")
        if len(errors) > 30:
            print(f"    ... 还有 {len(errors) - 30} 个")
    else:
        print("  0 个错误，标签格式全部正确 ✅")

    print("\n--- 最终统计 ---")
    print(f"  总 box 数: {total_boxes_all}")
    print(f"  空标签文件: {total_empty}")
    print(f"  图片-标签配对: {'全部通过 ✅' if all_paired else '有孤立文件 ⚠'}")

    if errors or not all_paired:
        print("\n⚠ 校验发现问题，请检查后再开始训练。")
        return 1
    else:
        print("\n✅ 全部校验通过，数据可以进入训练阶段。")
        return 0


if __name__ == "__main__":
    exit(main())
