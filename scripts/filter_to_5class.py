#!/usr/bin/env python3
"""筛选 COCO 80 类标签到 5 类并重映射 class_id。.

映射规则 (COCO id → 新 id):
  0 (person) → 0
  1 (bicycle) → 1
  2 (car) → 2
  3 (motorcycle) → 3
  5 (bus) → 4

其他 class_id 全部丢弃。
"""

import os

LABELS_DIR = "/home/wanglinkai/projects/Ultralytics_305597351/dataset/labels"
SPLITS = ["train", "val", "test"]

# COCO id → new id
MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
KEEP_IDS = set(MAPPING.keys())


def process_split(split_name):
    split_dir = os.path.join(LABELS_DIR, split_name)
    if not os.path.isdir(split_dir):
        print(f"  ⛔ {split_dir} 不存在，跳过")
        return

    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".txt")])
    total_files = len(files)
    total_kept = 0
    total_discarded = 0
    empty_files = 0

    for fname in files:
        fpath = os.path.join(split_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()

        kept_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue
            if cls_id in KEEP_IDS:
                new_id = MAPPING[cls_id]
                parts[0] = str(new_id)
                kept_lines.append(" ".join(parts) + "\n")
                total_kept += 1
            else:
                total_discarded += 1

        with open(fpath, "w") as f:
            f.writelines(kept_lines)

        if len(kept_lines) == 0:
            empty_files += 1

    print(f"  [{split_name}]")
    print(f"    处理文件: {total_files}")
    print(f"    保留 box: {total_kept}")
    print(f"    丢弃 box: {total_discarded}")
    print(f"    变空文件: {empty_files}")
    return total_files, total_kept, total_discarded, empty_files


def main():
    print("=== 筛选 5 类标签 ===\n")
    print(f"映射规则: {MAPPING}")
    print(f"保留 class_id: {sorted(KEEP_IDS)}\n")

    grand_files = 0
    grand_kept = 0
    grand_discarded = 0
    grand_empty = 0

    for split_name in SPLITS:
        result = process_split(split_name)
        if result:
            f, k, d, e = result
            grand_files += f
            grand_kept += k
            grand_discarded += d
            grand_empty += e

    print("\n=== 总计 ===")
    print(f"  处理文件: {grand_files}")
    print(f"  保留 box: {grand_kept}")
    print(f"  丢弃 box: {grand_discarded}")
    print(f"  变空文件: {grand_empty}")
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
