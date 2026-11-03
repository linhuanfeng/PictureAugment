"""把数据集里增强出来的文件剔除，只把原始图片和标签复制到新目录（不修改、不删除源目录）。

增强文件按文件名识别（与 yolo_augment 的输出命名一致）：
    xxx_aug3.jpg                 常规增强
    xxx_rot90.jpg / xxx_rotn30.jpg / xxx_rot90_j0.jpg   必转角度（rotate.fixed_angles）
其余文件视为原始文件。不用 augmented_merge_manifest.json：它不含后续批次的增强文件，
还记录了已被删掉的 val 文件。

用法：
    python util/restore_origin.py                      # 复制
    python util/restore_origin.py --dry-run            # 只统计，不写任何文件
    python util/restore_origin.py --link               # 用硬链接代替复制（不占额外空间，两边是同一份数据）
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
DATASET = Path("/media/xiaozhang/7b92dbd7-9b4b-479a-82aa-eea1734006a21/other_workspace/dataset")
DEFAULT_SRC = DATASET / "datasetsv3"
DEFAULT_DST = DATASET / "datasetsv3_origin"

AUG_STEM = re.compile(r"_(?:aug\d+|rotn?[\d.]+(?:_j\d+)?)$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def is_augmented(path: Path) -> bool:
    return bool(AUG_STEM.search(path.stem))


def plan(src: Path):
    """扫描源目录，返回 (要复制的 [(源, 相对路径)], 统计, 问题列表)。"""
    keep, stats, problems = [], {}, []
    for split in SPLITS:
        img_dir, lbl_dir = src / "images" / split, src / "labels" / split
        if not img_dir.is_dir():
            continue
        stats[split] = {"原图": 0, "增强(跳过)": 0, "原图无标签": 0}
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue  # 例如 train.cache
            if is_augmented(img):
                stats[split]["增强(跳过)"] += 1
                continue
            label = lbl_dir / f"{img.stem}.txt"
            keep.append((img, img.relative_to(src)))
            if label.exists():
                keep.append((label, label.relative_to(src)))
            else:
                problems.append(f"原图没有标签: {img.relative_to(src)}")
                stats[split]["原图无标签"] += 1
            stats[split]["原图"] += 1
    return keep, stats, problems


def transfer(src_file: Path, dst_file: Path, link: bool, overwrite: bool) -> str:
    """返回 'copied' / 'skipped'（已存在且大小一致） / 'conflict'（已存在但大小不同）。"""
    if dst_file.exists():
        if dst_file.stat().st_size == src_file.stat().st_size:
            return "skipped"
        if not overwrite:
            return "conflict"
        dst_file.unlink()
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(src_file, dst_file)
    else:
        shutil.copy2(src_file, dst_file)
    return "copied"


def write_data_yaml(src: Path, dst: Path) -> None:
    """复制 data.yaml，只改 path 一行，其余原样保留。"""
    text = (src / "data.yaml").read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^path:.*$", f"path: {dst}", text, count=1)
    if n == 0:
        text = f"path: {dst}\n" + text
    (dst / "data.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="剔除增强文件，把原始数据恢复到新目录")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"源数据集 (默认 {DEFAULT_SRC})")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"输出目录 (默认 {DEFAULT_DST})")
    ap.add_argument("--link", action="store_true", help="用硬链接代替复制（需同一文件系统；两边是同一份数据）")
    ap.add_argument("--overwrite", action="store_true", help="目标里已有同名但大小不同的文件时覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        print(f"源目录缺少 images/ 或 labels/: {src}")
        return 1
    if dst == src or src in dst.parents or dst in src.parents:
        print("源目录和输出目录不能相同或互相包含")
        return 1

    keep, stats, problems = plan(src)
    for split, s in stats.items():
        print(f"[{split}] 原图 {s['原图']}，跳过增强 {s['增强(跳过)']}，原图无标签 {s['原图无标签']}")
    print(f"将写入 {len(keep)} 个文件 -> {dst}（{'硬链接' if args.link else '复制'}）")
    for p in problems[:10]:
        print("警告:", p)
    if args.dry_run:
        print("dry-run：未写入任何文件")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **_: it  # noqa: E731

    counts = {"copied": 0, "skipped": 0, "conflict": 0}
    conflicts = []
    for src_file, rel in tqdm(keep, desc="restore"):
        result = transfer(src_file, dst / rel, args.link, args.overwrite)
        counts[result] += 1
        if result == "conflict":
            conflicts.append(str(rel))
    write_data_yaml(src, dst)

    report = {"src": str(src), "dst": str(dst), "mode": "link" if args.link else "copy",
              "splits": stats, "files": counts, "conflicts": conflicts, "warnings": problems}
    (dst / "restore_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：新复制 {counts['copied']}，已存在跳过 {counts['skipped']}，冲突 {counts['conflict']}")
    if conflicts:
        print(f"有 {len(conflicts)} 个文件已存在但大小不同，未覆盖（加 --overwrite 覆盖），例如: {conflicts[0]}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
