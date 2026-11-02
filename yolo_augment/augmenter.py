"""数据增强器模块"""

import multiprocessing as mp
import os
import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm

from .annotation import parse_yolo_label, save_yolo_label
from .transforms import build_transforms, seed_random

_worker = None  # 子进程里的 YOLOAugmenter，由 _init_worker 创建


def _init_worker(config: dict) -> None:
    # 每个进程只用 1 个 OpenCV 线程，否则 N 个进程 × N 个线程会互相抢 CPU
    cv2.setNumThreads(1)
    seed_random()  # fork 出来的子进程继承同一个随机状态，必须重新播种
    global _worker
    _worker = YOLOAugmenter(config)


def _run_job(job) -> int:
    img_path, labels_dir, out_images_dir, out_labels_dir = job
    return _worker._process_image(img_path, labels_dir, out_images_dir, out_labels_dir)


class YOLOAugmenter:
    """YOLO数据增强器"""

    def __init__(self, config: dict):
        self.config = config
        self.input_images_root = Path(config['input_images_root'])
        self.input_labels_root = Path(config['input_labels_root'])
        self.output_images_root = Path(config['output_images_root'])
        self.output_labels_root = Path(config['output_labels_root'])
        self.datasets = config.get('datasets', ['train'])
        self.num_augments = config.get('num_augments', 1)
        self.transforms = build_transforms(config.get('transforms', {}))
        rotate = self.transforms.rotate
        self.fixed_count = rotate.fixed_count if rotate else 0  # 每张图额外的必转角度输出数
        self.per_image = self.num_augments + self.fixed_count
        # 进程数：缺省用 CPU 核数的 3/4（留一些给磁盘读写和系统），1 表示不开子进程
        self.workers = config.get('workers') or max(1, (os.cpu_count() or 1) * 3 // 4)

    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """获取目录下所有图像文件"""
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(images_dir.glob(ext))
        return sorted(files)

    def run(self) -> None:
        """执行增强"""
        for dataset in self.datasets:
            self._process_dataset(dataset)

    def _process_dataset(self, dataset: str) -> None:
        """处理单个数据集"""
        images_dir = self.input_images_root / dataset
        labels_dir = self.input_labels_root / dataset
        out_images_dir = self.output_images_root / dataset
        out_labels_dir = self.output_labels_root / dataset

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        images = self._get_image_files(images_dir)
        if not images:
            print(f"[{dataset}] 未找到图像")
            return

        print(f"[{dataset}] 找到 {len(images)} 张图像")

        total = len(images) * self.per_image
        jobs = [(p, labels_dir, out_images_dir, out_labels_dir) for p in images]
        workers = min(self.workers, len(images))
        with tqdm(total=total, desc=f"[{dataset}]") as pbar:
            if workers <= 1:
                for job in jobs:
                    pbar.update(self._process_image(*job))
            else:
                # 一张图作为一个任务；imap_unordered 谁先做完先返回，进度条按输出张数推进
                with mp.get_context("fork").Pool(workers, _init_worker, (self.config,)) as pool:
                    for done in pool.imap_unordered(_run_job, jobs):
                        pbar.update(done)

    def _process_image(self, img_path: Path, labels_dir: Path,
                       out_images_dir: Path, out_labels_dir: Path) -> int:
        """处理单张图像，返回处理（含跳过）的输出张数，用于推进进度条"""
        label_path = labels_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            return self.per_image

        bboxes = parse_yolo_label(label_path)

        jobs = [(f"_aug{i}", None) for i in range(self.num_augments)]
        if self.fixed_count:
            # 必转角度：每张图都生成，文件名带角度标记（负号写作 n），例如 xxx_rot90.jpg、xxx_rot270_j0.jpg
            jobs += [(f"_{tag}", angle) for tag, angle in self.transforms.rotate.fixed_variants()]

        for suffix, angle in jobs:
            aug_img, aug_bboxes = self.transforms(image.copy(), bboxes.copy(), forced_angle=angle)
            out_img = out_images_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
            out_label = out_labels_dir / f"{img_path.stem}{suffix}.txt"

            cv2.imwrite(str(out_img), aug_img)
            save_yolo_label(aug_bboxes, out_label)
        return len(jobs)
