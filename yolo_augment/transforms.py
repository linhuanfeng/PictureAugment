"""图像增强变换模块"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .annotation import BBox

_rng = np.random.default_rng()  # 噪声用 Generator：直接生成 float32，比 np.random.normal 快


def seed_random(seed_seq: Optional[np.random.SeedSequence] = None) -> None:
    """给 numpy 全局随机数和模块里的 Generator 重新播种。
    多进程时每个子进程都要调用：fork 出来的子进程会继承同一个随机状态，不播种所有进程会生成一样的增强。"""
    global _rng
    seed_seq = seed_seq or np.random.SeedSequence()  # 缺省取系统熵，每个进程都不同
    np.random.seed(int(seed_seq.generate_state(1)[0]))
    _rng = np.random.default_rng(seed_seq)


class BaseTransform:
    """变换基类"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def should_apply(self) -> bool:
        return np.random.random() < self.p

    def __call__(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        if self.should_apply():
            return self.apply(image, bboxes)
        return image, bboxes

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        raise NotImplementedError


class Rotate(BaseTransform):
    """旋转变换

    常规增强：以概率 p 在 angle_range 内随机取一个角度。
    必转角度：fixed_angles 里的每个角度，每张图都会各生成一张精确旋转的图，
    再各生成 fixed_jitter_copies 张在该角度 ±fixed_jitter 度内随机偏移的图（见 fixed_variants）。
    """
    def __init__(self, angle_range: Tuple[float, float] = (-30, 30), p: float = 0.5,
                 fixed_angles: Tuple[float, ...] = (), fixed_jitter: float = 0,
                 fixed_jitter_copies: int = 0):
        super().__init__(p)
        self.angle_range = angle_range
        self.fixed_angles = tuple(fixed_angles)
        self.fixed_jitter = fixed_jitter
        self.fixed_jitter_copies = fixed_jitter_copies

    @property
    def fixed_count(self) -> int:
        """每张图额外生成的必转图数量"""
        return len(self.fixed_angles) * (1 + self.fixed_jitter_copies)

    def fixed_variants(self) -> List[Tuple[str, float]]:
        """返回 [(文件名标记, 角度)]：每个必转角度一张精确的，加若干张附近随机偏移的"""
        variants = []
        for base in self.fixed_angles:
            tag = f"{base:g}".replace('-', 'n')
            variants.append((f"rot{tag}", float(base)))
            for j in range(self.fixed_jitter_copies):
                offset = np.random.uniform(-self.fixed_jitter, self.fixed_jitter)
                variants.append((f"rot{tag}_j{j}", float(base) + offset))
        return variants

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        return self.rotate_by(image, bboxes, np.random.uniform(*self.angle_range))

    def rotate_by(self, image: np.ndarray, bboxes: List[BBox],
                  angle: float) -> Tuple[np.ndarray, List[BBox]]:
        """按指定角度旋转（逆时针为正），画布放大以容纳整张图"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)

        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(114, 114, 114))
        new_bboxes = self._rotate_bboxes(bboxes, M, w, h, new_w, new_h)
        return rotated, new_bboxes

    def _rotate_bboxes(self, bboxes: List[BBox], M: np.ndarray,
                       old_w: int, old_h: int, new_w: int, new_h: int) -> List[BBox]:
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.to_corners(old_w, old_h)
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])
            rotated_corners = (M @ corners_h.T).T

            rx1 = max(0, rotated_corners[:, 0].min())
            ry1 = max(0, rotated_corners[:, 1].min())
            rx2 = min(new_w, rotated_corners[:, 0].max())
            ry2 = min(new_h, rotated_corners[:, 1].max())

            if rx2 > rx1 and ry2 > ry1:
                new_bbox = BBox.from_corners(bbox.class_id, int(rx1), int(ry1),
                                             int(rx2), int(ry2), new_w, new_h)
                new_bboxes.append(new_bbox)
        return new_bboxes


class RandomMask(BaseTransform):
    """随机遮挡变换"""
    def __init__(self, num_masks: Tuple[int, int] = (1, 5),
                 mask_size_range: Tuple[float, float] = (0.02, 0.1),
                 p: float = 0.5):
        super().__init__(p)
        self.num_masks = num_masks
        self.mask_size_range = mask_size_range

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        h, w = image.shape[:2]
        result = image.copy()
        num = np.random.randint(*self.num_masks)

        for _ in range(num):
            mw = int(w * np.random.uniform(*self.mask_size_range))
            mh = int(h * np.random.uniform(*self.mask_size_range))
            x = np.random.randint(0, max(1, w - mw))
            y = np.random.randint(0, max(1, h - mh))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(result, (x, y), (x + mw, y + mh), color, -1)

        return result, bboxes


class Distort(BaseTransform):
    """透视畸变变换"""
    def __init__(self, distort_range: float = 0.1, p: float = 0.5):
        super().__init__(p)
        self.distort_range = distort_range

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        h, w = image.shape[:2]
        d = self.distort_range

        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [np.random.uniform(0, w * d), np.random.uniform(0, h * d)],
            [w - np.random.uniform(0, w * d), np.random.uniform(0, h * d)],
            [w - np.random.uniform(0, w * d), h - np.random.uniform(0, h * d)],
            [np.random.uniform(0, w * d), h - np.random.uniform(0, h * d)]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (w, h), borderValue=(114, 114, 114))
        new_bboxes = self._transform_bboxes(bboxes, M, w, h)
        return warped, new_bboxes

    def _transform_bboxes(self, bboxes: List[BBox], M: np.ndarray,
                          w: int, h: int) -> List[BBox]:
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.to_corners(w, h)
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            corners = corners.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

            tx1 = max(0, int(transformed[:, 0].min()))
            ty1 = max(0, int(transformed[:, 1].min()))
            tx2 = min(w, int(transformed[:, 0].max()))
            ty2 = min(h, int(transformed[:, 1].max()))

            if tx2 > tx1 and ty2 > ty1:
                new_bbox = BBox.from_corners(bbox.class_id, tx1, ty1, tx2, ty2, w, h)
                new_bboxes.append(new_bbox)
        return new_bboxes


class GaussianNoise(BaseTransform):
    """高斯噪声变换"""
    def __init__(self, mean: float = 0, std_range: Tuple[float, float] = (10, 50), p: float = 0.5):
        super().__init__(p)
        self.mean = mean
        self.std_range = std_range

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        std = np.random.uniform(*self.std_range)
        noise = _rng.standard_normal(image.shape, dtype=np.float32)
        noise *= std
        noise += self.mean
        noise += image  # uint8 原地加到 float32 上，少一次整图拷贝
        np.clip(noise, 0, 255, out=noise)
        return noise.astype(np.uint8), bboxes


class HSVAdjust(BaseTransform):
    """HSV色调变换"""
    def __init__(self,
                 hue_range: Tuple[float, float] = (-0.1, 0.1),
                 saturation_range: Tuple[float, float] = (0.7, 1.3),
                 value_range: Tuple[float, float] = (0.7, 1.3),
                 p: float = 0.5):
        super().__init__(p)
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        # 全程 uint8，每个通道的调整做成 256 项的查找表，比转 float32 逐像素算快
        h, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        levels = np.arange(256, dtype=np.float32)

        # 调整色调 (H通道范围是0-179)
        hue_shift = np.random.uniform(*self.hue_range)
        h = cv2.LUT(h, ((levels + hue_shift * 179) % 180).astype(np.uint8))

        # 调整饱和度
        sat_scale = np.random.uniform(*self.saturation_range)
        sat = cv2.LUT(sat, np.clip(levels * sat_scale, 0, 255).astype(np.uint8))

        # 调整明度
        val_scale = np.random.uniform(*self.value_range)
        val = cv2.LUT(val, np.clip(levels * val_scale, 0, 255).astype(np.uint8))

        result = cv2.cvtColor(cv2.merge([h, sat, val]), cv2.COLOR_HSV2BGR)
        return result, bboxes


class Grayscale(BaseTransform):
    """灰度化（黑白）变换"""
    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 转回3通道保持格式一致
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result, bboxes


class BrightnessBoost(BaseTransform):
    """强光（亮度增强）变换"""
    def __init__(self, brightness_range: Tuple[float, float] = (1.2, 1.8), p: float = 0.5):
        super().__init__(p)
        self.brightness_range = brightness_range

    def apply(self, image: np.ndarray, bboxes: List[BBox]) -> Tuple[np.ndarray, List[BBox]]:
        brightness_factor = np.random.uniform(*self.brightness_range)
        return cv2.convertScaleAbs(image, alpha=brightness_factor), bboxes  # 内部饱和到 0-255


class Compose:
    """组合多个变换"""
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    @property
    def rotate(self) -> Optional[Rotate]:
        """配置里启用的旋转变换，没有则为 None"""
        return next((t for t in self.transforms if isinstance(t, Rotate)), None)

    def __call__(self, image: np.ndarray, bboxes: List[BBox],
                 forced_angle: Optional[float] = None) -> Tuple[np.ndarray, List[BBox]]:
        """forced_angle 不为空时，旋转这一步固定用该角度（必转），其余变换仍按各自概率应用"""
        for t in self.transforms:
            if forced_angle is not None and isinstance(t, Rotate):
                image, bboxes = t.rotate_by(image, bboxes, forced_angle)
            else:
                image, bboxes = t(image, bboxes)
        return image, bboxes


def build_transforms(config: dict) -> Compose:
    """从配置构建变换"""
    transforms = []

    if config.get('rotate', {}).get('enabled', False):
        cfg = config['rotate']
        transforms.append(Rotate(
            angle_range=tuple(cfg.get('angle_range', [-30, 30])),
            p=cfg.get('p', 0.5),
            fixed_angles=tuple(cfg.get('fixed_angles', [])),
            fixed_jitter=cfg.get('fixed_jitter', 0),
            fixed_jitter_copies=cfg.get('fixed_jitter_copies', 0)
        ))

    if config.get('mask', {}).get('enabled', False):
        cfg = config['mask']
        transforms.append(RandomMask(
            num_masks=tuple(cfg.get('num_masks', [1, 5])),
            mask_size_range=tuple(cfg.get('size_range', [0.02, 0.1])),
            p=cfg.get('p', 0.5)
        ))

    if config.get('distort', {}).get('enabled', False):
        cfg = config['distort']
        transforms.append(Distort(
            distort_range=cfg.get('range', 0.1),
            p=cfg.get('p', 0.5)
        ))

    if config.get('gaussian_noise', {}).get('enabled', False):
        cfg = config['gaussian_noise']
        transforms.append(GaussianNoise(
            mean=cfg.get('mean', 0),
            std_range=tuple(cfg.get('std_range', [10, 50])),
            p=cfg.get('p', 0.5)
        ))

    if config.get('hsv_adjust', {}).get('enabled', False):
        cfg = config['hsv_adjust']
        transforms.append(HSVAdjust(
            hue_range=tuple(cfg.get('hue_range', [-0.1, 0.1])),
            saturation_range=tuple(cfg.get('saturation_range', [0.7, 1.3])),
            value_range=tuple(cfg.get('value_range', [0.7, 1.3])),
            p=cfg.get('p', 0.5)
        ))

    if config.get('grayscale', {}).get('enabled', False):
        cfg = config['grayscale']
        transforms.append(Grayscale(
            p=cfg.get('p', 0.5)
        ))

    if config.get('brightness_boost', {}).get('enabled', False):
        cfg = config['brightness_boost']
        transforms.append(BrightnessBoost(
            brightness_range=tuple(cfg.get('brightness_range', [1.2, 1.8])),
            p=cfg.get('p', 0.5)
        ))

    return Compose(transforms)
