"""配置文件解析模块"""

from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件"""
    required = ['input_images_root', 'input_labels_root',
                'output_images_root', 'output_labels_root']
    for key in required:
        if key not in config:
            raise ValueError(f"配置缺少必需字段: {key}")

    workers = config.get('workers')
    if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 1):
        raise ValueError("workers 必须是不小于 1 的整数（不填则自动取 CPU 核数的 3/4）")

    rotate = config.get('transforms', {}).get('rotate', {})
    angles = rotate.get('fixed_angles', [])
    if not isinstance(angles, list) or not all(isinstance(a, (int, float)) for a in angles):
        raise ValueError("rotate.fixed_angles 必须是数字列表，例如 [90, 270]")
    jitter, copies = rotate.get('fixed_jitter', 0), rotate.get('fixed_jitter_copies', 0)
    if not isinstance(jitter, (int, float)) or jitter < 0:
        raise ValueError("rotate.fixed_jitter 必须是不小于 0 的数字")
    if not isinstance(copies, int) or copies < 0:
        raise ValueError("rotate.fixed_jitter_copies 必须是不小于 0 的整数")
    if angles and rotate.get('enabled', False) is not True:
        raise ValueError("配置了 rotate.fixed_angles，但 rotate.enabled 不是 true")

    transforms = config.get('transforms', {})
    valid_transforms = ['rotate', 'mask', 'distort', 'gaussian_noise',
                        'hsv_adjust', 'grayscale', 'brightness_boost']
    for name in transforms:
        if name not in valid_transforms:
            raise ValueError(f"未知变换类型: {name}")
