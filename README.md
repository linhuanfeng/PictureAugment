# YOLO 数据增强工具

## 安装

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 目录结构

```
data/
├── images/
│   ├── train/     (训练集图像, 约80%)
│   ├── val/       (验证集图像, 约15%)
│   └── test/      (测试集图像, 约5%)
└── labels/
    ├── train/     (对应的TXT标注文件)
    ├── val/
    └── test/
```

## 命令

### 1. 数据增强

```bash
python run.py augment -c config.yaml
```

参数：

- `-c, --config`: 配置文件路径

#### 必转角度（rotate.fixed_angles）

`rotate` 的常规随机旋转只是"以概率 p 在 angle_range 里随机转一下"，不保证出现 90° 这类关键角度。需要保证时配置：

```yaml
rotate:
  enabled: true
  angle_range: [-30, 30]
  p: 0.5
  fixed_angles: [90, 270]   # 每张图、每个角度都必转（逆时针为正）
  fixed_jitter: 15          # 在必转角度附近随机偏移 ±15 度
  fixed_jitter_copies: 2    # 每个必转角度额外生成几张带偏移的图
```

- 每张图在 `num_augments` 张常规增强之外，额外输出 `len(fixed_angles) × (1 + fixed_jitter_copies)` 张。上面的配置是 2 × 3 = 6 张。
- 文件名带角度标记，负号写作 `n`：`xxx_rot90.jpg`（精确角度）、`xxx_rot90_j0.jpg`（附近偏移）、`xxx_rotn30.jpg`。
- 必转图的旋转角度固定，其余变换（遮挡、畸变、噪声、颜色等）仍按各自概率应用。
- 不需要时删掉这三项，或把 `fixed_angles` 设为 `[]`，行为和原来一致。
- 旋转后标注框取旋转框的外接矩形，非 90° 倍数的角度框会偏松。

### 2. 可视化查看

```bash
python run.py view -d <数据根目录> -s <数据集>
```

参数：

- `-d, --data`: 数据根目录 (默认: ./data)
- `-s, --split`: 数据集 train/val/test (默认: train)

示例：

```bash
# 查看训练集
python run.py view -s train

# 查看验证集
python run.py view -s val

# 查看测试集
python run.py view -s test

# 查看增强后的数据
python run.py view -d ./data/augmented -s train
```

键盘操作：

| 按键    | 功能       |
| ------- | ---------- |
| `←`  | 上一张图片 |
| `→`  | 下一张图片 |
| `ESC` | 退出       |
