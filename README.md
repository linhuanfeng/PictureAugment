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
